//! GLM-Dsa 单进程完整层链：8 卡执行 L0..L77，CPU/ROCm 在本进程内收口输出。

#![cfg(target_os = "linux")]

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::Value;

use super::super::protocol::{chat_prompt_glm52_with_template, request_reasoning_effort};
use super::*;
use crate::backend::Backend;
use crate::backend::cpu::{CpuContext, CpuWeight};
use crate::config::{Glm52NodeExecutionConfig, Glm52OutputBackend};
use crate::kernel::cpu::CpuTensor;
use crate::runtime::glm52::prepare_glm52_output_head_quantized;
use crate::runtime::glm52::rocm::{gather_embedding_rows, load_resident_embedding};
use crate::runtime::glm52::stage::{Glm52StageState, build_glm52_stage_states, build_pp_prefill_chunk, prepare_prefill_layers, run_glm52_stage_work_stateful};
use crate::runtime::session::{AtomicCounterU64, GenerationSummary, NodeCapabilities, RuntimeStatus};
use crate::server::node::{NodeBatchRequest, NodeBatchResult, NodeEngine};
use crate::tokenizer::Utf8StreamDecoder;

enum SingleOutputHead {
    Cpu(Glm52OutputHead<CpuWeight>),
    Rocm { context: RocmContext, head: Glm52OutputHead<RocmWeight> },
}

pub(super) struct Glm52SingleEngine {
    cfg: Glm52Config,
    mla: MlaSpec,
    contexts: Vec<RocmContext>,
    templates: Vec<Glm52StageState<RocmContext>>,
    weights: Arc<Glm52Weights>,
    rope: Arc<RopeTable>,
    tokenizer: Tokenizer,
    detokenizer: Detokenizer,
    output: SingleOutputHead,
    /// 首卡常驻 BF16 embedding 表;GGUF 源装载期一次上传。
    embedding_table: Option<std::sync::Arc<crate::kernel::rocm::hip::DeviceBuffer>>,
    max_seq_len: usize,
    prefill_chunk_size: usize,
    options: Glm52NodeExecutionConfig,
    capabilities: NodeCapabilities,
    runtime: Arc<Mutex<RuntimeStatus>>,
    compute_steps: Arc<AtomicCounterU64>,
}

impl Glm52SingleEngine {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn load(
        weights: Arc<Glm52Weights>,
        devices: Vec<i32>,
        layer_ends: Vec<usize>,
        tokenizer_path: &Path,
        max_seq_len: usize,
        options: Glm52NodeExecutionConfig,
        lm_head_quantization: crate::weight::LmHeadQuantization,
        runtime: Arc<Mutex<RuntimeStatus>>,
        compute_steps: Arc<AtomicCounterU64>,
    ) -> Result<Self, DynError> {
        if options.mtp || options.dspark_directory.is_some() || options.cooperative_expert_pairs || options.parallel_operator_pairs {
            return Err("GLM-Dsa 单进程首版要求关闭 MTP、DSpark 与双卡算子".into());
        }
        weights.validate_glm_dsa_gguf_types().map_err(|error| -> DynError { error.into() })?;
        let cfg = Glm52Config::standard();
        if devices.len() != layer_ends.len() || layer_ends.last().copied() != Some(cfg.layer_count - 1) {
            return Err(format!("单进程 devices={devices:?} 与 layer_ends={layer_ends:?} 未覆盖 L0..L{}", cfg.layer_count - 1).into());
        }
        let model = Glm52::standard();
        let mla = match &model.layer_spec(0)?.attention {
            AttentionSpec::Mla(spec) => spec.clone(),
            _ => return Err("GLM-Dsa L0 不是 MLA".into()),
        };
        let contexts = crate::runtime::rocm_chain::RocmDeviceChain::new(&devices, layer_ends.clone(), cfg.layer_count - 1, false).map_err(|error| -> DynError { format!("GLM-Dsa 单进程设备链: {error}").into() })?.contexts;
        let layers = prepare_prefill_layers(&contexts, &layer_ends, 0, cfg.layer_count, &cfg, &mla, &weights, crate::kernel::rocm::hip::options().prefill_attention_cpu)
            .map_err(|error| -> DynError { format!("准备 GLM-Dsa 完整层链: {error:?}").into() })?;
        let mut experts = (0..contexts.len())
            .map(|_| {
                if weights.source_is_gguf() {
                    weights.gguf_source().map(|source| RocmPrefillExperts::gguf(source))
                } else if weights.source_is_ct() {
                    weights.ct_source().map(RocmPrefillExperts::ct)
                } else if let Some(source) = weights.nvfp4_experts() {
                    Ok(RocmPrefillExperts::nvfp4(source))
                } else {
                    RocmPrefillExperts::fp8(weights.cache_source_path(), cfg.expert_intermediate_size, cfg.hidden_size, cfg.expert_count)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        if options.preload_experts {
            let mut resident = vec![0usize; contexts.len()];
            let limit = options.preload_layers_per_device.unwrap_or(usize::MAX);
            for layer in cfg.dense_layer_count..cfg.layer_count {
                let device = layer_ends.iter().position(|&end| layer <= end).ok_or_else(|| format!("L{layer} 没有设备"))?;
                if resident[device] == limit {
                    continue;
                }
                contexts[device].activate()?;
                experts[device].preload_layer(&contexts[device], layer, cfg.expert_count).map_err(|error| format!("预载 L{layer} experts: {error:?}"))?;
                resident[device] += 1;
            }
        }
        let templates = build_glm52_stage_states(&contexts, &layer_ends, 0, &cfg, layers, experts, max_seq_len).map_err(|error| -> DynError { format!("创建 GLM-Dsa 单进程 stage: {error:?}").into() })?;
        let rope = Arc::new(RopeTable::precompute(max_seq_len, mla.qk_rope_head_dim, mla.rope_theta));
        prepare_glm52_rope_resident(&contexts, &rope)?;
        let final_norm = weights.final_norm()?;
        let output = match options.output_backend {
            Glm52OutputBackend::Cpu => {
                let matrix = weights.gguf_lm_head().map_err(|error| -> DynError { format!("CPU LM head 要求 GGUF output.weight: {error}").into() })?;
                let head = prepare_glm52_output_head_quantized(&CpuContext, &cfg, &final_norm, LinearWeight::gguf(&matrix), lm_head_quantization).map_err(|error| -> DynError { format!("准备 CPU GLM output head: {error:?}").into() })?;
                eprintln!("[glm-dsa-output] backend=cpu format={} rows={} cols={}", matrix.tensor_type.name(), matrix.rows, matrix.columns);
                SingleOutputHead::Cpu(head)
            }
            Glm52OutputBackend::Rocm => {
                let context = *contexts.last().ok_or("GLM-Dsa 没有输出设备")?;
                let lm_head = weights.lm_head_bf16_bytes()?;
                let head =
                    prepare_glm52_output_head_quantized(&context, &cfg, &final_norm, LinearWeight::Bf16Bytes(&lm_head), lm_head_quantization).map_err(|error| -> DynError { format!("准备 ROCm GLM output head: {error:?}").into() })?;
                SingleOutputHead::Rocm { context, head }
            }
        };
        let tokenizer =
            if tokenizer_path.is_file() { Tokenizer::new(tokenizer_path).map_err(|error| -> DynError { error.to_string().into() })? } else { weights.gguf_source()?.tokenizer().map_err(|error| -> DynError { error.to_string().into() })? };
        let detokenizer = if tokenizer_path.is_file() {
            Detokenizer::load(tokenizer_path).map_err(|error| -> DynError { error.to_string().into() })?
        } else {
            weights.gguf_source()?.detokenizer().map_err(|error| -> DynError { error.to_string().into() })?
        };
        let capabilities = crate::runtime::rocm_chain::node_capabilities(&contexts, max_seq_len, "cpu-q8g64", weights.cache_quantization(), 0);
        eprintln!("[glm-dsa-single] devices={devices:?} layer_ends={layer_ends:?} layers=0..{}", cfg.layer_count);
        let embedding_table = load_resident_embedding(&contexts[0], &weights, &cfg).map_err(|error| -> DynError { error.into() })?;
        Ok(Self { cfg, mla, contexts, templates, weights, rope, tokenizer, detokenizer, output, embedding_table, max_seq_len, prefill_chunk_size: options.prefill_chunk_size, options, capabilities, runtime, compute_steps })
    }

    fn fresh_states(&self) -> Result<Vec<Glm52StageState<RocmContext>>, String> {
        self.templates.iter().map(|state| state.fresh_session(&self.cfg, self.max_seq_len).map_err(|error| format!("创建单进程 session: {error:?}"))).collect()
    }

    fn sample(&self, hidden: &RocmTensor, sampling: crate::backend::TokenSampling) -> Result<u32, String> {
        match &self.output {
            SingleOutputHead::Cpu(head) => {
                let context = *self.contexts.last().ok_or("缺少末卡")?;
                let data = context.tensor_to_f32(hidden).map_err(|error| format!("下载 CPU head hidden: {error:?}"))?;
                let tensor = CpuTensor { data, rows: hidden.rows, cols: hidden.cols };
                crate::runtime::glm52::glm52_sampled_token_ids(&CpuContext, &self.cfg, head, &tensor, &[sampling]).map_err(|error| format!("CPU LM head: {error:?}"))?.into_iter().next().ok_or("CPU LM head 没有 token".to_owned())
            }
            SingleOutputHead::Rocm { context, head } => {
                crate::runtime::glm52::glm52_sampled_token_ids(context, &self.cfg, head, hidden, &[sampling]).map_err(|error| format!("ROCm LM head: {error:?}"))?.into_iter().next().ok_or("ROCm LM head 没有 token".to_owned())
            }
        }
    }

    fn generate(&mut self, request_id: &str, request: &Value, cancellation: &AtomicBool, on_token: &mut dyn FnMut(u32, String) -> bool) -> Result<GenerationSummary, String> {
        let effort = request_reasoning_effort(request, self.options.reasoning_effort)?;
        let prompt = chat_prompt_glm52_with_template(request, self.options.diagnostics.official_chat_template, effort)?;
        let tokens = self.tokenizer.tokenize(prompt.as_bytes());
        let max_tokens = crate::runtime::session::requested_completion_tokens(request).min(self.max_seq_len.saturating_sub(tokens.len()));
        if tokens.is_empty() || max_tokens == 0 {
            return Err(format!("GLM-Dsa prompt/max_tokens 非法: prompt={} max={max_tokens}", tokens.len()));
        }
        let temperature = request.get("temperature").and_then(Value::as_f64).unwrap_or(0.0) as f32;
        let top_p = request.get("top_p").and_then(Value::as_f64).unwrap_or(1.0) as f32;
        let seed = request.get("seed").and_then(Value::as_u64).unwrap_or_else(|| u64::from_le_bytes(blake3::hash(request_id.as_bytes()).as_bytes()[..8].try_into().unwrap()));
        let mut sampling = SamplingState::new(SamplingConfig { temperature, top_p, seed })?;
        let mut states = self.fresh_states()?;
        let mut last = None;
        for (chunk, ids) in tokens.chunks(self.prefill_chunk_size).enumerate() {
            let position = chunk * self.prefill_chunk_size;
            let hidden = match self.embedding_table.as_ref() {
                Some(table) => gather_embedding_rows(&self.contexts[0], table, ids, self.cfg.hidden_size).map_err(|error| format!("构造 prefill chunk: {error}"))?,
                None => build_pp_prefill_chunk(&self.contexts[0], &self.weights, &self.cfg, ids, position).map_err(|error| format!("构造 prefill chunk: {error:?}"))?,
            };
            let (next, output) = run_glm52_stage_work_stateful(states, position, hidden, false, &self.cfg, &self.mla, &self.rope).map_err(|error| format!("完整 prefill: {error:?}"))?;
            states = next;
            last = Some(self.contexts.last().unwrap().select_row(&output, ids.len() - 1).map_err(|error| format!("选择 prefill 末行: {error:?}"))?);
        }
        let mut next = self.sample(last.as_ref().ok_or("prefill 没有输出")?, sampling.next())?;
        let mut utf8 = Utf8StreamDecoder::default();
        let mut completion = 0usize;
        let mut finish_reason = "length".to_owned();
        while completion < max_tokens {
            if cancellation.load(Ordering::Acquire) {
                finish_reason = "cancelled".to_owned();
                break;
            }
            let text = utf8.push(&self.detokenizer.decode_bytes(&[next], true).map_err(|error| format!("detokenize: {error}"))?);
            completion += 1;
            self.compute_steps.fetch_add(1);
            if !on_token(next, text) {
                finish_reason = "cancelled".to_owned();
                break;
            }
            if self.cfg.eos_token_ids.contains(&next) {
                finish_reason = "stop".to_owned();
                break;
            }
            if completion == max_tokens {
                break;
            }
            let position = tokens.len() + completion - 1;
            let hidden = match self.embedding_table.as_ref() {
                Some(table) => gather_embedding_rows(&self.contexts[0], table, &[next], self.cfg.hidden_size).map_err(|error| format!("构造 decode embedding: {error}"))?,
                None => build_pp_prefill_chunk(&self.contexts[0], &self.weights, &self.cfg, &[next], position).map_err(|error| format!("构造 decode embedding: {error:?}"))?,
            };
            let (next_states, output) = run_glm52_stage_work_stateful(states, position, hidden, true, &self.cfg, &self.mla, &self.rope).map_err(|error| format!("完整 decode: {error:?}"))?;
            states = next_states;
            next = self.sample(&output, sampling.next())?;
        }
        let _ = utf8.finish();
        if let Ok(mut runtime) = self.runtime.lock() {
            runtime.kv_cache_entries = 0;
            runtime.kv_cache_tokens = 0;
        }
        Ok(GenerationSummary { finish_reason, prompt_tokens: tokens.len(), completion_tokens: completion, cache: None, tool_calls: Vec::new() })
    }
}

impl NodeEngine for Glm52SingleEngine {
    fn model_key(&self) -> &'static str {
        "glm-5.3"
    }

    fn startup_info(&self) -> (NodeCapabilities, Arc<dyn Fn() -> u64 + Send + Sync>) {
        (self.capabilities.clone(), Arc::new(|| 0))
    }

    fn terminal_cache_infos(&self) -> Vec<crate::kv_cache::terminal_cache::TerminalInfo> {
        Vec::new()
    }

    fn max_concurrency(&self) -> usize {
        1
    }

    fn generate_one(&mut self, request_id: &str, request: &Value, cancellation: &AtomicBool, on_token: &mut dyn FnMut(u32, String) -> bool) -> Result<GenerationSummary, String> {
        self.generate(request_id, request, cancellation, on_token)
    }

    fn generate_batch(
        &mut self,
        requests: Vec<NodeBatchRequest>,
        _intake: &mut dyn FnMut(usize) -> Vec<NodeBatchRequest>,
        on_token: &mut dyn FnMut(&str, u32, String) -> bool,
        _on_tool_call_delta: &mut dyn FnMut(&str, crate::runtime::session::ToolCallDelta) -> bool,
        _on_runtime_changed: &mut dyn FnMut(),
        _on_result: &mut dyn FnMut(NodeBatchResult),
    ) -> Vec<NodeBatchResult> {
        requests
            .into_iter()
            .map(|request| {
                let request_id = request.request_id;
                let result = self.generate(&request_id, &request.request, &request.cancellation, &mut |token, text| on_token(&request_id, token, text));
                NodeBatchResult { request_id, result }
            })
            .collect()
    }
}
