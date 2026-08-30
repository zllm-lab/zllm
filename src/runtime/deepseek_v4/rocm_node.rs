//! DeepSeek-V4 ROCm 正式 node：只适配服务协议，执行体复用 rocm_engine。

use std::{
    cell::RefCell,
    collections::HashMap,
    sync::{Arc, Mutex, atomic::Ordering},
};

use serde_json::Value;

use crate::{
    config::{DeepSeekV4NodeModelConfig, RocmBackendConfig},
    kv_cache::terminal_cache::TerminalInfo as CacheInfo,
    runtime::session::{AtomicCounterU64, GenerationSummary, NodeCapabilities, RuntimeStatus as NodeRuntime, TerminalResume, ToolCall, ToolFunction, request_terminal_resume, terminal_cache_id},
    runtime::tool::{ParsedToolCall, ToolCallStream, ToolDialect, ToolOutput, dsml_tool_spec, request_tools, tool_argument_schemas, tool_call_id},
    server::node::{DynError, NodeBatchRequest, NodeBatchResult, NodeConfig, NodeEngine, run_node, text_content},
};

use super::{
    DeepSeekV4ChatMessage, deepseek_v4_chat_prompt, deepseek_v4_resume_suffix,
    rocm_engine::{DEEPSEEK_V4_MAX_CONCURRENCY, RocmBatchRequest, RocmDeepSeekV4Engine, RocmDeepSeekV4Options},
};

pub async fn run(model: DeepSeekV4NodeModelConfig, backend: RocmBackendConfig, config: NodeConfig) -> Result<(), DynError> {
    let options = options(model, backend, config.cache_dir.clone(), config.terminal_cache_global_entries, config.terminal_cache_prefix_rounds, config.persist_kv_cache);
    let factory = Box::new(move |runtime, compute_steps| {
        let engine = RocmDeepSeekV4Engine::load(options).map_err(|error| -> DynError { error.into() })?;
        Ok(Box::new(DeepSeekV4NodeEngine { engine, runtime, compute_steps }) as Box<dyn NodeEngine>)
    });
    run_node(config, factory).await
}

fn options(model: DeepSeekV4NodeModelConfig, backend: RocmBackendConfig, cache_directory: std::path::PathBuf, terminal_cache_global_entries: usize, terminal_cache_prefix_rounds: usize, persist_kv_cache: bool) -> RocmDeepSeekV4Options {
    RocmDeepSeekV4Options {
        weights_directory: model.weights_directory,
        cache_directory,
        devices: backend.devices,
        layer_ends: model.layer_ends,
        max_sequence_length: model.max_sequence_length,
        core_cache_gib: model.execution.core_cache_gib,
        prefill_chunk_size: model.execution.prefill_chunk_size,
        decode_priority_prefill_chunk_size: model.execution.decode_priority_prefill_chunk_size,
        decode_priority_prefill_chunk_ceiling: model.execution.decode_priority_prefill_chunk_ceiling,
        device_pool_gib: model.execution.device_pool_gib,
        kv_reservation_page_tokens: model.execution.kv_reservation_page_tokens,
        memory_reserve_bytes: model.execution.memory_reserve_bytes,
        long_prefill_threshold_tokens: model.execution.long_prefill_threshold_tokens,
        long_prefill_chunk_size: model.execution.long_prefill_chunk_size,
        dspark: model.execution.dspark,
        dspark_draft_tokens: model.execution.dspark_draft_tokens,
        dspark_min_sessions: model.execution.dspark_min_sessions,
        dspark_confidence_threshold: model.execution.dspark_confidence_threshold,
        decode_batch_limit: model.execution.decode_batch_limit,
        terminal_cache_global_entries,
        terminal_cache_prefix_rounds,
        persist_kv_cache,
        score_expert_top_k: model.execution.score_expert_top_k,
        profile: model.execution.profile,
    }
}

struct DeepSeekV4NodeEngine {
    engine: RocmDeepSeekV4Engine,
    runtime: Arc<Mutex<NodeRuntime>>,
    compute_steps: Arc<AtomicCounterU64>,
}

impl NodeEngine for DeepSeekV4NodeEngine {
    fn model_key(&self) -> &'static str {
        "deepseek-v4-flash"
    }

    fn startup_info(&self) -> (NodeCapabilities, Arc<dyn Fn() -> u64 + Send + Sync>) {
        let mut capabilities = crate::runtime::node::text_capabilities(
            crate::runtime::node::DeviceDescriptor {
                backend: "rocm",
                accelerator: "AMD ROCm multi-GPU".to_owned(),
                compute_units: None,
                compute_unit_kind: "CU",
                memory_kind: "VRAM",
                unified_memory: false,
                system_memory_bytes: None,
                accelerator_memory_bytes: None,
                recommended_working_set_bytes: None,
            },
            crate::runtime::node::SessionDescriptor { model_format: "safetensors-mxfp4", model_bytes: 0, max_seq_len: self.engine.max_sequence_length(), kv_cache_format: "q8g64-csa", input_modalities: &["text"] },
        );
        capabilities.kv_cache_devices = self.engine.kv_cache_devices().to_vec();
        capabilities.kv_reservation_page_tokens = self.engine.kv_reservation_page_tokens();
        (capabilities, Arc::new(|| 0))
    }

    fn refresh_runtime(&self) {
        let infos = self.terminal_cache_infos();
        if let Ok(mut runtime) = self.runtime.lock() {
            crate::runtime::session::refresh_cache_runtime(&mut runtime, &infos, self.engine.max_sequence_length());
        }
    }
    fn terminal_cache_infos(&self) -> Vec<CacheInfo> {
        self.engine
            .terminal_cache_infos()
            .into_iter()
            .map(|info| CacheInfo {
                cache_id: info.cache_id,
                model_key: "deepseek-v4-flash".to_owned(),
                cache_format: "deepseek-v4-rocm-q8-csa-v1".to_owned(),
                last_layer: 42,
                prompt_tokens: info.prompt_tokens,
                bytes: info.bytes,
                modified_unix: info.modified_unix,
            })
            .collect()
    }
    fn max_concurrency(&self) -> usize {
        DEEPSEEK_V4_MAX_CONCURRENCY
    }

    fn shutdown(&mut self) -> Result<(), String> {
        self.engine.shutdown();
        Ok(())
    }

    fn generate_batch(
        &mut self,
        mut requests: Vec<NodeBatchRequest>,
        intake: &mut dyn FnMut(usize) -> Vec<NodeBatchRequest>,
        on_token: &mut dyn FnMut(&str, u32, String) -> bool,
        _on_tool_call_delta: &mut dyn FnMut(&str, crate::runtime::session::ToolCallDelta) -> bool,
        on_result: &mut dyn FnMut(NodeBatchResult),
    ) -> Vec<NodeBatchResult> {
        requests.extend(intake(DEEPSEEK_V4_MAX_CONCURRENCY.saturating_sub(requests.len())));
        let mut prepared = Vec::new();
        let request_values = RefCell::new(HashMap::new());
        let response_texts = RefCell::new(HashMap::<String, String>::new());
        let tool_streams = RefCell::new(HashMap::<String, ToolCallStream>::new());
        let parsed_tools = RefCell::new(HashMap::<String, Vec<ParsedToolCall>>::new());
        let on_token = RefCell::new(on_token);
        let on_result = RefCell::new(on_result);
        for request in requests {
            let request_id = request.request_id.clone();
            request_values.borrow_mut().insert(request_id.clone(), request.request.clone());
            match prepare_one(request) {
                Ok((request, stream)) => {
                    if let Some(stream) = stream {
                        tool_streams.borrow_mut().insert(request_id, stream);
                    }
                    prepared.push(request);
                }
                Err(message) => (on_result.borrow_mut())(NodeBatchResult { request_id, result: Err(message) }),
            }
        }
        let mut finish_generated = |engine: &mut RocmDeepSeekV4Engine, generated: super::rocm_engine::RocmBatchResult| {
            let request_id = generated.request_id;
            let result = match generated.result {
                Ok(generated) => {
                    if let Some(mut stream) = tool_streams.borrow_mut().remove(&request_id) {
                        for output in stream.finish() {
                            match output {
                                ToolOutput::Text(text) => {
                                    response_texts.borrow_mut().entry(request_id.clone()).or_default().push_str(&text);
                                    (on_token.borrow_mut())(&request_id, 0, text);
                                }
                                ToolOutput::ToolCall(call) => parsed_tools.borrow_mut().entry(request_id.clone()).or_default().push(call),
                            }
                        }
                    }
                    let tool_calls = parsed_tools.borrow_mut().remove(&request_id).unwrap_or_default().into_iter().enumerate().map(|(index, call)| deepseek_tool_call(&request_id, index, call)).collect::<Vec<_>>();
                    let cache = if !matches!(generated.finish_reason.as_str(), "cancelled" | "repetition") {
                        let request = request_values.borrow_mut().remove(&request_id).ok_or_else(|| format!("DeepSeek request={request_id} 缺少原始请求"));
                        let response = response_texts.borrow_mut().remove(&request_id).unwrap_or_default();
                        request.and_then(|request| terminal_cache_id(&request, &response, &tool_calls)).map(|cache_id| {
                            engine.commit_cache(&request_id, cache_id).map(|info| CacheInfo {
                                cache_id: info.cache_id,
                                model_key: "deepseek-v4-flash".to_owned(),
                                cache_format: "deepseek-v4-rocm-q8-csa-v1".to_owned(),
                                last_layer: 42,
                                prompt_tokens: info.prompt_tokens,
                                bytes: info.bytes,
                                modified_unix: info.modified_unix,
                            })
                        })
                    } else {
                        engine.discard_pending(&request_id);
                        Ok(None)
                    };
                    match cache {
                        Ok(cache) => {
                            self.compute_steps.fetch_add(1);
                            let finish_reason = if tool_calls.is_empty() { generated.finish_reason } else { "tool_calls".to_owned() };
                            Ok(GenerationSummary { finish_reason, prompt_tokens: generated.prompt_tokens, completion_tokens: generated.completion_tokens, cache, tool_calls })
                        }
                        Err(error) => {
                            engine.discard_pending(&request_id);
                            Err(error)
                        }
                    }
                }
                Err(error) => {
                    engine.discard_pending(&request_id);
                    Err(error)
                }
            };
            (on_result.borrow_mut())(NodeBatchResult { request_id, result });
        };
        let generated = self.engine.generate_batch(
            prepared,
            &mut |available| {
                intake(available)
                    .into_iter()
                    .filter_map(|request| {
                        let request_id = request.request_id.clone();
                        request_values.borrow_mut().insert(request_id.clone(), request.request.clone());
                        match prepare_one(request) {
                            Ok((request, stream)) => {
                                if let Some(stream) = stream {
                                    tool_streams.borrow_mut().insert(request_id, stream);
                                }
                                Some(request)
                            }
                            Err(message) => {
                                (on_result.borrow_mut())(NodeBatchResult { request_id, result: Err(message) });
                                None
                            }
                        }
                    })
                    .collect()
            },
            &mut |request_id, token, text| {
                let output = tool_streams.borrow_mut().get_mut(request_id).map(|stream| stream.push(&text));
                let Some(output) = output else {
                    response_texts.borrow_mut().entry(request_id.to_owned()).or_default().push_str(&text);
                    return (on_token.borrow_mut())(request_id, token, text);
                };
                let mut running = true;
                for output in output {
                    match output {
                        ToolOutput::Text(text) => {
                            response_texts.borrow_mut().entry(request_id.to_owned()).or_default().push_str(&text);
                            running &= (on_token.borrow_mut())(request_id, token, text);
                        }
                        ToolOutput::ToolCall(call) => parsed_tools.borrow_mut().entry(request_id.to_owned()).or_default().push(call),
                    }
                }
                running
            },
            &mut finish_generated,
        );
        for generated in generated {
            finish_generated(&mut self.engine, generated);
        }
        self.refresh_runtime();
        Vec::new()
    }
}

fn prepare_one(input: NodeBatchRequest) -> Result<(RocmBatchRequest, Option<ToolCallStream>), String> {
    let request = &input.request;
    let messages = request.get("messages").and_then(Value::as_array).ok_or("messages 必须是数组")?;
    let tools = request_tools(request)?;
    let schemas = tool_argument_schemas(tools);
    let instructions = ToolDialect::DeepseekDsml.instructions(tools, request.get("tool_choice"))?;
    let tool_fence = dsml_tool_spec(tools, request.get("tool_choice"))?;
    let tool_stream = instructions.is_some().then(|| ToolDialect::DeepseekDsml.stream(schemas));
    let mut prompt_messages = Vec::<(String, String, Option<String>, String, Option<String>, Vec<String>)>::new();
    for message in messages {
        let role = message.get("role").and_then(Value::as_str).ok_or("message.role 必须是字符串")?;
        if !matches!(role, "developer" | "system" | "user" | "assistant" | "tool") {
            return Err(format!("DeepSeek-V4 不支持 message.role={role}"));
        }
        let content = text_content(message.get("content"))?;
        let reasoning_content = message.get("reasoning_content").and_then(Value::as_str).map(str::to_owned);
        let tool_calls = if role == "assistant" { ToolDialect::DeepseekDsml.render_history(message.get("tool_calls"))? } else { String::new() };
        let tool_call_ids = if role == "assistant" {
            message
                .get("tool_calls")
                .and_then(Value::as_array)
                .map(Vec::as_slice)
                .unwrap_or_default()
                .iter()
                .filter_map(|call| call.get("id").or_else(|| call.get("function").and_then(|function| function.get("id"))).and_then(Value::as_str).map(str::to_owned))
                .collect()
        } else {
            Vec::new()
        };
        let tool_call_id = (role == "tool").then(|| message.get("tool_call_id").and_then(Value::as_str).map(str::to_owned)).flatten();
        prompt_messages.push((role.to_owned(), content, reasoning_content, tool_calls, tool_call_id, tool_call_ids));
    }
    let thinking = match request.get("thinking") {
        None | Some(Value::Null) => None,
        Some(Value::Object(value)) => Some(value.get("type").and_then(Value::as_str).ok_or("thinking.type 必须是字符串")?),
        Some(_) => return Err("thinking 必须是对象".to_owned()),
    };
    let reasoning_effort = match request.get("reasoning_effort") {
        None | Some(Value::Null) => None,
        Some(Value::String(value)) => Some(value.as_str()),
        Some(_) => return Err("reasoning_effort 必须是字符串".to_owned()),
    };
    let prompt = deepseek_v4_chat_prompt(
        prompt_messages.iter().map(|(role, content, reasoning_content, tool_calls, tool_call_id, tool_call_ids)| DeepSeekV4ChatMessage {
            role,
            content,
            reasoning_content: reasoning_content.as_deref(),
            tool_calls,
            tool_call_id: tool_call_id.as_deref(),
            tool_call_ids,
        }),
        instructions.as_deref(),
        thinking,
        reasoning_effort,
    )?;
    let resume_suffix = match request_terminal_resume(request)? {
        TerminalResume::Match { assistant, .. } => {
            let assistant_prompt = deepseek_v4_chat_prompt(
                prompt_messages[..=assistant].iter().map(|(role, content, reasoning_content, tool_calls, tool_call_id, tool_call_ids)| DeepSeekV4ChatMessage {
                    role,
                    content,
                    reasoning_content: reasoning_content.as_deref(),
                    tool_calls,
                    tool_call_id: tool_call_id.as_deref(),
                    tool_call_ids,
                }),
                instructions.as_deref(),
                thinking,
                reasoning_effort,
            )?;
            Some(deepseek_v4_resume_suffix(&prompt, &assistant_prompt)?)
        }
        TerminalResume::None | TerminalResume::Mismatch { .. } => None,
    };
    let max_tokens = crate::runtime::session::requested_completion_tokens(request);
    if max_tokens == 0 {
        return Err("max_tokens 必须大于 0".to_owned());
    }
    if input.cancellation.load(Ordering::Acquire) {
        return Err("请求已取消".to_owned());
    }
    let repeat_loop_breaker = match request.get("repeat_loop_breaker") {
        None | Some(Value::Null) => true,
        Some(Value::Bool(enabled)) => *enabled,
        Some(_) => return Err("repeat_loop_breaker 必须是 bool".to_owned()),
    };
    Ok((
        RocmBatchRequest {
            request_id: input.request_id,
            prompt,
            requested_tokens: max_tokens,
            cancellation: input.cancellation,
            cache_id: request.get("cache_id").and_then(Value::as_str).map(str::to_owned),
            resume_suffix,
            cache_namespace: request.get("_zllm_cache_namespace").and_then(Value::as_str).map(str::to_owned),
            tool_fence,
            repeat_loop_breaker,
        },
        tool_stream,
    ))
}

fn deepseek_tool_call(scope: &str, index: usize, call: ParsedToolCall) -> ToolCall {
    ToolCall { id: tool_call_id(scope, index, &call.name), kind: "function".to_owned(), function: ToolFunction { name: call.name, arguments: call.arguments } }
}
