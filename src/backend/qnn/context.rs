use std::{
    path::{Path, PathBuf},
    sync::{
        Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

use crate::{
    attention::{gqa::GqaSpec, rope::RotaryLayout},
    backend::cpu::{CpuContext, CpuKvCache},
    backend::{Backend, BackendError, BackendResources, GqaPrefillBackend, LinearWeight},
    kernel::cpu::CpuTensor,
    moe::Activation,
};

use super::{QnnDualLinear, QnnLinear, QnnMlpGraph, QnnQuantWeight, QnnTripleLinear, quantize_linear_weight};

/// HTP decode + CPU reference/prefill 的共享上下文。
///
/// Linear 走 QNN 静态 W8A8-S32 图(输入逐行动态量化,输出 S32 精确还原),
/// 其余算子复用 CPU backend。prefill 的 Linear 同样逐行执行,后续融合图不改变模型 runtime。
pub struct QnnContext {
    cpu: CpuContext,
    backend: PathBuf,
    graph_capacity_available: AtomicBool,
    /// HTP 静态权重预算记账。SM8635 实测 ~800MB 后逐图 6031 abort;装载期
    /// 按层序累计 INT8 副本字节,超预算的层直接不量化,省 host 内存并避免
    /// finalize+abort 试错(大模型下试错一次就是一个权重一份 CPU 校准)。
    int8_resident_bytes: AtomicU64,
    /// GGUF 量化字节常驻总量,装载日志用。
    gguf_resident_bytes: AtomicU64,
}

/// HTP 静态权重预算上限,留 ~100MB 余量覆盖图结构与中间缓冲。
const HTP_INT8_BUDGET_BYTES: u64 = 700 * 1024 * 1024;
/// vocab 级输出头 F32 常驻的上限(K2 lm_head 394MB 在内,E4B 2.7GB 保持量化形态)。
const CPU_F32_HEAD_LIMIT_BYTES: u64 = 512 * 1024 * 1024;

pub struct QnnWeight {
    cpu: crate::backend::cpu::CpuWeight,
    q: Option<QnnQuantWeight>,
    decode: Mutex<Option<CachedLinear>>,
    dual_decode: Mutex<Option<CachedDualLinear>>,
    triple_decode: Mutex<Option<CachedTripleLinear>>,
    mlp: Mutex<Option<CachedMlp>>,
}

/// 图内烘焙输入 scale 常量;真实输入逐行动态量化后按比值校正。
const BAKED_INPUT_SCALE: f32 = 1.0;
/// 一次性校准的输出 scale:max/50 → 127/50 = 2.54× clamp 余量。
const OUTPUT_HEADROOM: f32 = 50.0;

/// 执行中止(status=6031,个别权重×输入组合触发 HTP 栈 bug)时该图降级 CPU,不再重试。
struct CachedLinear {
    graph: QnnLinear,
    output_scale: f32,
    degraded: bool,
}
struct CachedDualLinear {
    graph: QnnDualLinear,
    first_scale: f32,
    second_scale: f32,
    degraded: bool,
}
struct CachedTripleLinear {
    graph: QnnTripleLinear,
    output_scales: [f32; 3],
    degraded: bool,
}
struct CachedMlp {
    graph: QnnMlpGraph,
    degraded: bool,
}

fn is_execute_aborted(msg: &str) -> bool {
    msg.contains("QnnGraph_execute") && msg.contains("status=6031")
}

impl QnnContext {
    pub fn new(backend: impl AsRef<Path>) -> Result<Self, String> {
        Ok(Self { cpu: CpuContext, backend: backend.as_ref().to_owned(), graph_capacity_available: AtomicBool::new(true), int8_resident_bytes: AtomicU64::new(0), gguf_resident_bytes: AtomicU64::new(0) })
    }

    /// 装载统计：INT8 进 HTP 的字节数与 GGUF 常驻字节数。
    pub fn resident_stats(&self) -> (u64, u64) {
        (self.int8_resident_bytes.load(Ordering::Relaxed), self.gguf_resident_bytes.load(Ordering::Relaxed))
    }

    fn cpu_weight<'a>(&self, weight: &'a QnnWeight) -> &'a crate::backend::cpu::CpuWeight {
        &weight.cpu
    }
}

/// 图内量纲的输出 scale:真实输出 max ÷ 校准行输入 scale ÷ 余量。
/// 输出 clamp 容限随之与输入幅度同向缩放(比值校正),对漂移不敏感。
fn graph_output_scale(output: &[f32], input_scale: f32) -> f32 {
    ((output.iter().copied().map(f32::abs).fold(0.0_f32, f32::max) / input_scale) / OUTPUT_HEADROOM).max(1.0e-9)
}

/// 行动态量化 scale(与 super::quantize_row 一致):max/100。
fn row_scale(row: &[f32]) -> f32 {
    row.iter().copied().map(f32::abs).fold(0.0_f32, f32::max).max(1.0e-6) / 100.0
}

impl BackendResources for QnnContext {
    type Tensor = CpuTensor;
    type Weight = QnnWeight;
    type Cache = CpuKvCache;
    type LayerScope<'a>
        = ()
    where
        Self: 'a;

    fn layer_scope(&self) -> Self::LayerScope<'_> {}
    fn token_rows(&self, tensor: &Self::Tensor) -> usize {
        tensor.rows
    }
    fn token_cols(&self, tensor: &Self::Tensor) -> usize {
        tensor.cols
    }
    fn tensor_allocated_bytes(&self, tensor: &Self::Tensor) -> u64 {
        self.cpu.tensor_allocated_bytes(tensor)
    }
    fn begin_batch(&self) {}
    fn finish_batch(&self) {}

    fn prepare_weight(&self, weight: LinearWeight<'_>, rows: usize, cols: usize) -> Result<Self::Weight, BackendError> {
        // lm_head(vocab 级)且 F32 常驻量可控时走 F32 gemv 快路径(实测比 BF16 GGUF
        // 快 ~20×);超限(E4B 2.7GB 级)保持原始量化形态。INT8 进 HTP 受预算记账
        // 约束,超预算层纯 CPU。decoded 惰性求值:只有真的要量化或转 F32 头时才
        // dequant——decode 会把 GGUF 整表缓存进 OnceLock,大模型下预算外矩阵
        // 白做 decode 等于把全部权重再常驻一份(真机实测直接把 11GB 手机压死)。
        let elements = rows.checked_mul(cols).ok_or_else(|| BackendError::Compute { msg: "QNN 权重 shape 溢出".to_owned() })?;
        let element_bytes = u64::try_from(elements).map_err(|_| BackendError::Compute { msg: "QNN 权重 shape 溢出".to_owned() })?;
        let needs_int8 = rows > 1 && self.int8_resident_bytes.load(Ordering::Relaxed) + element_bytes <= HTP_INT8_BUDGET_BYTES;
        // 宽 FFN 也会超过 8192 行，不能仅凭行数把压缩权重误当输出头展开。
        // 仅为原本就是浮点的权重保留 F32 快路径，量化矩阵始终保留压缩形态。
        let floating_weight = match weight {
            LinearWeight::Quantized(crate::weight::format::quantization::QuantizedMatrixRef::Gguf(matrix)) => matches!(matrix.tensor_type.0, 0 | 1 | 30),
            LinearWeight::Quantized(_) => false,
            _ => true,
        };
        let needs_f32_head = floating_weight && rows > 8192 && element_bytes * 4 <= CPU_F32_HEAD_LIMIT_BYTES;
        let decoded = (needs_int8 || needs_f32_head)
            .then(|| match weight {
                LinearWeight::F32(values) => Ok(values.to_vec()),
                LinearWeight::F16(values) => Ok(values.iter().map(|value| value.to_f32()).collect()),
                LinearWeight::Bf16Bytes(values) => Ok(values.chunks_exact(2).map(|bytes| half::bf16::from_le_bytes([bytes[0], bytes[1]]).to_f32()).collect()),
                LinearWeight::Quantized(matrix) => matrix.decode().map_err(|msg| BackendError::Compute { msg }),
            })
            .transpose()?;
        let quantized = needs_int8
            .then(|| {
                let weight = quantize_linear_weight(decoded.as_deref().expect("needs_int8 时 decoded 已生成"), rows, cols, 8).map_err(|msg| BackendError::Compute { msg })?;
                self.int8_resident_bytes.fetch_add(element_bytes, Ordering::Relaxed);
                Ok(weight)
            })
            .transpose()?;
        let cpu = if needs_f32_head {
            self.cpu.prepare_f32(decoded.as_deref().expect("needs_f32_head 时 decoded 已生成"), rows, cols)?
        } else {
            let mut cpu = self.cpu.prepare_weight(weight, rows, cols)?;
            // GGUF 量化矩阵一次读入常驻:CPU 路径每次 matvec 都 read_bytes() 重读文件,
            // E4B 级模型每 chunk 15GB IO,prefill/decode 都不可行。常驻换 IO。
            let resident = cpu.make_gguf_resident().map_err(|msg| BackendError::Compute { msg })?;
            if resident > 0 {
                self.gguf_resident_bytes.fetch_add(resident as u64, Ordering::Relaxed);
            }
            cpu
        };
        Ok(QnnWeight { cpu, q: quantized, decode: Mutex::new(None), dual_decode: Mutex::new(None), triple_decode: Mutex::new(None), mlp: Mutex::new(None) })
    }

    fn prepare_f32(&self, values: &[f32], rows: usize, cols: usize) -> Result<Self::Weight, BackendError> {
        let cpu = self.cpu.prepare_f32(values, rows, cols)?;
        Ok(QnnWeight { cpu, q: None, decode: Mutex::new(None), dual_decode: Mutex::new(None), triple_decode: Mutex::new(None), mlp: Mutex::new(None) })
    }
}

impl Backend for QnnContext {
    fn linear(&self, input: &CpuTensor, weight: &QnnWeight) -> Result<CpuTensor, BackendError> {
        let Some(q) = &weight.q else {
            return self.cpu.linear(input, &weight.cpu);
        };
        if !self.graph_capacity_available.load(Ordering::Relaxed) {
            return self.cpu.linear(input, &weight.cpu);
        }
        let mut cached = weight.decode.lock().map_err(|_| BackendError::Compute { msg: "QNN Linear mutex poisoned".to_owned() })?;
        let calibrated = if cached.is_none() { Some(self.cpu.linear(input, &weight.cpu)?) } else { None };
        if cached.is_none() {
            let input_scale = row_scale(&input.data[(input.rows - 1) * input.cols..]);
            let output_scale = calibrated.as_ref().map_or(1.0, |output| graph_output_scale(&output.data, input_scale));
            match QnnLinear::new(&self.backend, 1, q, BAKED_INPUT_SCALE, output_scale) {
                Ok(graph) => *cached = Some(CachedLinear { graph, output_scale, degraded: false }),
                Err(msg) if msg.contains("status=6020") => {
                    self.graph_capacity_available.store(false, Ordering::Relaxed);
                    return calibrated.ok_or_else(|| BackendError::Compute { msg: "QNN 图容量降级缺少 CPU 输出".to_owned() });
                }
                Err(msg) => return Err(BackendError::Compute { msg }),
            }
        }
        if let Some(output) = calibrated {
            return Ok(output);
        }
        let cached = cached.as_mut().expect("QNN 图已创建");
        if cached.degraded {
            return self.cpu.linear(input, &weight.cpu);
        }
        let mut output = Vec::with_capacity(input.rows * q.output_columns);
        for row in input.data.chunks_exact(input.cols) {
            match cached.graph.execute_f32(row) {
                Ok(values) => output.extend(values),
                Err(msg) if is_execute_aborted(&msg) => {
                    eprintln!("QNN Linear 执行中止(K={}),该权重降级 CPU: {msg}", q.input_columns);
                    cached.degraded = true;
                    return self.cpu.linear(input, &weight.cpu);
                }
                Err(msg) => return Err(BackendError::Compute { msg }),
            }
        }
        Ok(CpuTensor { data: output, rows: input.rows, cols: q.output_columns })
    }

    fn dual_linear(&self, input: &CpuTensor, first: &QnnWeight, second: &QnnWeight) -> Result<(CpuTensor, CpuTensor), BackendError> {
        let (Some(first_q), Some(second_q)) = (&first.q, &second.q) else {
            return Ok((self.cpu.linear(input, &first.cpu)?, self.cpu.linear(input, &second.cpu)?));
        };
        if !self.graph_capacity_available.load(Ordering::Relaxed) {
            return Ok((self.cpu.linear(input, &first.cpu)?, self.cpu.linear(input, &second.cpu)?));
        }
        let mut cached = first.dual_decode.lock().map_err(|_| BackendError::Compute { msg: "QNN DualLinear mutex poisoned".to_owned() })?;
        let calibrated = if cached.is_none() { Some((self.cpu.linear(input, &first.cpu)?, self.cpu.linear(input, &second.cpu)?)) } else { None };
        if cached.is_none() {
            let input_scale = row_scale(&input.data[(input.rows - 1) * input.cols..]);
            let scales = calibrated.as_ref().map(|(first, second)| [graph_output_scale(&first.data, input_scale), graph_output_scale(&second.data, input_scale)]).unwrap_or([1.0, 1.0]);
            let first_scale = scales[0];
            let second_scale = scales[1];
            match QnnDualLinear::new(&self.backend, 1, first_q, second_q, BAKED_INPUT_SCALE, first_scale, second_scale) {
                Ok(graph) => *cached = Some(CachedDualLinear { graph, first_scale, second_scale, degraded: false }),
                Err(msg) if msg.contains("status=6020") => {
                    self.graph_capacity_available.store(false, Ordering::Relaxed);
                    return calibrated.ok_or_else(|| BackendError::Compute { msg: "QNN 图容量降级缺少 CPU 输出".to_owned() });
                }
                Err(msg) => return Err(BackendError::Compute { msg }),
            }
        }
        if let Some(outputs) = calibrated {
            return Ok(outputs);
        }
        let cached = cached.as_mut().expect("QNN dual 图已创建");
        if cached.degraded {
            return Ok((self.cpu.linear(input, &first.cpu)?, self.cpu.linear(input, &second.cpu)?));
        }
        let mut first_output = Vec::with_capacity(input.rows * first_q.output_columns);
        let mut second_output = Vec::with_capacity(input.rows * second_q.output_columns);
        for row in input.data.chunks_exact(input.cols) {
            match cached.graph.execute_f32(row) {
                Ok((first_values, second_values)) => {
                    first_output.extend(first_values);
                    second_output.extend(second_values);
                }
                Err(msg) if is_execute_aborted(&msg) => {
                    eprintln!("QNN DualLinear 执行中止(K={}),该权重降级 CPU: {msg}", first_q.input_columns);
                    cached.degraded = true;
                    return Ok((self.cpu.linear(input, &first.cpu)?, self.cpu.linear(input, &second.cpu)?));
                }
                Err(msg) => return Err(BackendError::Compute { msg }),
            }
        }
        Ok((CpuTensor { data: first_output, rows: input.rows, cols: first_q.output_columns }, CpuTensor { data: second_output, rows: input.rows, cols: second_q.output_columns }))
    }

    fn triple_linear(&self, input: &CpuTensor, first: &QnnWeight, second: &QnnWeight, third: &QnnWeight) -> Result<(CpuTensor, CpuTensor, CpuTensor), BackendError> {
        let (Some(a), Some(b), Some(c)) = (&first.q, &second.q, &third.q) else {
            return Ok((self.cpu.linear(input, &first.cpu)?, self.cpu.linear(input, &second.cpu)?, self.cpu.linear(input, &third.cpu)?));
        };
        if a.input_columns != b.input_columns || a.input_columns != c.input_columns || !self.graph_capacity_available.load(Ordering::Relaxed) {
            return Ok((self.cpu.linear(input, &first.cpu)?, self.cpu.linear(input, &second.cpu)?, self.cpu.linear(input, &third.cpu)?));
        }
        let mut cached = first.triple_decode.lock().map_err(|_| BackendError::Compute { msg: "QNN TripleLinear mutex poisoned".to_owned() })?;
        let calibrated = if cached.is_none() { Some((self.cpu.linear(input, &first.cpu)?, self.cpu.linear(input, &second.cpu)?, self.cpu.linear(input, &third.cpu)?)) } else { None };
        if cached.is_none() {
            let input_scale = row_scale(&input.data[(input.rows - 1) * input.cols..]);
            let output_scales =
                calibrated.as_ref().map(|(first, second, third)| [graph_output_scale(&first.data, input_scale), graph_output_scale(&second.data, input_scale), graph_output_scale(&third.data, input_scale)]).unwrap_or([1.0; 3]);
            match QnnTripleLinear::new(&self.backend, 1, [a, b, c], BAKED_INPUT_SCALE, output_scales) {
                Ok(graph) => *cached = Some(CachedTripleLinear { graph, output_scales, degraded: false }),
                Err(msg) if msg.contains("status=6020") => {
                    self.graph_capacity_available.store(false, Ordering::Relaxed);
                    return calibrated.ok_or_else(|| BackendError::Compute { msg: "QNN 图容量降级缺少 CPU 输出".to_owned() });
                }
                Err(msg) => return Err(BackendError::Compute { msg }),
            }
        }
        if let Some(outputs) = calibrated {
            return Ok(outputs);
        }
        let cached = cached.as_mut().expect("QNN triple 图已创建");
        if cached.degraded {
            return Ok((self.cpu.linear(input, &first.cpu)?, self.cpu.linear(input, &second.cpu)?, self.cpu.linear(input, &third.cpu)?));
        }
        let mut outputs = [Vec::with_capacity(input.rows * a.output_columns), Vec::with_capacity(input.rows * b.output_columns), Vec::with_capacity(input.rows * c.output_columns)];
        for row in input.data.chunks_exact(input.cols) {
            match cached.graph.execute_f32(row) {
                Ok(values) => {
                    for (output, values) in outputs.iter_mut().zip(values) {
                        output.extend(values);
                    }
                }
                Err(msg) if is_execute_aborted(&msg) => {
                    eprintln!("QNN TripleLinear 执行中止(K={}),该权重降级 CPU: {msg}", a.input_columns);
                    cached.degraded = true;
                    return Ok((self.cpu.linear(input, &first.cpu)?, self.cpu.linear(input, &second.cpu)?, self.cpu.linear(input, &third.cpu)?));
                }
                Err(msg) => return Err(BackendError::Compute { msg }),
            }
        }
        Ok((
            CpuTensor { data: std::mem::take(&mut outputs[0]), rows: input.rows, cols: a.output_columns },
            CpuTensor { data: std::mem::take(&mut outputs[1]), rows: input.rows, cols: b.output_columns },
            CpuTensor { data: std::mem::take(&mut outputs[2]), rows: input.rows, cols: c.output_columns },
        ))
    }

    fn rmsnorm(&self, input: &CpuTensor, weight: &QnnWeight, eps: f32) -> Result<CpuTensor, BackendError> {
        self.cpu.rmsnorm(input, self.cpu_weight(weight), eps)
    }
    fn gemma_rmsnorm(&self, input: &CpuTensor, weight: &QnnWeight, eps: f32) -> Result<CpuTensor, BackendError> {
        self.cpu.gemma_rmsnorm(input, self.cpu_weight(weight), eps)
    }
    fn layernorm_bias(&self, input: &CpuTensor, weight: &QnnWeight, bias: &QnnWeight, eps: f32) -> Result<CpuTensor, BackendError> {
        self.cpu.layernorm_bias(input, self.cpu_weight(weight), self.cpu_weight(bias), eps)
    }
    fn split_columns(&self, input: &CpuTensor, left: usize) -> Result<(CpuTensor, CpuTensor), BackendError> {
        self.cpu.split_columns(input, left)
    }
    fn split_interleaved_columns(&self, input: &CpuTensor, block: usize) -> Result<(CpuTensor, CpuTensor), BackendError> {
        self.cpu.split_interleaved_columns(input, block)
    }
    fn concat_columns(&self, left: &CpuTensor, right: &CpuTensor) -> Result<CpuTensor, BackendError> {
        self.cpu.concat_columns(left, right)
    }
    fn rope(&self, input: &CpuTensor, heads: usize, dim: usize, layout: RotaryLayout, position: usize, cos: &[f32], sin: &[f32]) -> Result<CpuTensor, BackendError> {
        self.cpu.rope(input, heads, dim, layout, position, cos, sin)
    }
    fn rope_prefix(&self, input: &CpuTensor, heads: usize, dim: usize, layout: RotaryLayout, position: usize, cos: &[f32], sin: &[f32]) -> Result<CpuTensor, BackendError> {
        self.cpu.rope_prefix(input, heads, dim, layout, position, cos, sin)
    }
    fn add(&self, left: &CpuTensor, right: &CpuTensor) -> Result<CpuTensor, BackendError> {
        self.cpu.add(left, right)
    }
    fn add_scaled(&self, left: &CpuTensor, right: &CpuTensor, scale: f32) -> Result<CpuTensor, BackendError> {
        self.cpu.add_scaled(left, right, scale)
    }
    fn sigmoid_gate(&self, input: &CpuTensor, gate: &CpuTensor) -> Result<CpuTensor, BackendError> {
        self.cpu.sigmoid_gate(input, gate)
    }
    fn select_row(&self, input: &CpuTensor, row: usize) -> Result<CpuTensor, BackendError> {
        self.cpu.select_row(input, row)
    }
    fn argmax(&self, input: &CpuTensor) -> Result<u32, BackendError> {
        self.cpu.argmax(input)
    }

    fn argmax_excluding(&self, input: &CpuTensor, excluded: &[u32]) -> Result<u32, BackendError> {
        self.cpu.argmax_excluding(input, excluded)
    }
    fn sample_top_p(&self, input: &CpuTensor, temperature: f32, top_p: f32, random: f32) -> Result<u32, BackendError> {
        self.cpu.sample_top_p(input, temperature, top_p, random)
    }
    fn gated_activation(&self, gate: &CpuTensor, up: &CpuTensor, activation: &Activation) -> Result<CpuTensor, BackendError> {
        self.cpu.gated_activation(gate, up, activation)
    }

    /// MLP 整段(gate/up GEMM + SiLU·mul + down GEMM)单图融合;首调 CPU 校准五级 scale。
    fn gated_mlp(&self, input: &CpuTensor, gate: &QnnWeight, up: &QnnWeight, down: &QnnWeight, activation: &Activation) -> Result<CpuTensor, BackendError> {
        if !matches!(activation, Activation::Silu) {
            return self.cpu.gated_mlp(input, &gate.cpu, &up.cpu, &down.cpu, activation);
        }
        let (Some(gq), Some(uq), Some(dq)) = (&gate.q, &up.q, &down.q) else {
            return self.cpu.gated_mlp(input, &gate.cpu, &up.cpu, &down.cpu, activation);
        };
        if !self.graph_capacity_available.load(Ordering::Relaxed) {
            return self.cpu.gated_mlp(input, &gate.cpu, &up.cpu, &down.cpu, activation);
        }
        let mut cached = gate.mlp.lock().map_err(|_| BackendError::Compute { msg: "QNN MLP mutex poisoned".to_owned() })?;
        let calibrated = if cached.is_none() {
            // CPU 参考链:gate/up → silu·mul → down,取各级 max(图内量纲,s_a 幂次)。
            let (gate_out, up_out) = (self.cpu.linear(input, &gate.cpu)?, self.cpu.linear(input, &up.cpu)?);
            let input_scale = row_scale(&input.data[(input.rows - 1) * input.cols..]);
            let mut silu_max = 0.0_f32;
            let mut act_max = 0.0_f32;
            let mut activated = vec![0.0_f32; gate_out.data.len()];
            for ((value, activated), up_value) in gate_out.data.iter().zip(&mut activated).zip(&up_out.data) {
                let silu = *value / (1.0 + (-value).exp());
                silu_max = silu_max.max(silu.abs());
                let product = silu * up_value;
                act_max = act_max.max(product.abs());
                *activated = product;
            }
            let activated_tensor = CpuTensor { data: activated, rows: input.rows, cols: gate_out.cols };
            let down_out = self.cpu.linear(&activated_tensor, &down.cpu)?;
            // 全链真实量纲校准(除以冻结输入 scale 还原真实值域),Sigmoid 才不饱和。
            let scale = |max: f32| (max / OUTPUT_HEADROOM).max(1.0e-9);
            let scales = [
                scale(gate_out.data.iter().copied().map(f32::abs).fold(0.0_f32, f32::max)),
                scale(up_out.data.iter().copied().map(f32::abs).fold(0.0_f32, f32::max)),
                scale(silu_max),
                scale(act_max),
                scale(down_out.data.iter().copied().map(f32::abs).fold(0.0_f32, f32::max)),
            ];
            Some((down_out, scales, input_scale))
        } else {
            None
        };
        if cached.is_none() {
            let (_, scales, input_scale) = calibrated.as_ref().expect("校准输出已生成");
            match QnnMlpGraph::new(&self.backend, 1, gq, uq, dq, *input_scale, scales[0], scales[1], scales[2], scales[3], scales[4]) {
                Ok(graph) => *cached = Some(CachedMlp { graph, degraded: false }),
                Err(msg) if msg.contains("status=6020") => {
                    self.graph_capacity_available.store(false, Ordering::Relaxed);
                    return calibrated.map(|(output, ..)| output).ok_or_else(|| BackendError::Compute { msg: "QNN 图容量降级缺少 CPU 输出".to_owned() });
                }
                Err(msg) => return Err(BackendError::Compute { msg }),
            }
        }
        if let Some((output, ..)) = calibrated {
            return Ok(output);
        }
        let cached = cached.as_mut().expect("QNN MLP 图已创建");
        if cached.degraded {
            return self.cpu.gated_mlp(input, &gate.cpu, &up.cpu, &down.cpu, activation);
        }
        let mut output = Vec::with_capacity(input.rows * dq.output_columns);
        for row in input.data.chunks_exact(input.cols) {
            match cached.graph.execute_f32(row) {
                Ok(values) => output.extend(values),
                Err(msg) if is_execute_aborted(&msg) => {
                    eprintln!("QNN MLP 执行中止,该 MLP 降级 CPU: {msg}");
                    cached.degraded = true;
                    return self.cpu.gated_mlp(input, &gate.cpu, &up.cpu, &down.cpu, activation);
                }
                Err(msg) => return Err(BackendError::Compute { msg }),
            }
        }
        Ok(CpuTensor { data: output, rows: input.rows, cols: dq.output_columns })
    }
}

impl GqaPrefillBackend for QnnContext {
    fn gemma_rmsnorm_heads(&self, input: &CpuTensor, weight: &QnnWeight, heads: usize, dim: usize, eps: f32) -> Result<CpuTensor, BackendError> {
        self.cpu.gemma_rmsnorm_heads(input, self.cpu_weight(weight), heads, dim, eps)
    }
    fn gqa_prefill_attention(&self, query: &CpuTensor, key: &CpuTensor, value: &CpuTensor, spec: &GqaSpec) -> Result<CpuTensor, BackendError> {
        self.cpu.gqa_prefill_attention(query, key, value, spec)
    }
    fn gqa_prefill_attention_cached(&self, cache: &mut CpuKvCache, layer: usize, position: usize, query: &CpuTensor, key: &CpuTensor, value: &CpuTensor, spec: &GqaSpec, retain: bool) -> Result<CpuTensor, BackendError> {
        self.cpu.gqa_prefill_attention_cached(cache, layer, position, query, key, value, spec, retain)
    }
    fn gqa_prefill_attention_cached_from(&self, cache: &CpuKvCache, layer: usize, position: usize, query: &CpuTensor, spec: &GqaSpec) -> Result<CpuTensor, BackendError> {
        self.cpu.gqa_prefill_attention_cached_from(cache, layer, position, query, spec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight::{container::gguf::GgufReader, format::quantization::QuantizedMatrixRef};

    #[test]
    fn excluded_tokens_are_not_selected() {
        let ctx = QnnContext::new("unused-in-cpu-test").unwrap();
        let logits = CpuTensor { data: vec![1.0, 9.0, 3.0], rows: 1, cols: 3 };
        assert_eq!(ctx.argmax_excluding(&logits, &[1]).unwrap(), 2);
        assert!(ctx.argmax_excluding(&logits, &[0, 1, 2]).is_err());
    }

    #[test]
    fn wide_quantized_matrix_stays_packed_after_htp_budget_is_exhausted() {
        // 超过旧输出头阈值的 Q4_K FFN，不能隐式常驻成 F32。
        let rows = 8193usize;
        let cols = 256usize;
        let name = b"wide.weight";
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(3u32.to_le_bytes());
        bytes.extend(1u64.to_le_bytes());
        bytes.extend(0u64.to_le_bytes());
        bytes.extend((name.len() as u64).to_le_bytes());
        bytes.extend(name);
        bytes.extend(2u32.to_le_bytes());
        bytes.extend((cols as u64).to_le_bytes());
        bytes.extend((rows as u64).to_le_bytes());
        bytes.extend(12u32.to_le_bytes());
        bytes.extend(0u64.to_le_bytes());
        bytes.resize(bytes.len().next_multiple_of(32) + rows * 144, 0);
        let path = std::env::temp_dir().join(format!("zllm-qnn-wide-{}.gguf", std::process::id()));
        std::fs::write(&path, bytes).unwrap();
        let reader = GgufReader::open(&path).unwrap();
        let matrix = reader.read_matrix("wide.weight").unwrap();
        let ctx = QnnContext::new("unused-in-cpu-test").unwrap();
        ctx.int8_resident_bytes.store(HTP_INT8_BUDGET_BYTES, Ordering::Relaxed);
        let weight = ctx.prepare_weight(LinearWeight::Quantized(QuantizedMatrixRef::Gguf(&matrix)), rows, cols).unwrap();
        assert!(weight.q.is_none());
        assert!(weight.cpu.data.is_empty());
        assert!(weight.cpu.gguf.is_some());
        let output = ctx.linear(&CpuTensor { data: vec![1.0; cols], rows: 1, cols }, &weight).unwrap();
        assert_eq!(output.data, vec![0.0; rows]);
        std::fs::remove_file(path).unwrap();
    }
}
