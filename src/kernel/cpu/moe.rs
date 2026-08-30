//! CPU MoE:router logits/top-k reference 与专家矩阵计算,不包含批次或 scatter 编排。
//!
//! 路由算法(DeepSeek noaux_tc,GLM-5.2 / MiniMax-M3 共用):
//! 1. 每 expert 算 `logit = input · weight[expert]`
//! 2. `raw = sigmoid(logit)`
//! 3. `corrected = raw + bias[expert]`
//! 4. 按 corrected 降序选 top_k(平局 expert id 升序)
//! 5. 归一化:`weight = raw / sum(raw_topk) × scaling_factor`
//!
//! 输出 `(expert_id, weight)` 对,长度 = top_k。

use crate::{
    kernel::cpu::{
        CpuTensor, blas,
        matmul::matmul,
        silu::{gelu_tanh_mul, silu_clamped_mul, silu_mul, situ_mul, swiglu_oai_mul},
    },
    moe::{
        Activation,
        routing::{route_sigmoid_bias_logits, route_softmax_logits, route_sqrt_softplus_bias_logits},
    },
    weight::{
        container::gguf::GgufMatrix,
        expert_source::GgufExpertWeights,
        format::nvfp4::{Nvfp4ExpertWeights, Nvfp4Matrix},
    },
};
use wide::f32x8;

pub use crate::moe::routing::Routing;

const SIMD_LANES: usize = 8;

/// 路由一个 token。
/// `input`:[hidden] `weight`:[num_experts, hidden] 行优先 `bias`:[num_experts]
fn router_logits(input: &[f32], weight: &[f32], num_experts: usize) -> Vec<f32> {
    let hidden = input.len();
    assert!(hidden.is_multiple_of(SIMD_LANES));

    let input_v: Vec<f32x8> = input.chunks_exact(SIMD_LANES).map(|chunk| f32x8::from(<[f32; SIMD_LANES]>::try_from(chunk).unwrap())).collect();

    let mut logits = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        let w_row = &weight[e * hidden..(e + 1) * hidden];
        let mut acc = f32x8::splat(0.0);
        for (i, wc) in w_row.chunks_exact(SIMD_LANES).enumerate() {
            let weight = f32x8::from(<[f32; SIMD_LANES]>::try_from(wc).unwrap());
            acc += input_v[i] * weight;
        }
        logits.push(acc.reduce_add());
    }
    logits
}

pub fn route_sigmoid_bias(input: &[f32], weight: &[f32], bias: &[f32], num_experts: usize, top_k: usize, scaling_factor: f32) -> Routing {
    route_sigmoid_bias_logits(&router_logits(input, weight, num_experts), bias, top_k, scaling_factor).expect("CPU router 参数应由 backend 校验")
}

pub fn route_softmax(input: &[f32], weight: &[f32], num_experts: usize, top_k: usize, scaling_factor: f32, normalize_selected: bool) -> Routing {
    route_softmax_logits(&router_logits(input, weight, num_experts), top_k, scaling_factor, normalize_selected).expect("CPU router 参数应由 backend 校验")
}

pub fn route_sqrt_softplus_bias(input: &[f32], weight: &[f32], bias: &[f32], num_experts: usize, top_k: usize, scaling_factor: f32) -> Routing {
    route_sqrt_softplus_bias_logits(&router_logits(input, weight, num_experts), bias, top_k, scaling_factor).expect("CPU router 参数应由 backend 校验")
}

fn matmul_batch(input: &CpuTensor, weight: &[f32], output_columns: usize) -> CpuTensor {
    let mut output = CpuTensor { data: vec![0.0; input.rows * output_columns], rows: input.rows, cols: output_columns };
    if !blas::sgemm_nt(input.rows, output_columns, input.cols, 1.0, &input.data, weight, &mut output.data) {
        matmul(&input.data, weight, input.rows, input.cols, output_columns, &mut output.data);
    }
    output
}

fn activate(gate: &CpuTensor, up: &CpuTensor, activation: &Activation) -> CpuTensor {
    let mut output = CpuTensor { data: vec![0.0; gate.data.len()], rows: gate.rows, cols: gate.cols };
    match activation {
        Activation::Silu => silu_mul(&gate.data, &up.data, &mut output.data),
        Activation::SiluClamped { limit } => silu_clamped_mul(&gate.data, &up.data, *limit, &mut output.data),
        Activation::Situ { beta, linear_beta } => situ_mul(&gate.data, &up.data, *beta, *linear_beta, &mut output.data),
        Activation::GeluTanh => gelu_tanh_mul(&gate.data, &up.data, &mut output.data),
        Activation::SwigluOai { alpha, limit } => swiglu_oai_mul(&gate.data, &up.data, *alpha, *limit, &mut output.data),
    }
    output
}

pub fn f32_expert_batch(input: &CpuTensor, gate_weight: &[f32], up_weight: &[f32], down_weight: &[f32], intermediate: usize, activation: &Activation) -> CpuTensor {
    let gate = matmul_batch(input, gate_weight, intermediate);
    let up = matmul_batch(input, up_weight, intermediate);
    let activated = activate(&gate, &up, activation);
    matmul_batch(&activated, down_weight, input.cols)
}

pub fn nvfp4_expert_batch(input: &CpuTensor, weights: &Nvfp4ExpertWeights, activation: &Activation) -> CpuTensor {
    let mut scratch = Vec::new();
    let gate = nvfp4_matmul_batch(input, &weights.gate, &mut scratch);
    let up = nvfp4_matmul_batch(input, &weights.up, &mut scratch);
    let activated = activate(&gate, &up, activation);
    nvfp4_matmul_batch(&activated, &weights.down, &mut scratch)
}

pub fn gguf_expert_batch(input: &CpuTensor, weights: &GgufExpertWeights, activation: &Activation) -> Result<CpuTensor, String> {
    let gate = gguf_matmul_batch(input, &weights.gate)?;
    let up = gguf_matmul_batch(input, &weights.up)?;
    let activated = activate(&gate, &up, activation);
    gguf_matmul_batch(&activated, &weights.down)
}

fn gguf_matmul_batch(input: &CpuTensor, weight: &GgufMatrix) -> Result<CpuTensor, String> {
    if input.cols != weight.columns {
        return Err(format!("GGUF expert input cols={}，weight=[{},{}]", input.cols, weight.rows, weight.columns));
    }
    let mut output = CpuTensor { data: vec![0.0; input.rows * weight.rows], rows: input.rows, cols: weight.rows };
    for (input, output) in input.data.chunks_exact(input.cols).zip(output.data.chunks_exact_mut(weight.rows)) {
        super::ggml_quant::matvec(weight.tensor_type.0, weight.bytes()?, weight.rows, weight.columns, input, output)?;
    }
    Ok(output)
}

fn nvfp4_matmul_batch(input: &CpuTensor, weight: &Nvfp4Matrix, scratch: &mut Vec<f32>) -> CpuTensor {
    let mut output = CpuTensor { data: vec![0.0; input.rows * weight.rows], rows: input.rows, cols: weight.rows };
    if input.rows == 1 {
        super::nvfp4::matvec_nvfp4_matrix(weight.codes(), weight.scales(), weight.global_scale, weight.rows, weight.cols, &input.data, &mut output.data).expect("已校验的 NVFP4 权重无法执行 GEMV");
        return output;
    }

    scratch.resize(weight.rows * weight.cols, 0.0);
    super::nvfp4::decode_nvfp4_matrix(weight.codes(), weight.scales(), weight.global_scale, weight.rows, weight.cols, scratch).expect("已校验的 NVFP4 权重无法解码");
    if !blas::sgemm_nt(input.rows, weight.rows, weight.cols, 1.0, &input.data, scratch, &mut output.data) {
        matmul(&input.data, scratch, input.rows, weight.cols, weight.rows, &mut output.data);
    }
    output
}
