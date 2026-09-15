//! CPU MoE 路由、专家执行与 decode 状态。

use rayon::prelude::*;

use crate::{
    backend::{BackendError, ExpertDecodeBackend, MoePrefillBackend, MoePrefillRouting, checked_elements, compute_error as compute},
    kernel::cpu::{
        CpuTensor,
        moe::{route_sigmoid_bias, route_softmax, route_sqrt_softplus_bias},
    },
    moe::{
        routing::{execute_routed_experts, route_sqrt_softplus_selected},
        topk_moe::{ScoringFunc, TopkMoeSpec},
    },
    weight::expert_source::{GgufExpertSource, Mxfp8ExpertSource, Nvfp4ExpertSource},
    weight::format::mxfp8::{Mxfp8Matrix, Mxfp8MatrixBufferMut, mxfp8_storage_lengths},
};

use super::context::{CpuContext, CpuWeight};

fn route_cpu(input: &[f32], weight: &[f32], bias: &[f32], spec: &TopkMoeSpec) -> crate::moe::routing::Routing {
    match spec.scoring_func {
        ScoringFunc::Softmax => route_softmax(input, weight, spec.num_experts, spec.top_k, spec.routed_scaling_factor, spec.normalize_selected),
        ScoringFunc::SigmoidBias => route_sigmoid_bias(input, weight, bias, spec.num_experts, spec.top_k, spec.routed_scaling_factor),
        ScoringFunc::SqrtSoftplusBias => route_sqrt_softplus_bias(input, weight, bias, spec.num_experts, spec.top_k, spec.routed_scaling_factor),
    }
}

impl MoePrefillBackend for CpuContext {
    type MoeAccumulator = CpuTensor;

    fn moe_route(&self, input: &CpuTensor, router_weight: &CpuWeight, router_bias: &CpuWeight, spec: &TopkMoeSpec) -> Result<MoePrefillRouting, BackendError> {
        self.moe_route_rows(input, router_weight, router_bias, None, None, spec)
    }

    fn moe_route_rows(&self, input: &CpuTensor, router_weight: &CpuWeight, router_bias: &CpuWeight, router_bias_vl: Option<&CpuWeight>, image_rows: Option<&[bool]>, spec: &TopkMoeSpec) -> Result<MoePrefillRouting, BackendError> {
        if router_weight.rows != spec.num_experts || router_weight.cols != input.cols || router_bias.data.len() != spec.num_experts {
            return Err(compute(format!("MoE router shape 异常: input=[{},{}], weight=[{},{}], bias={}, experts={}", input.rows, input.cols, router_weight.rows, router_weight.cols, router_bias.data.len(), spec.num_experts)));
        }
        if let (Some(bias_vl), Some(rows)) = (router_bias_vl, image_rows)
            && (bias_vl.data.len() != spec.num_experts || rows.len() != input.rows)
        {
            return Err(compute(format!("MoE VL 路由 shape 异常: bias_vl={}, mask={}, rows={}", bias_vl.data.len(), rows.len(), input.rows)));
        }
        let routes: Vec<_> = input
            .data
            .par_chunks_exact(input.cols)
            .enumerate()
            .map(|(row, values)| {
                let bias = match (router_bias_vl, image_rows) {
                    (Some(bias_vl), Some(rows)) if rows[row] => &bias_vl.data,
                    _ => &router_bias.data,
                };
                route_cpu(values, &router_weight.data, bias, spec)
            })
            .collect();
        let mut expert_ids = Vec::with_capacity(input.rows * spec.top_k);
        let mut weights = Vec::with_capacity(input.rows * spec.top_k);
        for routing in routes {
            expert_ids.extend(routing.experts);
            weights.extend(routing.weights);
        }
        Ok(MoePrefillRouting { expert_ids, weights, rows: input.rows, top_k: spec.top_k })
    }

    fn moe_route_selected(&self, input: &CpuTensor, router_weight: &CpuWeight, selected_experts: &[u32], spec: &TopkMoeSpec) -> Result<MoePrefillRouting, BackendError> {
        if spec.scoring_func != ScoringFunc::SqrtSoftplusBias {
            return Err(compute("固定专家路由只支持 sqrt-softplus"));
        }
        route_sqrt_softplus_selected(&input.data, input.rows, input.cols, &router_weight.data, spec.num_experts, selected_experts, spec.top_k, spec.routed_scaling_factor).map_err(compute)
    }

    fn moe_zeros(&self, rows: usize, cols: usize) -> Result<CpuTensor, BackendError> {
        Ok(CpuTensor { data: vec![0.0; checked_elements(rows, cols, "MoE output")?], rows, cols })
    }

    fn moe_gather_rows(&self, input: &CpuTensor, rows: &[u32]) -> Result<CpuTensor, BackendError> {
        let mut data = Vec::with_capacity(checked_elements(rows.len(), input.cols, "MoE gather")?);
        for &row in rows {
            let row = row as usize;
            if row >= input.rows {
                return Err(compute(format!("MoE gather row {row} 越界，rows={}", input.rows)));
            }
            data.extend_from_slice(&input.data[row * input.cols..(row + 1) * input.cols]);
        }
        Ok(CpuTensor { data, rows: rows.len(), cols: input.cols })
    }

    fn moe_gather_rows_batch(&self, input: &CpuTensor, batches: &[Vec<u32>]) -> Result<Vec<CpuTensor>, BackendError> {
        batches.par_iter().map(|rows| self.moe_gather_rows(input, rows)).collect()
    }

    fn moe_scatter_add_rows(&self, output: &mut CpuTensor, input: &CpuTensor, rows: &[u32], weights: &[f32]) -> Result<(), BackendError> {
        if input.rows != rows.len() || rows.len() != weights.len() || input.cols != output.cols {
            return Err(compute(format!("MoE scatter shape 异常: output=[{},{}], input=[{},{}], rows={}, weights={}", output.rows, output.cols, input.rows, input.cols, rows.len(), weights.len())));
        }
        for (source_row, (&target_row, &weight)) in rows.iter().zip(weights).enumerate() {
            let target_row = target_row as usize;
            if target_row >= output.rows {
                return Err(compute(format!("MoE scatter row {target_row} 越界，rows={}", output.rows)));
            }
            let source = &input.data[source_row * input.cols..(source_row + 1) * input.cols];
            let target = &mut output.data[target_row * output.cols..(target_row + 1) * output.cols];
            for (target, source) in target.iter_mut().zip(source) {
                *target += source * weight;
            }
        }
        Ok(())
    }

    fn moe_scatter_add_rows_batch(&self, output: &mut CpuTensor, inputs: &[CpuTensor], rows: &[Vec<u32>], weights: &[Vec<f32>]) -> Result<(), BackendError> {
        if inputs.len() != rows.len() || rows.len() != weights.len() {
            return Err(compute(format!("MoE batch scatter 数量异常: inputs={}, rows={}, weights={}", inputs.len(), rows.len(), weights.len())));
        }

        let mut contributions = (0..output.rows).map(|_| Vec::new()).collect::<Vec<_>>();
        for (input_index, ((input, rows), weights)) in inputs.iter().zip(rows).zip(weights).enumerate() {
            if input.rows != rows.len() || rows.len() != weights.len() || input.cols != output.cols {
                return Err(compute(format!("MoE batch scatter shape 异常: output=[{},{}], input=[{},{}], rows={}, weights={}", output.rows, output.cols, input.rows, input.cols, rows.len(), weights.len())));
            }
            for (source_row, (&target_row, &weight)) in rows.iter().zip(weights).enumerate() {
                let target_row = target_row as usize;
                if target_row >= output.rows {
                    return Err(compute(format!("MoE batch scatter row {target_row} 越界，rows={}", output.rows)));
                }
                contributions[target_row].push((input_index, source_row, weight));
            }
        }

        let cols = output.cols;
        output.data.par_chunks_exact_mut(cols).zip(contributions).for_each(|(target, contributions)| {
            for (input_index, source_row, weight) in contributions {
                let input = &inputs[input_index];
                let source = &input.data[source_row * cols..(source_row + 1) * cols];
                for (target, source) in target.iter_mut().zip(source) {
                    *target += source * weight;
                }
            }
        });
        Ok(())
    }

    fn moe_finish(&self, output: CpuTensor) -> Result<CpuTensor, BackendError> {
        Ok(output)
    }
}

impl ExpertDecodeBackend for CpuContext {
    type MoeState = crate::moe::UncachedMoeState;
    type DecodeRouting = ();

    fn decode_route(&self, input: &CpuTensor, router_weight: &CpuWeight, router_bias: &CpuWeight, spec: &TopkMoeSpec) -> Result<(MoePrefillRouting, Self::DecodeRouting), BackendError> {
        Ok((self.moe_route(input, router_weight, router_bias, spec)?, ()))
    }

    fn decode_route_selected(&self, input: &CpuTensor, router_weight: &CpuWeight, selected_experts: &[u32], spec: &TopkMoeSpec) -> Result<(MoePrefillRouting, Self::DecodeRouting), BackendError> {
        Ok((self.moe_route_selected(input, router_weight, selected_experts, spec)?, ()))
    }

    fn prefetch_experts(&self, _spec: &TopkMoeSpec, _state: &mut Self::MoeState, _request: crate::backend::ExpertPrefetchRequest<'_>) -> Result<usize, BackendError> {
        Ok(0)
    }

    fn decode_routed_experts<'a, F>(
        &self,
        spec: &TopkMoeSpec,
        layer: usize,
        source: crate::weight::expert_source::ExpertSource<'_>,
        state: &mut Self::MoeState,
        input: &CpuTensor,
        assignments: &crate::moe::routing::ExpertAssignments,
        _routing: &Self::DecodeRouting,
        on_ready: F,
    ) -> Result<CpuTensor, BackendError>
    where
        F: FnOnce(&mut Self::MoeState) -> Result<Option<crate::backend::ExpertPrefetchRequest<'a>>, BackendError>,
    {
        let active = assignments.iter().filter(|rows| !rows.is_empty()).count();
        state.record_routed_experts(active);
        if let Some(request) = on_ready(state)? {
            self.prefetch_experts(spec, state, request)?;
        }
        match source {
            crate::weight::expert_source::ExpertSource::Fp8(source) => decode_f32_routed(self, spec, input, assignments, |expert| {
                let weights = source.load_expert_fp8(layer, expert).map_err(BackendError::ExpertLoad)?;
                Ok((weights.gate.decode(), weights.up.decode(), weights.down.decode()))
            }),
            crate::weight::expert_source::ExpertSource::Nvfp4(source) => decode_nvfp4_routed(self, spec, layer, source, input, assignments),
            crate::weight::expert_source::ExpertSource::Gguf(source) => decode_gguf_routed(self, spec, layer, source, input, assignments),
            crate::weight::expert_source::ExpertSource::Mxfp8(source) => decode_mxfp8_routed(self, spec, layer, source, input, assignments),
            crate::weight::expert_source::ExpertSource::Mxfp4(source) => {
                if source.hidden() != input.cols || source.intermediate() != spec.intermediate_size {
                    return Err(BackendError::ExpertLoad(format!("MXFP4 expert shape hidden={}/{} intermediate={}/{}", source.hidden(), input.cols, source.intermediate(), spec.intermediate_size,)));
                }
                execute_routed_experts(self, input, assignments, input.rows, input.cols, |expert, expert_input| {
                    let weights = source.load_expert_mxfp4(layer, expert).map_err(BackendError::ExpertLoad)?;
                    crate::kernel::cpu::moe::mxfp4_expert_batch(expert_input, &weights, &spec.activation).map_err(BackendError::ExpertLoad)
                })
            }
            crate::weight::expert_source::ExpertSource::W4A16(source) => decode_f32_routed(self, spec, input, assignments, |expert| {
                let weights = source.load_expert_w4a16(layer, expert).map_err(BackendError::ExpertLoad)?;
                let decode = |matrix: &crate::weight::format::quantization::W4A16Matrix| matrix.decode().map_err(BackendError::ExpertLoad);
                Ok((decode(&weights.gate)?, decode(&weights.up)?, decode(&weights.down)?))
            }),
        }
    }
}

/// fp8 / MXFP4 / W4A16 三种 routed expert decode 的共用骨架:执行流程完全相同,
/// 仅 expert 权重的 load 与解码不同,收敛为一个入口;load 闭包返回已解码的
/// (gate, up, down) f32 权重。
fn decode_f32_routed(
    ctx: &CpuContext,
    spec: &TopkMoeSpec,
    input: &CpuTensor,
    assignments: &crate::moe::routing::ExpertAssignments,
    load: impl Fn(usize) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), BackendError>,
) -> Result<CpuTensor, BackendError> {
    execute_routed_experts(ctx, input, assignments, input.rows, input.cols, |expert, expert_input| {
        let (gate, up, down) = load(expert)?;
        Ok(crate::kernel::cpu::moe::f32_expert_batch(expert_input, &gate, &up, &down, spec.intermediate_size, &spec.activation))
    })
}

fn decode_nvfp4_routed(ctx: &CpuContext, spec: &TopkMoeSpec, layer: usize, source: &dyn Nvfp4ExpertSource, input: &CpuTensor, assignments: &crate::moe::routing::ExpertAssignments) -> Result<CpuTensor, BackendError> {
    if source.hidden() != input.cols || source.intermediate() != spec.intermediate_size {
        return Err(BackendError::ExpertLoad(format!("NVFP4 expert shape hidden={}/{} intermediate={}/{}", source.hidden(), input.cols, source.intermediate(), spec.intermediate_size,)));
    }
    execute_routed_experts(ctx, input, assignments, input.rows, input.cols, |expert, expert_input| {
        let weights = source.load_expert_nvfp4(layer, expert).map_err(BackendError::ExpertLoad)?;
        Ok(crate::kernel::cpu::moe::nvfp4_expert_batch(expert_input, &weights, &spec.activation))
    })
}

fn decode_mxfp8_routed(ctx: &CpuContext, spec: &TopkMoeSpec, layer: usize, source: &dyn Mxfp8ExpertSource, input: &CpuTensor, assignments: &crate::moe::routing::ExpertAssignments) -> Result<CpuTensor, BackendError> {
    // Mxfp8ExpertSource 不暴露 shape,分配尺寸由 MoE spec 推导
    let hidden = input.cols;
    let intermediate = spec.intermediate_size;
    execute_routed_experts(ctx, input, assignments, input.rows, input.cols, |expert, expert_input| {
        let (gate, up, down) = load_mxfp8_expert(source, layer, expert, hidden, intermediate).map_err(BackendError::ExpertLoad)?;
        crate::kernel::cpu::moe::mxfp8_expert_batch(expert_input, &gate, &up, &down, &spec.activation).map_err(BackendError::ExpertLoad)
    })
}

/// 按 `load_expert_into` 契约分配 gate/up/down 三个矩阵并加载。
fn load_mxfp8_expert(source: &dyn Mxfp8ExpertSource, layer: usize, expert: usize, hidden: usize, intermediate: usize) -> Result<(Mxfp8Matrix, Mxfp8Matrix, Mxfp8Matrix), String> {
    let alloc = |rows: usize, cols: usize| -> Result<(Vec<u8>, Vec<u8>), String> {
        let (code_bytes, scale_bytes) = mxfp8_storage_lengths(rows, cols)?;
        Ok((vec![0; code_bytes], vec![0; scale_bytes]))
    };
    let mut gate = alloc(intermediate, hidden)?;
    let mut up = alloc(intermediate, hidden)?;
    let mut down = alloc(hidden, intermediate)?;
    source.load_expert_into(
        layer,
        expert,
        Mxfp8MatrixBufferMut { codes: &mut gate.0, scale_inv: &mut gate.1, rows: intermediate, cols: hidden },
        Mxfp8MatrixBufferMut { codes: &mut up.0, scale_inv: &mut up.1, rows: intermediate, cols: hidden },
        Mxfp8MatrixBufferMut { codes: &mut down.0, scale_inv: &mut down.1, rows: hidden, cols: intermediate },
    )?;
    Ok((Mxfp8Matrix::new(gate.0, gate.1, intermediate, hidden)?, Mxfp8Matrix::new(up.0, up.1, intermediate, hidden)?, Mxfp8Matrix::new(down.0, down.1, hidden, intermediate)?))
}

#[allow(clippy::too_many_arguments)]
fn decode_gguf_routed(ctx: &CpuContext, spec: &TopkMoeSpec, layer: usize, source: &dyn GgufExpertSource, input: &CpuTensor, assignments: &crate::moe::routing::ExpertAssignments) -> Result<CpuTensor, BackendError> {
    if source.hidden() != input.cols || source.intermediate() != spec.intermediate_size {
        return Err(BackendError::ExpertLoad(format!("GGUF expert shape hidden={}/{} intermediate={}/{}", source.hidden(), input.cols, source.intermediate(), spec.intermediate_size,)));
    }
    execute_routed_experts(ctx, input, assignments, input.rows, input.cols, |expert, expert_input| {
        let weights = source.load_expert_gguf(layer, expert).map_err(BackendError::ExpertLoad)?;
        crate::kernel::cpu::moe::gguf_expert_batch(expert_input, &weights, &spec.activation).map_err(BackendError::ExpertLoad)
    })
}

#[cfg(test)]
mod route_tests {
    use super::*;
    use crate::backend::{BackendResources, LinearWeight, MoePrefillBackend};

    fn dense(values: &[f32], rows: usize, cols: usize) -> CpuWeight {
        CpuContext::default().prepare_weight(LinearWeight::F32(values), rows, cols).unwrap()
    }

    /// DeepSeek-V4.1 VL 双偏置:image 行按 bias_vl 选专家,文本行按 bias。
    #[test]
    fn moe_route_rows双偏置按行选bias() {
        let (rows, columns, experts) = (4usize, 16usize, 8usize);
        let spec = TopkMoeSpec { num_experts: experts, top_k: 2, num_shared_experts: 0, scoring_func: crate::moe::topk_moe::ScoringFunc::SqrtSoftplusBias, normalize_selected: true, routed_scaling_factor: 1.5, intermediate_size: 4, shared_intermediate_size: 0, activation: crate::moe::Activation::Silu };
        let row: Vec<f32> = (0..columns).map(|index| (index as f32 * 0.37).sin()).collect();
        let input = CpuTensor { data: row.repeat(rows), rows, cols: columns };
        // bias 把 top-1 强推到专家 0;bias_vl 强推到专家 7。
        let bias = vec![100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0];
        let bias_vl = vec![-100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0, 100.0];
        let router = dense(&(0..experts * columns).map(|index| (index as f32 * 0.11).cos()).collect::<Vec<_>>(), experts, columns);
        let bias_weight = dense(&bias, 1, experts);
        let bias_vl_weight = dense(&bias_vl, 1, experts);
        let mask = vec![false, true, false, true];

        let routing = CpuContext::default().moe_route_rows(&input, &router, &bias_weight, Some(&bias_vl_weight), Some(&mask), &spec).unwrap();
        for row in 0..rows {
            let first = routing.expert_ids[row * spec.top_k];
            assert_eq!(first, if mask[row] { 7 } else { 0 }, "row={row} 首选专家应由 {} 决定", if mask[row] { "bias_vl" } else { "bias" });
        }
        // 文本行为 None 时与 moe_route 逐位一致。
        let plain = CpuContext::default().moe_route(&input, &router, &bias_weight, &spec).unwrap();
        for row in [0usize, 2] {
            assert_eq!(routing.expert_ids[row * spec.top_k..(row + 1) * spec.top_k], plain.expert_ids[row * spec.top_k..(row + 1) * spec.top_k]);
        }
    }
}
