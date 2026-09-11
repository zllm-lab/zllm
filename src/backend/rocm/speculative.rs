//! ROCm 块卷积与候选链算子，所有中间张量留在设备上。
use super::*;
use crate::backend::BlockConvolutionBackend;

impl BlockConvolutionBackend for RocmContext {
    fn grouped_block_conv(&self, input: &RocmTensor, delta: &RocmTensor, base: &RocmWeight, block_size: usize, group_size: usize, taps: usize, side: usize) -> Result<RocmTensor, BackendError> {
        let delta_cols = input.cols.checked_div(group_size).and_then(|g| taps.checked_mul(2)?.checked_mul(g));
        if delta.rows != input.rows || Some(delta.cols) != delta_cols || Some(base.rows) != taps.checked_mul(2) || base.cols != input.cols {
            return Err(compute_error("ROCm block conv tensor shape 不兼容"));
        }
        let input = self.tensor_as_f32(input.clone())?;
        let delta = self.tensor_as_f32(delta.clone())?;
        let base_device = dense_device(base, "block conv base")?;
        let output = ops::hip::grouped_block_conv(self.device_id, resident(&input)?, resident(&delta)?, base_device, base.resident_bf16(), input.rows, input.cols, block_size, group_size, taps, side).map_err(compute_error)?;
        Ok(device_tensor_f32(output, input.rows, input.cols))
    }
}

impl RocmContext {
    pub fn candidate_greedy(&self, logits: &RocmTensor, gate: &RocmTensor, predecessor: &RocmWeight, successor: &RocmWeight, anchor: u32, top_k: usize) -> Result<Vec<u32>, BackendError> {
        if logits.rows != gate.rows || predecessor.rows != logits.cols || successor.rows != logits.cols || predecessor.cols != gate.cols || successor.cols != gate.cols {
            return Err(compute_error("ROCm candidate selector shape 不兼容"));
        }
        let logits = self.tensor_as_f32(logits.clone())?;
        let gate = self.tensor_as_f32(gate.clone())?;
        ops::hip::candidate_greedy(
            self.device_id,
            resident(&logits)?,
            resident(&gate)?,
            dense_device(predecessor, "predecessor")?,
            dense_device(successor, "successor")?,
            predecessor.resident_bf16(),
            successor.resident_bf16(),
            logits.rows,
            logits.cols,
            gate.cols,
            top_k,
            anchor,
        )
        .map_err(compute_error)
    }
}

fn resident(tensor: &RocmTensor) -> Result<&ops::hip::DeviceBuffer, BackendError> {
    tensor.device.as_deref().ok_or_else(|| compute_error("ROCm speculative tensor 缺少 device buffer"))
}
fn dense_device<'a>(weight: &'a RocmWeight, name: &str) -> Result<&'a ops::hip::DeviceBuffer, BackendError> {
    weight.resident().map(Arc::as_ref).ok_or_else(|| compute_error(format!("ROCm {name} 要求 dense resident 权重")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        backend::{BackendResources, cpu::CpuContext},
        kernel::cpu::{CpuTensor, block_conv},
        runtime::dflash2::cpu::greedy_selector,
    };

    #[test]
    fn dflash2_device_kernels_match_cpu() {
        let Ok(gpu) = RocmContext::new(0) else {
            eprintln!("ROCm 不可用，跳过 GPU oracle");
            return;
        };
        eprintln!("执行 ROCm device kernel oracle，device=0");
        let cpu = CpuContext;
        let values = |n: usize| (0..n).map(|i| ((i * 7 % 17) as f32 - 8.) / 8.).collect::<Vec<_>>();
        let x = values(48);
        let delta = values(48);
        let input = gpu.tensor_from_f32(x.clone(), 6, 8).unwrap();
        // 两个三行块、四通道一组、两侧各两 tap。
        let base = values(32);
        let dynamic = gpu.tensor_from_f32(delta.clone(), 6, 8).unwrap();
        for bf16 in [false, true] {
            let bytes: Vec<_> = base.iter().flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes()).collect();
            let weight = if bf16 { gpu.prepare_weight(crate::backend::LinearWeight::Bf16Bytes(&bytes), 4, 8).unwrap() } else { gpu.prepare_f32(&base, 4, 8).unwrap() };
            for side in 0..2 {
                let expected = block_conv::grouped_block_conv(&x, &delta, &base, 6, 8, 3, 4, 2, side).unwrap();
                let result = gpu.grouped_block_conv(&input, &dynamic, &weight, 3, 4, 2, side).unwrap();
                for (a, b) in gpu.tensor_to_f32(&result).unwrap().iter().zip(expected) {
                    assert!((a - b).abs() < 1e-5);
                }
            }
        }
        // 跨越多个 wave 的词表、非平凡前驱链，以及 BF16/F32 codebook。
        let vocab = 521;
        let rank = 8;
        let rows = 3;
        let pred = values(vocab * rank);
        let succ: Vec<_> = pred.iter().rev().copied().collect();
        let logits = CpuTensor { data: values(rows * vocab), rows, cols: vocab };
        let gate = CpuTensor { data: values(rows * rank), rows, cols: rank };
        let cp = cpu.prepare_f32(&pred, vocab, rank).unwrap();
        let cs = cpu.prepare_f32(&succ, vocab, rank).unwrap();
        let gl = gpu.tensor_from_f32(logits.data.clone(), rows, vocab).unwrap();
        let gg = gpu.tensor_from_f32(gate.data.clone(), rows, rank).unwrap();
        for bf16 in [false, true] {
            let prepare = |v: &[f32]| {
                let bytes: Vec<_> = v.iter().flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes()).collect();
                if bf16 { gpu.prepare_weight(crate::backend::LinearWeight::Bf16Bytes(&bytes), vocab, rank).unwrap() } else { gpu.prepare_f32(v, vocab, rank).unwrap() }
            };
            let gp = prepare(&pred);
            let gs = prepare(&succ);
            for k in [1, 16, 300] {
                assert_eq!(gpu.candidate_greedy(&gl, &gg, &gp, &gs, 7, k).unwrap(), greedy_selector(&logits, &gate, &cp, &cs, 7, k).unwrap());
            }
            for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let invalid = gpu.tensor_from_f32(vec![value; rows * vocab], rows, vocab).unwrap();
                assert!(gpu.candidate_greedy(&invalid, &gg, &gp, &gs, 7, 16).is_err());
            }
        }
    }
}
