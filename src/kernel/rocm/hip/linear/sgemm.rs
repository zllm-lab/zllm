use super::*;

/// F32 dense linear：`out[n,k] = x[n,m] · W[k,m]ᵀ`，host 切片进出。
/// 复用 `try_f32_gemv_resident_f32` 的确定性 F32 GEMV kernel（固定归约顺序，无原子），
/// 不再依赖 rocBLAS；`download_f32` 在活跃 compute stream 上同步回读，顺序安全。
pub fn try_sgemm_f32(device_id: i32, x: &[f32], w: &[f32], n: usize, m: usize, k: usize, out: &mut [f32]) -> Result<(), String> {
    let output_expected = n.checked_mul(k).ok_or_else(|| "matmul output 尺寸溢出".to_string())?;
    let expected_w = k.checked_mul(m).ok_or_else(|| "matmul weight 尺寸溢出".to_string())?;
    let expected_x = n.checked_mul(m).ok_or_else(|| "matmul input 尺寸溢出".to_string())?;
    if x.len() != expected_x {
        return Err(format!("HIP matmul input 长度={}，期望 {expected_x}", x.len()));
    }
    if w.len() != expected_w {
        return Err(format!("HIP matmul weight 长度={}，期望 {expected_w}", w.len()));
    }
    if out.len() < output_expected {
        return Err(format!("HIP matmul out 长度={}，期望至少 {output_expected}", out.len()));
    }
    if output_expected == 0 {
        return Ok(());
    }
    set_device(device_id)?;
    let input = DeviceBuffer::upload_f32(device_id, x)?;
    let weight = DeviceBuffer::upload_f32(device_id, w)?;
    let output = try_f32_gemv_resident_f32(device_id, &input, &weight, n, m, k)?;
    let values = output.download_f32(output_expected)?;
    out[..output_expected].copy_from_slice(&values);
    Ok(())
}

/// 同一数学的 device resident 版本；与 host 版共享同一 kernel，结果逐位一致。
pub fn try_sgemm_resident_f32(device_id: i32, x: &DeviceBuffer, w: &DeviceBuffer, n: usize, m: usize, k: usize) -> Result<DeviceBuffer, String> {
    if n == 0 || m == 0 || k == 0 {
        return Err("resident sgemm shape 为空".to_owned());
    }
    try_f32_gemv_resident_f32(device_id, x, w, n, m, k)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// n>8 覆盖 small-n kernel 的 8 行分块尾部，k/m 取非对齐值覆盖尾循环。
    #[test]
    fn rocm_sgemm_f32_matches_cpu_oracle() {
        if !super::super::super::is_hip_available() {
            eprintln!("[sgemm] 跳过：本机未检测到 ROCm 运行时");
            return;
        }
        let n = 21usize;
        let m = 257usize;
        let k = 64usize;
        let x = (0..n * m).map(|index| (index as f32 * 0.013).sin()).collect::<Vec<_>>();
        let w = (0..k * m).map(|index| (index as f32 * 0.019).cos()).collect::<Vec<_>>();
        let mut out = vec![0.0_f32; n * k];
        try_sgemm_f32(0, &x, &w, n, m, k, &mut out).expect("sgemm f32");
        for row in 0..n {
            for col in 0..k {
                let expected = x[row * m..(row + 1) * m].iter().zip(&w[col * m..(col + 1) * m]).map(|(&a, &b)| a * b).sum::<f32>();
                let actual = out[row * k + col];
                assert!((actual - expected).abs() <= expected.abs() * 1.0e-4 + 1.0e-4, "row={row} col={col} actual={actual} expected={expected}");
            }
        }
    }

    /// resident 与 host 版走同一 kernel，必须逐位一致。
    #[test]
    fn rocm_sgemm_resident_matches_host_bitwise() {
        if !super::super::super::is_hip_available() {
            eprintln!("[sgemm-resident] 跳过：本机未检测到 ROCm 运行时");
            return;
        }
        let n = 13usize;
        let m = 129usize;
        let k = 37usize;
        let x = (0..n * m).map(|index| (index as f32 * 0.017).sin()).collect::<Vec<_>>();
        let w = (0..k * m).map(|index| (index as f32 * 0.023).cos()).collect::<Vec<_>>();
        let mut host = vec![0.0_f32; n * k];
        try_sgemm_f32(0, &x, &w, n, m, k, &mut host).expect("sgemm f32 host");
        let input = DeviceBuffer::upload_f32(0, &x).expect("upload sgemm input");
        let weight = DeviceBuffer::upload_f32(0, &w).expect("upload sgemm weight");
        let resident = try_sgemm_resident_f32(0, &input, &weight, n, m, k).expect("sgemm resident");
        let device = resident.download_f32(n * k).expect("download sgemm resident");
        for (host, device) in host.iter().zip(&device) {
            assert_eq!(host.to_bits(), device.to_bits(), "host 与 resident sgemm 必须逐位一致");
        }
    }
}
