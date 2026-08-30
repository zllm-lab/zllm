//! 华为 NPU kernel 入口与跨平台正确性 oracle。

pub const Q5_K_BLOCK_ELEMENTS: usize = 256;
pub const Q5_K_BLOCK_BYTES: usize = 176;
pub const Q6_K_BLOCK_ELEMENTS: usize = 256;
pub const Q6_K_BLOCK_BYTES: usize = 210;
pub const Q8_0_BLOCK_ELEMENTS: usize = 32;
pub const Q8_0_BLOCK_BYTES: usize = 34;

/// GGUF K-quant 通用布局校验：packed 权重直接下发设备，不在 host 展开权重。
/// 三种格式（Q5_K/Q6_K/Q8_0）共享相同的 AIV decode → Cube tile 架构。
///
/// 该校验只覆盖 raw packed 布局（layout 0）。设备侧重排后的 decode 布局
/// kernel（q5/q6 的 compact/repacked/dense gemv）另有 batch=1、rows 为
/// output_tile(16) 倍数、columns 被 block_batch*256 整除的约束；这些 kernel
/// 无错误通道，约束不满足时静默返回且不写输出，装配期必须在上传侧完成
/// 同样的校验（见各 kernel Process 开头的注释）。
#[allow(clippy::too_many_arguments)]
fn validate_packed_layout(weight_bytes: usize, rows: usize, columns: usize, batch: usize, input_elements: usize, block_elements: usize, block_bytes: usize, kind: &str) -> Result<(), String> {
    if rows == 0 || columns == 0 || batch == 0 || batch > 4096 || !columns.is_multiple_of(block_elements) {
        return Err(format!("Huawei {kind} shape [{batch},{rows},{columns}] 非法"));
    }
    // decode kernel（q5_k/q5_decode_tiles/q6_decode_tiles/q8_decode_tiles）按
    // 16x16 Cube tile 寻址，rows 不是 16 的整数倍时尾部行会写错 tile。
    if !rows.is_multiple_of(16) {
        return Err(format!("Huawei {kind} shape [{batch},{rows},{columns}] 非法：rows={rows} 不是 16 的整数倍，decode kernel 按 16x16 Cube tile 寻址"));
    }
    let expected = rows.checked_mul(columns / block_elements).and_then(|blocks| blocks.checked_mul(block_bytes)).ok_or_else(|| format!("Huawei {kind} 权重大小溢出"))?;
    if weight_bytes != expected || input_elements != batch * columns {
        return Err(format!("Huawei {kind} 数据不匹配: weight={weight_bytes}/{expected} input={input_elements}/{}", batch * columns));
    }
    Ok(())
}

pub fn validate_q5_k_layout(weight_bytes: usize, rows: usize, columns: usize, batch: usize, input_elements: usize) -> Result<(), String> {
    validate_packed_layout(weight_bytes, rows, columns, batch, input_elements, Q5_K_BLOCK_ELEMENTS, Q5_K_BLOCK_BYTES, "Q5_K")
}

pub fn validate_q6_k_layout(weight_bytes: usize, rows: usize, columns: usize, batch: usize, input_elements: usize) -> Result<(), String> {
    validate_packed_layout(weight_bytes, rows, columns, batch, input_elements, Q6_K_BLOCK_ELEMENTS, Q6_K_BLOCK_BYTES, "Q6_K")
}

pub fn validate_q8_0_layout(weight_bytes: usize, rows: usize, columns: usize, batch: usize, input_elements: usize) -> Result<(), String> {
    validate_packed_layout(weight_bytes, rows, columns, batch, input_elements, Q8_0_BLOCK_ELEMENTS, Q8_0_BLOCK_BYTES, "Q8_0")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extract_last_half_reference(input: &[u16], rows: usize, columns: usize) -> Vec<u16> {
        let output_columns = columns / 2;
        input.chunks_exact(columns).flat_map(|row| row[output_columns..].iter().copied()).take(rows * output_columns).collect()
    }

    #[test]
    fn extracts_gdn_output_from_qkv_tail() {
        let input: Vec<u16> = (0..32).collect();
        assert_eq!(extract_last_half_reference(&input, 4, 8), vec![4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31]);
    }

    #[test]
    fn validates_packed_k_quant_without_host_decode() {
        // Q5_K：256 元素 / 176 字节超块
        assert!(validate_q5_k_layout(Q5_K_BLOCK_BYTES * 16, 16, Q5_K_BLOCK_ELEMENTS, 1, Q5_K_BLOCK_ELEMENTS).is_ok());
        assert!(validate_q5_k_layout(Q5_K_BLOCK_BYTES * 16, 16, Q5_K_BLOCK_ELEMENTS, 15, 15 * Q5_K_BLOCK_ELEMENTS).is_ok());
        assert!(validate_q5_k_layout(Q5_K_BLOCK_BYTES, 16, 255, 1, 255).is_err());
        // decode kernel 按 16x16 tile 寻址：rows 必须是 16 的整数倍
        assert!(validate_q5_k_layout(Q5_K_BLOCK_BYTES * 2, 2, Q5_K_BLOCK_ELEMENTS, 1, Q5_K_BLOCK_ELEMENTS).is_err());

        // Q6_K：256 元素 / 210 字节超块
        assert!(validate_q6_k_layout(Q6_K_BLOCK_BYTES * 16, 16, Q6_K_BLOCK_ELEMENTS, 1, Q6_K_BLOCK_ELEMENTS).is_ok());
        assert!(validate_q6_k_layout(Q6_K_BLOCK_BYTES, 16, 255, 1, 255).is_err());
        assert!(validate_q6_k_layout(Q6_K_BLOCK_BYTES * 16, 16, Q6_K_BLOCK_ELEMENTS, 1024, 1024 * Q6_K_BLOCK_ELEMENTS).is_ok());

        // Q8_0：32 元素 / 34 字节块
        assert!(validate_q8_0_layout(Q8_0_BLOCK_BYTES * 16, 16, Q8_0_BLOCK_ELEMENTS, 1, Q8_0_BLOCK_ELEMENTS).is_ok());
        assert!(validate_q8_0_layout(Q8_0_BLOCK_BYTES, 16, 31, 1, 31).is_err());
        assert!(validate_q8_0_layout(Q8_0_BLOCK_BYTES * 16, 16, Q8_0_BLOCK_ELEMENTS, 1024, 1024 * Q8_0_BLOCK_ELEMENTS).is_ok());
    }
}
