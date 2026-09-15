//! 标准权重格式的布局与 reference decode，不包含平台 kernel。

pub mod fp8;
pub mod ggml;
pub mod groupwise;
mod iq2s_grid;
mod iq3xxs_grid;
pub mod mxfp8;

/// 两种16位浮点格式等长，原址查表避免大权重逐元素重复转换；舍入沿用half。
pub fn f16_to_bf16_in_place(data: &mut [u8]) -> Result<(), String> {
    if !data.len().is_multiple_of(2) {
        return Err(format!("F16→BF16 字节数必须为偶数，实际 {}", data.len()));
    }
    static TABLE: std::sync::OnceLock<Box<[u16]>> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| (0..=u16::MAX).map(|bits| half::bf16::from_f32(half::f16::from_bits(bits).to_f32()).to_bits()).collect());
    for bytes in data.chunks_exact_mut(2) {
        let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
        bytes.copy_from_slice(&table[bits as usize].to_le_bytes());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn f16_conversion_matches_every_bit_pattern() {
        let mut bytes = (0..=u16::MAX).rev().flat_map(u16::to_le_bytes).collect::<Vec<_>>();
        super::f16_to_bf16_in_place(&mut bytes).unwrap();
        for (actual, bits) in bytes.chunks_exact(2).zip((0..=u16::MAX).rev()) {
            let expected = half::bf16::from_f32(half::f16::from_bits(bits).to_f32()).to_le_bytes();
            assert_eq!(actual, expected, "F16 bits={bits:04x}");
        }
        let mut odd = [1, 2, 3];
        assert!(super::f16_to_bf16_in_place(&mut odd).is_err());
        assert_eq!(odd, [1, 2, 3]);
    }
}
