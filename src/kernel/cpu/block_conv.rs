//! 分组动态 depthwise 卷积；相邻 block 的历史严格隔离。

#[allow(clippy::too_many_arguments)]
pub fn grouped_block_conv(input: &[f32], delta: &[f32], base: &[f32], rows: usize, columns: usize, block_size: usize, group_size: usize, taps: usize, side: usize) -> Result<Vec<f32>, String> {
    if columns == 0 || block_size == 0 || group_size == 0 || taps == 0 || taps > block_size || side > 1 || !columns.is_multiple_of(group_size) || !rows.is_multiple_of(block_size) {
        return Err(format!("grouped block conv 规格非法: rows={rows} columns={columns} block={block_size} group={group_size} taps={taps} side={side}"));
    }
    let groups = columns / group_size;
    let product = |dimensions: &[usize]| dimensions.iter().try_fold(1usize, |n, d| n.checked_mul(*d)).ok_or("grouped block conv shape 溢出");
    let input_size = product(&[rows, columns])?;
    let delta_size = product(&[rows, 2, taps, groups])?;
    let base_size = product(&[2, taps, columns])?;
    if input.len() != input_size || delta.len() != delta_size || base.len() != base_size {
        return Err(format!("grouped block conv shape: input={}/{} delta={}/{} base={}/{}", input.len(), input_size, delta.len(), delta_size, base.len(), base_size));
    }
    let mut output = vec![0.0; input.len()];
    for row in 0..rows {
        for column in 0..columns {
            for tap in 0..taps.min(row % block_size + 1) {
                let coefficient = base[(side * taps + tap) * columns + column] + delta[((row * 2 + side) * taps + tap) * groups + column / group_size];
                output[row * columns + column] += coefficient * input[(row - tap) * columns + column];
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_sides_and_block_boundaries() {
        let input = [1., 2., 3., 4., 10., 20., 30., 40.];
        let base = [1., 2., 3., 4., 2., 3., 4., 5.];
        // 每行 input/output 各两 tap；同组共享动态项，但 base 按 channel 区分。
        let delta = [0., 0., 1., 1., 1., 2., 0., 0., 0., 0., 1., 1., 1., 2., 0., 0.];
        assert_eq!(grouped_block_conv(&input, &delta, &base, 4, 2, 2, 2, 2, 0).unwrap(), [1., 4., 11., 24., 10., 40., 110., 240.]);
        assert_eq!(grouped_block_conv(&input, &delta, &base, 4, 2, 2, 2, 2, 1).unwrap(), [3., 8., 10., 22., 30., 80., 100., 220.]);
    }

    #[test]
    fn rejects_partial_blocks_and_bad_shapes() {
        assert!(grouped_block_conv(&[1.], &[0.; 4], &[0.; 4], 1, 1, 2, 1, 2, 0).is_err());
        assert!(grouped_block_conv(&[1.; 4], &[0.; 8], &[0.; 8], 2, 2, 2, 3, 2, 0).is_err());
        assert!(grouped_block_conv(&[1.; 4], &[0.; 8], &[0.; 7], 2, 2, 2, 1, 2, 0).is_err());
    }
}
