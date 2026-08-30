use super::*;

pub fn try_add_f32(left: &[f32], right: &[f32], out: &mut [f32]) -> Result<(), String> {
    try_scale_add(left, right, 1.0, out)
}

pub fn try_add_scaled_f32(left: &[f32], right: &[f32], scale: f32, out: &mut [f32]) -> Result<(), String> {
    if !scale.is_finite() {
        return Err("add_scaled scale 非法".to_owned());
    }
    try_scale_add(left, right, scale, out)
}

fn try_scale_add(left: &[f32], right: &[f32], scale: f32, out: &mut [f32]) -> Result<(), String> {
    if left.len() != right.len() || left.len() != out.len() {
        return Err(format!("add 输入形状不一致: left={}, right={}, out={}", left.len(), right.len(), out.len()));
    }
    if left.is_empty() {
        return Ok(());
    }

    let device_id = current_device()?;
    let functions = tensor_functions(device_id)?;
    let bytes = std::mem::size_of_val(left);
    with_tensor_workspace(device_id, &[bytes, bytes, bytes], |workspace| {
        workspace.buffer(0).copy_from_host(unsafe { std::slice::from_raw_parts(left.as_ptr().cast(), bytes) })?;
        workspace.buffer(1).copy_from_host(unsafe { std::slice::from_raw_parts(right.as_ptr().cast(), bytes) })?;
        let mut d_left = workspace.buffer(0).pointer;
        let mut d_right = workspace.buffer(1).pointer;
        let mut d_output = workspace.buffer(2).pointer;
        let mut elements = u32::try_from(left.len()).map_err(|_| "add 长度超过 u32".to_owned())?;
        let mut scale = scale;
        let mut arguments = [(&mut d_left as *mut *mut c_void).cast(), (&mut d_right as *mut *mut c_void).cast(), (&mut d_output as *mut *mut c_void).cast(), (&mut elements as *mut u32).cast(), (&mut scale as *mut f32).cast()];
        launch_tensor_kernel(functions.add_scaled, elements.div_ceil(256), 256, &mut arguments, "HIP add_scaled")?;
        workspace.buffer(2).copy_to_host(unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast(), bytes) })
    })
}

pub fn try_sigmoid_gate_f32(input: &[f32], gate: &[f32], rows: usize, cols: usize, out: &mut [f32]) -> Result<(), String> {
    let expected = rows.checked_mul(cols).ok_or_else(|| "sigmoid_gate 元素数溢出".to_owned())?;
    if input.len() != expected || out.len() != expected {
        return Err(format!("sigmoid_gate input/output 长度不匹配: input={} out={} 预期={}", input.len(), out.len(), expected));
    }
    if gate.len() != rows && gate.len() != expected {
        return Err(format!("sigmoid_gate gate 长度={} 不匹配 rows={} cols={}", gate.len(), rows, cols));
    }
    if cols == 0 {
        return Ok(());
    }
    let gate_cols = if gate.len() == rows { 1 } else { cols };
    for row in 0..rows {
        let row_base = row * cols;
        let gate_base = if gate_cols == 1 { row } else { row_base };
        for col in 0..cols {
            let gate_value = gate[gate_base + if gate_cols == 1 { 0 } else { col }];
            let scale = 1.0 / (1.0 + (-gate_value).exp());
            out[row_base + col] = input[row_base + col] * scale;
        }
    }
    Ok(())
}

pub fn try_layernorm_bias_f32(input: &[f32], weight: &[f32], bias: &[f32], rows: usize, cols: usize, eps: f32, out: &mut [f32]) -> Result<(), String> {
    let expected = rows.checked_mul(cols).ok_or_else(|| "layernorm_bias 元素数溢出".to_owned())?;
    if input.len() != expected || out.len() != expected {
        return Err("layernorm_bias input/output 长度不匹配".to_owned());
    }
    if weight.len() != cols || bias.len() != cols {
        return Err("layernorm_bias weight/bias 长度不匹配".to_owned());
    }
    if rows == 0 || cols == 0 {
        return Ok(());
    }
    let device_id = current_device()?;
    let functions = tensor_functions(device_id)?;
    let input_bytes = std::mem::size_of_val(input);
    let parameter_bytes = std::mem::size_of_val(weight);
    with_tensor_workspace(device_id, &[input_bytes, parameter_bytes, parameter_bytes, input_bytes], |workspace| {
        workspace.buffer(0).copy_from_host(unsafe { std::slice::from_raw_parts(input.as_ptr().cast(), input_bytes) })?;
        workspace.buffer(1).copy_from_host(unsafe { std::slice::from_raw_parts(weight.as_ptr().cast(), parameter_bytes) })?;
        workspace.buffer(2).copy_from_host(unsafe { std::slice::from_raw_parts(bias.as_ptr().cast(), parameter_bytes) })?;
        let mut d_input = workspace.buffer(0).pointer;
        let mut d_weight = workspace.buffer(1).pointer;
        let mut d_bias = workspace.buffer(2).pointer;
        let mut d_output = workspace.buffer(3).pointer;
        let mut rows = u32::try_from(rows).map_err(|_| "layernorm rows 超过 u32".to_owned())?;
        let mut cols = u32::try_from(cols).map_err(|_| "layernorm cols 超过 u32".to_owned())?;
        let mut eps = eps;
        let mut arguments = [
            (&mut d_input as *mut *mut c_void).cast(),
            (&mut d_weight as *mut *mut c_void).cast(),
            (&mut d_bias as *mut *mut c_void).cast(),
            (&mut d_output as *mut *mut c_void).cast(),
            (&mut rows as *mut u32).cast(),
            (&mut cols as *mut u32).cast(),
            (&mut eps as *mut f32).cast(),
        ];
        launch_tensor_kernel(functions.layernorm_bias, rows, 256, &mut arguments, "HIP layernorm_bias")?;
        workspace.buffer(3).copy_to_host(unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast(), input_bytes) })
    })
}

pub fn try_split_columns_f32(input: &[f32], rows: usize, cols: usize, left_columns: usize) -> Result<(Vec<f32>, Vec<f32>), String> {
    let expected = rows.checked_mul(cols).ok_or_else(|| "split columns 元素数溢出".to_owned())?;
    if input.len() != expected {
        return Err(format!("split columns input 长度={}，期望 {}", input.len(), expected));
    }
    if left_columns > cols {
        return Err(format!("split left={left_columns} 超过 cols={cols}"));
    }
    let right_columns = cols - left_columns;
    let mut left = vec![0.0; rows * left_columns];
    let mut right = vec![0.0; rows * right_columns];
    if input.is_empty() {
        return Ok((left, right));
    }
    let device_id = current_device()?;
    let functions = tensor_functions(device_id)?;
    let input_bytes = std::mem::size_of_val(input);
    let left_bytes = std::mem::size_of_val(left.as_slice());
    let right_bytes = std::mem::size_of_val(right.as_slice());
    with_tensor_workspace(device_id, &[input_bytes, left_bytes, right_bytes], |workspace| {
        workspace.buffer(0).copy_from_host(unsafe { std::slice::from_raw_parts(input.as_ptr().cast(), input_bytes) })?;
        let mut d_input = workspace.buffer(0).pointer;
        let mut d_left = workspace.buffer(1).pointer;
        let mut d_right = workspace.buffer(2).pointer;
        let mut rows = u32::try_from(rows).map_err(|_| "split rows 超过 u32".to_owned())?;
        let mut cols = u32::try_from(cols).map_err(|_| "split cols 超过 u32".to_owned())?;
        let mut left_columns = u32::try_from(left_columns).map_err(|_| "split left_columns 超过 u32".to_owned())?;
        let mut arguments = [
            (&mut d_input as *mut *mut c_void).cast(),
            (&mut d_left as *mut *mut c_void).cast(),
            (&mut d_right as *mut *mut c_void).cast(),
            (&mut rows as *mut u32).cast(),
            (&mut cols as *mut u32).cast(),
            (&mut left_columns as *mut u32).cast(),
        ];
        let elements = rows.checked_mul(cols).ok_or("split elements 溢出")?;
        launch_tensor_kernel(functions.split_columns, elements.div_ceil(256), 256, &mut arguments, "HIP split_columns")?;
        workspace.buffer(1).copy_to_host(unsafe { std::slice::from_raw_parts_mut(left.as_mut_ptr().cast(), left_bytes) })?;
        workspace.buffer(2).copy_to_host(unsafe { std::slice::from_raw_parts_mut(right.as_mut_ptr().cast(), right_bytes) })
    })?;
    Ok((left, right))
}

pub fn try_split_interleaved_columns_f32(input: &[f32], rows: usize, cols: usize, block_columns: usize) -> Result<(Vec<f32>, Vec<f32>), String> {
    let expected = rows.checked_mul(cols).ok_or_else(|| "split_interleaved columns 元素数溢出".to_owned())?;
    if input.len() != expected {
        return Err(format!("split_interleaved input 长度={}，期望 {}", input.len(), expected));
    }
    let pair_columns = block_columns.checked_mul(2).ok_or_else(|| "split_interleaved block 溢出".to_owned())?;
    if block_columns == 0 || !cols.is_multiple_of(pair_columns) {
        return Err(format!("split_interleaved cols={cols} block={block_columns} 非法"));
    }
    let mut left = Vec::with_capacity(rows * cols / 2);
    let mut right = Vec::with_capacity(rows * cols / 2);
    for row in 0..rows {
        for pair in input[row * cols..(row + 1) * cols].chunks_exact(pair_columns) {
            left.extend_from_slice(&pair[..block_columns]);
            right.extend_from_slice(&pair[block_columns..]);
        }
    }
    Ok((left, right))
}

pub fn try_concat_columns_f32(left: &[f32], right: &[f32], rows: usize, left_cols: usize, right_cols: usize) -> Result<Vec<f32>, String> {
    let left_expected = rows.checked_mul(left_cols).ok_or_else(|| "concat left 元素数溢出".to_owned())?;
    if left.len() != left_expected {
        return Err(format!("concat left 长度={}，期望 {left_expected}", left.len()));
    }
    let right_expected = rows.checked_mul(right_cols).ok_or_else(|| "concat right 元素数溢出".to_owned())?;
    if right.len() != right_expected {
        return Err(format!("concat right 长度={}，期望 {right_expected}", right.len()));
    }
    if rows == 0 {
        return Ok(Vec::new());
    }
    let mut output = Vec::with_capacity(left.len() + right.len());
    left_cols.checked_add(right_cols).ok_or_else(|| "concat columns 溢出".to_owned())?;
    for row in 0..rows {
        let left_row = &left[row * left_cols..(row + 1) * left_cols];
        let right_row = &right[row * right_cols..(row + 1) * right_cols];
        output.extend_from_slice(left_row);
        output.extend_from_slice(right_row);
    }
    Ok(output)
}

pub fn try_select_row_f32(input: &[f32], rows: usize, cols: usize, row: usize) -> Result<Vec<f32>, String> {
    let expected = rows.checked_mul(cols).ok_or_else(|| "select_row 元素数溢出".to_owned())?;
    if input.len() != expected {
        return Err(format!("select_row input 长度={}，期望 {}", input.len(), expected));
    }
    if row >= rows {
        return Err(format!("select_row 行 {row} 越界，rows={rows}"));
    }
    if cols == 0 {
        return Ok(Vec::new());
    }
    let base = row * cols;
    Ok(input[base..base + cols].to_vec())
}

pub fn try_rope_f32(input: &[f32], rows: usize, cols: usize, head_count: usize, rotary_dim: usize, layout: RotaryLayout, position: usize, cos: &[f32], sin: &[f32], prefix: bool) -> Result<Vec<f32>, String> {
    if input.len() != rows.checked_mul(cols).ok_or("RoPE shape 溢出")? || head_count == 0 || rotary_dim == 0 || !rotary_dim.is_multiple_of(2) || !cols.is_multiple_of(head_count) || rotary_dim > cols / head_count || cos.len() != sin.len()
    {
        return Err(format!("ROCm RoPE 参数非法: shape=[{rows},{cols}] heads={head_count} rotary={rotary_dim}"));
    }
    let half = rotary_dim / 2;
    let table_begin = position.checked_mul(half).ok_or("RoPE table offset 溢出")?;
    let table_end = position.checked_add(rows).and_then(|end| end.checked_mul(half)).ok_or("RoPE table 长度溢出")?;
    if table_end > cos.len() {
        return Err(format!("RoPE table 长度 {}，至少需要 {table_end}", cos.len()));
    }
    let cosine = &cos[table_begin..table_end];
    let sine = &sin[table_begin..table_end];
    let mut output = vec![0.0; input.len()];
    if input.is_empty() {
        return Ok(output);
    }
    let device_id = current_device()?;
    let functions = tensor_functions(device_id)?;
    let input_bytes = std::mem::size_of_val(input);
    let table_bytes = std::mem::size_of_val(cosine);
    with_tensor_workspace(device_id, &[input_bytes, table_bytes, table_bytes, input_bytes], |workspace| {
        workspace.buffer(0).copy_from_host(unsafe { std::slice::from_raw_parts(input.as_ptr().cast(), input_bytes) })?;
        workspace.buffer(1).copy_from_host(unsafe { std::slice::from_raw_parts(cosine.as_ptr().cast(), table_bytes) })?;
        workspace.buffer(2).copy_from_host(unsafe { std::slice::from_raw_parts(sine.as_ptr().cast(), table_bytes) })?;
        let mut d_input = workspace.buffer(0).pointer;
        let mut d_cosine = workspace.buffer(1).pointer;
        let mut d_sine = workspace.buffer(2).pointer;
        let mut d_output = workspace.buffer(3).pointer;
        let mut rows = u32::try_from(rows).map_err(|_| "RoPE rows 超过 u32".to_owned())?;
        let mut cols = u32::try_from(cols).map_err(|_| "RoPE cols 超过 u32".to_owned())?;
        let mut head_count = u32::try_from(head_count).map_err(|_| "RoPE heads 超过 u32".to_owned())?;
        let mut rotary_dim = u32::try_from(rotary_dim).map_err(|_| "RoPE rotary_dim 超过 u32".to_owned())?;
        let mut table_position = 0_u32;
        let mut split_half = u32::from(layout == RotaryLayout::SplitHalf);
        let mut prefix = u32::from(prefix);
        let mut arguments = [
            (&mut d_input as *mut *mut c_void).cast(),
            (&mut d_cosine as *mut *mut c_void).cast(),
            (&mut d_sine as *mut *mut c_void).cast(),
            (&mut d_output as *mut *mut c_void).cast(),
            (&mut rows as *mut u32).cast(),
            (&mut cols as *mut u32).cast(),
            (&mut head_count as *mut u32).cast(),
            (&mut rotary_dim as *mut u32).cast(),
            (&mut table_position as *mut u32).cast(),
            (&mut split_half as *mut u32).cast(),
            (&mut prefix as *mut u32).cast(),
        ];
        let elements = rows.checked_mul(cols).ok_or("RoPE elements 溢出")?;
        launch_tensor_kernel(functions.rope, elements.div_ceil(256), 256, &mut arguments, "HIP rope")?;
        workspace.buffer(3).copy_to_host(unsafe { std::slice::from_raw_parts_mut(output.as_mut_ptr().cast(), input_bytes) })
    })?;
    Ok(output)
}

pub fn try_gated_activation_f32(gate: &[f32], up: &[f32], rows: usize, cols: usize, activation: &Activation, out: &mut [f32]) -> Result<(), String> {
    let expected = rows.checked_mul(cols).ok_or_else(|| "gated activation 元素数溢出".to_owned())?;
    if gate.len() != expected || up.len() != expected || out.len() != expected {
        return Err(format!("gated activation 长度不一致: gate={} up={} out={} 预期={}", gate.len(), up.len(), out.len(), expected));
    }
    if cols == 0 || expected == 0 {
        return Ok(());
    }
    let (kind, alpha, beta, limit, linear_beta, has_linear_beta) = match activation {
        Activation::Silu => (0_u32, 0.0, 0.0, 0.0, 0.0, 0_u32),
        Activation::SiluClamped { limit } => (1, 0.0, 0.0, *limit, 0.0, 0),
        Activation::Situ { beta, linear_beta } => (2, 0.0, *beta, 0.0, linear_beta.unwrap_or(0.0), u32::from(linear_beta.is_some())),
        Activation::SwigluOai { alpha, limit } => (3, *alpha, 0.0, *limit, 0.0, 0),
        Activation::GeluTanh => (4, 0.0, 0.0, 0.0, 0.0, 0),
    };
    let device_id = current_device()?;
    let functions = tensor_functions(device_id)?;
    let bytes = std::mem::size_of_val(gate);
    with_tensor_workspace(device_id, &[bytes, bytes, bytes], |workspace| {
        workspace.buffer(0).copy_from_host(unsafe { std::slice::from_raw_parts(gate.as_ptr().cast(), bytes) })?;
        workspace.buffer(1).copy_from_host(unsafe { std::slice::from_raw_parts(up.as_ptr().cast(), bytes) })?;
        let mut d_gate = workspace.buffer(0).pointer;
        let mut d_up = workspace.buffer(1).pointer;
        let mut d_output = workspace.buffer(2).pointer;
        let mut elements = u32::try_from(expected).map_err(|_| "gated activation 元素数超过 u32".to_owned())?;
        let mut kind = kind;
        let mut alpha = alpha;
        let mut beta = beta;
        let mut limit = limit;
        let mut linear_beta = linear_beta;
        let mut has_linear_beta = has_linear_beta;
        let mut arguments = [
            (&mut d_gate as *mut *mut c_void).cast(),
            (&mut d_up as *mut *mut c_void).cast(),
            (&mut d_output as *mut *mut c_void).cast(),
            (&mut elements as *mut u32).cast(),
            (&mut kind as *mut u32).cast(),
            (&mut alpha as *mut f32).cast(),
            (&mut beta as *mut f32).cast(),
            (&mut limit as *mut f32).cast(),
            (&mut linear_beta as *mut f32).cast(),
            (&mut has_linear_beta as *mut u32).cast(),
        ];
        launch_tensor_kernel(functions.gated_activation, elements.div_ceil(256), 256, &mut arguments, "HIP gated_activation")?;
        workspace.buffer(2).copy_to_host(unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast(), bytes) })
    })
}

pub fn try_select_rows_f32(input: &[f32], rows: usize, cols: usize, selected: &[u32]) -> Result<Vec<f32>, String> {
    if rows == 0 || cols == 0 {
        if selected.is_empty() {
            return Ok(Vec::new());
        }
        return Err("select_rows 输入为空但行列表非空".to_owned());
    }
    if input.len() != rows.checked_mul(cols).ok_or("select_rows shape 溢出")? {
        return Err("select_rows input 长度不匹配 rows*cols".to_owned());
    }
    if selected.is_empty() {
        return Ok(Vec::new());
    }
    let mut output = Vec::with_capacity(selected.len() * cols);
    for &raw in selected {
        let row = usize::try_from(raw).map_err(|_| "select_rows index 超出 usize".to_owned())?;
        if row >= rows {
            return Err("select_rows 行越界".to_owned());
        }
        let begin = row * cols;
        output.extend_from_slice(&input[begin..begin + cols]);
    }
    Ok(output)
}

pub fn try_rmsnorm_f32(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) -> Result<(), String> {
    try_rmsnorm_impl(x, w, eps, false, out)
}

pub fn try_gemma_rmsnorm_f32(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) -> Result<(), String> {
    try_rmsnorm_impl(x, w, eps, true, out)
}

fn try_rmsnorm_impl(x: &[f32], w: &[f32], eps: f32, gemma: bool, out: &mut [f32]) -> Result<(), String> {
    if x.is_empty() {
        return Ok(());
    }
    if x.len() != out.len() {
        return Err(format!("RMSNorm 输入长度={} 与输出长度={} 不一致", x.len(), out.len()));
    }
    if w.is_empty() || !x.len().is_multiple_of(w.len()) {
        return Err("RMSNorm 输入长度与权重长度不构成行对齐".to_owned());
    }
    let rows = x.len() / w.len();
    let cols = w.len();
    let device_id = current_device()?;
    let functions = tensor_functions(device_id)?;
    let input_bytes = std::mem::size_of_val(x);
    let weight_bytes = std::mem::size_of_val(w);
    with_tensor_workspace(device_id, &[input_bytes, weight_bytes, input_bytes], |workspace| {
        workspace.buffer(0).copy_from_host(unsafe { std::slice::from_raw_parts(x.as_ptr().cast(), input_bytes) })?;
        workspace.buffer(1).copy_from_host(unsafe { std::slice::from_raw_parts(w.as_ptr().cast(), weight_bytes) })?;
        let mut d_input = workspace.buffer(0).pointer;
        let mut d_weight = workspace.buffer(1).pointer;
        let mut d_output = workspace.buffer(2).pointer;
        let mut rows = u32::try_from(rows).map_err(|_| "RMSNorm rows 超过 u32".to_owned())?;
        let mut cols = u32::try_from(cols).map_err(|_| "RMSNorm cols 超过 u32".to_owned())?;
        let mut eps = eps;
        let mut gemma = u32::from(gemma);
        let mut arguments = [
            (&mut d_input as *mut *mut c_void).cast(),
            (&mut d_weight as *mut *mut c_void).cast(),
            (&mut d_output as *mut *mut c_void).cast(),
            (&mut rows as *mut u32).cast(),
            (&mut cols as *mut u32).cast(),
            (&mut eps as *mut f32).cast(),
            (&mut gemma as *mut u32).cast(),
        ];
        launch_tensor_kernel(functions.rmsnorm, rows, 256, &mut arguments, "HIP rmsnorm")?;
        workspace.buffer(2).copy_to_host(unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast(), input_bytes) })
    })
}

#[cfg(test)]
mod tests {
    use super::super::{
        DeviceBuffer, try_argmax_add_rows_resident_f32, try_argmax_excluding_resident_f32, try_argmax_rows_excluding_resident_f32, try_rmsnorm_resident_weight_to_f32, try_rmsnorm_resident_weight_to_f32_bf16,
        try_sample_top_p_rows_excluding_resident_f32, try_select_rows_f32,
    };

    fn bytes<T>(values: &[T]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
    }

    fn bytes_mut<T>(values: &mut [T]) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast(), std::mem::size_of_val(values)) }
    }

    fn bf16(value: f32) -> u16 {
        let bits = value.to_bits();
        ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
    }

    #[test]
    fn test_try_select_rows_f32() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let output = try_select_rows_f32(&input, 2, 3, &[1, 0]).unwrap();
        assert_eq!(output, vec![4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);
        assert!(try_select_rows_f32(&input, 2, 3, &[2]).is_err());
    }

    #[test]
    fn rocm_row_argmax_matches_single_rows() {
        let rows = 3;
        let columns = 7;
        let input = [1.0, 9.0, 8.0, 7.0, f32::NAN, 6.0, 5.0, 3.0, 2.0, 4.0, 1.0, 0.0, 8.0, 7.0, 5.0, 4.0, 5.0, 3.0, 2.0, 1.0, 0.0];
        let excluded = [1, 5];
        let device = std::sync::Arc::new(DeviceBuffer::upload(0, bytes(&input)).expect("upload row argmax input"));
        let actual = try_argmax_rows_excluding_resident_f32(0, &device, rows, columns, &excluded).expect("row argmax");
        let expected = (0..rows)
            .map(|row| {
                let view = DeviceBuffer::view(device.clone(), row * columns * 4, columns * 4).expect("view argmax row");
                try_argmax_excluding_resident_f32(0, &view, columns, &excluded).expect("single row argmax")
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
        assert_eq!(actual, vec![2, 6, 0]);
    }

    #[test]
    fn rocm_add_rows_argmax_matches_cpu() {
        let logits_rows = 6;
        let columns = 4097;
        let selected = [1u32, 3, 5];
        let logits = (0..logits_rows * columns).map(|index| ((index * 29 % 101) as f32 - 50.0) / 19.0).collect::<Vec<_>>();
        let bias = (0..selected.len() * columns).map(|index| ((index * 11 % 37) as f32 - 18.0) / 13.0).collect::<Vec<_>>();
        let expected = selected
            .iter()
            .enumerate()
            .map(|(row, &selected_row)| {
                (0..columns)
                    .map(|column| (column, logits[selected_row as usize * columns + column] + bias[row * columns + column]))
                    .max_by(|(left_index, left), (right_index, right)| left.total_cmp(right).then_with(|| right_index.cmp(left_index)))
                    .expect("add rows argmax row")
                    .0 as u32
            })
            .collect::<Vec<_>>();
        let logits = DeviceBuffer::upload(0, bytes(&logits)).expect("upload add rows logits");
        let bias = DeviceBuffer::upload(0, bytes(&bias)).expect("upload add rows bias");
        let actual = try_argmax_add_rows_resident_f32(0, &logits, logits_rows, columns, &selected, &bias).expect("add rows argmax");
        assert_eq!(actual, expected);
    }

    #[test]
    fn rocm_large_argmax_preserves_exclusion_nan_and_tie_break() {
        let columns = 131_072;
        let mut input = (0..columns).map(|index| ((index * 97 % 701) as f32 - 350.0) / 73.0).collect::<Vec<_>>();
        input[17] = f32::NAN;
        input[129_001] = 1000.0;
        input[73_003] = 999.0;
        input[81_007] = 999.0;
        let excluded = [129_001];
        let device = DeviceBuffer::upload(0, bytes(&input)).expect("upload large argmax input");
        let actual = try_argmax_excluding_resident_f32(0, &device, columns, &excluded).expect("large argmax");
        assert_eq!(actual, 73_003);
    }

    #[test]
    fn rocm_row_top_p_matches_cpu_rows() {
        let rows = 4;
        let columns = 1024;
        let input = (0..rows * columns).map(|index| ((index * 97 % 701) as f32 - 350.0) / 73.0).collect::<Vec<_>>();
        let sampling = [
            crate::backend::TokenSampling { temperature: 0.6, top_p: 0.95, random: 0.1 },
            crate::backend::TokenSampling { temperature: 1.0, top_p: 0.8, random: 0.5 },
            crate::backend::TokenSampling { temperature: 0.0, top_p: 1.0, random: 0.0 },
            crate::backend::TokenSampling { temperature: 1.7, top_p: 1.0, random: 0.9 },
        ];
        let excluded = [5, 706];
        let device = DeviceBuffer::upload(0, bytes(&input)).expect("upload row top-p input");
        let actual = try_sample_top_p_rows_excluding_resident_f32(0, &device, rows, columns, &sampling, &excluded).expect("row top-p");
        let expected = (0..rows)
            .map(|row| {
                let sample = sampling[row];
                let values = &input[row * columns..(row + 1) * columns];
                if sample.temperature == 0.0 {
                    values.iter().enumerate().filter(|(index, _)| !excluded.contains(&(*index as u32))).max_by(|left, right| left.1.total_cmp(right.1).then_with(|| right.0.cmp(&left.0))).map(|(index, _)| index as u32).unwrap()
                } else {
                    crate::kernel::cpu::sample_top_p_excluding(values, sample.temperature, sample.top_p, sample.random, &excluded).unwrap()
                }
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn rocm_large_vocab_top_p_matches_cpu_rows() {
        let rows = 3;
        let columns = 8192;
        let input = (0..rows * columns)
            .map(|index| {
                let row = index / columns;
                let column = index % columns;
                -(((column * 97 + row * 53) % columns) as f32) / 256.0
            })
            .collect::<Vec<_>>();
        let sampling = [
            crate::backend::TokenSampling { temperature: 0.6, top_p: 0.95, random: 0.1 },
            crate::backend::TokenSampling { temperature: 1.0, top_p: 0.8, random: 0.5 },
            crate::backend::TokenSampling { temperature: 1.7, top_p: 0.5, random: 0.9 },
        ];
        let excluded = [5, 706, 4097];
        let device = DeviceBuffer::upload(0, bytes(&input)).expect("upload large-vocab top-p input");
        let actual = try_sample_top_p_rows_excluding_resident_f32(0, &device, rows, columns, &sampling, &excluded).expect("large-vocab top-p");
        let expected = (0..rows)
            .map(|row| {
                let sample = sampling[row];
                crate::kernel::cpu::sample_top_p_excluding(&input[row * columns..(row + 1) * columns], sample.temperature, sample.top_p, sample.random, &excluded).unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn rocm_batched_rmsnorm_matches_single_rows_bitwise() {
        let rows = 8;
        let columns = 2048;
        let input = (0..rows * columns).map(|index| bf16((index as f32 * 0.013).sin())).collect::<Vec<_>>();
        let weight = (0..columns).map(|index| 0.75 + (index as f32 * 0.017).cos() * 0.25).collect::<Vec<_>>();
        let input_device = std::sync::Arc::new(DeviceBuffer::upload(0, bytes(&input)).expect("upload batched RMSNorm input"));
        let weight_device = DeviceBuffer::upload(0, bytes(&weight)).expect("upload RMSNorm weight");
        let actual = try_rmsnorm_resident_weight_to_f32(0, &input_device, &weight_device, rows, columns, 1.0e-6, false).expect("batched RMSNorm").download_f32(rows * columns).expect("download batched RMSNorm");
        let (dual, quantized) = try_rmsnorm_resident_weight_to_f32_bf16(0, &input_device, &weight_device, rows, columns, 1.0e-6, false).expect("dual RMSNorm");
        let dual = dual.download_f32(rows * columns).expect("download dual RMSNorm");
        let mut quantized_host = vec![0_u16; rows * columns];
        quantized.copy_to_host(bytes_mut(&mut quantized_host)).expect("download dual RMSNorm BF16");
        assert_eq!(dual, actual);
        assert_eq!(quantized_host, actual.iter().map(|&value| bf16(value)).collect::<Vec<_>>());
        for row in 0..rows {
            let input_row = DeviceBuffer::view(input_device.clone(), row * columns * 2, columns * 2).expect("view RMSNorm row");
            let expected = try_rmsnorm_resident_weight_to_f32(0, &input_row, &weight_device, 1, columns, 1.0e-6, false).expect("single RMSNorm").download_f32(columns).expect("download single RMSNorm");
            assert_eq!(&actual[row * columns..(row + 1) * columns], expected, "row={row}");
        }
    }

    #[test]
    fn rocm_single_row_dual_rmsnorm_matches_separate_cast_bitwise() {
        let columns = 6144;
        let input = (0..columns).map(|index| (index as f32 * 0.013).sin()).collect::<Vec<_>>();
        let weight = (0..columns).map(|index| 0.75 + (index as f32 * 0.017).cos() * 0.25).collect::<Vec<_>>();
        let input_device = DeviceBuffer::upload(0, bytes(&input)).expect("upload single-row RMSNorm input");
        let weight_device = DeviceBuffer::upload(0, bytes(&weight)).expect("upload single-row RMSNorm weight");
        let expected = try_rmsnorm_resident_weight_to_f32(0, &input_device, &weight_device, 1, columns, 1.0e-6, false).expect("single-row RMSNorm").download_f32(columns).expect("download single-row RMSNorm");
        let (actual, quantized) = try_rmsnorm_resident_weight_to_f32_bf16(0, &input_device, &weight_device, 1, columns, 1.0e-6, false).expect("single-row dual RMSNorm");
        let actual = actual.download_f32(columns).expect("download single-row dual RMSNorm");
        let mut quantized_host = vec![0_u16; columns];
        quantized.copy_to_host(bytes_mut(&mut quantized_host)).expect("download single-row dual RMSNorm BF16");
        assert_eq!(actual, expected);
        assert_eq!(quantized_host, expected.iter().map(|&value| bf16(value)).collect::<Vec<_>>());
    }
}

#[cfg(test)]
mod mixed_add_tests {
    use super::*;

    fn bytes<T>(values: &[T]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
    }

    fn bf16(value: f32) -> u16 {
        let bits = value.to_bits();
        ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
    }

    #[test]
    fn rocm_mixed_add_matches_expanded_bf16() {
        let left = [-3.25_f32, -0.1, 0.0, 1.25, 17.75];
        let left_bf16 = left.map(bf16);
        let right = [0.5_f32, 0.25, -0.75, 2.0, -7.5];
        let d_left = DeviceBuffer::upload(0, bytes(&left_bf16)).expect("upload BF16 left");
        let d_right = DeviceBuffer::upload(0, bytes(&right)).expect("upload F32 right");
        let output = try_add_resident_bf16_f32(0, &d_left, &d_right, left.len()).expect("ROCm BF16+F32 add").download_f32(left.len()).expect("download mixed add");
        for (index, actual) in output.into_iter().enumerate() {
            let expanded = f32::from_bits(u32::from(left_bf16[index]) << 16);
            assert_eq!(actual, expanded + right[index], "mixed add index={index}");
        }
    }
}

#[cfg(test)]
mod gated_activation_bf16_tests {
    use super::*;

    fn bytes<T>(values: &[T]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
    }

    #[test]
    fn rocm_gated_activation_bf16_matches_separate_cast_bitwise() {
        let elements = 2048;
        let gate = (0..elements).map(|index| (index as f32 * 0.03125).sin() * 12.0).collect::<Vec<_>>();
        let up = (0..elements).map(|index| (index as f32 * 0.046875).cos() * 8.0).collect::<Vec<_>>();
        let gate_device = DeviceBuffer::upload(0, bytes(&gate)).expect("upload gated BF16 gate");
        let up_device = DeviceBuffer::upload(0, bytes(&up)).expect("upload gated BF16 up");
        let activations = [Activation::Silu, Activation::SiluClamped { limit: 7.0 }, Activation::Situ { beta: 2.0, linear_beta: Some(1.5) }, Activation::SwigluOai { alpha: 1.702, limit: 7.0 }, Activation::GeluTanh];
        for activation in activations {
            let f32_output = try_gated_activation_resident_f32(0, &gate_device, &up_device, elements, &activation).expect("ROCm gated F32");
            let expected = crate::kernel::rocm::hip::try_cast_f32_to_bf16_resident(0, &f32_output, elements).expect("ROCm separate BF16 cast").download_u16(elements).expect("download separate BF16 cast");
            let actual = try_gated_activation_resident_bf16(0, &gate_device, &up_device, elements, &activation).expect("ROCm direct gated BF16").download_u16(elements).expect("download direct gated BF16");
            assert_eq!(actual, expected, "activation={activation:?}");
        }
    }
}

#[cfg(test)]
mod full_attention_tests {
    use super::*;

    fn bytes(values: &[f32]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
    }

    #[test]
    fn rocm_full_attention_matches_cpu() {
        let rows = 17_usize;
        let heads = 2_usize;
        let dim = 128_usize;
        let columns = heads * dim;
        let elements = rows * columns;
        let query = (0..elements).map(|i| ((i as f32) * 0.071).sin()).collect::<Vec<_>>();
        let key = (0..elements).map(|i| ((i as f32) * 0.053).cos()).collect::<Vec<_>>();
        let value = (0..elements).map(|i| ((i as f32) * 0.037).sin()).collect::<Vec<_>>();
        let d_query = DeviceBuffer::upload(0, bytes(&query)).expect("upload query");
        let d_key = DeviceBuffer::upload(0, bytes(&key)).expect("upload key");
        let d_value = DeviceBuffer::upload(0, bytes(&value)).expect("upload value");
        let scale = (dim as f32).sqrt().recip();
        let output = try_full_attention_resident_f32(0, d_query, d_key, d_value, rows, heads, dim, scale).expect("ROCm full attention").download_f32(elements).expect("download attention");

        let mut reference = vec![0.0_f32; elements];
        for row in 0..rows {
            for head in 0..heads {
                let mut scores = vec![0.0_f32; rows];
                for key_row in 0..rows {
                    let mut dot = 0.0_f32;
                    for feature in 0..dim {
                        dot += query[row * columns + head * dim + feature] * key[key_row * columns + head * dim + feature];
                    }
                    scores[key_row] = dot * scale;
                }
                let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let denominator = scores.iter().map(|score| (*score - maximum).exp()).sum::<f32>();
                for feature in 0..dim {
                    let mut sum = 0.0_f32;
                    for key_row in 0..rows {
                        let probability = (scores[key_row] - maximum).exp() / denominator;
                        sum += probability * value[key_row * columns + head * dim + feature];
                    }
                    reference[row * columns + head * dim + feature] = sum;
                }
            }
        }

        let mut maximum_error = 0.0_f32;
        for (actual, expected) in output.iter().zip(&reference) {
            assert!(actual.is_finite(), "attention 输出不是有限值: {actual}");
            maximum_error = maximum_error.max((actual - expected).abs());
        }
        assert!(maximum_error < 0.08, "attention 最大绝对误差 {maximum_error}");
    }

    #[test]
    fn rocm_rope_pair_matches_separate_rope() {
        let rows = 3_usize;
        let heads = 2_usize;
        let dim = 16_usize;
        let columns = heads * dim;
        let rotary_dim = 8_usize;
        let elements = rows * columns;
        let query = (0..elements).map(|i| (i as f32 * 0.13).sin()).collect::<Vec<_>>();
        let key = (0..elements).map(|i| (i as f32 * 0.17).cos()).collect::<Vec<_>>();
        let cosine = vec![0.8_f32; rows * rotary_dim / 2];
        let sine = vec![0.6_f32; rows * rotary_dim / 2];

        let reference_query = try_rope_resident_f32(0, &DeviceBuffer::upload(0, bytes(&query)).unwrap(), rows, columns, heads, rotary_dim, RotaryLayout::SplitHalf, 0, &cosine, &sine, true).unwrap().download_f32(elements).unwrap();
        let reference_key = try_rope_resident_f32(0, &DeviceBuffer::upload(0, bytes(&key)).unwrap(), rows, columns, heads, rotary_dim, RotaryLayout::SplitHalf, 0, &cosine, &sine, true).unwrap().download_f32(elements).unwrap();
        let (actual_query, actual_key) =
            try_rope_pair_resident_f32(0, DeviceBuffer::upload(0, bytes(&query)).unwrap(), DeviceBuffer::upload(0, bytes(&key)).unwrap(), rows, columns, heads, rotary_dim, RotaryLayout::SplitHalf, 0, &cosine, &sine, true).unwrap();
        let actual_query = actual_query.download_f32(elements).unwrap();
        let actual_key = actual_key.download_f32(elements).unwrap();
        for (actual, expected) in actual_query.iter().zip(&reference_query).chain(actual_key.iter().zip(&reference_key)) {
            assert!((actual - expected).abs() < 1e-6, "paired RoPE actual={actual} expected={expected}");
        }
    }

    #[test]
    fn rocm_segmented_rope_pair_matches_separate_rope() {
        let rows = 5_usize;
        let query_heads = 2_usize;
        let query_cols = 32_usize;
        let key_cols = 8_usize;
        let rotary_dim = 8_usize;
        let query = (0..rows * query_cols).map(|i| (i as f32 * 0.13).sin()).collect::<Vec<_>>();
        let key = (0..rows * key_cols).map(|i| (i as f32 * 0.17).cos()).collect::<Vec<_>>();
        let cosine = (0..40).map(|i| (i as f32 * 0.07).cos()).collect::<Vec<_>>();
        let sine = (0..40).map(|i| (i as f32 * 0.07).sin()).collect::<Vec<_>>();
        let mut reference_query = Vec::new();
        let mut reference_key = Vec::new();
        let mut row_offset = 0usize;
        for (segment_rows, position) in [(2_usize, 2_usize), (3, 7)] {
            let query_begin = row_offset * query_cols;
            let query_end = query_begin + segment_rows * query_cols;
            reference_query.extend(
                try_rope_resident_f32(0, &DeviceBuffer::upload(0, bytes(&query[query_begin..query_end])).unwrap(), segment_rows, query_cols, query_heads, rotary_dim, RotaryLayout::SplitHalf, position, &cosine, &sine, false)
                    .unwrap()
                    .download_f32(segment_rows * query_cols)
                    .unwrap(),
            );
            let key_begin = row_offset * key_cols;
            let key_end = key_begin + segment_rows * key_cols;
            reference_key.extend(
                try_rope_resident_f32(0, &DeviceBuffer::upload(0, bytes(&key[key_begin..key_end])).unwrap(), segment_rows, key_cols, 1, rotary_dim, RotaryLayout::SplitHalf, position, &cosine, &sine, false)
                    .unwrap()
                    .download_f32(segment_rows * key_cols)
                    .unwrap(),
            );
            row_offset += segment_rows;
        }
        let positions = [2_u32, 3, 7, 8, 9];
        let (actual_query, actual_key) = try_rope_segmented_pair_resident_f32(
            0,
            &DeviceBuffer::upload(0, bytes(&query)).unwrap(),
            rows,
            query_cols,
            query_heads,
            &DeviceBuffer::upload(0, bytes(&key)).unwrap(),
            key_cols,
            1,
            rotary_dim,
            RotaryLayout::SplitHalf,
            &positions,
            &cosine,
            &sine,
        )
        .unwrap();
        let actual_query = actual_query.download_f32(rows * query_cols).unwrap();
        let actual_key = actual_key.download_f32(rows * key_cols).unwrap();
        for (actual, expected) in actual_query.iter().zip(&reference_query).chain(actual_key.iter().zip(&reference_key)) {
            assert!((actual - expected).abs() < 1e-6, "segmented RoPE actual={actual} expected={expected}");
        }
    }

    #[test]
    fn rocm_fused_qkv_attention_matches_composed_path() {
        let rows = 5_usize;
        let heads = 2_usize;
        let dim = 16_usize;
        let columns = heads * dim;
        let rotary_dim = 8_usize;
        let eps = 1e-6_f32;
        let elements = rows * columns;
        let mut qkv = vec![0.0_f32; elements * 3];
        let mut query = vec![0.0_f32; elements];
        let mut key = vec![0.0_f32; elements];
        let mut value = vec![0.0_f32; elements];
        for row in 0..rows {
            for column in 0..columns {
                let index = row * columns + column;
                query[index] = (index as f32 * 0.071).sin();
                key[index] = (index as f32 * 0.053).cos();
                value[index] = (index as f32 * 0.037).sin();
                qkv[row * columns * 3 + column] = query[index];
                qkv[row * columns * 3 + columns + column] = key[index];
                qkv[row * columns * 3 + columns * 2 + column] = value[index];
            }
        }
        let query_weight = (0..dim).map(|i| 0.8 + i as f32 * 0.01).collect::<Vec<_>>();
        let key_weight = (0..dim).map(|i| 0.9 + i as f32 * 0.005).collect::<Vec<_>>();
        let cosine = vec![0.8_f32; rows * rotary_dim / 2];
        let sine = vec![0.6_f32; rows * rotary_dim / 2];
        let scale = (dim as f32).sqrt().recip();

        let normalized_query = try_rmsnorm_resident_to_f32(0, &DeviceBuffer::upload(0, bytes(&query)).unwrap(), &query_weight, rows * heads, dim, eps, false).unwrap();
        let normalized_key = try_rmsnorm_resident_to_f32(0, &DeviceBuffer::upload(0, bytes(&key)).unwrap(), &key_weight, rows * heads, dim, eps, false).unwrap();
        let (query, key) = try_rope_pair_resident_f32(0, normalized_query, normalized_key, rows, columns, heads, rotary_dim, RotaryLayout::SplitHalf, 0, &cosine, &sine, true).unwrap();
        let reference = try_full_attention_resident_f32(0, query, key, DeviceBuffer::upload(0, bytes(&value)).unwrap(), rows, heads, dim, scale).unwrap().download_f32(elements).unwrap();

        let actual = try_full_attention_qkv_resident_f32(0, DeviceBuffer::upload(0, bytes(&qkv)).unwrap(), &query_weight, &key_weight, rows, heads, dim, rotary_dim, eps, &cosine, &sine, scale).unwrap().download_f32(elements).unwrap();
        for (index, (actual, expected)) in actual.iter().zip(&reference).enumerate() {
            assert!(actual.is_finite() && (actual - expected).abs() < 0.03, "index={index} actual={actual} expected={expected}");
        }
    }
}
