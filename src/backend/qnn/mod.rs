//! Qualcomm QNN/HTP 资源与静态 INT8 Linear 图。
//!
//! QNN SDK 只在构建期提供 header,运行期通过 `dlopen` 使用手机 QAIRT,避免把
//! 厂商 SDK 或模型语义带入仓库。图持有 backend/device/context/graph 与静态权重,
//! activation 只在 `execute` 调用期间借用。
//!
//! 量化布局(W8A8):权重 per-output-channel INT8;输入逐行 S8 动态 scale
//! (max 映射 ±100,无裁剪),读回按 实际/烘焙 输入 scale 比值校正——clamp 容限
//! 随输入幅度同向缩放,对残差流漂移天然免疫;输出 S8 用一次性校准 scale(留
//! 2.5× 余量,context.rs 负责),这是本栈 MatMul 唯一可用的输出类型。
//!
//! 已证伪并放弃的路线(真机 SM8635/QAIRT-2.38 实测):
//! - Float(F16/F32)与 S32 输出的 MatMul:addNode 6005 拒绝或静默输出全零。
//! - W4A8:per-channel max/7 的 INT4 每矩阵 cos≈0.98,56 个矩阵累积后生成乱码。
//! - 冻结 activation scale:幅度漂移 8× 时 cos 掉到 0.85(o_proj/down 输入未归一化)。

use std::{
    ffi::{CStr, CString, c_char, c_void},
    path::Path,
    ptr::NonNull,
};

mod context;
pub use context::{QnnContext, QnnWeight};

/// QNN MatMul 需要的 K×N 行优先布局;每元素一个 i8 的低 `bits` 位。
#[derive(Debug, Clone)]
pub struct QnnQuantWeight {
    pub values: Vec<i8>,
    pub scales: Vec<f32>,
    pub input_columns: usize,
    pub output_columns: usize,
    pub bits: u8,
}

/// 把 zLLM `linear` 使用的 `[out, in]` F32 权重量化、转置为 QNN `[in, out]`。
///
/// `bits = 4` 对称范围 [-7,7](scale = max/7),`bits = 8` 范围 [-127,127](scale = max/127)。
pub fn quantize_linear_weight(weights: &[f32], output_columns: usize, input_columns: usize, bits: u8) -> Result<QnnQuantWeight, String> {
    let level = match bits {
        4 => 7.0_f32,
        8 => 127.0,
        _ => return Err(format!("QNN 权重量化只支持 4/8 bit,实际 {bits}")),
    };
    if output_columns == 0 || input_columns == 0 || weights.len() != output_columns.checked_mul(input_columns).ok_or("QNN weight shape 溢出")? {
        return Err(format!("QNN weight shape/data 不匹配: out={output_columns} in={input_columns} values={}", weights.len()));
    }
    let mut values = vec![0_i8; weights.len()];
    let mut scales = vec![1.0_f32; output_columns];
    for output in 0..output_columns {
        let row = &weights[output * input_columns..(output + 1) * input_columns];
        let max_abs = row.iter().copied().map(f32::abs).fold(0.0_f32, f32::max);
        let scale = if max_abs <= f32::EPSILON { 1.0 } else { max_abs / level };
        scales[output] = scale;
        for input in 0..input_columns {
            values[input * output_columns + output] = (row[input] / scale).round().clamp(-level, level) as i8;
        }
    }
    Ok(QnnQuantWeight { values, scales, input_columns, output_columns, bits })
}

#[repr(C)]
struct NativeError {
    message: [c_char; 512],
}

unsafe extern "C" {
    fn zllm_qnn_linear_create(backend: *const c_char, rows: u32, inner: u32, columns: u32, weights: *const i8, scales: *const f32, weight_bits: u32, baked_input_scale: f32, output_scale: f32, error: *mut NativeError) -> *mut c_void;
    fn zllm_qnn_linear_execute(graph: *mut c_void, input: *const i8, output: *mut i8, error: *mut NativeError) -> i32;
    fn zllm_qnn_dual_linear_create(
        backend: *const c_char,
        rows: u32,
        inner: u32,
        columns: u32,
        weights: *const i8,
        scales: *const f32,
        columns2: u32,
        weights2: *const i8,
        scales2: *const f32,
        weight_bits: u32,
        baked_input_scale: f32,
        output_scale: f32,
        output2_scale: f32,
        error: *mut NativeError,
    ) -> *mut c_void;
    fn zllm_qnn_dual_linear_execute(graph: *mut c_void, input: *const i8, output: *mut i8, output2: *mut i8, error: *mut NativeError) -> i32;
    fn zllm_qnn_triple_linear_create(
        backend: *const c_char,
        rows: u32,
        inner: u32,
        columns: u32,
        weights: *const i8,
        scales: *const f32,
        columns2: u32,
        weights2: *const i8,
        scales2: *const f32,
        columns3: u32,
        weights3: *const i8,
        scales3: *const f32,
        weight_bits: u32,
        baked_input_scale: f32,
        output_scale: f32,
        output2_scale: f32,
        output3_scale: f32,
        error: *mut NativeError,
    ) -> *mut c_void;
    fn zllm_qnn_triple_linear_execute(graph: *mut c_void, input: *const i8, output: *mut i8, output2: *mut i8, output3: *mut i8, error: *mut NativeError) -> i32;
    fn zllm_qnn_linear_context_bytes(graph: *const c_void) -> u64;
    fn zllm_qnn_linear_destroy(graph: *mut c_void);
    fn zllm_qnn_mlp_create(
        backend: *const c_char,
        rows: u32,
        inner: u32,
        gate_columns: u32,
        gate_weights: *const i8,
        gate_scales: *const f32,
        up_columns: u32,
        up_weights: *const i8,
        up_scales: *const f32,
        down_columns: u32,
        down_weights: *const i8,
        down_scales: *const f32,
        weight_bits: u32,
        baked_input_scale: f32,
        gate_out_scale: f32,
        up_out_scale: f32,
        silu_scale: f32,
        act_scale: f32,
        out_scale: f32,
        error: *mut NativeError,
    ) -> *mut c_void;
    fn zllm_qnn_mlp_execute(graph: *mut c_void, input: *const i8, output: *mut i8, error: *mut NativeError) -> i32;
    fn zllm_qnn_mlp_destroy(graph: *mut c_void);
}

/// 逐行 S8 量化:max 映射到 ±100,留 27% headroom;返回 (量化值, 实际 scale)。
fn quantize_row(row: &[f32]) -> (Vec<i8>, f32) {
    let max_abs = row.iter().copied().map(f32::abs).fold(0.0_f32, f32::max).max(1.0e-6);
    let scale = max_abs / 100.0;
    let quantized = row.iter().map(|value| (value / scale).round().clamp(-127.0, 127.0) as i8).collect();
    (quantized, scale)
}

/// S8 输出 → F32:yq × 烘焙输出 scale × (实际输入 scale / 烘焙输入 scale)。
/// 比值校正使输出 clamp 容限随输入幅度同向缩放,残差流漂移不触发裁剪。
fn restore_output(raw: &[i8], output_scale: f32, ratio: f32) -> Vec<f32> {
    raw.iter().map(|value| *value as f32 * output_scale * ratio).collect()
}

#[derive(Debug)]
pub struct QnnTripleLinear {
    handle: NonNull<c_void>,
    inner: usize,
    columns: [usize; 3],
    baked_input_scale: f32,
    output_scales: [f32; 3],
}
unsafe impl Send for QnnTripleLinear {}

impl QnnTripleLinear {
    #[allow(clippy::too_many_arguments)]
    pub fn new(backend: &Path, rows: usize, weights: [&QnnQuantWeight; 3], baked_input_scale: f32, output_scales: [f32; 3]) -> Result<Self, String> {
        if rows == 0 || weights.iter().any(|weight| weight.input_columns != weights[0].input_columns || weight.bits != weights[0].bits) {
            return Err("QNN TripleLinear 输入 shape 不一致".to_owned());
        }
        let backend = CString::new(backend.as_os_str().as_encoded_bytes()).map_err(|_| "QNN backend 路径包含 NUL".to_owned())?;
        let mut error = NativeError { message: [0; 512] };
        let handle = unsafe {
            zllm_qnn_triple_linear_create(
                backend.as_ptr(),
                rows as u32,
                weights[0].input_columns as u32,
                weights[0].output_columns as u32,
                weights[0].values.as_ptr(),
                weights[0].scales.as_ptr(),
                weights[1].output_columns as u32,
                weights[1].values.as_ptr(),
                weights[1].scales.as_ptr(),
                weights[2].output_columns as u32,
                weights[2].values.as_ptr(),
                weights[2].scales.as_ptr(),
                weights[0].bits as u32,
                baked_input_scale,
                output_scales[0],
                output_scales[1],
                output_scales[2],
                &mut error,
            )
        };
        Ok(Self {
            handle: NonNull::new(handle).ok_or_else(|| native_error(&error))?,
            inner: weights[0].input_columns,
            columns: [weights[0].output_columns, weights[1].output_columns, weights[2].output_columns],
            baked_input_scale,
            output_scales,
        })
    }

    pub fn execute_f32(&mut self, input: &[f32]) -> Result<[Vec<f32>; 3], String> {
        if input.len() != self.inner {
            return Err("QNN TripleLinear execute 输入长度不匹配".to_owned());
        }
        let (quantized, input_scale) = quantize_row(input);
        let ratio = input_scale / self.baked_input_scale;
        let mut buffers = [vec![0_i8; self.columns[0]], vec![0_i8; self.columns[1]], vec![0_i8; self.columns[2]]];
        let mut error = NativeError { message: [0; 512] };
        let status = unsafe { zllm_qnn_triple_linear_execute(self.handle.as_ptr(), quantized.as_ptr(), buffers[0].as_mut_ptr(), buffers[1].as_mut_ptr(), buffers[2].as_mut_ptr(), &mut error) };
        if status != 0 {
            return Err(native_error(&error));
        }
        Ok([restore_output(&buffers[0], self.output_scales[0], ratio), restore_output(&buffers[1], self.output_scales[1], ratio), restore_output(&buffers[2], self.output_scales[2], ratio)])
    }
}

impl Drop for QnnTripleLinear {
    fn drop(&mut self) {
        unsafe { zllm_qnn_linear_destroy(self.handle.as_ptr()) }
    }
}

#[derive(Debug)]
pub struct QnnDualLinear {
    handle: NonNull<c_void>,
    inner: usize,
    first_columns: usize,
    second_columns: usize,
    baked_input_scale: f32,
    first_output_scale: f32,
    second_output_scale: f32,
}

unsafe impl Send for QnnDualLinear {}

impl QnnDualLinear {
    pub fn new(backend: &Path, rows: usize, first: &QnnQuantWeight, second: &QnnQuantWeight, baked_input_scale: f32, first_output_scale: f32, second_output_scale: f32) -> Result<Self, String> {
        if rows == 0 || first.input_columns != second.input_columns || first.bits != second.bits {
            return Err("QNN DualLinear 两份权重 shape 不一致".to_owned());
        }
        let backend = CString::new(backend.as_os_str().as_encoded_bytes()).map_err(|_| "QNN backend 路径包含 NUL".to_owned())?;
        let mut error = NativeError { message: [0; 512] };
        let handle = unsafe {
            zllm_qnn_dual_linear_create(
                backend.as_ptr(),
                rows as u32,
                first.input_columns as u32,
                first.output_columns as u32,
                first.values.as_ptr(),
                first.scales.as_ptr(),
                second.output_columns as u32,
                second.values.as_ptr(),
                second.scales.as_ptr(),
                first.bits as u32,
                baked_input_scale,
                first_output_scale,
                second_output_scale,
                &mut error,
            )
        };
        let handle = NonNull::new(handle).ok_or_else(|| native_error(&error))?;
        Ok(Self { handle, inner: first.input_columns, first_columns: first.output_columns, second_columns: second.output_columns, baked_input_scale, first_output_scale, second_output_scale })
    }

    pub fn execute_f32(&mut self, input: &[f32]) -> Result<(Vec<f32>, Vec<f32>), String> {
        if input.len() != self.inner {
            return Err("QNN DualLinear execute 输入长度不匹配".to_owned());
        }
        let (quantized, input_scale) = quantize_row(input);
        let ratio = input_scale / self.baked_input_scale;
        let mut first_buffer = vec![0_i8; self.first_columns];
        let mut second_buffer = vec![0_i8; self.second_columns];
        let mut error = NativeError { message: [0; 512] };
        let status = unsafe { zllm_qnn_dual_linear_execute(self.handle.as_ptr(), quantized.as_ptr(), first_buffer.as_mut_ptr(), second_buffer.as_mut_ptr(), &mut error) };
        if status != 0 {
            return Err(native_error(&error));
        }
        Ok((restore_output(&first_buffer, self.first_output_scale, ratio), restore_output(&second_buffer, self.second_output_scale, ratio)))
    }
}

impl Drop for QnnDualLinear {
    fn drop(&mut self) {
        unsafe { zllm_qnn_linear_destroy(self.handle.as_ptr()) }
    }
}

/// MLP 整段融合图:`normed → gate/up GEMM → sigmoid·gate·up → down GEMM`,单次 dispatch。
///
/// 中间量纲随输入动态 scale 的幂次缩放(gate/up/silu ~ s_a,act/down ~ s_a²),
/// 读回按 s_a² 比值校正;五级 scale 由首次调用的 CPU 参考一次性校准(÷50 余量)。
#[derive(Debug)]
pub struct QnnMlpGraph {
    handle: NonNull<c_void>,
    inner: usize,
    columns: usize,
    baked_input_scale: f32,
    output_scale: f32,
}
unsafe impl Send for QnnMlpGraph {}

impl QnnMlpGraph {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        backend: &Path,
        rows: usize,
        gate: &QnnQuantWeight,
        up: &QnnQuantWeight,
        down: &QnnQuantWeight,
        baked_input_scale: f32,
        gate_out_scale: f32,
        up_out_scale: f32,
        silu_scale: f32,
        act_scale: f32,
        output_scale: f32,
    ) -> Result<Self, String> {
        if rows == 0 || gate.input_columns != up.input_columns || gate.output_columns != up.output_columns || down.input_columns != gate.output_columns || gate.bits != up.bits || gate.bits != down.bits {
            return Err("QNN MLP 图三份权重 shape 不一致".to_owned());
        }
        let backend = CString::new(backend.as_os_str().as_encoded_bytes()).map_err(|_| "QNN backend 路径包含 NUL".to_owned())?;
        let mut error = NativeError { message: [0; 512] };
        let handle = unsafe {
            zllm_qnn_mlp_create(
                backend.as_ptr(),
                rows as u32,
                gate.input_columns as u32,
                gate.output_columns as u32,
                gate.values.as_ptr(),
                gate.scales.as_ptr(),
                up.output_columns as u32,
                up.values.as_ptr(),
                up.scales.as_ptr(),
                down.output_columns as u32,
                down.values.as_ptr(),
                down.scales.as_ptr(),
                gate.bits as u32,
                baked_input_scale,
                gate_out_scale,
                up_out_scale,
                silu_scale,
                act_scale,
                output_scale,
                &mut error,
            )
        };
        let handle = NonNull::new(handle).ok_or_else(|| native_error(&error))?;
        Ok(Self { handle, inner: gate.input_columns, columns: down.output_columns, baked_input_scale, output_scale })
    }

    pub fn execute_f32(&mut self, input: &[f32]) -> Result<Vec<f32>, String> {
        if input.len() != self.inner {
            return Err(format!("QNN MLP execute 输入长度不匹配: input={} inner={}", input.len(), self.inner));
        }
        let (quantized, input_scale) = quantize_row(input);
        // down 输出实值 = y_true / s_a²(中间两次 s_a 缩放),按平方比值还原。
        let ratio = (input_scale / self.baked_input_scale).powi(2);
        let mut buffer = vec![0_i8; self.columns];
        let mut error = NativeError { message: [0; 512] };
        let status = unsafe { zllm_qnn_mlp_execute(self.handle.as_ptr(), quantized.as_ptr(), buffer.as_mut_ptr(), &mut error) };
        if status != 0 {
            return Err(native_error(&error));
        }
        Ok(buffer.iter().map(|value| *value as f32 * self.output_scale * ratio).collect())
    }
}

impl Drop for QnnMlpGraph {
    fn drop(&mut self) {
        unsafe { zllm_qnn_mlp_destroy(self.handle.as_ptr()) }
    }
}

#[derive(Debug)]
pub struct QnnLinear {
    handle: NonNull<c_void>,
    inner: usize,
    columns: usize,
    baked_input_scale: f32,
    output_scale: f32,
}

unsafe impl Send for QnnLinear {}

impl QnnLinear {
    /// `weights` 是 K×N 行优先量化值,scales 按输出列 N 提供;finalize 后权重归图所有。
    /// `baked_input_scale` 固定为常量 1.0 即可(动态比值校正在 execute 内完成);
    /// `output_scale` 由调用方一次性校准(参考输出 max / 50)。
    pub fn new(backend: &Path, rows: usize, weight: &QnnQuantWeight, baked_input_scale: f32, output_scale: f32) -> Result<Self, String> {
        let expected = weight.input_columns.checked_mul(weight.output_columns).ok_or("QNN weight shape 溢出")?;
        if rows == 0 || weight.input_columns == 0 || weight.output_columns == 0 || weight.values.len() != expected || weight.scales.len() != weight.output_columns {
            return Err(format!("QNN Linear shape/data 不匹配: M={rows} K={} N={} values={} scales={}", weight.input_columns, weight.output_columns, weight.values.len(), weight.scales.len()));
        }
        let level = match weight.bits {
            4 => 7_i8,
            8 => 127,
            _ => return Err(format!("QNN 权重量化只支持 4/8 bit,实际 {}", weight.bits)),
        };
        if weight.values.iter().any(|&value| !(-level - 1..=level).contains(&value)) || weight.scales.iter().any(|scale| !scale.is_finite() || *scale <= 0.0) {
            return Err(format!("QNN Linear 权重必须位于 [{},{}],scales 必须为有限正数", -level - 1, level));
        }
        let backend = CString::new(backend.as_os_str().as_encoded_bytes()).map_err(|_| "QNN backend 路径包含 NUL".to_owned())?;
        let mut error = NativeError { message: [0; 512] };
        let handle = unsafe {
            zllm_qnn_linear_create(backend.as_ptr(), rows as u32, weight.input_columns as u32, weight.output_columns as u32, weight.values.as_ptr(), weight.scales.as_ptr(), weight.bits as u32, baked_input_scale, output_scale, &mut error)
        };
        let handle = NonNull::new(handle).ok_or_else(|| native_error(&error))?;
        Ok(Self { handle, inner: weight.input_columns, columns: weight.output_columns, baked_input_scale, output_scale })
    }

    pub fn execute_f32(&mut self, input: &[f32]) -> Result<Vec<f32>, String> {
        if input.len() != self.inner {
            return Err(format!("QNN execute 输入长度不匹配: input={} inner={}", input.len(), self.inner));
        }
        let (quantized, input_scale) = quantize_row(input);
        let ratio = input_scale / self.baked_input_scale;
        let mut buffer = vec![0_i8; self.columns];
        let mut error = NativeError { message: [0; 512] };
        let status = unsafe { zllm_qnn_linear_execute(self.handle.as_ptr(), quantized.as_ptr(), buffer.as_mut_ptr(), &mut error) };
        if status != 0 {
            return Err(native_error(&error));
        }
        Ok(restore_output(&buffer, self.output_scale, ratio))
    }

    pub fn context_bytes(&self) -> u64 {
        unsafe { zllm_qnn_linear_context_bytes(self.handle.as_ptr()) }
    }
}

impl Drop for QnnLinear {
    fn drop(&mut self) {
        unsafe { zllm_qnn_linear_destroy(self.handle.as_ptr()) }
    }
}

fn native_error(error: &NativeError) -> String {
    unsafe { CStr::from_ptr(error.message.as_ptr()) }.to_string_lossy().into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 拒绝错误量化权重与shape() {
        let weight = quantize_linear_weight(&[0.0; 6], 2, 3, 8).unwrap();
        // 合法权重通过全部校验后卡在 dlopen(主机无 QNN),不应是 shape/scale 错误。
        assert!(!QnnLinear::new(Path::new("x"), 1, &weight, 1.0, 1.0).unwrap_err().contains("shape"));
        assert!(QnnLinear::new(Path::new("x"), 0, &weight, 1.0, 1.0).unwrap_err().contains("shape"));
        let mut overflow = weight.clone();
        overflow.values.push(0);
        assert!(QnnLinear::new(Path::new("x"), 1, &overflow, 1.0, 1.0).unwrap_err().contains("shape"));
        let mut bad_level = weight.clone();
        bad_level.values[0] = 127;
        bad_level.scales[0] = 0.0;
        assert!(QnnLinear::new(Path::new("x"), 1, &bad_level, 1.0, 1.0).unwrap_err().contains("有限正数"));
    }

    #[test]
    fn 量化转成qnn矩阵方向() {
        let weight = quantize_linear_weight(&[7.0, -7.0, 3.5, 0.0, 1.0, -1.0], 2, 3, 4).unwrap();
        assert_eq!(weight.values, vec![7, 0, -7, 7, 4, -7]);
        assert_eq!(weight.scales, vec![1.0, 1.0 / 7.0]);
        assert_eq!((weight.input_columns, weight.output_columns, weight.bits), (3, 2, 4));
        let weight = quantize_linear_weight(&[7.0, -7.0], 1, 2, 8).unwrap();
        assert_eq!(weight.values, vec![127, -127]);
        assert_eq!(weight.scales, vec![7.0 / 127.0]);
        assert!(quantize_linear_weight(&[1.0], 1, 1, 6).unwrap_err().contains("4/8"));
    }

    #[test]
    fn dual_linear拒绝不同shape() {
        let first = quantize_linear_weight(&[1.0; 6], 2, 3, 8).unwrap();
        let second = quantize_linear_weight(&[1.0; 8], 2, 4, 8).unwrap();
        assert!(QnnDualLinear::new(Path::new("x"), 1, &first, &second, 1.0, 1.0, 1.0).unwrap_err().contains("shape"));
    }

    #[test]
    fn 逐行量化留headroom不裁剪() {
        let (values, scale) = quantize_row(&[100.0, -50.0, 25.0]);
        assert_eq!(values, vec![100, -50, 25]);
        assert!((scale - 1.0).abs() < 1.0e-6);
    }
}
