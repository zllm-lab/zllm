//! Apple MPS 大矩阵乘。用于已经物化的 F16/F32 prefill 矩阵。

use std::mem::size_of;

use objc2::AnyThread;
use objc2::rc::Retained;
use objc2_metal_performance_shaders::{MPSDataType, MPSMatrix, MPSMatrixDescriptor, MPSMatrixMultiplication};

use crate::backend::metal::api::{BufferRef, CommandBufferRef, DeviceRef};

const MPS_DATA_TYPE_FLOAT16: u32 = 0x1000_0010;
const MPS_DATA_TYPE_FLOAT32: u32 = 0x1000_0020;

/// `output = input @ weight^T`，输入为 F16，结果用 F32 保存。
#[allow(clippy::too_many_arguments)]
pub fn encode_f16_matmul_transposed_f32(command: &CommandBufferRef, device: &DeviceRef, input: &BufferRef, rows: usize, input_columns: usize, weight: &BufferRef, output_columns: usize, output: &BufferRef) -> Result<(), String> {
    encode_matmul_mixed(
        command,
        device,
        input,
        0,
        input_columns * size_of::<u16>(),
        weight,
        0,
        input_columns * size_of::<u16>(),
        output,
        0,
        output_columns * size_of::<f32>(),
        rows,
        input_columns,
        output_columns,
        true,
        1.0,
        MPS_DATA_TYPE_FLOAT16,
        MPS_DATA_TYPE_FLOAT32,
    )
}

/// `output = input @ weight^T`，input 为 F32，weight 为 F16，结果保持 F32。
#[allow(clippy::too_many_arguments)]
pub fn encode_f32_f16_matmul_transposed_f32(command: &CommandBufferRef, device: &DeviceRef, input: &BufferRef, rows: usize, input_columns: usize, weight: &BufferRef, output_columns: usize, output: &BufferRef) -> Result<(), String> {
    encode_matmul_mixed_inputs(
        command,
        device,
        input,
        0,
        input_columns * size_of::<f32>(),
        weight,
        0,
        input_columns * size_of::<u16>(),
        output,
        0,
        output_columns * size_of::<f32>(),
        rows,
        input_columns,
        output_columns,
        true,
        1.0,
        MPS_DATA_TYPE_FLOAT32,
        MPS_DATA_TYPE_FLOAT16,
        MPS_DATA_TYPE_FLOAT32,
    )
}

/// `output = input @ weight^T`，三个矩阵都是 F32。
#[allow(clippy::too_many_arguments)]
pub fn encode_f32_matmul_transposed(command: &CommandBufferRef, device: &DeviceRef, input: &BufferRef, rows: usize, input_columns: usize, weight: &BufferRef, output_columns: usize, output: &BufferRef) -> Result<(), String> {
    encode_matmul(command, device, input, 0, input_columns * size_of::<f32>(), weight, 0, input_columns * size_of::<f32>(), output, 0, output_columns * size_of::<f32>(), rows, input_columns, output_columns, true, 1.0, MPS_DATA_TYPE_FLOAT32)
}

/// 用带 stride 的 F16 矩阵 view 编码乘法，结果保存为 F32。
#[allow(clippy::too_many_arguments)]
pub fn encode_f16_matmul_f32(
    command: &CommandBufferRef,
    device: &DeviceRef,
    left: &BufferRef,
    left_offset: usize,
    left_row_bytes: usize,
    right: &BufferRef,
    right_offset: usize,
    right_row_bytes: usize,
    result: &BufferRef,
    result_offset: usize,
    result_row_bytes: usize,
    rows: usize,
    interior_columns: usize,
    output_columns: usize,
    transpose_right: bool,
    alpha: f64,
) -> Result<(), String> {
    encode_matmul_mixed(
        command,
        device,
        left,
        left_offset,
        left_row_bytes,
        right,
        right_offset,
        right_row_bytes,
        result,
        result_offset,
        result_row_bytes,
        rows,
        interior_columns,
        output_columns,
        transpose_right,
        alpha,
        MPS_DATA_TYPE_FLOAT16,
        MPS_DATA_TYPE_FLOAT32,
    )
}

/// 用带 stride 的矩阵 view 编码 F32 @ F16，结果保持 F32。
#[allow(clippy::too_many_arguments)]
pub fn encode_f32_f16_matmul_f32(
    command: &CommandBufferRef,
    device: &DeviceRef,
    left: &BufferRef,
    left_offset: usize,
    left_row_bytes: usize,
    right: &BufferRef,
    right_offset: usize,
    right_row_bytes: usize,
    result: &BufferRef,
    result_offset: usize,
    result_row_bytes: usize,
    rows: usize,
    interior_columns: usize,
    output_columns: usize,
    transpose_right: bool,
    alpha: f64,
) -> Result<(), String> {
    encode_matmul_mixed_inputs(
        command,
        device,
        left,
        left_offset,
        left_row_bytes,
        right,
        right_offset,
        right_row_bytes,
        result,
        result_offset,
        result_row_bytes,
        rows,
        interior_columns,
        output_columns,
        transpose_right,
        alpha,
        MPS_DATA_TYPE_FLOAT32,
        MPS_DATA_TYPE_FLOAT16,
        MPS_DATA_TYPE_FLOAT32,
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_matmul(
    command: &CommandBufferRef,
    device: &DeviceRef,
    left: &BufferRef,
    left_offset: usize,
    left_row_bytes: usize,
    right: &BufferRef,
    right_offset: usize,
    right_row_bytes: usize,
    result: &BufferRef,
    result_offset: usize,
    result_row_bytes: usize,
    rows: usize,
    interior_columns: usize,
    output_columns: usize,
    transpose_right: bool,
    alpha: f64,
    data_type: u32,
) -> Result<(), String> {
    encode_matmul_mixed(command, device, left, left_offset, left_row_bytes, right, right_offset, right_row_bytes, result, result_offset, result_row_bytes, rows, interior_columns, output_columns, transpose_right, alpha, data_type, data_type)
}

#[allow(clippy::too_many_arguments)]
fn encode_matmul_mixed(
    command: &CommandBufferRef,
    device: &DeviceRef,
    left: &BufferRef,
    left_offset: usize,
    left_row_bytes: usize,
    right: &BufferRef,
    right_offset: usize,
    right_row_bytes: usize,
    result: &BufferRef,
    result_offset: usize,
    result_row_bytes: usize,
    rows: usize,
    interior_columns: usize,
    output_columns: usize,
    transpose_right: bool,
    alpha: f64,
    data_type: u32,
    result_data_type: u32,
) -> Result<(), String> {
    encode_matmul_mixed_inputs(
        command,
        device,
        left,
        left_offset,
        left_row_bytes,
        right,
        right_offset,
        right_row_bytes,
        result,
        result_offset,
        result_row_bytes,
        rows,
        interior_columns,
        output_columns,
        transpose_right,
        alpha,
        data_type,
        data_type,
        result_data_type,
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_matmul_mixed_inputs(
    command: &CommandBufferRef,
    device: &DeviceRef,
    left: &BufferRef,
    left_offset: usize,
    left_row_bytes: usize,
    right: &BufferRef,
    right_offset: usize,
    right_row_bytes: usize,
    result: &BufferRef,
    result_offset: usize,
    result_row_bytes: usize,
    rows: usize,
    interior_columns: usize,
    output_columns: usize,
    transpose_right: bool,
    alpha: f64,
    left_data_type: u32,
    right_data_type: u32,
    result_data_type: u32,
) -> Result<(), String> {
    let left = matrix(left, left_offset, rows, interior_columns, left_row_bytes, left_data_type);
    let (right_rows, right_columns) = if transpose_right { (output_columns, interior_columns) } else { (interior_columns, output_columns) };
    let right = matrix(right, right_offset, right_rows, right_columns, right_row_bytes, right_data_type);
    let result = matrix(result, result_offset, rows, output_columns, result_row_bytes, result_data_type);
    let multiplication = unsafe {
        MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(
            MPSMatrixMultiplication::alloc(),
            device.raw(),
            false,
            transpose_right,
            rows,
            output_columns,
            interior_columns,
            alpha,
            0.0,
        )
    };
    unsafe { multiplication.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(command.raw(), &left, &right, &result) };
    Ok(())
}

fn matrix(buffer: &BufferRef, offset: usize, rows: usize, columns: usize, row_bytes: usize, data_type: u32) -> Retained<MPSMatrix> {
    let descriptor = unsafe { MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(rows, columns, row_bytes, MPSDataType(data_type)) };
    unsafe { MPSMatrix::initWithBuffer_offset_descriptor(MPSMatrix::alloc(), buffer.raw(), offset, &descriptor) }
}
