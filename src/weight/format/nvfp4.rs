//! NVIDIA ModelOpt NVFP4 routed-expert 权重。

use std::{
    collections::HashMap,
    fs::File,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use crate::weight::container::file_ext::FileExt;

use crate::{
    kernel::cpu::nvfp4::{NVFP4_BLOCK, decode_nvfp4_matrix},
    weight::{
        container::safetensor::{SafetensorStore, TensorData, TensorDestination, TensorInfo},
        format::official_fp8::ExpertF32,
    },
};

#[derive(Clone)]
pub struct Nvfp4Matrix {
    codes: Vec<u8>,
    scales: Vec<u8>,
    pub global_scale: f32,
    pub rows: usize,
    pub cols: usize,
}

impl Nvfp4Matrix {
    pub fn new(codes: Vec<u8>, scales: Vec<u8>, global_scale: f32, rows: usize, cols: usize) -> Result<Self, String> {
        let (code_bytes, scale_bytes) = nvfp4_storage_lengths(rows, cols)?;
        if codes.len() != code_bytes || scales.len() != scale_bytes {
            return Err(format!("NVFP4 字节数 codes={}/{} scales={}/{}", codes.len(), code_bytes, scales.len(), scale_bytes));
        }
        if !global_scale.is_finite() || global_scale <= 0.0 {
            return Err(format!("NVFP4 global scale 无效: {global_scale}"));
        }
        Ok(Self { codes, scales, global_scale, rows, cols })
    }

    pub fn decode(&self) -> Result<Vec<f32>, String> {
        let mut output = vec![0.0; self.rows * self.cols];
        decode_nvfp4_matrix(&self.codes, &self.scales, self.global_scale, self.rows, self.cols, &mut output)?;
        Ok(output)
    }

    pub fn codes(&self) -> &[u8] {
        &self.codes
    }

    pub fn scales(&self) -> &[u8] {
        &self.scales
    }
}

pub struct Nvfp4MatrixBufferMut<'a> {
    pub codes: &'a mut [u8],
    pub scales: &'a mut [u8],
    pub global_scale: &'a mut [u8],
    pub rows: usize,
    pub cols: usize,
}

pub struct Nvfp4ExpertBufferMut<'a> {
    pub expert: usize,
    pub gate: Nvfp4MatrixBufferMut<'a>,
    pub up: Nvfp4MatrixBufferMut<'a>,
    pub down: Nvfp4MatrixBufferMut<'a>,
}

/// 每层单文件 NVFP4 expert-major 布局；只描述字节，不包含线程或设备策略。
#[derive(Debug, Clone, Copy)]
pub struct Nvfp4ArchiveLayout {
    expert_count: usize,
    code_bytes: usize,
    scale_bytes: usize,
    matrix_bytes: usize,
    expert_bytes: usize,
}

impl Nvfp4ArchiveLayout {
    pub fn new(expert_count: usize, intermediate: usize, hidden: usize) -> Result<Self, String> {
        if expert_count == 0 {
            return Err("NVFP4 archive expert_count 必须大于 0".to_owned());
        }
        let (code_bytes, scale_bytes) = nvfp4_storage_lengths(intermediate, hidden)?;
        let (down_codes, down_scales) = nvfp4_storage_lengths(hidden, intermediate)?;
        if (code_bytes, scale_bytes) != (down_codes, down_scales) {
            return Err("NVFP4 archive gate/up/down 矩阵存储长度不一致".to_owned());
        }
        let matrix_bytes = code_bytes.checked_add(scale_bytes).and_then(|bytes| bytes.checked_add(std::mem::size_of::<f32>())).ok_or_else(|| "NVFP4 archive matrix 大小溢出".to_owned())?;
        let expert_bytes = matrix_bytes.checked_mul(3).ok_or_else(|| "NVFP4 archive expert 大小溢出".to_owned())?;
        Ok(Self { expert_count, code_bytes, scale_bytes, matrix_bytes, expert_bytes })
    }

    pub fn expert_count(self) -> usize {
        self.expert_count
    }
    pub fn code_bytes(self) -> usize {
        self.code_bytes
    }
    pub fn scale_bytes(self) -> usize {
        self.scale_bytes
    }
    pub fn matrix_bytes(self) -> usize {
        self.matrix_bytes
    }
    pub fn expert_bytes(self) -> usize {
        self.expert_bytes
    }

    pub fn total_bytes(self) -> Result<usize, String> {
        self.expert_count.checked_mul(self.expert_bytes).ok_or_else(|| "NVFP4 archive 总大小溢出".to_owned())
    }

    pub fn expert_offset(self, expert: usize) -> Result<usize, String> {
        if expert >= self.expert_count {
            return Err(format!("NVFP4 archive expert 越界: {expert} >= {}", self.expert_count));
        }
        expert.checked_mul(self.expert_bytes).ok_or_else(|| "NVFP4 archive expert offset 溢出".to_owned())
    }

    pub fn matrix_offset(self, expert: usize, matrix: usize) -> Result<usize, String> {
        if matrix >= 3 {
            return Err(format!("NVFP4 archive matrix 越界: {matrix}"));
        }
        self.expert_offset(expert)?.checked_add(matrix.checked_mul(self.matrix_bytes).ok_or_else(|| "NVFP4 archive matrix offset 溢出".to_owned())?).ok_or_else(|| "NVFP4 archive matrix offset 溢出".to_owned())
    }
}

impl<'a> Nvfp4MatrixBufferMut<'a> {
    pub fn new(codes: &'a mut [u8], scales: &'a mut [u8], global_scale: &'a mut [u8], rows: usize, cols: usize) -> Result<Self, String> {
        let (code_bytes, scale_bytes) = nvfp4_storage_lengths(rows, cols)?;
        if codes.len() != code_bytes || scales.len() != scale_bytes || global_scale.len() != 4 {
            return Err(format!("NVFP4 destination 长度 codes={}/{code_bytes} scales={}/{scale_bytes} global_scale={}/4", codes.len(), scales.len(), global_scale.len(),));
        }
        Ok(Self { codes, scales, global_scale, rows, cols })
    }

    pub fn copy_from(&mut self, weight: &Nvfp4Matrix) -> Result<(), String> {
        if weight.rows != self.rows || weight.cols != self.cols {
            return Err(format!("NVFP4 destination shape [{},{}] 与权重 [{},{}] 不符", self.rows, self.cols, weight.rows, weight.cols));
        }
        self.codes.copy_from_slice(weight.codes());
        self.scales.copy_from_slice(weight.scales());
        self.global_scale.copy_from_slice(&weight.global_scale.to_le_bytes());
        Ok(())
    }
}

pub struct Nvfp4ExpertWeights {
    pub gate: Nvfp4Matrix,
    pub up: Nvfp4Matrix,
    pub down: Nvfp4Matrix,
}

impl Nvfp4ExpertWeights {
    pub fn decode(&self) -> Result<ExpertF32, String> {
        Ok(ExpertF32 { gate: self.gate.decode()?, up: self.up.decode()?, down: self.down.decode()? })
    }
}

#[derive(Clone)]
pub struct NvidiaNvfp4Experts {
    store: SafetensorStore,
    intermediate: usize,
    hidden: usize,
    expert_count: usize,
    layout: NvidiaNvfp4ExpertLayout,
    archive_dir: Option<PathBuf>,
    archives: Arc<Mutex<HashMap<usize, Arc<File>>>>,
}

#[derive(Clone, Copy)]
enum NvidiaNvfp4ExpertLayout {
    Glm52,
    MiniMaxM3,
}

impl NvidiaNvfp4Experts {
    pub fn open(root: &Path, intermediate: usize, hidden: usize, expert_count: usize) -> Result<Self, String> {
        Self::open_with_layout(root, intermediate, hidden, expert_count, NvidiaNvfp4ExpertLayout::Glm52)
    }

    pub fn open_minimax_m3(root: &Path, intermediate: usize, hidden: usize, expert_count: usize) -> Result<Self, String> {
        Self::open_with_layout(root, intermediate, hidden, expert_count, NvidiaNvfp4ExpertLayout::MiniMaxM3)
    }

    fn open_with_layout(root: &Path, intermediate: usize, hidden: usize, expert_count: usize, layout: NvidiaNvfp4ExpertLayout) -> Result<Self, String> {
        if expert_count == 0 {
            return Err("NVFP4 experts expert_count 必须大于 0".to_owned());
        }
        Ok(Self { store: SafetensorStore::open(root)?, intermediate, hidden, expert_count, layout, archive_dir: None, archives: Arc::new(Mutex::new(HashMap::new())) })
    }

    pub fn with_archive_dir(mut self, archive_dir: impl AsRef<Path>) -> Self {
        self.archive_dir = Some(archive_dir.as_ref().to_path_buf());
        self
    }

    pub fn intermediate(&self) -> usize {
        self.intermediate
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    pub fn load_expert(&self, layer: usize, expert: usize) -> Result<Nvfp4ExpertWeights, String> {
        if let Some(file) = self.archive_file(layer)? {
            return self.load_archive_expert(&file, expert);
        }
        let [gate, up, down] = self.expert_matrix_names(layer, expert);
        Ok(Nvfp4ExpertWeights { gate: self.load_matrix(&gate, self.intermediate, self.hidden)?, up: self.load_matrix(&up, self.intermediate, self.hidden)?, down: self.load_matrix(&down, self.hidden, self.intermediate)? })
    }

    pub fn load_expert_into(&self, layer: usize, expert: usize, gate: Nvfp4MatrixBufferMut<'_>, up: Nvfp4MatrixBufferMut<'_>, down: Nvfp4MatrixBufferMut<'_>) -> Result<(), String> {
        if let Some(file) = self.archive_file(layer)? {
            return self.load_archive_expert_into(&file, expert, gate, up, down);
        }
        let [gate_name, up_name, down_name] = self.expert_matrix_names(layer, expert);
        self.load_matrix_into(&gate_name, gate)?;
        self.load_matrix_into(&up_name, up)?;
        self.load_matrix_into(&down_name, down)
    }

    pub fn load_experts_into(&self, layer: usize, experts: Vec<Nvfp4ExpertBufferMut<'_>>) -> Result<(), String> {
        if let Some(file) = self.archive_file(layer)? {
            for expert in experts {
                self.load_archive_expert_into(&file, expert.expert, expert.gate, expert.up, expert.down)?;
            }
            return Ok(());
        }
        enum Expected {
            Weight { rows: usize, cols: usize, bytes: usize },
            Scales { rows: usize, cols: usize, bytes: usize },
            GlobalScale { address: usize },
        }

        fn append_matrix<'a>(loads: &mut Vec<TensorDestination<'a>>, expected: &mut Vec<Expected>, base: String, matrix: Nvfp4MatrixBufferMut<'a>) {
            let Nvfp4MatrixBufferMut { codes, scales, global_scale, rows, cols } = matrix;
            let global_scale_address = global_scale.as_ptr() as usize;
            expected.push(Expected::Weight { rows, cols, bytes: codes.len() });
            loads.push(TensorDestination { name: format!("{base}.weight"), data: codes });
            expected.push(Expected::Scales { rows, cols, bytes: scales.len() });
            loads.push(TensorDestination { name: format!("{base}.weight_scale"), data: scales });
            expected.push(Expected::GlobalScale { address: global_scale_address });
            loads.push(TensorDestination { name: format!("{base}.weight_scale_2"), data: global_scale });
        }

        let mut loads = Vec::with_capacity(experts.len() * 9);
        let mut expected = Vec::with_capacity(experts.len() * 9);
        for expert in experts {
            let [gate, up, down] = self.expert_matrix_names(layer, expert.expert);
            append_matrix(&mut loads, &mut expected, gate, expert.gate);
            append_matrix(&mut loads, &mut expected, up, expert.up);
            append_matrix(&mut loads, &mut expected, down, expert.down);
        }
        let infos = self.store.load_many_into(loads)?;
        for (info, expected) in infos.iter().zip(expected) {
            match expected {
                Expected::Weight { rows, cols, bytes } => validate_weight_info(info, bytes, rows, cols)?,
                Expected::Scales { rows, cols, bytes } => validate_scales_info(info, bytes, rows, cols)?,
                Expected::GlobalScale { address } => {
                    if info.dtype != "F32" || !(info.shape.is_empty() || info.shape == [1]) {
                        return Err(format!("{} 需要 F32 scalar，dtype={} shape={:?} bytes=4", info.name, info.dtype, info.shape));
                    }
                    let bytes = unsafe { std::slice::from_raw_parts(address as *const u8, 4) };
                    let value = f32::from_le_bytes(bytes.try_into().expect("global scale 长度固定为 4"));
                    if !value.is_finite() || value <= 0.0 {
                        return Err(format!("{} global scale 无效: {value}", info.name));
                    }
                }
            }
        }
        Ok(())
    }

    fn expert_matrix_names(&self, layer: usize, expert: usize) -> [String; 3] {
        match self.layout {
            NvidiaNvfp4ExpertLayout::Glm52 => {
                let prefix = format!("model.layers.{layer}.mlp.experts.{expert}");
                [format!("{prefix}.gate_proj"), format!("{prefix}.up_proj"), format!("{prefix}.down_proj")]
            }
            NvidiaNvfp4ExpertLayout::MiniMaxM3 => {
                let prefix = format!("language_model.model.layers.{layer}.block_sparse_moe.experts.{expert}");
                [format!("{prefix}.w1"), format!("{prefix}.w3"), format!("{prefix}.w2")]
            }
        }
    }

    pub fn layer_archive_layout(&self, layer: usize, expert_count: usize) -> Result<Option<Nvfp4ArchiveLayout>, String> {
        let Some(file) = self.archive_file(layer)? else { return Ok(None) };
        let layout = Nvfp4ArchiveLayout::new(expert_count, self.intermediate, self.hidden)?;
        let expected = u64::try_from(layout.total_bytes()?).map_err(|_| "NVFP4 archive 大小超出 u64".to_owned())?;
        let actual = file.metadata().map_err(|error| format!("读取 L{layer} NVFP4 archive metadata: {error}"))?.len();
        if actual != expected {
            return Err(format!("L{layer} NVFP4 archive 大小 {actual}，期望 {expected}"));
        }
        Ok(Some(layout))
    }

    pub fn read_layer_archive_range(&self, layer: usize, offset: usize, destination: &mut [u8]) -> Result<(), String> {
        let file = self.archive_file(layer)?.ok_or_else(|| format!("L{layer} NVFP4 archive 不存在"))?;
        let end = offset.checked_add(destination.len()).ok_or_else(|| format!("L{layer} NVFP4 archive range 溢出"))?;
        let file_len = usize::try_from(file.metadata().map_err(|error| format!("读取 L{layer} NVFP4 archive metadata: {error}"))?.len()).map_err(|_| format!("L{layer} NVFP4 archive 大小超出 usize"))?;
        if end > file_len {
            return Err(format!("L{layer} NVFP4 archive range [{offset},{end}) 超过 {file_len}"));
        }
        file.read_exact_at(destination, offset as u64).map_err(|error| format!("读取 L{layer} NVFP4 archive [{offset},{end}): {error}"))
    }

    fn archive_file(&self, layer: usize) -> Result<Option<Arc<File>>, String> {
        if let Some(file) = self.archives.lock().map_err(|_| "NVFP4 archive cache 锁中毒".to_owned())?.get(&layer).cloned() {
            return Ok(Some(file));
        }
        let Some(dir) = &self.archive_dir else { return Ok(None) };
        let path = dir.join(format!("nvfp4-layer{layer}.bin"));
        if !path.is_file() {
            return Ok(None);
        }
        let file = Arc::new(File::open(&path).map_err(|error| format!("打开 NVFP4 archive {}: {error}", path.display()))?);
        self.archives.lock().map_err(|_| "NVFP4 archive cache 锁中毒".to_owned())?.insert(layer, file.clone());
        Ok(Some(file))
    }

    fn load_archive_expert(&self, file: &File, expert: usize) -> Result<Nvfp4ExpertWeights, String> {
        let layout = Nvfp4ArchiveLayout::new(self.expert_count, self.intermediate, self.hidden)?;
        Ok(Nvfp4ExpertWeights {
            gate: self.load_archive_matrix(file, layout, expert, 0, self.intermediate, self.hidden)?,
            up: self.load_archive_matrix(file, layout, expert, 1, self.intermediate, self.hidden)?,
            down: self.load_archive_matrix(file, layout, expert, 2, self.hidden, self.intermediate)?,
        })
    }

    fn load_archive_matrix(&self, file: &File, layout: Nvfp4ArchiveLayout, expert: usize, matrix: usize, rows: usize, cols: usize) -> Result<Nvfp4Matrix, String> {
        let base = layout.matrix_offset(expert, matrix)?;
        let mut codes = vec![0u8; layout.code_bytes()];
        let mut scales = vec![0u8; layout.scale_bytes()];
        let mut global_scale = [0u8; 4];
        file.read_exact_at(&mut codes, base as u64).map_err(|error| format!("读取 NVFP4 archive expert {expert} matrix {matrix} codes: {error}"))?;
        file.read_exact_at(&mut scales, (base + layout.code_bytes()) as u64).map_err(|error| format!("读取 NVFP4 archive expert {expert} matrix {matrix} scales: {error}"))?;
        file.read_exact_at(&mut global_scale, (base + layout.code_bytes() + layout.scale_bytes()) as u64).map_err(|error| format!("读取 NVFP4 archive expert {expert} matrix {matrix} global scale: {error}"))?;
        Nvfp4Matrix::new(codes, scales, f32::from_le_bytes(global_scale), rows, cols)
    }

    fn load_archive_expert_into(&self, file: &File, expert: usize, gate: Nvfp4MatrixBufferMut<'_>, up: Nvfp4MatrixBufferMut<'_>, down: Nvfp4MatrixBufferMut<'_>) -> Result<(), String> {
        let layout = Nvfp4ArchiveLayout::new(self.expert_count, self.intermediate, self.hidden)?;
        for (matrix, destination) in [gate, up, down].into_iter().enumerate() {
            let base = layout.matrix_offset(expert, matrix)?;
            file.read_exact_at(destination.codes, base as u64).map_err(|error| format!("读取 NVFP4 archive expert {expert} matrix {matrix} codes: {error}"))?;
            file.read_exact_at(destination.scales, (base + layout.code_bytes()) as u64).map_err(|error| format!("读取 NVFP4 archive expert {expert} matrix {matrix} scales: {error}"))?;
            file.read_exact_at(destination.global_scale, (base + layout.code_bytes() + layout.scale_bytes()) as u64).map_err(|error| format!("读取 NVFP4 archive expert {expert} matrix {matrix} global scale: {error}"))?;
        }
        Ok(())
    }

    fn load_matrix(&self, base: &str, rows: usize, cols: usize) -> Result<Nvfp4Matrix, String> {
        load_modelopt_matrix(&self.store, base, rows, cols)
    }

    fn load_matrix_into(&self, base: &str, destination: Nvfp4MatrixBufferMut<'_>) -> Result<(), String> {
        let weight = self.store.load_into(&format!("{base}.weight"), destination.codes)?;
        let scales = self.store.load_into(&format!("{base}.weight_scale"), destination.scales)?;
        let scale_2 = self.store.load_into(&format!("{base}.weight_scale_2"), destination.global_scale)?;
        validate_weight_info(&weight, destination.codes.len(), destination.rows, destination.cols)?;
        validate_scales_info(&scales, destination.scales.len(), destination.rows, destination.cols)?;
        if scale_2.dtype != "F32" || !(scale_2.shape.is_empty() || scale_2.shape == [1]) {
            return Err(format!("{} 需要 F32 scalar，dtype={} shape={:?} bytes=4", scale_2.name, scale_2.dtype, scale_2.shape));
        }
        let global_scale = f32::from_le_bytes(destination.global_scale.try_into().expect("长度已经校验"));
        if !global_scale.is_finite() || global_scale <= 0.0 {
            return Err(format!("{} global scale 无效: {global_scale}", scale_2.name));
        }
        Ok(())
    }
}

/// 从 ModelOpt HF safetensors 布局加载单个 NVFP4 矩阵。
///
/// 命名与校验和 expert 路径一致:`{base}.weight`(U8/F4 packed `[rows, cols/2]`)、
/// `{base}.weight_scale`(F8_E4M3 block scale `[rows, cols/16]`)、`{base}.weight_scale_2`
/// (F32 scalar 全局 scale)。dense Linear(如经典 Qwen3)与 MoE expert 共用此布局。
pub fn load_modelopt_matrix(store: &SafetensorStore, base: &str, rows: usize, cols: usize) -> Result<Nvfp4Matrix, String> {
    let weight = store.load(&format!("{base}.weight"))?;
    let scales = store.load(&format!("{base}.weight_scale"))?;
    let scale_2 = store.load(&format!("{base}.weight_scale_2"))?;
    validate_weight(&weight, rows, cols)?;
    validate_scales(&scales, rows, cols)?;
    if scale_2.dtype != "F32" || scale_2.data.len() != 4 || !(scale_2.shape.is_empty() || scale_2.shape == [1]) {
        return Err(format!("{} 需要 F32 scalar，dtype={} shape={:?} bytes={}", scale_2.name, scale_2.dtype, scale_2.shape, scale_2.data.len()));
    }
    let global_scale = f32::from_le_bytes(scale_2.data[..4].try_into().expect("长度已经校验"));
    Nvfp4Matrix::new(weight.data, scales.data, global_scale, rows, cols)
}

fn validate_weight(tensor: &TensorData, rows: usize, cols: usize) -> Result<(), String> {
    validate_weight_layout(&tensor.name, &tensor.dtype, &tensor.shape, tensor.data.len(), rows, cols)
}

fn validate_weight_info(tensor: &TensorInfo, bytes: usize, rows: usize, cols: usize) -> Result<(), String> {
    validate_weight_layout(&tensor.name, &tensor.dtype, &tensor.shape, bytes, rows, cols)
}

fn validate_weight_layout(name: &str, dtype: &str, shape: &[usize], bytes: usize, rows: usize, cols: usize) -> Result<(), String> {
    let expected = rows.checked_mul(cols).and_then(|n| n.checked_div(2)).ok_or_else(|| "NVFP4 weight 大小溢出".to_owned())?;
    let packed_shape = shape == [rows, cols / 2];
    let logical_shape = shape == [rows, cols];
    let supported_dtype = matches!(dtype, "U8" | "F4" | "F4_E2M1" | "F4_E2M1FN_X2");
    if !supported_dtype || (!packed_shape && !logical_shape) || bytes != expected {
        return Err(format!("{} 不是 ModelOpt packed NVFP4: dtype={} shape={:?} bytes={}，期望 [{rows},{}] U8/F4、{expected} bytes", name, dtype, shape, bytes, cols / 2,));
    }
    Ok(())
}

fn validate_scales(tensor: &TensorData, rows: usize, cols: usize) -> Result<(), String> {
    validate_scales_layout(&tensor.name, &tensor.dtype, &tensor.shape, tensor.data.len(), rows, cols)
}

fn validate_scales_info(tensor: &TensorInfo, bytes: usize, rows: usize, cols: usize) -> Result<(), String> {
    validate_scales_layout(&tensor.name, &tensor.dtype, &tensor.shape, bytes, rows, cols)
}

fn validate_scales_layout(name: &str, dtype: &str, shape: &[usize], bytes: usize, rows: usize, cols: usize) -> Result<(), String> {
    let expected = rows.checked_mul(cols / NVFP4_BLOCK).ok_or_else(|| "NVFP4 scale 大小溢出".to_owned())?;
    if dtype != "F8_E4M3" || bytes != expected || shape.iter().product::<usize>() != expected {
        return Err(format!("{} 不是 NVFP4 E4M3 block scale: dtype={} shape={:?} bytes={}，期望 {expected}", name, dtype, shape, bytes,));
    }
    Ok(())
}

pub fn nvfp4_storage_lengths(rows: usize, cols: usize) -> Result<(usize, usize), String> {
    if rows == 0 || cols == 0 || !cols.is_multiple_of(NVFP4_BLOCK) || !cols.is_multiple_of(2) {
        return Err(format!("NVFP4 shape [{rows},{cols}] 无效，cols 必须按 {NVFP4_BLOCK} 对齐"));
    }
    let elements = rows.checked_mul(cols).ok_or_else(|| "NVFP4 矩阵大小溢出".to_owned())?;
    let scales = rows.checked_mul(cols / NVFP4_BLOCK).ok_or_else(|| "NVFP4 scale 大小溢出".to_owned())?;
    Ok((elements / 2, scales))
}
