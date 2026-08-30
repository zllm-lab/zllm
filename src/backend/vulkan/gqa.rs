use super::{BackendError, VulkanContext, VulkanTensor, compute};

pub struct VulkanKvCache {
    layers: Vec<Option<VulkanKvLayer>>,
    max_seq_len: usize,
    q8: bool,
}

struct VulkanKvLayer {
    key: wgpu::Buffer,
    value: wgpu::Buffer,
    key_scales: Option<wgpu::Buffer>,
    value_scales: Option<wgpu::Buffer>,
    rows: usize,
    cols: usize,
}

impl VulkanKvCache {
    pub fn new(layer_count: usize, max_seq_len: usize) -> Result<Self, BackendError> {
        if layer_count == 0 || max_seq_len == 0 {
            return Err(compute(format!("Vulkan KV cache 参数无效: layers={layer_count} max_seq={max_seq_len}")));
        }
        Ok(Self { layers: (0..layer_count).map(|_| None).collect(), max_seq_len, q8: false })
    }

    pub fn new_q8(layer_count: usize, max_seq_len: usize) -> Result<Self, BackendError> {
        let mut cache = Self::new(layer_count, max_seq_len)?;
        cache.q8 = true;
        Ok(cache)
    }

    pub fn rows(&self, layer: usize) -> Option<usize> {
        self.layers.get(layer).and_then(Option::as_ref).map(|layer| layer.rows)
    }
}

impl VulkanContext {
    pub(super) fn append_gqa_cache(&self, cache: &mut VulkanKvCache, layer: usize, position: usize, key: &VulkanTensor, value: &VulkanTensor) -> Result<(), BackendError> {
        if key.rows != value.rows || key.cols != value.cols || position.checked_add(key.rows).is_none_or(|end| end > cache.max_seq_len) {
            return Err(compute(format!("Vulkan GQA cache append 无效: L{layer} position={position} K=[{},{}] V=[{},{}] max={}", key.rows, key.cols, value.rows, value.cols, cache.max_seq_len)));
        }
        let slot = cache.layers.get_mut(layer).ok_or(BackendError::UnsupportedLayer { layer })?;
        if slot.is_none() {
            let elements = cache.max_seq_len.checked_mul(key.cols).ok_or_else(|| compute("Vulkan KV cache 大小溢出"))?;
            let bytes = if cache.q8 { elements.div_ceil(4) * 4 } else { elements * size_of::<f32>() } as u64;
            let create = |label, size| self.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            let (key_scales, value_scales) = if cache.q8 {
                if !key.cols.is_multiple_of(crate::kv_cache::DEFAULT_GROUP_SIZE) {
                    return Err(compute(format!("Vulkan Q8 KV cols={} 不能按 Q8G{} 分组", key.cols, crate::kv_cache::DEFAULT_GROUP_SIZE)));
                }
                // 每个 scale 独占一个 u32，低 16 位保存 f16；避免并行 append 时两个 workgroup 修改同一 packed word。
                let scale_bytes = cache.max_seq_len * key.cols / crate::kv_cache::DEFAULT_GROUP_SIZE * 4;
                (Some(create("Vulkan GQA key scales", scale_bytes as u64)), Some(create("Vulkan GQA value scales", scale_bytes as u64)))
            } else {
                (None, None)
            };
            *slot = Some(VulkanKvLayer { key: create("Vulkan GQA key cache", bytes), value: create("Vulkan GQA value cache", bytes), key_scales, value_scales, rows: 0, cols: key.cols });
        }
        let cached = slot.as_mut().expect("KV layer 已创建");
        if cached.rows != position || cached.cols != key.cols {
            return Err(compute(format!("Vulkan GQA cache L{layer} 状态不连续: rows={} position={position} cols={}/{}", cached.rows, cached.cols, key.cols)));
        }
        if let (Some(key_scales), Some(value_scales)) = (&cached.key_scales, &cached.value_scales) {
            let params = [position as u32, key.rows as u32, key.cols as u32, crate::kv_cache::DEFAULT_GROUP_SIZE as u32];
            let params_buffer = self.uniform_buffer("Vulkan Q8 KV append 参数", super::u32_bytes(&params));
            let layout = self.gqa_q8_append_pipeline.get_bind_group_layout(0);
            let bind_group = self.bind_group(
                "Vulkan Q8 KV append",
                &layout,
                &[
                    super::binding(0, &key.buffer),
                    super::binding(1, &value.buffer),
                    super::binding(2, &cached.key),
                    super::binding(3, &cached.value),
                    super::binding(4, key_scales),
                    super::binding(5, value_scales),
                    super::binding(6, &params_buffer),
                ],
            );
            let mut encoder = self.device.create_command_encoder(&Default::default());
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.gqa_q8_append_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((key.rows * key.cols / crate::kv_cache::DEFAULT_GROUP_SIZE) as u32, 1, 1);
            drop(pass);
            self.queue.submit([encoder.finish()]);
        } else {
            let bytes = key.rows * key.cols * size_of::<f32>();
            let offset = position * key.cols * size_of::<f32>();
            let mut encoder = self.device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(&key.buffer, 0, &cached.key, offset as u64, bytes as u64);
            encoder.copy_buffer_to_buffer(&value.buffer, 0, &cached.value, offset as u64, bytes as u64);
            self.queue.submit([encoder.finish()]);
        }
        cached.rows += key.rows;
        Ok(())
    }

    pub(super) fn gqa_cached(&self, cache: &VulkanKvCache, layer: usize, position: usize, query: &VulkanTensor, spec: &crate::attention::gqa::GqaSpec) -> Result<VulkanTensor, BackendError> {
        let cached = cache.layers.get(layer).and_then(Option::as_ref).ok_or(BackendError::UnsupportedLayer { layer })?;
        if let (Some(key_scales), Some(value_scales)) = (&cached.key_scales, &cached.value_scales) {
            self.gqa_attention_q8(query, &cached.key, &cached.value, key_scales, value_scales, cached.rows, cached.cols, position, spec)
        } else {
            self.gqa_attention(query, &cached.key, &cached.value, cached.rows, cached.cols, position, spec)
        }
    }

    pub(super) fn gqa_uncached(&self, query: &VulkanTensor, key: &VulkanTensor, value: &VulkanTensor, spec: &crate::attention::gqa::GqaSpec) -> Result<VulkanTensor, BackendError> {
        if key.rows != value.rows || key.cols != value.cols {
            return Err(compute("Vulkan uncached GQA K/V shape 不一致"));
        }
        self.gqa_attention(query, &key.buffer, &value.buffer, key.rows, key.cols, 0, spec)
    }

    fn gqa_attention(&self, query: &VulkanTensor, key: &wgpu::Buffer, value: &wgpu::Buffer, kv_rows: usize, kv_cols: usize, position: usize, spec: &crate::attention::gqa::GqaSpec) -> Result<VulkanTensor, BackendError> {
        let query_cols = spec.num_heads * spec.head_dim;
        if query.cols != query_cols
            || kv_cols != spec.num_kv_heads * spec.head_dim
            || spec.head_dim > 256
            || spec.num_heads == 0
            || !spec.num_heads.is_multiple_of(spec.num_kv_heads)
            || !matches!(spec.window, crate::attention::gqa::CausalWindow::Full)
        {
            return Err(compute(format!("Vulkan GQA spec/shape 不支持: Q=[{},{}] KV=[{kv_rows},{kv_cols}] heads={}/{} dim={} window={:?}", query.rows, query.cols, spec.num_heads, spec.num_kv_heads, spec.head_dim, spec.window)));
        }
        let output = self.output_buffer(query.rows, query.cols, "Vulkan GQA output")?;
        let params = [query.rows as u32, spec.num_heads as u32, spec.num_kv_heads as u32, spec.head_dim as u32, kv_rows as u32, position as u32, spec.score_scale.to_bits(), 0];
        let params_buffer = self.uniform_buffer("Vulkan GQA 参数", super::u32_bytes(&params));
        let layout = self.gqa_pipeline.get_bind_group_layout(0);
        let bind_group = self.bind_group("Vulkan GQA", &layout, &[super::binding(0, &query.buffer), super::binding(1, key), super::binding(2, value), super::binding(3, &output), super::binding(4, &params_buffer)]);
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.gqa_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(spec.num_heads as u32, query.rows as u32, 1);
        }
        self.queue.submit([encoder.finish()]);
        Ok(VulkanTensor { buffer: output, rows: query.rows, cols: query.cols })
    }

    fn gqa_attention_q8(
        &self,
        query: &VulkanTensor,
        key: &wgpu::Buffer,
        value: &wgpu::Buffer,
        key_scales: &wgpu::Buffer,
        value_scales: &wgpu::Buffer,
        kv_rows: usize,
        kv_cols: usize,
        position: usize,
        spec: &crate::attention::gqa::GqaSpec,
    ) -> Result<VulkanTensor, BackendError> {
        let query_cols = spec.num_heads * spec.head_dim;
        if query.cols != query_cols
            || kv_cols != spec.num_kv_heads * spec.head_dim
            || spec.head_dim > 256
            || !spec.head_dim.is_multiple_of(crate::kv_cache::DEFAULT_GROUP_SIZE)
            || spec.num_heads == 0
            || !spec.num_heads.is_multiple_of(spec.num_kv_heads)
            || !matches!(spec.window, crate::attention::gqa::CausalWindow::Full)
        {
            return Err(compute(format!("Vulkan Q8 GQA spec/shape 不支持: Q=[{},{}] KV=[{kv_rows},{kv_cols}] heads={}/{} dim={}", query.rows, query.cols, spec.num_heads, spec.num_kv_heads, spec.head_dim)));
        }
        let output = self.output_buffer(query.rows, query.cols, "Vulkan Q8 GQA output")?;
        let params = [query.rows as u32, spec.num_heads as u32, spec.num_kv_heads as u32, spec.head_dim as u32, kv_rows as u32, position as u32, spec.score_scale.to_bits(), crate::kv_cache::DEFAULT_GROUP_SIZE as u32];
        let params_buffer = self.uniform_buffer("Vulkan Q8 GQA 参数", super::u32_bytes(&params));
        let layout = self.gqa_q8_pipeline.get_bind_group_layout(0);
        let bind_group = self.bind_group(
            "Vulkan Q8 GQA",
            &layout,
            &[super::binding(0, &query.buffer), super::binding(1, key), super::binding(2, value), super::binding(3, key_scales), super::binding(4, value_scales), super::binding(5, &output), super::binding(6, &params_buffer)],
        );
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.gqa_q8_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(spec.num_heads as u32, query.rows as u32, 1);
        drop(pass);
        self.queue.submit([encoder.finish()]);
        Ok(VulkanTensor { buffer: output, rows: query.rows, cols: query.cols })
    }
}

pub const SHADER: &str = r#"
struct Params { query_rows: u32, query_heads: u32, kv_heads: u32, head_dim: u32, kv_rows: u32, position: u32, score_scale: f32, _pad: u32 }
@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> key: array<f32>;
@group(0) @binding(2) var<storage, read> value: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;
var<workgroup> partial: array<f32, 256>;
var<workgroup> shared_max: f32;
var<workgroup> shared_exp: f32;
var<workgroup> denominator: f32;

fn reduce_sum(local: u32) {
    var stride = 128u;
    loop {
        if local < stride { partial[local] += partial[local + stride]; }
        workgroupBarrier();
        if stride == 1u { break; }
        stride /= 2u;
    }
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let head = group.x;
    let token = group.y;
    let local = local_id.x;
    let group_size = params.query_heads / params.kv_heads;
    let kv_head = head / group_size;
    let query_base = token * params.query_heads * params.head_dim + head * params.head_dim;
    let causal_end = min(params.kv_rows, params.position + token + 1u);
    if local == 0u { shared_max = -3.402823466e+38; }
    workgroupBarrier();
    for (var source = 0u; source < causal_end; source += 1u) {
        let kv_base = source * params.kv_heads * params.head_dim + kv_head * params.head_dim;
        if local < params.head_dim { partial[local] = query[query_base + local] * key[kv_base + local]; } else { partial[local] = 0.0; }
        workgroupBarrier();
        reduce_sum(local);
        if local == 0u { shared_max = max(shared_max, partial[0] * params.score_scale); }
        workgroupBarrier();
    }
    var accumulated = 0.0;
    if local == 0u { denominator = 0.0; }
    workgroupBarrier();
    for (var source = 0u; source < causal_end; source += 1u) {
        let kv_base = source * params.kv_heads * params.head_dim + kv_head * params.head_dim;
        if local < params.head_dim { partial[local] = query[query_base + local] * key[kv_base + local]; } else { partial[local] = 0.0; }
        workgroupBarrier();
        reduce_sum(local);
        if local == 0u {
            shared_exp = exp(partial[0] * params.score_scale - shared_max);
            denominator += shared_exp;
        }
        workgroupBarrier();
        if local < params.head_dim { accumulated += shared_exp * value[kv_base + local]; }
        workgroupBarrier();
    }
    if local < params.head_dim {
        output[token * params.query_heads * params.head_dim + head * params.head_dim + local] = accumulated / denominator;
    }
}
"#;

pub const Q8_APPEND_SHADER: &str = r#"
struct Params { position: u32, rows: u32, cols: u32, group_size: u32 }
@group(0) @binding(0) var<storage, read> key: array<f32>;
@group(0) @binding(1) var<storage, read> value: array<f32>;
@group(0) @binding(2) var<storage, read_write> key_codes: array<u32>;
@group(0) @binding(3) var<storage, read_write> value_codes: array<u32>;
@group(0) @binding(4) var<storage, read_write> key_scales: array<u32>;
@group(0) @binding(5) var<storage, read_write> value_scales: array<u32>;
@group(0) @binding(6) var<uniform> params: Params;
var<workgroup> key_abs: array<f32, 64>;
var<workgroup> value_abs: array<f32, 64>;
var<workgroup> key_scale: f32;
var<workgroup> value_scale: f32;

fn signed_byte(value: f32, scale: f32) -> u32 {
    return u32(i32(clamp(round(value / scale), -127.0, 127.0))) & 255u;
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let local = local_id.x;
    let groups_per_row = params.cols / params.group_size;
    let row = group.x / groups_per_row;
    let column = (group.x % groups_per_row) * params.group_size + local;
    let input = row * params.cols + column;
    key_abs[local] = abs(key[input]);
    value_abs[local] = abs(value[input]);
    workgroupBarrier();
    var stride = 32u;
    loop {
        if local < stride {
            key_abs[local] = max(key_abs[local], key_abs[local + stride]);
            value_abs[local] = max(value_abs[local], value_abs[local + stride]);
        }
        workgroupBarrier();
        if stride == 1u { break; }
        stride /= 2u;
    }
    if local == 0u {
        key_scale = max(key_abs[0] / 127.0, 1.0e-12);
        value_scale = max(value_abs[0] / 127.0, 1.0e-12);
        let scale_index = (params.position + row) * groups_per_row + group.x % groups_per_row;
        let packed_key = pack2x16float(vec2<f32>(key_scale, key_scale));
        let packed_value = pack2x16float(vec2<f32>(value_scale, value_scale));
        key_scales[scale_index] = packed_key & 0xffffu;
        value_scales[scale_index] = packed_value & 0xffffu;
    }
    workgroupBarrier();
    if local < 16u {
        let base = input + local * 3u;
        let cache_base = (params.position + row) * params.cols + (group.x % groups_per_row) * params.group_size + local * 4u;
        var kp = 0u;
        var vp = 0u;
        for (var lane = 0u; lane < 4u; lane += 1u) {
            kp |= signed_byte(key[base + lane], key_scale) << (lane * 8u);
            vp |= signed_byte(value[base + lane], value_scale) << (lane * 8u);
        }
        key_codes[cache_base / 4u] = kp;
        value_codes[cache_base / 4u] = vp;
    }
}
"#;

pub const Q8_SHADER: &str = r#"
struct Params { query_rows: u32, query_heads: u32, kv_heads: u32, head_dim: u32, kv_rows: u32, position: u32, score_scale: f32, group_size: u32 }
@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> key: array<u32>;
@group(0) @binding(2) var<storage, read> value: array<u32>;
@group(0) @binding(3) var<storage, read> key_scales: array<u32>;
@group(0) @binding(4) var<storage, read> value_scales: array<u32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;
var<workgroup> partial: array<f32, 256>;
var<workgroup> shared_max: f32;
var<workgroup> shared_exp: f32;
var<workgroup> denominator: f32;

fn signed_code(word: u32, index: u32) -> f32 {
    let byte = (word >> ((index & 3u) * 8u)) & 255u;
    return f32(select(i32(byte), i32(byte) - 256, byte >= 128u));
}

fn key_code(index: u32) -> f32 { return signed_code(key[index / 4u], index); }
fn value_code(index: u32) -> f32 { return signed_code(value[index / 4u], index); }
fn key_scale(index: u32) -> f32 { return unpack2x16float(key_scales[index]).x; }
fn value_scale(index: u32) -> f32 { return unpack2x16float(value_scales[index]).x; }

fn reduce_sum(local: u32) {
    var stride = 128u;
    loop {
        if local < stride { partial[local] += partial[local + stride]; }
        workgroupBarrier();
        if stride == 1u { break; }
        stride /= 2u;
    }
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let head = group.x;
    let token = group.y;
    let local = local_id.x;
    let kv_head = head / (params.query_heads / params.kv_heads);
    let query_base = token * params.query_heads * params.head_dim + head * params.head_dim;
    let causal_end = min(params.kv_rows, params.position + token + 1u);
    let groups_per_row = params.kv_heads * params.head_dim / params.group_size;
    if local == 0u { shared_max = -3.402823466e+38; }
    workgroupBarrier();
    for (var source = 0u; source < causal_end; source += 1u) {
        let kv_base = source * params.kv_heads * params.head_dim + kv_head * params.head_dim;
        if local < params.head_dim {
            let group_index = source * groups_per_row + kv_head * params.head_dim / params.group_size + local / params.group_size;
            partial[local] = query[query_base + local] * key_code(kv_base + local) * key_scale(group_index);
        } else { partial[local] = 0.0; }
        workgroupBarrier();
        reduce_sum(local);
        if local == 0u { shared_max = max(shared_max, partial[0] * params.score_scale); }
        workgroupBarrier();
    }
    var accumulated = 0.0;
    if local == 0u { denominator = 0.0; }
    workgroupBarrier();
    for (var source = 0u; source < causal_end; source += 1u) {
        let kv_base = source * params.kv_heads * params.head_dim + kv_head * params.head_dim;
        let group_index = source * groups_per_row + kv_head * params.head_dim / params.group_size + local / params.group_size;
        if local < params.head_dim { partial[local] = query[query_base + local] * key_code(kv_base + local) * key_scale(group_index); } else { partial[local] = 0.0; }
        workgroupBarrier();
        reduce_sum(local);
        if local == 0u { shared_exp = exp(partial[0] * params.score_scale - shared_max); denominator += shared_exp; }
        workgroupBarrier();
        if local < params.head_dim { accumulated += shared_exp * value_code(kv_base + local) * value_scale(group_index); }
        workgroupBarrier();
    }
    if local < params.head_dim { output[query_base + local] = accumulated / denominator; }
}
"#;
