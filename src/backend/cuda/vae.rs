//! CUDA 后端的扩散(DiT)与 VAE 能力。
//!
//! 对称 `backend/metal/vae.rs` 与 `backend/rocm/vae.rs`。逐方法对齐
//! `backend/rocm/vae.rs`,操作 `CudaWeight.data`(`CudaSlice<f16>`)与 `CudaTensor.slice`。
//!
//! **M1**:实现视频 VAE 解码所需算子(`kernel::cuda::vae` 的 6 个 kernel +
//! `add_row_bias` + `full_attention` 复用 `gqa_attention_f16`)。`conv3d`/`group_norm`/
//! 尚未接入的 VAE 能力返回 `Err`，由模型 runtime 在启动时拒绝不完整组合。
//! DiT 的 `adaln_modulate`/`timestep_embedding` 等 M2 补齐,此处仍为 `Err`。

use crate::attention::gqa::{CausalWindow, GqaSpec};
use crate::backend::{Backend, BackendError, DiffusionBackend, VaeBackend};
use crate::diffusion::ModulationSegment;
use crate::kernel::cuda as ops;
use crate::vae::{Conv1dSpec, Conv3dSpec, PixelShuffleSpec};

use super::{CudaContext, CudaTensor, CudaWeight, compute_error};

impl DiffusionBackend for CudaContext {
    fn adaln_modulate(&self, input: &CudaTensor, shift: &CudaTensor, scale: &CudaTensor) -> Result<CudaTensor, BackendError> {
        // out = input * (1 + scale) + shift。shift/scale 按列广播或逐元素。
        ops::diffusion::adaln_modulate_f16(self, input, shift, scale).map_err(compute_error)
    }

    fn adaln_modulate_segmented(&self, input: &CudaTensor, shift: &CudaTensor, scale: &CudaTensor, segments: &[ModulationSegment]) -> Result<CudaTensor, BackendError> {
        // 分段广播 AdaLN(按时间步/模态段选调制行)。DiT block 主路径。
        // input 为 f32(残差流传播):输出 f32 以继续传播到 MLP linear。
        if input.slice_f32.is_some() {
            return ops::diffusion::adaln_modulate_segmented_f32(self, input, shift, scale, segments).map_err(compute_error);
        }
        ops::diffusion::adaln_modulate_segmented_f16(self, input, shift, scale, segments).map_err(compute_error)
    }

    fn gated_residual_segmented(&self, residual: &CudaTensor, update: &CudaTensor, gate: &CudaTensor, segments: &[ModulationSegment]) -> Result<CudaTensor, BackendError> {
        // DiT 残差流以 f32 累积(可达 ~5e5,超 f16)。首层 residual 仍是 f16(packed_hidden),
        // 输出 f32 开启残差路径;后续层 residual 已是 f32。update:MLP 路径为 f32(down 投影 ~6e4),
        // attention 路径为 f16(attn_update ~8e3)。gate 始终 f16。
        if let Some(residual_f32) = &residual.slice_f32 {
            if update.slice_f32.is_some() {
                return ops::diffusion::gated_residual_segmented_f32_f32update(self, residual_f32, update, gate, segments, residual.rows, residual.cols).map_err(compute_error);
            }
            return ops::diffusion::gated_residual_segmented_f32(self, residual_f32, update, gate, segments, residual.rows, residual.cols).map_err(compute_error);
        }
        ops::diffusion::gated_residual_segmented_f32_from_f16(self, residual, update, gate, segments).map_err(compute_error)
    }

    fn flow_step(&self, sample: &CudaTensor, velocity: &CudaTensor, scale: f32) -> Result<CudaTensor, BackendError> {
        // data-ward Euler 更新:sample + scale * velocity。
        ops::diffusion::flow_step_f16(self, sample, velocity, scale).map_err(compute_error)
    }

    fn silu(&self, input: &CudaTensor) -> Result<CudaTensor, BackendError> {
        // 时间步嵌入 MLP 的激活。
        ops::diffusion::silu_f16(self, input).map_err(compute_error)
    }

    fn timestep_embedding(&self, timesteps: &[f32], dim: usize) -> Result<CudaTensor, BackendError> {
        // 正弦时间步嵌入 → [timesteps.len(), dim]。
        ops::diffusion::timestep_embedding_f16(self, timesteps, dim).map_err(compute_error)
    }

    fn diffusion_tensor_from_f32(&self, values: &[f32], rows: usize, cols: usize) -> Result<CudaTensor, BackendError> {
        // host F32 → f16 tensor 上传(pruned 时间曲线插值结果)。
        self.tensor_from_f32(values, rows, cols).map_err(compute_error)
    }

    fn add_row_bias(&self, input: &CudaTensor, bias: &CudaWeight) -> Result<CudaTensor, BackendError> {
        // linear_bias(每个线性层后接)依赖此算子;DiT/VAE 共用。
        ops::diffusion::add_row_bias_f16(self, input, &bias.data).map_err(compute_error)
    }

    fn concat_rows(&self, left: &CudaTensor, right: &CudaTensor) -> Result<CudaTensor, BackendError> {
        // 沿 token 行拼接(对称 Backend::concat_columns 的行向版本)。DiT 打包序列依赖。
        ops::diffusion::concat_rows_f16(self, left, right).map_err(compute_error)
    }

    fn modulation_chunks(&self, input: &CudaTensor, modalities: usize, chunks: usize, hidden: usize) -> Result<Vec<CudaTensor>, BackendError> {
        // adaln 投影输出切片为 chunks 个 [rows*modalities, hidden] 张量。
        ops::diffusion::modulation_chunks_f16(self, input, modalities, chunks, hidden).map_err(compute_error)
    }

    fn rmsnorm_heads(&self, input: &CudaTensor, weight: &CudaWeight, head_count: usize, head_dim: usize, eps: f32) -> Result<CudaTensor, BackendError> {
        // 逐头 RMSNorm(带 weight,无 (1+weight));DiT attention 前 Q/K 归一化。
        ops::diffusion::rmsnorm_heads_f16(self, input, &weight.data, head_count, head_dim, eps).map_err(compute_error)
    }

    fn full_attention(&self, query: CudaTensor, key: CudaTensor, value: CudaTensor, head_count: usize, head_dim: usize, score_scale: f32) -> Result<CudaTensor, BackendError> {
        // 复用 GQA 在线 softmax kernel:令 num_kv_heads==num_heads(无 GQA 映射)、
        // position=rows-1,则因果钳制 `last=min(rows-1+row, kv_rows-1)=rows-1` 对所有
        // query row 成立 → 每 row attend KV[0,rows) 即全双向 attention。Q/K 的 RoPE 已在
        // 调用前(rope_prefix)施加,kernel 内部不施加(gqa 无内建 RoPE),故 rope_dim=0。
        let attention_dim = head_count.checked_mul(head_dim).ok_or_else(|| compute_error("CUDA full attention columns 溢出"))?;
        if query.rows != key.rows || query.rows != value.rows || query.cols != attention_dim || key.cols != query.cols || value.cols != query.cols {
            return Err(compute_error(format!("CUDA full attention Q/K/V 不兼容: query=[{},{}] key=[{},{}] value=[{},{}] heads={head_count} head_dim={head_dim}", query.rows, query.cols, key.rows, key.cols, value.rows, value.cols)));
        }
        let rows = query.rows;
        let spec = GqaSpec { num_heads: head_count, num_kv_heads: head_count, head_dim, rope_dim: 0, rope_theta: 0.0, use_qk_norm: false, window: CausalWindow::Full, score_scale, output_gate: false };
        ops::attention::gqa_attention_f16(self, &query, &key.slice, &value.slice, rows, rows.saturating_sub(1), &spec, rows).map_err(compute_error)
    }

    #[allow(clippy::too_many_arguments)]
    fn full_attention_batched(&self, query: CudaTensor, key: CudaTensor, value: CudaTensor, batch: usize, rows: usize, head_count: usize, head_dim: usize, score_scale: f32) -> Result<CudaTensor, BackendError> {
        // batched 全序列 self-attention:Q/K/V 连续存放 batch 个等长序列,attention 在 batch 边界
        // 处严格隔离。VAE tiled 解码(384px+ → batch>1)依赖此路径。
        let total_rows = batch.checked_mul(rows).ok_or_else(|| compute_error("CUDA batched full attention rows 溢出"))?;
        let attention_dim = head_count.checked_mul(head_dim).ok_or_else(|| compute_error("CUDA batched full attention columns 溢出"))?;
        if batch == 0 || query.rows != total_rows || query.rows != key.rows || query.rows != value.rows || query.cols != attention_dim || key.cols != query.cols || value.cols != query.cols {
            return Err(compute_error(format!(
                "CUDA batched full attention Q/K/V 不兼容: query=[{},{}] key=[{},{}] value=[{},{}] batch={batch} rows={rows} heads={head_count} head_dim={head_dim}",
                query.rows, query.cols, key.rows, key.cols, value.rows, value.cols
            )));
        }
        // VAE 路径 Q/K/V 为 f16(rmsnorm_heads_unit 产出);若 DiT 以 f32 残差流传入则先转回 f16。
        let (query, key, value) = if query.slice_f32.is_some() {
            let q = ops::diffusion::cast_f32_to_f16(self, &query).map_err(compute_error)?;
            let k = ops::diffusion::cast_f32_to_f16(self, &key).map_err(compute_error)?;
            let v = ops::diffusion::cast_f32_to_f16(self, &value).map_err(compute_error)?;
            (q, k, v)
        } else {
            (query, key, value)
        };
        ops::attention::full_attention_batched_f16(self, &query, &key.slice, &value.slice, batch, rows, head_count, head_dim, score_scale).map_err(compute_error)
    }

    #[allow(clippy::too_many_arguments)]
    fn full_attention_qkv(
        &self,
        qkv: CudaTensor,
        query_norm: &CudaWeight,
        key_norm: &CudaWeight,
        head_count: usize,
        head_dim: usize,
        rotary_dim: usize,
        eps: f32,
        cosine: &[f32],
        sine: &[f32],
        score_scale: f32,
    ) -> Result<CudaTensor, BackendError> {
        // 扩散 qkv 由 f32 linear 产出(残差流 f32 传播下来):attention 值 ~1e2 适合 f16,故转回 f16
        // 再跑既有 f16 拆分/qk_norm/RoPE/gqa 路径(attn_update ~8e3,安全 f16)。
        let qkv = if qkv.slice_f32.is_some() { ops::diffusion::cast_f32_to_f16(self, &qkv).map_err(compute_error)? } else { qkv };
        let attention_dim = head_count.checked_mul(head_dim).ok_or_else(|| compute_error("CUDA full attention QKV columns 溢出"))?;
        let (query, key_value) = self.split_columns(&qkv, attention_dim)?;
        let (key, value) = self.split_columns(&key_value, attention_dim)?;
        let query = self.rmsnorm_heads(&query, query_norm, head_count, head_dim, eps)?;
        let key = self.rmsnorm_heads(&key, key_norm, head_count, head_dim, eps)?;
        let (query, key) = self.rope_pair_prefix(query, key, head_count, rotary_dim, crate::attention::rope::RotaryLayout::SplitHalf, 0, cosine, sine)?;
        self.full_attention(query, key, value, head_count, head_dim, score_scale)
    }
}

impl VaeBackend for CudaContext {
    fn vae_tensor_from_f32(&self, values: Vec<f32>, rows: usize, cols: usize) -> Result<CudaTensor, BackendError> {
        self.tensor_from_f32(&values, rows, cols).map_err(compute_error)
    }

    fn vae_tensor_to_f32(&self, input: &CudaTensor) -> Result<Vec<f32>, BackendError> {
        self.tensor_to_f32(input).map_err(compute_error)
    }

    fn layer_norm(&self, input: &CudaTensor, weight: &CudaWeight, bias: &CudaWeight, eps: f32) -> Result<CudaTensor, BackendError> {
        ops::tensor::layer_norm_f16(self, input, &weight.data, &bias.data, eps).map_err(compute_error)
    }

    fn rms_norm_heads_unit(&self, input: &CudaTensor, heads: usize, head_dim: usize, eps: f32) -> Result<CudaTensor, BackendError> {
        ops::vae::rmsnorm_heads_unit_f16(self, input, heads, head_dim, eps).map_err(compute_error)
    }

    fn scaled_residual(&self, input: &CudaTensor, update: &CudaTensor, scale: &CudaWeight) -> Result<CudaTensor, BackendError> {
        ops::vae::scaled_residual_columns_f16(self, input, update, &scale.data).map_err(compute_error)
    }

    fn unpatch_affine(&self, input: &CudaTensor, scale: &CudaWeight, bias: &CudaWeight, shape: [usize; 3], patch: [usize; 3], channels: usize) -> Result<CudaTensor, BackendError> {
        ops::vae::unpatch_affine_f16(self, input, &scale.data, &bias.data, shape, patch, channels).map_err(compute_error)
    }

    fn concat_weight_rows_batched(&self, input: &CudaTensor, weight: &CudaWeight, rows: usize, batch: usize) -> Result<CudaTensor, BackendError> {
        if batch == 0 {
            return Err(compute_error("CUDA concat_weight_rows_batched batch=0"));
        }
        let input_rows = input.rows / batch;
        ops::vae::concat_weight_rows_batched_f16(self, input, &weight.data, input_rows, rows, batch).map_err(compute_error)
    }

    fn take_rows_batched(&self, input: &CudaTensor, rows: usize, batch: usize) -> Result<CudaTensor, BackendError> {
        ops::vae::take_rows_batched_f16(self, input, rows, batch).map_err(compute_error)
    }

    // —— 音频 VAE 解码(M5)。布局 [batch*channels, time],f16 存储 + f32 累加。 ——

    fn audio_unpack_affine(&self, input: &CudaTensor, scale: &CudaWeight, bias: &CudaWeight, batch: usize, time: usize, channels: usize) -> Result<CudaTensor, BackendError> {
        if input.rows != batch * time || input.cols != channels {
            return Err(compute_error(format!("CUDA audio unpack input=[{},{}]，期望 [{},{}]", input.rows, input.cols, batch * time, channels)));
        }
        ops::vae::audio_unpack_affine_f16(self, input, &scale.data, &bias.data, batch, time, channels).map_err(compute_error)
    }

    #[allow(clippy::too_many_arguments)]
    fn conv1d(&self, input: &CudaTensor, weight_g: Option<&CudaWeight>, weight_v: &CudaWeight, bias: Option<&CudaWeight>, spec: &Conv1dSpec) -> Result<CudaTensor, BackendError> {
        let Conv1dSpec { batch, input_channels, output_channels, kernel, dilation, padding, .. } = *spec;
        if input.rows != batch * input_channels {
            return Err(compute_error(format!("CUDA audio Conv1D input rows={}，期望 {}x{}", input.rows, batch, input_channels)));
        }
        if weight_v.data.len() != output_channels * input_channels * kernel {
            return Err(compute_error(format!("CUDA audio Conv1D weight_v={} 期望 {}x{}x{}", weight_v.data.len(), output_channels, input_channels, kernel)));
        }
        // bias 可选:无 bias 时零填,weight-norm 与 plain 两条路径都接 bias 指针,统一处理。
        let zero_bias;
        let bias_slice: &ops::CudaSliceF16 = match bias {
            Some(weight) => {
                if weight.data.len() != output_channels {
                    return Err(compute_error(format!("CUDA audio Conv1D bias={} 期望 {}", weight.data.len(), output_channels)));
                }
                &weight.data
            }
            None => {
                zero_bias = self.stream().alloc_zeros::<half::f16>(output_channels).map_err(|e| compute_error(format!("CUDA audio Conv1D 零 bias 分配失败: {e:?}")))?;
                &zero_bias
            }
        };
        match weight_g {
            // weight-norm:factor = g[r] / max(||v[r]||, 1e-12),kernel 内联(每 out_channel 重算)。
            Some(scale) => {
                if scale.data.len() != output_channels {
                    return Err(compute_error(format!("CUDA audio Conv1D weight_g={} 期望 {}", scale.data.len(), output_channels)));
                }
                ops::vae::audio_conv1d_f16(self, input, &scale.data, &weight_v.data, bias_slice, batch, input_channels, output_channels, kernel, dilation, padding).map_err(compute_error)
            }
            // 无 weight-norm(普通 conv,normalized=false):weight_v 直接用,factor=1。
            None => ops::vae::audio_conv1d_plain_f16(self, input, &weight_v.data, bias_slice, batch, input_channels, output_channels, kernel, dilation, padding).map_err(compute_error),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn conv_transpose1d(&self, input: &CudaTensor, weight_g: &CudaWeight, weight_v: &CudaWeight, bias: &CudaWeight, spec: &Conv1dSpec) -> Result<CudaTensor, BackendError> {
        let Conv1dSpec { batch, input_channels, output_channels, kernel, stride, padding, .. } = *spec;
        if input.rows != batch * input_channels {
            return Err(compute_error(format!("CUDA audio ConvTranspose1D input rows={}，期望 {}x{}", input.rows, batch, input_channels)));
        }
        ops::vae::audio_conv_transpose1d_f16(self, input, &weight_g.data, &weight_v.data, &bias.data, batch, input_channels, output_channels, kernel, stride, padding).map_err(compute_error)
    }

    fn snake_beta(&self, input: &CudaTensor, alpha: &CudaWeight, beta: &CudaWeight, up_filter: &CudaWeight, down_filter: &CudaWeight, channels: usize) -> Result<CudaTensor, BackendError> {
        if channels == 0 || !input.rows.is_multiple_of(channels) {
            return Err(compute_error(format!("CUDA audio SnakeBeta input rows={} 无法按 channels={channels} 分组", input.rows)));
        }
        let batch = input.rows / channels;
        ops::vae::audio_snake_beta_f16(self, input, &alpha.data, &beta.data, &up_filter.data, &down_filter.data, batch, channels).map_err(compute_error)
    }

    fn scale_tensor(&self, input: &CudaTensor, scale: f32) -> Result<CudaTensor, BackendError> {
        ops::vae::scale_tensor_f16(self, input, scale).map_err(compute_error)
    }

    fn tanh(&self, input: &CudaTensor) -> Result<CudaTensor, BackendError> {
        ops::vae::tanh_f16(self, input).map_err(compute_error)
    }

    fn conv3d(&self, _input: &CudaTensor, _weight: &CudaWeight, _bias: Option<&CudaWeight>, _spec: &Conv3dSpec) -> Result<CudaTensor, BackendError> {
        Err(compute_error("CUDA VAE conv3d 暂未实现"))
    }

    fn group_norm(&self, _input: &CudaTensor, _weight: &CudaWeight, _bias: &CudaWeight, _num_groups: usize, _eps: f32) -> Result<CudaTensor, BackendError> {
        Err(compute_error("CUDA VAE group_norm 暂未实现"))
    }

    fn pixel_shuffle(&self, _input: &CudaTensor, _spec: &PixelShuffleSpec) -> Result<CudaTensor, BackendError> {
        Err(compute_error("CUDA VAE pixel_shuffle 暂未实现"))
    }
}
