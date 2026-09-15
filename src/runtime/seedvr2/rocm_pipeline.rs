//! SeedVR2 单个视频的八卡 ROCm 常驻执行；层间只有设备 P2P，不回读激活。

use super::{SeedVr2PreparedBlock, SeedVr2PreparedGlobal, SeedVr2WindowPlan, linear_bias, patchify, prepare_block, prepare_global, prepare_linear, rocm::SeedVr2RocmGroup, stream_finish, stream_qkv, timestep, unpatchify};
use crate::{
    backend::{
        BackendResources, DiffusionBackend, SegmentedTensorBackend,
        rocm::{RocmTensor, RocmWeight},
    },
    model_spec::seedvr2::SeedVr2Config,
    weight::{
        container::safetensor::TensorData,
        model::seedvr2::{SeedVr2DitSource, SeedVr2Linear},
    },
};
use std::{path::Path, time::Instant};

pub struct SeedVr2RocmPipeline {
    pub group: SeedVr2RocmGroup,
    pub config: SeedVr2Config,
    globals: Vec<SeedVr2PreparedGlobal<RocmWeight>>,
    blocks: Vec<Vec<SeedVr2PreparedBlock<RocmWeight, RocmTensor>>>,
    outputs: Vec<SeedVr2Linear<RocmWeight>>,
    text: Vec<RocmTensor>,
    frequencies: Vec<f32>,
}

impl SeedVr2RocmPipeline {
    pub fn load(root: &Path, devices: &[i32]) -> Result<Self, String> {
        let started = Instant::now();
        let config = SeedVr2Config::standard_7b();
        let source = SeedVr2DitSource::open(root.join("seedvr2_ema_7b_fp16.safetensors"), config.clone())?;
        let group = SeedVr2RocmGroup::new(devices)?;
        // 八卡共享同一份已转换的矩阵字节；向量保留原精度，避免每份副本
        // 在 prepare_tensor 里重复执行整模型 F16→BF16 转换。
        let prepare_replica = |mut tensor: TensorData| {
            if tensor.dtype == "F16" && tensor.shape.len() == 2 && tensor.shape.iter().all(|&d| d > 1) {
                crate::weight::codec::f16_to_bf16_in_place(&mut tensor.data)?;
                tensor.dtype = "BF16".to_owned();
            }
            Ok(tensor)
        };
        let raw_global = source.load_global_with(&prepare_replica)?;
        let mut globals = Vec::with_capacity(devices.len());
        let mut embeddings = Vec::with_capacity(devices.len());
        for context in group.contexts() {
            context.activate()?;
            let mut global = prepare_global(context, &config, &raw_global).map_err(|e| e.to_string())?;
            // 33×2×2=132 的 patch 输入未对齐 WMMA K16；这一个小投影用
            // resident F32 GEMM，保留相同的 BF16 权重数值，其余大矩阵保持 BF16。
            let patch_weight = &raw_global.video_input.weight;
            if !patch_weight.shape[1].is_multiple_of(16) {
                let values = patch_weight.to_f32()?.into_iter().map(|v| half::bf16::from_f32(v).to_f32()).collect::<Vec<_>>();
                global.weights.video_input.weight = context.prepare_f32(&values, patch_weight.shape[0], patch_weight.shape[1]).map_err(|e| e.to_string())?;
            }
            let embedding = timestep(context, &config, &global, 1000.0).map_err(|e| e.to_string())?;
            embeddings.push(context.tensor_to_f32(&embedding).map_err(|e| e.to_string())?);
            globals.push(global);
        }
        drop(raw_global);
        let fixed = source.load_fixed_embeddings(root.join("fixed_embeddings.safetensors"))?;
        let text_values = fixed.positive.to_f32()?;
        let text_input = group.primary().diffusion_tensor_from_f32(&text_values, config.positive_text_tokens, config.text_dim).map_err(|e| e.to_string())?;
        let text_input = group.scatter_sequence(&text_input).map_err(|e| e.to_string())?;
        let mut text = Vec::with_capacity(devices.len());
        for (rank, context) in group.contexts().iter().enumerate() {
            context.activate()?;
            text.push(linear_bias(context, &text_input[rank], &globals[rank].weights.text_input).map_err(|e| e.to_string())?);
        }
        group.synchronize_all().map_err(|e| e.to_string())?;
        let mut blocks = Vec::with_capacity(config.num_layers);
        let mut frequencies = Vec::new();
        for layer in 0..config.num_layers {
            let raw = source.load_block_with(layer, &prepare_replica)?;
            let layer_frequencies = raw.rope_frequencies.to_f32()?;
            if layer == 0 {
                frequencies = layer_frequencies;
            } else if frequencies != layer_frequencies {
                return Err(format!("SeedVR2 layer={layer} RoPE buffer 与第 0 层不同，不能共用窗口表"));
            }
            // 各卡上传同一份只读矩阵，没有数据依赖；线程结束前排空本卡上传，
            // 保证下一层释放host字节及后续跨线程消费时设备数据已就绪。
            let rank_weights = std::thread::scope(|scope| {
                let handles = group
                    .contexts()
                    .iter()
                    .enumerate()
                    .map(|(rank, context)| {
                        let raw = &raw;
                        let config = &config;
                        let embedding = &embeddings[rank];
                        scope.spawn(move || {
                            context.activate()?;
                            let result = prepare_block(context, config, raw, embedding).map_err(|e| format!("SeedVR2 准备 layer={layer} rank={rank}: {e}"));
                            context.synchronize_compute_stream().map_err(|e| format!("SeedVR2 准备同步 layer={layer} rank={rank}: {e}"))?;
                            result
                        })
                    })
                    .collect::<Vec<_>>();
                handles.into_iter().enumerate().map(|(rank, handle)| handle.join().map_err(|_| format!("SeedVR2 准备 layer={layer} rank={rank} 线程panic"))?).collect::<Result<Vec<_>, String>>()
            })?;
            blocks.push(rank_weights);
            if (layer + 1) % 6 == 0 {
                eprintln!("[seedvr2-native] prepare layers={}/{} ranks={} wall={:.3}s", layer + 1, config.num_layers, devices.len(), started.elapsed().as_secs_f64());
            }
        }
        let raw_output = source.load_final_with(&prepare_replica)?;
        let mut outputs = Vec::with_capacity(devices.len());
        for context in group.contexts() {
            context.activate()?;
            outputs.push(prepare_linear(context, &raw_output).map_err(|e| e.to_string())?);
        }
        group.synchronize_all().map_err(|e| e.to_string())?;
        Ok(Self { group, config, globals, blocks, outputs, text, frequencies })
    }

    /// 输入为 `[noise(16), condition(16), mask(1)]` 的 patch 行。
    /// 用同一入口接受已保存的输入，验证时无需重跑外部框架或依赖其 RNG。
    pub fn forward_packed(&self, packed: &[f32], shape: [usize; 3], on_layer: &mut dyn FnMut(usize, usize) -> Result<(), String>) -> Result<Vec<f32>, String> {
        let config = &self.config;
        if shape.contains(&0) || shape.into_iter().zip(config.patch).any(|(s, p)| !s.is_multiple_of(p)) {
            return Err(format!("SeedVR2 latent shape={shape:?} 与 patch 不兼容"));
        }
        let patch_shape = std::array::from_fn(|i| shape[i] / config.patch[i]);
        let rows = patch_shape.into_iter().try_fold(1usize, |n, d| n.checked_mul(d)).ok_or("SeedVR2 patch 行数溢出")?;
        let plans = [false, true].map(|shifted| SeedVr2WindowPlan::new(patch_shape, config.positive_text_tokens, config.window, shifted, &self.frequencies));
        let [ordinary, shifted] = plans;
        let plans = [ordinary?, shifted?];
        let input = self.group.primary().diffusion_tensor_from_f32(packed, rows, config.input_channels * config.patch_volume()).map_err(|e| e.to_string())?;
        let input = self.group.scatter_sequence(&input).map_err(|e| e.to_string())?;
        let mut video = Vec::with_capacity(input.len());
        for (rank, context) in self.group.contexts().iter().enumerate() {
            context.activate()?;
            video.push(linear_bias(context, &input[rank], &self.globals[rank].weights.video_input).map_err(|e| e.to_string())?);
        }
        drop(input);
        let mut text = self.text.clone();
        for (layer, weights) in self.blocks.iter().enumerate() {
            let mut video_qkv = Vec::with_capacity(video.len());
            let mut text_qkv = Vec::with_capacity(text.len());
            for (rank, context) in self.group.contexts().iter().enumerate() {
                context.activate()?;
                video_qkv.push(stream_qkv(context, config, &self.globals[rank].unit_norm, &video[rank], &weights[rank].video).map_err(|e| format!("SeedVR2 layer={layer} rank={rank} video QKV: {e}"))?);
                text_qkv.push(stream_qkv(context, config, &self.globals[rank].unit_norm, &text[rank], &weights[rank].text).map_err(|e| format!("SeedVR2 layer={layer} rank={rank} text QKV: {e}"))?);
            }
            let norms = weights.iter().map(|w| (&w.video.attention.query_norm, &w.video.attention.key_norm, &w.text.attention.query_norm, &w.text.attention.key_norm)).collect::<Vec<_>>();
            let (video_attention, text_attention) =
                self.group.attention_qkv(&video_qkv, &text_qkv, &norms, &plans[layer % 2], config.num_heads, config.head_dim, config.rotary_dim(), config.norm_eps).map_err(|e| format!("SeedVR2 layer={layer} 8 卡 attention: {e}"))?;
            drop(video_qkv);
            drop(text_qkv);
            // 各 rank 的 MLP 没有相互依赖；独立提交避免一张卡的分配或
            // 设备等待阻止其余卡启动，线程返回前排空本卡并稳定输出生命周期。
            let outputs = std::thread::scope(|scope| {
                let handles = self
                    .group
                    .contexts()
                    .iter()
                    .enumerate()
                    .map(|(rank, context)| {
                        let video = &video;
                        let text = &text;
                        let video_attention = &video_attention;
                        let text_attention = &text_attention;
                        scope.spawn(move || {
                            context.activate()?;
                            // attention 之后各 token 的残差、归一化和 MLP 互不依赖。
                            // 限制中间激活为约 1GiB，避免长视频跨过分配器的同步释放阈值；
                            // 这里只切设备行视图，不重置时间窗口、VAE 历史或随机序列。
                            let row_limit = ((1024 * 1024 * 1024 / std::mem::size_of::<f32>() / config.mlp_hidden_size) / 32 * 32).max(32);
                            let mut parts = Vec::new();
                            for start in (0..video[rank].rows).step_by(row_limit) {
                                let rows = row_limit.min(video[rank].rows - start);
                                let input = context.slice_token_rows(&video[rank], start, rows).map_err(|e| e.to_string())?;
                                let attention = context.slice_token_rows(&video_attention[rank], start, rows).map_err(|e| e.to_string())?;
                                parts.push(
                                    stream_finish(context, config, &self.globals[rank].unit_norm, &input, &attention, &weights[rank].video)
                                        .map_err(|e| format!("SeedVR2 layer={layer} rank={rank} rows={start}..{} video MLP: {e}", start + rows))?,
                                );
                            }
                            let video_output = if parts.len() == 1 { parts.pop().unwrap() } else { context.concat_token_rows(&parts.iter().collect::<Vec<_>>()).map_err(|e| e.to_string())? };
                            let text_output = stream_finish(context, config, &self.globals[rank].unit_norm, &text[rank], &text_attention[rank], &weights[rank].text).map_err(|e| format!("SeedVR2 layer={layer} rank={rank} text MLP: {e}"))?;
                            let outputs = (context.tensor_to_stable_deferred(video_output).map_err(|e| e.to_string())?, context.tensor_to_stable_deferred(text_output).map_err(|e| e.to_string())?);
                            context.synchronize_compute_stream().map_err(|e| e.to_string())?;
                            Ok::<_, String>(outputs)
                        })
                    })
                    .collect::<Vec<_>>();
                handles.into_iter().map(|h| h.join().unwrap_or_else(|_| Err(format!("SeedVR2 layer={layer} MLP worker panic")))).collect::<Vec<_>>()
            });
            let (next_video, next_text) = outputs.into_iter().collect::<Result<Vec<_>, _>>()?.into_iter().unzip();
            video = next_video;
            text = next_text;
            on_layer(layer + 1, config.num_layers)?;
        }
        drop(text);
        for (rank, context) in self.group.contexts().iter().enumerate() {
            context.activate()?;
            video[rank] = linear_bias(context, &video[rank], &self.outputs[rank]).map_err(|e| e.to_string())?;
        }
        let output = self.group.gather_sequence(&video).map_err(|e| e.to_string())?;
        let values = self.group.primary().tensor_to_f32(&output).map_err(|e| e.to_string())?;
        unpatchify(&values, shape, config.output_channels, config.patch)
    }

    pub fn denoise_with_noise(&self, latent: &[f32], noise: &[f32], shape: [usize; 3], on_layer: &mut dyn FnMut(usize, usize) -> Result<(), String>) -> Result<Vec<f32>, String> {
        let channels = self.config.output_channels;
        let rows = shape.into_iter().try_fold(1usize, |n, d| n.checked_mul(d)).ok_or("SeedVR2 latent 行数溢出")?;
        if rows.checked_mul(channels) != Some(latent.len()) || noise.len() != latent.len() {
            return Err("SeedVR2 condition/noise 形状不匹配".to_owned());
        }
        let mut input = Vec::with_capacity(rows * self.config.input_channels);
        for (noise, condition) in noise.chunks_exact(channels).zip(latent.chunks_exact(channels)) {
            input.extend_from_slice(noise);
            input.extend_from_slice(condition);
            input.push(1.0);
        }
        let packed = patchify(&input, shape, self.config.input_channels, self.config.patch)?;
        drop(input);
        let velocity = self.forward_packed(&packed, shape, on_layer)?;
        // 1000→0 单步 v_lerp，CFG=1，x0=xT-v；这是 SeedVR2 蒸馏权重的采样定义。
        Ok(noise.iter().zip(velocity).map(|(&x, v)| x - v).collect())
    }
}
