//! ROCm runtime FFI 封装。
//!
//! 说明：当前只覆盖线性代数主路径。矩阵乘由 HIP runtime 上的自研 kernel 执行，
//! 不同实现共用同一组 shape 校验，失败时返回可读错误并由上层决定是否回退。

use std::sync::OnceLock;

#[derive(Clone)]
pub struct RocmOptions {
    pub(crate) kernel_sync: bool,
    pub(crate) kernel_profile: bool,
    pub(crate) decode_graph: bool,
    pub(crate) generic_full_attention: bool,
    pub(crate) native_full_attention_kv: bool,
    pub(crate) precise_router: bool,
    pub(crate) trace_moe_route: bool,
    pub(crate) block_fp8_profile: bool,
    pub(crate) memory_pool: bool,
    pub(crate) legacy_pool: bool,
    pub(crate) log_memory: bool,
    pub(crate) rocm_root: String,
    pub(crate) hiprtc_cache_dir: std::path::PathBuf,
    pub(crate) profile_dsa: bool,
    pub(crate) native_dsa_wmma: bool,
    pub(crate) dsa_hadamard_i8: bool,
    pub(crate) dsa_hadamard_shadow_samples: usize,
    pub(crate) dsa_hisa_shadow_samples: usize,
    pub(crate) dsa_cpu_select: bool,
    pub(crate) mla_cpu_hot_rows: usize,
    pub(crate) prefill_attention_cpu: bool,
    pub(crate) mla_hot_trace: bool,
    pub(crate) debug_dsa_sync: bool,
    pub(crate) debug_selection: bool,
    pub(crate) debug_finite: bool,
    pub(crate) log_mla_pointers: bool,
    pub(crate) mla_absorb_wmma: bool,
    pub(crate) force_dense_prefill: bool,
    pub(crate) mla_decode_split: Option<bool>,
    pub(crate) mla_decode_split_threshold: usize,
    pub(crate) mla_decode_tile_size: usize,
    pub(crate) mla_decode_wmma: bool,
    // decode split 的总 block 数目标（≈ 2×CU）。cooperative 半头拆分下每卡
    // grid.x 减半，tile 目标按该值除以本次 launch 的 grid.x 重新定标。
    pub(crate) mla_decode_target_blocks: usize,
    /// cooperative decode 用序列拆分（两卡全头各扫半段序列 + LSE 合并）替代半头拆分。
    pub(crate) cooperative_mla_sequence_split: bool,
    /// decode 两卡各自产生完整 attention hidden，并把副本交给双 router/MoE。
    pub(crate) cooperative_mla_decode_replicated: bool,
    /// prefill 按 query 行分工，每卡只为自己负责的 token 行执行完整 o_proj。
    pub(crate) cooperative_mla_prefill_row_output: bool,
    /// owner 汇合两侧 sequence shard 后直接生成完整 attention 输出，跳过
    /// peer full-hidden partial 的第二次 P2P 与归约。
    pub(crate) cooperative_mla_decode_full_merge: bool,
    pub(crate) sparse_prefill_heads4: bool,
    pub(crate) mla_prefill_target_blocks: usize,
    pub(crate) dense_profile: bool,
    pub(crate) w8_profile: bool,
    pub(crate) convrot_cached_quant: bool,
    pub(crate) convrot_force_cached_quant: bool,
    pub(crate) convrot_baseline: bool,
    pub(crate) convrot_tiled: bool,
    pub(crate) convrot_force_tiled: bool,
    pub(crate) log_expert_pointers_device: Option<i32>,
    pub(crate) debug_decode_moe_finite: bool,
    pub(crate) decode_moe_fused: bool,
    pub(crate) grouped_wmma: bool,
    pub(crate) grouped_down_route_buffer: bool,
    pub(crate) release_batch_workspace: bool,
    pub(crate) debug_decode_layer_finite: bool,
    pub(crate) dual_w8: bool,
    pub(crate) kv_f16: bool,
    pub(crate) attention_devices: Option<Vec<i32>>,
    pub(crate) trace_moe_segment_rows: Option<(usize, usize)>,
    pub(crate) trace_moe_layer: Option<usize>,
    pub(crate) expert_load_width: usize,
    pub(crate) kda_sequential: bool,
}

impl Default for RocmOptions {
    fn default() -> Self {
        Self {
            kernel_sync: false,
            kernel_profile: false,
            decode_graph: false,
            generic_full_attention: false,
            native_full_attention_kv: true,
            precise_router: true,
            trace_moe_route: false,
            block_fp8_profile: false,
            memory_pool: true,
            legacy_pool: false,
            log_memory: false,
            rocm_root: "/opt/rocm".to_owned(),
            hiprtc_cache_dir: std::path::PathBuf::from("/tmp/zllm-hiprtc"),
            profile_dsa: false,
            native_dsa_wmma: true,
            dsa_hadamard_i8: false,
            dsa_hadamard_shadow_samples: 0,
            dsa_hisa_shadow_samples: 0,
            dsa_cpu_select: false,
            mla_cpu_hot_rows: 0,
            prefill_attention_cpu: false,
            mla_hot_trace: false,
            debug_dsa_sync: false,
            debug_selection: false,
            debug_finite: false,
            log_mla_pointers: false,
            mla_absorb_wmma: true,
            force_dense_prefill: false,
            mla_decode_split: None,
            mla_decode_split_threshold: 1024,
            mla_decode_tile_size: 32,
            mla_decode_wmma: true,
            mla_decode_target_blocks: 608,
            cooperative_mla_sequence_split: false,
            cooperative_mla_decode_replicated: false,
            cooperative_mla_prefill_row_output: false,
            cooperative_mla_decode_full_merge: false,
            sparse_prefill_heads4: false,
            mla_prefill_target_blocks: 8192,
            dense_profile: false,
            w8_profile: false,
            convrot_cached_quant: true,
            convrot_force_cached_quant: false,
            convrot_baseline: false,
            convrot_tiled: true,
            convrot_force_tiled: false,
            log_expert_pointers_device: None,
            debug_decode_moe_finite: false,
            decode_moe_fused: true,
            grouped_wmma: true,
            grouped_down_route_buffer: false,
            release_batch_workspace: false,
            debug_decode_layer_finite: false,
            dual_w8: true,
            kv_f16: false,
            attention_devices: None,
            trace_moe_segment_rows: None,
            trace_moe_layer: None,
            expert_load_width: 4,
            kda_sequential: false,
        }
    }
}

impl RocmOptions {
    pub fn configured(
        kernel_sync: bool,
        kernel_profile: bool,
        decode_graph: bool,
        memory_pool: bool,
        grouped_down_route_buffer: bool,
        rocm_root: String,
        hiprtc_cache_dir: std::path::PathBuf,
        kv_f16: bool,
        mla_decode_split_threshold: usize,
        mla_decode_wmma: bool,
        cooperative_mla_sequence_split: bool,
        cooperative_mla_decode_replicated: bool,
        cooperative_mla_prefill_row_output: bool,
        cooperative_mla_decode_full_merge: bool,
        dsa_hadamard_i8: bool,
        dsa_hadamard_shadow_samples: usize,
        dsa_hisa_shadow_samples: usize,
        dsa_cpu_select: bool,
        mla_cpu_hot_rows: usize,
        prefill_attention_cpu: bool,
        mla_hot_trace: bool,
        precise_router: bool,
        w8_profile: bool,
        mla_decode_tile_size: usize,
    ) -> Self {
        Self {
            kernel_sync,
            kernel_profile,
            decode_graph,
            memory_pool,
            grouped_down_route_buffer,
            rocm_root,
            hiprtc_cache_dir,
            profile_dsa: kernel_profile,
            kv_f16,
            mla_decode_split_threshold,
            mla_decode_wmma,
            mla_decode_tile_size,
            cooperative_mla_sequence_split,
            cooperative_mla_decode_replicated,
            cooperative_mla_prefill_row_output,
            cooperative_mla_decode_full_merge,
            dsa_hadamard_i8,
            // shadow 会同步下载整行 exact/coarse score，只允许显式 profile 使用。
            dsa_hadamard_shadow_samples: if kernel_profile { dsa_hadamard_shadow_samples } else { 0 },
            dsa_hisa_shadow_samples: if kernel_profile { dsa_hisa_shadow_samples } else { 0 },
            dsa_cpu_select,
            mla_cpu_hot_rows,
            prefill_attention_cpu,
            mla_hot_trace,
            precise_router,
            w8_profile,
            ..Self::default()
        }
    }
}

static ROCM_OPTIONS: OnceLock<RocmOptions> = OnceLock::new();

pub fn configure(options: RocmOptions) -> Result<(), String> {
    ROCM_OPTIONS.set(options).map_err(|_| "ROCm options 必须在创建 context 前且只能配置一次".to_owned())
}

pub fn options() -> &'static RocmOptions {
    ROCM_OPTIONS.get_or_init(RocmOptions::default)
}
