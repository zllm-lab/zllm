use super::OrnithRuntimeOptions;
use crate::attention::hybrid::HybridAttentionOptions;
use crate::config::{KvCacheFormat, OrnithNodeExecutionConfig};

#[derive(Clone, Copy)]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub struct OrnithOptions {
    pub runtime: OrnithRuntimeOptions,
    pub kv_f16: bool,
    pub lazy_experts: bool,
    pub expert_cache_gib: usize,
    pub expert_prefetch_count: Option<usize>,
    pub terminal_cache_entries: usize,
}

impl From<OrnithNodeExecutionConfig> for OrnithOptions {
    fn from(config: OrnithNodeExecutionConfig) -> Self {
        Self {
            runtime: OrnithRuntimeOptions { attention: HybridAttentionOptions { precise_prefill: config.precise_gqa_prefill }, expert_batch_size: config.expert_batch_size },
            kv_f16: config.kv_cache_format == KvCacheFormat::F16,
            lazy_experts: config.lazy_experts,
            expert_cache_gib: config.expert_cache_gib,
            expert_prefetch_count: config.expert_prefetch_count,
            terminal_cache_entries: config.terminal_cache_entries,
        }
    }
}
