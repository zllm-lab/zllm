//! GLM-5.2 terminal cache 拥有的完整会话状态。

use super::*;

/// 状态中的 backend tensor 与 cache 跟随 terminal entry 同生命周期。
pub(super) struct Glm52HeadState {
    pub(super) states: Vec<Glm52StageState>,
    pub(super) last_hidden: RocmTensor,
    /// None 表示旧快照缺少元数据；Some(empty) 表示 EOS 后无待前向 token。
    pub(super) pending_tokens: Option<Vec<u32>>,
    pub(super) mtp: Option<RocmMtpSession>,
    pub(super) dspark_aux_history: Option<RocmTensor>,
    pub(super) dspark_aux_history_start: usize,
    pub(super) dspark_target_cache: DsparkTargetCache<RocmTensor>,
    pub(super) cache_namespace: Option<String>,
    pub(super) info: CacheInfo,
    pub(super) decode_finished_at: SystemTime,
    /// 当前轮输入长度，不用包含生成结果的 KV 水位决定短 prompt 是否落盘。
    pub(super) prompt_tokens: usize,
}
