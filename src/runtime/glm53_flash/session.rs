//! GLM-5.3-Flash 跨 prefill/decode round 持久化的 session 状态。
//!
//! KDA recurrent state、MLA cache 与 DSA 索引状态是三份独立生命周期,
//! 只有真正跨 token 的状态进 session;mHC 展开态与池化中间量都是单轮 scratch。

use crate::{
    attention::kda::{KdaSpec, KdaState, KdaStorage},
    backend::BackendError,
};

pub struct Glm53FlashSessionState<S, C, D> {
    kda: KdaState<S>,
    mla_cache: C,
    dsa: D,
    next_position: usize,
    poisoned: bool,
}

impl<S: KdaStorage, C, D> Glm53FlashSessionState<S, C, D> {
    pub fn new(layer_count: usize, kda_spec: KdaSpec, mla_cache: C, dsa: D) -> Result<Self, BackendError> {
        if layer_count == 0 {
            return Err(BackendError::Compute { msg: "GLM-5.3-Flash session layer_count 不能为 0".to_owned() });
        }
        Ok(Self { kda: KdaState::new(layer_count, kda_spec)?, mla_cache, dsa, next_position: 0, poisoned: false })
    }

    pub fn next_position(&self) -> usize {
        self.next_position
    }

    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    pub fn begin_round(&self, start_position: usize) -> Result<(), BackendError> {
        if self.poisoned {
            return Err(BackendError::Compute { msg: "GLM-5.3-Flash session 已因不完整 round 失效".to_owned() });
        }
        if start_position != self.next_position {
            return Err(BackendError::Compute { msg: format!("GLM-5.3-Flash session position 不连续: next={} start={start_position}", self.next_position) });
        }
        Ok(())
    }

    pub fn states_mut(&mut self) -> (&mut KdaState<S>, &mut C, &mut D) {
        (&mut self.kda, &mut self.mla_cache, &mut self.dsa)
    }
}

impl<S, C, D> Glm53FlashSessionState<S, C, D> {
    pub fn poison(&mut self) {
        self.poisoned = true;
    }

    /// 只有完整模型轮次成功后才提交 position;失败 round 必须先标记 poisoned。
    pub fn commit(&mut self, start_position: usize, token_count: usize) -> Result<(), BackendError> {
        if self.poisoned {
            return Err(BackendError::Compute { msg: "GLM-5.3-Flash session 已 poisoned,不能 commit".to_owned() });
        }
        let expected = self.next_position;
        if start_position != expected {
            return Err(BackendError::Compute { msg: format!("GLM-5.3-Flash commit position {start_position} 与 next {expected} 不一致") });
        }
        self.next_position = start_position.checked_add(token_count).ok_or_else(|| BackendError::Compute { msg: "GLM-5.3-Flash session position 溢出".to_owned() })?;
        Ok(())
    }
}
