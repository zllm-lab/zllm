//! K3 跨 prefill/decode round 持久化的 session 状态。

use crate::{
    attention::kda::{KdaSpec, KdaState, KdaStorage},
    backend::BackendError,
};

/// AttnRes 属于单轮 scratch，不进入 session；这里只持有真正跨 token 的状态。
pub struct KimiK3SessionState<S, C> {
    kda: KdaState<S>,
    mla_cache: C,
    next_position: usize,
    poisoned: bool,
}

impl<S, C> KimiK3SessionState<S, C> {
    pub fn new(layer_count: usize, kda_spec: KdaSpec, mla_cache: C) -> Result<Self, BackendError> {
        if layer_count == 0 {
            return Err(BackendError::Compute { msg: "Kimi K3 session layer_count 不能为 0".to_owned() });
        }
        Ok(Self { kda: KdaState::new(layer_count, kda_spec)?, mla_cache, next_position: 0, poisoned: false })
    }

    pub fn next_position(&self) -> usize {
        self.next_position
    }

    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    pub fn begin_round(&self, start_position: usize) -> Result<(), BackendError> {
        if self.poisoned {
            return Err(BackendError::Compute { msg: "Kimi K3 session 已因不完整 round 失效".to_owned() });
        }
        if start_position != self.next_position {
            return Err(BackendError::Compute { msg: format!("Kimi K3 session position 不连续: next={}, start={start_position}", self.next_position) });
        }
        Ok(())
    }

    pub fn kda(&self) -> &KdaState<S> {
        &self.kda
    }

    pub fn mla_cache(&self) -> &C {
        &self.mla_cache
    }

    pub fn states_mut(&mut self) -> (&mut KdaState<S>, &mut C) {
        (&mut self.kda, &mut self.mla_cache)
    }

    pub fn poison(&mut self) {
        self.poisoned = true;
    }

    /// 只有完整模型轮次成功后才提交 position；失败 round 必须先标记 poisoned。
    pub fn commit(&mut self, start_position: usize, token_count: usize) -> Result<(), BackendError> {
        if self.poisoned {
            return Err(BackendError::Compute { msg: "Kimi K3 session 已失效，不能提交 position".to_owned() });
        }
        if start_position != self.next_position {
            return Err(BackendError::Compute { msg: format!("Kimi K3 session position 不连续: next={}, start={start_position}", self.next_position) });
        }
        if token_count == 0 {
            return Err(BackendError::Compute { msg: "Kimi K3 session 不能提交空 token round".to_owned() });
        }
        self.next_position = self.next_position.checked_add(token_count).ok_or_else(|| BackendError::Compute { msg: "Kimi K3 session position 溢出".to_owned() })?;
        Ok(())
    }
}

impl<S: KdaStorage, C> KimiK3SessionState<S, C> {}

#[cfg(test)]
mod tests {
    use crate::{
        attention::{AttentionSpec, attn_res::AttnResState, kda::KdaStorage},
        moe::FeedforwardSpec,
        runtime::{
            Model,
            kimi_k3::{KimiK3, KimiK3Config},
        },
    };

    use super::KimiK3SessionState;

    struct DummyKdaStorage;

    impl KdaStorage for DummyKdaStorage {
        fn allocated_bytes(&self) -> usize {
            0
        }
    }

    #[test]
    fn 无权重模拟一次prefill和decode() {
        let model = KimiK3::new(KimiK3Config::standard()).unwrap();
        let kda_spec = match &model.layer_spec(0).unwrap().attention {
            AttentionSpec::Kda(spec) => *spec,
            other => panic!("K3 L0 期望 KDA，实际 {other:?}"),
        };
        let mut session = KimiK3SessionState::<DummyKdaStorage, ()>::new(model.layer_count(), kda_spec, ()).unwrap();

        let prefill = simulate_round(&model, 0, 128);
        session.begin_round(0).unwrap();
        session.commit(0, 128).unwrap();
        let decode = simulate_round(&model, session.next_position(), 1);
        session.begin_round(128).unwrap();
        session.commit(128, 1).unwrap();

        assert_eq!(prefill, (69, 24, 1, 92));
        assert_eq!(decode, prefill);
        assert_eq!(session.next_position(), 129);
        assert!(!session.is_poisoned());
        println!("K3 dry-run: layers=93 kda={} mla={} dense={} latent_moe={} prefill=128 decode=1 next_position={}", prefill.0, prefill.1, prefill.2, prefill.3, session.next_position(),);
    }

    fn simulate_round(model: &KimiK3, position: usize, token_count: usize) -> (usize, usize, usize, usize) {
        assert!(token_count > 0);
        let mut attn_res = AttnResState::new(model.config().attn_res_block_size).unwrap();
        let mut counts = (0, 0, 0, 0);
        for layer in 0..model.layer_count() {
            attn_res.begin_layer(layer).unwrap();
            if attn_res.is_block_start(layer) {
                attn_res.push((position, token_count, layer));
            }
            let spec = model.layer_spec(layer).unwrap();
            match &spec.attention {
                AttentionSpec::Kda(_) => counts.0 += 1,
                AttentionSpec::GatedMla(_) => counts.1 += 1,
                other => panic!("K3 L{layer} 不支持 attention {other:?}"),
            }
            match &spec.feedforward {
                FeedforwardSpec::Dense(_) => counts.2 += 1,
                FeedforwardSpec::LatentTopkMoe(_) => counts.3 += 1,
                other => panic!("K3 L{layer} 不支持 FFN {other:?}"),
            }
            attn_res.finish_layer();
        }
        assert!(attn_res.block_count() > 0);
        attn_res.finish_round(model.layer_count()).unwrap();
        assert_eq!(attn_res.block_count(), 0);
        counts
    }
}
