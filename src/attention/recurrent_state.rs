//! 线性注意力逐层 recurrent state 的公共生命周期。

use crate::backend::BackendError;

pub(crate) struct LayerState<S> {
    pub position: usize,
    pub storage: S,
}

pub(crate) struct RecurrentState<S, Spec> {
    name: &'static str,
    spec: Spec,
    layers: Vec<Option<LayerState<S>>>,
}

impl<S, Spec: PartialEq> RecurrentState<S, Spec> {
    pub fn new(name: &'static str, layer_count: usize, spec: Spec) -> Self {
        Self { name, spec, layers: (0..layer_count).map(|_| None).collect() }
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn layer_position(&self, layer: usize) -> Option<usize> {
        self.layers.get(layer).and_then(|state| state.as_ref().map(|state| state.position))
    }

    pub fn layer_storage(&self, layer: usize) -> Option<&S> {
        self.layers.get(layer).and_then(|state| state.as_ref().map(|state| &state.storage))
    }

    pub fn layer_storage_mut(&mut self, layer: usize) -> Option<&mut S> {
        self.layers.get_mut(layer).and_then(|state| state.as_mut().map(|state| &mut state.storage))
    }

    /// 快照恢复：直接注入 storage 与已推进的 position，绕过 position 连续性检查。
    pub fn restore_layer(&mut self, layer: usize, position: usize, storage: S) -> Result<(), BackendError> {
        let slot = self.layers.get_mut(layer).ok_or(BackendError::UnsupportedLayer { layer })?;
        *slot = Some(LayerState { position, storage });
        Ok(())
    }

    /// 只回退 position 不动 storage(字节级快照恢复后对齐游标)。
    pub fn rewind_layer(&mut self, layer: usize, position: usize) -> Result<(), BackendError> {
        let slot = self.layers.get_mut(layer).ok_or(BackendError::UnsupportedLayer { layer })?;
        if let Some(state) = slot.as_mut() {
            state.position = position;
        }
        Ok(())
    }

    pub fn layer_mut<F>(&mut self, layer: usize, position: usize, spec: &Spec, allocate: F) -> Result<&mut LayerState<S>, BackendError>
    where
        F: FnOnce() -> Result<S, BackendError>,
    {
        if self.spec != *spec {
            return Err(BackendError::Compute { msg: format!("{} state spec 在 session 内发生变化", self.name) });
        }
        let layer_count = self.layers.len();
        let slot = self.layers.get_mut(layer).ok_or(BackendError::UnsupportedLayer { layer })?;
        if slot.is_none() {
            if position != 0 {
                return Err(BackendError::Compute { msg: format!("{} L{layer}/{layer_count} 尚无 state，不能从 position={position} 开始", self.name) });
            }
            *slot = Some(LayerState { position: 0, storage: allocate()? });
        }
        let state = slot.as_mut().expect("recurrent state 已初始化");
        if state.position != position {
            return Err(BackendError::Compute { msg: format!("{} L{layer}/{layer_count} position 不连续: state={}, input={position}", self.name, state.position) });
        }
        Ok(state)
    }

    pub fn allocated_bytes<F>(&self, bytes: F) -> usize
    where
        F: Fn(&S) -> usize,
    {
        self.layers.iter().filter_map(Option::as_ref).map(|layer| bytes(&layer.storage)).sum()
    }
}
