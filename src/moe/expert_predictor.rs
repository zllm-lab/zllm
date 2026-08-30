//! 后端无关的路由专家预测。
//!
//! 预测器只学习模型产生的真实路由，不持有权重、不执行 I/O，也不依赖计算后端。

#[derive(Clone, Copy, Debug)]
pub struct ExpertPredictorWeights {
    pub request_frequency: f32,
    pub temporal_transition: f32,
    pub spatial_transition: f32,
    pub future_router: f32,
}

impl Default for ExpertPredictorWeights {
    fn default() -> Self {
        Self { request_frequency: 0.15, temporal_transition: 0.30, spatial_transition: 0.20, future_router: 0.35 }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ExpertPredictorConfig {
    pub first_layer: usize,
    pub layer_count: usize,
    pub expert_count: usize,
    pub routed_top_k: usize,
    pub prefetch_count: usize,
    pub weights: ExpertPredictorWeights,
}

impl ExpertPredictorConfig {
    pub fn validate(self) -> Result<Self, String> {
        if self.layer_count == 0 {
            return Err("expert predictor requires at least one layer".to_owned());
        }
        if self.expert_count == 0 || self.expert_count > usize::from(u16::MAX) + 1 {
            return Err("expert predictor expert count must fit in u16".to_owned());
        }
        if self.routed_top_k == 0 || self.routed_top_k > self.expert_count {
            return Err("expert predictor routed Top-K is invalid".to_owned());
        }
        if self.prefetch_count > self.expert_count {
            return Err("expert predictor prefetch count is invalid".to_owned());
        }
        self.first_layer.checked_add(self.layer_count).ok_or_else(|| "expert predictor layer range overflow".to_owned())?;
        let weights = [self.weights.request_frequency, self.weights.temporal_transition, self.weights.spatial_transition, self.weights.future_router];
        if weights.iter().any(|weight| !weight.is_finite() || *weight < 0.0) || weights.iter().all(|weight| *weight == 0.0) {
            return Err("expert predictor weights must be finite, non-negative, and non-zero".to_owned());
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PredictionEvidence {
    pub request_frequency: bool,
    pub temporal_transition: bool,
    pub spatial_transition: bool,
    pub future_router: bool,
}

#[derive(Clone, Debug)]
pub struct ExpertPrediction {
    pub layer: usize,
    pub experts: Vec<u16>,
    pub priorities: Vec<f32>,
    pub evidence: PredictionEvidence,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ExpertPredictionStats {
    pub evaluated: u64,
    pub predicted_experts: u64,
    pub hits: u64,
}

const ROUTE_TRACE_MAGIC_V1: &[u8; 8] = b"ZLLMRT01";
const ROUTE_TRACE_MAGIC: &[u8; 8] = b"ZLLMRT02";

/// Prefill 按层产生路由，预测器按 token 学习；该结构负责两种顺序之间的转换。
pub struct ExpertRouteTrace {
    first_layer: usize,
    layer_count: usize,
    expert_count: usize,
    routed_top_k: usize,
    token_count: Option<usize>,
    routes: Vec<Option<Vec<u16>>>,
}

pub struct ExpertPredictionCalibration {
    pub evaluated: u64,
    pub rank_hits: Vec<u64>,
    pub routed_top_k: usize,
}

impl ExpertPredictionCalibration {
    pub fn cumulative_hits(&self, count: usize) -> u64 {
        self.rank_hits.iter().take(count).sum()
    }

    pub fn precision(&self, count: usize) -> f64 {
        if self.evaluated == 0 || count == 0 { 0.0 } else { self.cumulative_hits(count) as f64 / (self.evaluated * count as u64) as f64 }
    }

    pub fn recall(&self, count: usize) -> f64 {
        if self.evaluated == 0 { 0.0 } else { self.cumulative_hits(count) as f64 / (self.evaluated * self.routed_top_k as u64) as f64 }
    }
}

impl ExpertRouteTrace {
    pub fn new(first_layer: usize, layer_count: usize, expert_count: usize, routed_top_k: usize) -> Result<Self, String> {
        ExpertPredictorConfig { first_layer, layer_count, expert_count, routed_top_k, prefetch_count: routed_top_k, weights: ExpertPredictorWeights::default() }.validate()?;
        Ok(Self { first_layer, layer_count, expert_count, routed_top_k, token_count: None, routes: (0..layer_count).map(|_| None).collect() })
    }

    pub fn record_layer(&mut self, layer: usize, rows: usize, top_k: usize, expert_ids: &[u32]) -> Result<(), String> {
        let layer_index = layer.checked_sub(self.first_layer).filter(|index| *index < self.layer_count).ok_or_else(|| format!("prefill route layer {layer} 超出记录范围"))?;
        if top_k != self.routed_top_k || expert_ids.len() != rows * top_k {
            return Err(format!("prefill route L{layer} 形状错误: rows={rows},top_k={top_k},ids={}", expert_ids.len()));
        }
        if let Some(token_count) = self.token_count {
            if rows != token_count {
                return Err(format!("prefill route L{layer} token 数 {rows} 与前层 {token_count} 不一致"));
            }
        } else {
            self.token_count = Some(rows);
        }
        if self.routes[layer_index].is_some() {
            return Err(format!("prefill route L{layer} 重复记录"));
        }
        let mut route = Vec::with_capacity(expert_ids.len());
        for &expert in expert_ids {
            let expert = usize::try_from(expert).map_err(|_| format!("prefill route expert {expert} 超出 usize"))?;
            if expert >= self.expert_count {
                return Err(format!("prefill route expert 越界: {expert} >= {}", self.expert_count));
            }
            route.push(u16::try_from(expert).map_err(|_| format!("prefill route expert {expert} 超出 u16"))?);
        }
        self.routes[layer_index] = Some(route);
        Ok(())
    }

    pub fn is_complete(&self) -> bool {
        self.token_count.is_some() && self.routes.iter().all(Option::is_some)
    }

    pub fn replay(&self, predictor: &mut ExpertPredictor) -> Result<(), String> {
        let config = predictor.config();
        if config.first_layer != self.first_layer || config.layer_count != self.layer_count || config.expert_count != self.expert_count || config.routed_top_k != self.routed_top_k {
            return Err("prefill route trace 与 expert predictor 配置不一致".to_owned());
        }
        let token_count = self.token_count.filter(|_| self.is_complete()).ok_or_else(|| "prefill route trace 不完整".to_owned())?;
        predictor.reset_request();
        for token in 0..token_count {
            predictor.begin_token()?;
            let begin = token * self.routed_top_k;
            let end = begin + self.routed_top_k;
            for (layer_index, routes) in self.routes.iter().enumerate() {
                predictor.observe_route(self.first_layer + layer_index, &routes.as_ref().expect("已检查完整性")[begin..end])?;
            }
            predictor.finish_token()?;
        }
        Ok(())
    }

    /// 用完整 prefill 路由模拟在线 decode，统计每个预测排名的真实命中次数。
    pub fn calibrate(&self, prefetch_count: usize, weights: ExpertPredictorWeights) -> Result<ExpertPredictionCalibration, String> {
        let token_count = self.token_count.filter(|_| self.is_complete()).ok_or_else(|| "prefill route trace 不完整".to_owned())?;
        let mut predictor = ExpertPredictor::new(ExpertPredictorConfig { first_layer: self.first_layer, layer_count: self.layer_count, expert_count: self.expert_count, routed_top_k: self.routed_top_k, prefetch_count, weights })?;
        let mut rank_hits = vec![0u64; prefetch_count];
        let mut evaluated = 0u64;
        for token in 0..token_count {
            predictor.begin_token()?;
            let begin = token * self.routed_top_k;
            let end = begin + self.routed_top_k;
            for (layer_index, routes) in self.routes.iter().enumerate() {
                let layer = self.first_layer + layer_index;
                let actual = &routes.as_ref().expect("已检查完整性")[begin..end];
                if let Some(prediction) = predictor.predict(layer, None)? {
                    evaluated += 1;
                    for (rank, expert) in prediction.experts.iter().enumerate() {
                        rank_hits[rank] += u64::from(actual.contains(expert));
                    }
                }
                predictor.observe_route(layer, actual)?;
            }
            predictor.finish_token()?;
        }
        Ok(ExpertPredictionCalibration { evaluated, rank_hits, routed_top_k: self.routed_top_k })
    }

    /// checkpoint 只保存原始路由，避免绑定预测器内部表布局。
    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        let token_count = self.token_count.ok_or_else(|| "prefill route trace 尚无 MoE 层".to_owned())?;
        let header = [self.first_layer, self.layer_count, self.expert_count, self.routed_top_k, token_count];
        let values_per_layer = token_count.checked_mul(self.routed_top_k).ok_or_else(|| "prefill route trace 大小溢出".to_owned())?;
        let present_layers = self.routes.iter().filter(|routes| routes.is_some()).count();
        let value_count = present_layers.checked_mul(values_per_layer).ok_or_else(|| "prefill route trace 大小溢出".to_owned())?;
        let mut bytes = Vec::with_capacity(ROUTE_TRACE_MAGIC.len() + 5 * 8 + self.layer_count + value_count * 2);
        bytes.extend_from_slice(ROUTE_TRACE_MAGIC);
        for value in header {
            bytes.extend_from_slice(&u64::try_from(value).map_err(|_| "prefill route header 超出 u64".to_owned())?.to_le_bytes());
        }
        for routes in &self.routes {
            bytes.push(u8::from(routes.is_some()));
            if let Some(routes) = routes {
                for &expert in routes {
                    bytes.extend_from_slice(&expert.to_le_bytes());
                }
            }
        }
        Ok(bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        const HEADER_BYTES: usize = 8 + 5 * 8;
        if bytes.len() < HEADER_BYTES || (&bytes[..8] != ROUTE_TRACE_MAGIC && &bytes[..8] != ROUTE_TRACE_MAGIC_V1) {
            return Err("prefill route trace magic/header 错误".to_owned());
        }
        let partial = &bytes[..8] == ROUTE_TRACE_MAGIC;
        let mut cursor = 8;
        let mut read_u64 = || {
            let value = u64::from_le_bytes(bytes[cursor..cursor + 8].try_into().expect("header 长度已检查"));
            cursor += 8;
            usize::try_from(value).map_err(|_| "prefill route header 超出 usize".to_owned())
        };
        let first_layer = read_u64()?;
        let layer_count = read_u64()?;
        let expert_count = read_u64()?;
        let routed_top_k = read_u64()?;
        let token_count = read_u64()?;
        let values_per_layer = token_count.checked_mul(routed_top_k).ok_or_else(|| "prefill route layer 大小溢出".to_owned())?;
        let mut trace = Self::new(first_layer, layer_count, expert_count, routed_top_k)?;
        trace.token_count = Some(token_count);
        let mut data_cursor = HEADER_BYTES;
        for layer in &mut trace.routes {
            let present = if partial {
                let marker = *bytes.get(data_cursor).ok_or_else(|| "prefill route trace 缺少 layer marker".to_owned())?;
                data_cursor += 1;
                match marker {
                    0 => false,
                    1 => true,
                    _ => return Err(format!("prefill route trace layer marker {marker} 无效")),
                }
            } else {
                true
            };
            if !present {
                continue;
            }
            let layer_bytes = values_per_layer.checked_mul(2).ok_or_else(|| "prefill route layer 字节数溢出".to_owned())?;
            if data_cursor.checked_add(layer_bytes).is_none_or(|end| end > bytes.len()) {
                return Err("prefill route trace layer 数据截断".to_owned());
            }
            let mut routes = Vec::with_capacity(values_per_layer);
            for _ in 0..values_per_layer {
                routes.push(u16::from_le_bytes(bytes[data_cursor..data_cursor + 2].try_into().expect("route trace 长度已检查")));
                data_cursor += 2;
            }
            *layer = Some(routes);
        }
        if data_cursor != bytes.len() {
            return Err(format!("prefill route trace 尾部多出 {} bytes", bytes.len() - data_cursor));
        }
        Ok(trace)
    }
}

impl ExpertPredictionStats {
    #[inline]
    pub fn precision(self) -> f64 {
        if self.predicted_experts == 0 { 0.0 } else { self.hits as f64 / self.predicted_experts as f64 }
    }

    #[inline]
    pub fn recall(self, routed_top_k: usize) -> f64 {
        let possible = self.evaluated.saturating_mul(routed_top_k as u64);
        if possible == 0 { 0.0 } else { self.hits as f64 / possible as f64 }
    }
}

/// 请求级时空专家预测器。
///
/// 每个 token 先调用 `begin_token`，真实路由产生后调用 `observe_route`。
/// `finish_token` 把当前路由提交为下一 token 的时间历史。
pub struct ExpertPredictor {
    config: ExpertPredictorConfig,
    frequency: Box<[u32]>,
    temporal: Box<[u16]>,
    temporal_totals: Box<[u32]>,
    spatial: Box<[u16]>,
    spatial_totals: Box<[u32]>,
    recent_spatial: Box<[u16]>,
    recent_spatial_valid: Box<[bool]>,
    previous: Vec<Option<Vec<u16>>>,
    current: Vec<Option<Vec<u16>>>,
    pending: Vec<Option<Vec<u16>>>,
    stats: Vec<ExpertPredictionStats>,
    active_token: bool,
    // decode 热路径每 token 每层都会 predict/observe；scratch 复用避免每次堆分配。
    scratch_combined: Vec<f32>,
    scratch_signal: Vec<f32>,
    scratch_order: Vec<usize>,
    scratch_seen: Vec<bool>,
}

impl ExpertPredictor {
    pub fn new(config: ExpertPredictorConfig) -> Result<Self, String> {
        let config = config.validate()?;
        let layer_experts = config.layer_count.checked_mul(config.expert_count).ok_or_else(|| "expert predictor frequency size overflow".to_owned())?;
        let transition_rows = layer_experts;
        let transition_values = transition_rows.checked_mul(config.expert_count).ok_or_else(|| "expert predictor temporal table size overflow".to_owned())?;
        let spatial_rows = config.layer_count.saturating_sub(1).checked_mul(config.expert_count).ok_or_else(|| "expert predictor spatial row size overflow".to_owned())?;
        let spatial_values = spatial_rows.checked_mul(config.expert_count).ok_or_else(|| "expert predictor spatial table size overflow".to_owned())?;

        Ok(Self {
            config,
            frequency: vec![0; layer_experts].into_boxed_slice(),
            temporal: vec![0; transition_values].into_boxed_slice(),
            temporal_totals: vec![0; transition_rows].into_boxed_slice(),
            spatial: vec![0; spatial_values].into_boxed_slice(),
            spatial_totals: vec![0; spatial_rows].into_boxed_slice(),
            recent_spatial: vec![0; spatial_rows * config.routed_top_k].into_boxed_slice(),
            recent_spatial_valid: vec![false; spatial_rows].into_boxed_slice(),
            previous: vec![None; config.layer_count],
            current: vec![None; config.layer_count],
            pending: vec![None; config.layer_count],
            stats: vec![ExpertPredictionStats::default(); config.layer_count],
            active_token: false,
            scratch_combined: vec![0.0; config.expert_count],
            scratch_signal: vec![0.0; config.expert_count],
            scratch_order: vec![0; config.expert_count],
            scratch_seen: vec![false; config.expert_count],
        })
    }

    #[inline]
    pub fn config(&self) -> ExpertPredictorConfig {
        self.config
    }

    pub fn begin_token(&mut self) -> Result<(), String> {
        if self.active_token {
            return Err("expert predictor token is already active".to_owned());
        }
        self.current.fill(None);
        self.pending.fill(None);
        self.active_token = true;
        Ok(())
    }

    pub fn observe_route(&mut self, layer: usize, route: &[u16]) -> Result<(), String> {
        if !self.active_token {
            return Err("expert predictor requires begin_token before observing routes".to_owned());
        }
        let layer_index = self.layer_index(layer)?;
        self.validate_route(route)?;
        if self.current[layer_index].is_some() {
            return Err(format!("expert predictor already observed layer {layer}"));
        }

        if let Some(predicted) = self.pending[layer_index].take() {
            let actual = &mut self.scratch_seen;
            actual.fill(false);
            for &expert in route {
                actual[usize::from(expert)] = true;
            }
            let hits = predicted.iter().filter(|expert| actual[usize::from(**expert)]).count() as u64;
            let stats = &mut self.stats[layer_index];
            stats.evaluated = stats.evaluated.saturating_add(1);
            stats.predicted_experts = stats.predicted_experts.saturating_add(predicted.len() as u64);
            stats.hits = stats.hits.saturating_add(hits);
        }

        let frequency_base = layer_index * self.config.expert_count;
        for &expert in route {
            let value = &mut self.frequency[frequency_base + usize::from(expert)];
            *value = value.saturating_add(1);
        }

        // take/restore 代替 clone：路由 Vec 留在 predictor 内，不为遍历整段复制。
        if let Some(previous) = self.previous[layer_index].take() {
            for from in &previous {
                let row = layer_index * self.config.expert_count + usize::from(*from);
                update_transition(&mut self.temporal, &mut self.temporal_totals, row, self.config.expert_count, route);
            }
            self.previous[layer_index] = Some(previous);
        }

        if layer_index > 0 && self.current[layer_index - 1].is_some() {
            let source = self.current[layer_index - 1].take().expect("已检查非空");
            for from in &source {
                let row = (layer_index - 1) * self.config.expert_count + usize::from(*from);
                update_transition(&mut self.spatial, &mut self.spatial_totals, row, self.config.expert_count, route);
                let begin = row * self.config.routed_top_k;
                self.recent_spatial[begin..begin + self.config.routed_top_k].copy_from_slice(route);
                self.recent_spatial_valid[row] = true;
            }
            self.current[layer_index - 1] = Some(source);
        }

        self.current[layer_index] = Some(route.to_vec());
        Ok(())
    }

    /// 预测指定层专家。可选 future-router 分数必须包含每个专家的一个分数。
    pub fn predict(&mut self, layer: usize, future_router_scores: Option<&[f32]>) -> Result<Option<ExpertPrediction>, String> {
        if !self.active_token {
            return Err("expert predictor requires begin_token before prediction".to_owned());
        }
        let layer_index = self.layer_index(layer)?;
        if let Some(scores) = future_router_scores
            && scores.len() != self.config.expert_count
        {
            return Err(format!("future Router returned {} scores, expected {}", scores.len(), self.config.expert_count));
        }
        if self.config.prefetch_count == 0 {
            return Ok(None);
        }

        let count = self.config.expert_count;
        let combined = &mut self.scratch_combined;
        let signal = &mut self.scratch_signal;
        combined.fill(0.0);
        let mut evidence = PredictionEvidence::default();

        let frequency_base = layer_index * count;
        for (target, value) in signal.iter_mut().enumerate() {
            *value = self.frequency[frequency_base + target] as f32;
        }
        evidence.request_frequency = add_normalized_signal(combined, signal, self.config.weights.request_frequency);

        signal.fill(0.0);
        if let Some(previous) = &self.previous[layer_index] {
            accumulate_transition(signal, &self.temporal, &self.temporal_totals, layer_index, previous, count);
            if !signal.iter().any(|score| *score > 0.0) {
                for &expert in previous {
                    signal[usize::from(expert)] = 1.0;
                }
            }
            evidence.temporal_transition = add_normalized_signal(combined, signal, self.config.weights.temporal_transition);
        }

        signal.fill(0.0);
        if layer_index > 0
            && let Some(source) = &self.current[layer_index - 1]
        {
            if !accumulate_recent_transition(signal, &self.recent_spatial, &self.recent_spatial_valid, self.config.routed_top_k, layer_index - 1, source, count) {
                accumulate_transition(signal, &self.spatial, &self.spatial_totals, layer_index - 1, source, count);
            }
            evidence.spatial_transition = add_normalized_signal(combined, signal, self.config.weights.spatial_transition);
        }

        if let Some(router) = future_router_scores {
            evidence.future_router = add_normalized_signal(combined, router, self.config.weights.future_router);
        }

        if !combined.iter().any(|score| *score > 0.0) {
            return Ok(None);
        }
        // 只需要前 prefetch_count 名：select_nth 取 top-k 再排序，替代全排序。
        let prefetch = self.config.prefetch_count.min(count);
        let order = &mut self.scratch_order;
        order.clear();
        order.extend(0..count);
        let compare = |left: &usize, right: &usize| combined[*right].total_cmp(&combined[*left]).then_with(|| left.cmp(right));
        if prefetch < order.len() {
            order.select_nth_unstable_by(prefetch, compare);
            order.truncate(prefetch);
        }
        order.sort_unstable_by(compare);
        let experts: Vec<u16> = order.iter().map(|expert| *expert as u16).collect();
        let priorities: Vec<f32> = order.iter().map(|expert| combined[*expert]).collect();
        self.pending[layer_index] = Some(experts.clone());
        Ok(Some(ExpertPrediction { layer, experts, priorities, evidence }))
    }

    pub fn finish_token(&mut self) -> Result<(), String> {
        if !self.active_token {
            return Err("expert predictor has no active token".to_owned());
        }
        for layer in 0..self.config.layer_count {
            if let Some(route) = self.current[layer].take() {
                self.previous[layer] = Some(route);
            }
        }
        self.pending.fill(None);
        self.active_token = false;
        Ok(())
    }

    pub fn stats(&self, layer: usize) -> Result<ExpertPredictionStats, String> {
        Ok(self.stats[self.layer_index(layer)?])
    }

    fn reset_request(&mut self) {
        self.frequency.fill(0);
        self.temporal.fill(0);
        self.temporal_totals.fill(0);
        self.spatial.fill(0);
        self.spatial_totals.fill(0);
        self.recent_spatial.fill(0);
        self.recent_spatial_valid.fill(false);
        self.previous.fill(None);
        self.current.fill(None);
        self.pending.fill(None);
        self.stats.fill(ExpertPredictionStats::default());
        self.active_token = false;
    }

    pub fn storage_bytes(&self) -> usize {
        self.frequency.len() * size_of::<u32>()
            + self.temporal.len() * size_of::<u16>()
            + self.temporal_totals.len() * size_of::<u32>()
            + self.spatial.len() * size_of::<u16>()
            + self.spatial_totals.len() * size_of::<u32>()
            + self.recent_spatial.len() * size_of::<u16>()
            + self.recent_spatial_valid.len() * size_of::<bool>()
    }

    fn layer_index(&self, layer: usize) -> Result<usize, String> {
        layer.checked_sub(self.config.first_layer).filter(|index| *index < self.config.layer_count).ok_or_else(|| format!("expert predictor layer {layer} is outside the MoE range"))
    }

    fn validate_route(&mut self, route: &[u16]) -> Result<(), String> {
        if route.len() != self.config.routed_top_k {
            return Err(format!("expert route has {} experts, expected {}", route.len(), self.config.routed_top_k));
        }
        let seen = &mut self.scratch_seen;
        seen.fill(false);
        for &expert in route {
            let expert = usize::from(expert);
            if expert >= self.config.expert_count {
                return Err(format!("expert ID {expert} is out of range"));
            }
            if std::mem::replace(&mut seen[expert], true) {
                return Err(format!("expert route contains duplicate ID {expert}"));
            }
        }
        Ok(())
    }
}

fn update_transition(table: &mut [u16], totals: &mut [u32], row: usize, expert_count: usize, targets: &[u16]) {
    const RESCALE_AT: u32 = 60_000;
    let base = row * expert_count;
    if totals[row] >= RESCALE_AT {
        let values = &mut table[base..base + expert_count];
        for value in values {
            *value /= 2;
        }
        totals[row] /= 2;
    }
    for &target in targets {
        let value = &mut table[base + usize::from(target)];
        *value = value.saturating_add(1);
    }
    totals[row] = totals[row].saturating_add(targets.len() as u32);
}

fn accumulate_transition(output: &mut [f32], table: &[u16], totals: &[u32], layer_index: usize, sources: &[u16], expert_count: usize) {
    for &source in sources {
        let row = layer_index * expert_count + usize::from(source);
        let total = totals[row];
        if total == 0 {
            continue;
        }
        let base = row * expert_count;
        let scale = 1.0 / total as f32;
        for (target, output) in output.iter_mut().enumerate() {
            *output += table[base + target] as f32 * scale;
        }
    }
}

fn accumulate_recent_transition(output: &mut [f32], table: &[u16], valid: &[bool], routed_top_k: usize, layer_index: usize, sources: &[u16], expert_count: usize) -> bool {
    let mut found = false;
    for &source in sources {
        let row = layer_index * expert_count + usize::from(source);
        if !valid[row] {
            continue;
        }
        found = true;
        let begin = row * routed_top_k;
        for &target in &table[begin..begin + routed_top_k] {
            output[usize::from(target)] += 1.0;
        }
    }
    found
}

fn add_normalized_signal(combined: &mut [f32], signal: &[f32], weight: f32) -> bool {
    if weight == 0.0 {
        return false;
    }
    let mut minimum = f32::INFINITY;
    let mut maximum = f32::NEG_INFINITY;
    for &value in signal {
        if value.is_finite() {
            minimum = minimum.min(value);
            maximum = maximum.max(value);
        }
    }
    let range = maximum - minimum;
    if !range.is_finite() || range <= f32::EPSILON {
        return false;
    }
    let scale = weight / range;
    for (combined, &value) in combined.iter_mut().zip(signal) {
        if value.is_finite() {
            *combined += (value - minimum) * scale;
        }
    }
    true
}

use std::mem::size_of;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_frequency_predicts_previous_route() {
        let mut predictor = ExpertPredictor::new(ExpertPredictorConfig {
            first_layer: 3,
            layer_count: 2,
            expert_count: 8,
            routed_top_k: 2,
            prefetch_count: 2,
            weights: ExpertPredictorWeights { request_frequency: 1.0, temporal_transition: 0.0, spatial_transition: 0.0, future_router: 0.0 },
        })
        .unwrap();
        predictor.begin_token().unwrap();
        predictor.observe_route(3, &[2, 5]).unwrap();
        predictor.observe_route(4, &[1, 7]).unwrap();
        predictor.finish_token().unwrap();

        predictor.begin_token().unwrap();
        let prediction = predictor.predict(3, None).unwrap().unwrap();
        assert_eq!(prediction.experts, vec![2, 5]);
    }

    #[test]
    fn temporal_prediction_uses_previous_route_before_transitions_exist() {
        let mut predictor = ExpertPredictor::new(ExpertPredictorConfig {
            first_layer: 3,
            layer_count: 1,
            expert_count: 8,
            routed_top_k: 2,
            prefetch_count: 2,
            weights: ExpertPredictorWeights { request_frequency: 0.0, temporal_transition: 1.0, spatial_transition: 0.0, future_router: 0.0 },
        })
        .unwrap();
        predictor.begin_token().unwrap();
        predictor.observe_route(3, &[2, 5]).unwrap();
        predictor.finish_token().unwrap();
        predictor.begin_token().unwrap();
        let prediction = predictor.predict(3, None).unwrap().unwrap();
        assert_eq!(prediction.experts, vec![2, 5]);
    }

    #[test]
    fn route_trace_roundtrip_and_replay() {
        let mut trace = ExpertRouteTrace::new(3, 2, 8, 2).unwrap();
        trace.record_layer(3, 2, 2, &[2, 5, 1, 6]).unwrap();
        trace.record_layer(4, 2, 2, &[1, 7, 0, 4]).unwrap();
        let decoded = ExpertRouteTrace::from_bytes(&trace.to_bytes().unwrap()).unwrap();
        let mut predictor = ExpertPredictor::new(ExpertPredictorConfig { first_layer: 3, layer_count: 2, expert_count: 8, routed_top_k: 2, prefetch_count: 2, weights: ExpertPredictorWeights::default() }).unwrap();
        decoded.replay(&mut predictor).unwrap();
        predictor.begin_token().unwrap();
        assert!(predictor.predict(3, None).unwrap().is_some());
    }

    #[test]
    fn zero_prefetch_disables_prediction() {
        let mut predictor = ExpertPredictor::new(ExpertPredictorConfig { first_layer: 0, layer_count: 1, expert_count: 8, routed_top_k: 2, prefetch_count: 0, weights: ExpertPredictorWeights::default() }).unwrap();
        predictor.begin_token().unwrap();
        assert!(predictor.predict(0, None).unwrap().is_none());
        predictor.observe_route(0, &[1, 3]).unwrap();
        predictor.finish_token().unwrap();
    }

    #[test]
    fn external_route_trace_prediction_curve() {
        let Some(path) = std::env::var_os("ZLLM_ROUTE_TRACE") else {
            return;
        };
        let bytes = std::fs::read(path).unwrap();
        let trace = ExpertRouteTrace::from_bytes(&bytes).unwrap();
        let calibration = trace.calibrate(20, ExpertPredictorWeights::default()).unwrap();
        for count in 1..=20 {
            eprintln!("prefetch_count={count} precision={:.6} recall={:.6} cumulative_hits={}", calibration.precision(count), calibration.recall(count), calibration.cumulative_hits(count),);
        }
        let candidates = [
            ("frequency", ExpertPredictorWeights { request_frequency: 1.0, temporal_transition: 0.0, spatial_transition: 0.0, future_router: 0.0 }),
            ("temporal", ExpertPredictorWeights { request_frequency: 0.0, temporal_transition: 1.0, spatial_transition: 0.0, future_router: 0.0 }),
            ("spatial", ExpertPredictorWeights { request_frequency: 0.0, temporal_transition: 0.0, spatial_transition: 1.0, future_router: 0.0 }),
            ("frequency_temporal", ExpertPredictorWeights { request_frequency: 0.3, temporal_transition: 0.7, spatial_transition: 0.0, future_router: 0.0 }),
            ("frequency_spatial", ExpertPredictorWeights { request_frequency: 0.3, temporal_transition: 0.0, spatial_transition: 0.7, future_router: 0.0 }),
            ("temporal_spatial", ExpertPredictorWeights { request_frequency: 0.0, temporal_transition: 0.6, spatial_transition: 0.4, future_router: 0.0 }),
            ("temporal_heavy", ExpertPredictorWeights { request_frequency: 0.1, temporal_transition: 0.7, spatial_transition: 0.2, future_router: 0.0 }),
            ("spatial_heavy", ExpertPredictorWeights { request_frequency: 0.1, temporal_transition: 0.2, spatial_transition: 0.7, future_router: 0.0 }),
            ("balanced", ExpertPredictorWeights { request_frequency: 0.2, temporal_transition: 0.4, spatial_transition: 0.4, future_router: 0.0 }),
            ("spatial_temporal_fallback", ExpertPredictorWeights { request_frequency: 0.0, temporal_transition: 0.01, spatial_transition: 1.0, future_router: 0.0 }),
            ("spatial_frequency_fallback", ExpertPredictorWeights { request_frequency: 0.01, temporal_transition: 0.0, spatial_transition: 1.0, future_router: 0.0 }),
            ("spatial_both_fallback", ExpertPredictorWeights { request_frequency: 0.01, temporal_transition: 0.01, spatial_transition: 1.0, future_router: 0.0 }),
        ];
        for (name, weights) in candidates {
            let calibration2 = trace.calibrate(2, weights).unwrap();
            let calibration8 = trace.calibrate(8, weights).unwrap();
            eprintln!(
                "weights={name} precision2={:.6} recall2={:.6} hits2={} precision8={:.6} recall8={:.6} hits8={}",
                calibration2.precision(2),
                calibration2.recall(2),
                calibration2.cumulative_hits(2),
                calibration8.precision(8),
                calibration8.recall(8),
                calibration8.cumulative_hits(8)
            );
        }
    }
}
