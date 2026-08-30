//! GLM-5.2 ROCm 连续 stage pipeline 节点。

use super::*;

struct Glm52PipelineSession {
    caches: Vec<RocmKvCache>,
    dsa: Vec<RocmDsaState>,
}

struct Glm52PipelineNode {
    cfg: Glm52Config,
    mla: MlaSpec,
    contexts: Vec<RocmContext>,
    start_layer: usize,
    end_layer: usize,
    layers: Vec<(usize, PrefillLayer)>,
    experts: Vec<RocmPrefillExperts>,
    rope: RopeTable,
    output_head: Option<Glm52OutputHead<RocmWeight>>,
    sessions: HashMap<String, Glm52PipelineSession>,
    max_seq_len: usize,
}

impl Glm52PipelineNode {
    fn load(root_context: &RocmContext, args: &Args, cfg: &Glm52Config, mla: &MlaSpec, weights: &Glm52Weights, start_layer: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let end_layer = args.end_layer.unwrap_or(cfg.layer_count);
        if start_layer == 0 || start_layer >= end_layer || end_layer > cfg.layer_count {
            return Err(format!("非入口 pipeline 节点要求 0 < start-layer < end-layer <= {}", cfg.layer_count).into());
        }
        if !weights.source_is_ct() && !weights.source_is_gguf() {
            return Err("GLM pipeline node 当前仅支持 compressed-tensors/GGUF 权重".into());
        }
        let devices = args.prefill_devices.clone().unwrap_or_else(|| vec![0]);
        let layer_ends = args.prefill_layer_ends.clone().unwrap_or_else(|| vec![end_layer - 1]);
        if devices.len() != layer_ends.len() || layer_ends.last().copied() != Some(end_layer - 1) || layer_ends.windows(2).any(|pair| pair[0] >= pair[1]) || layer_ends.first().copied().unwrap_or(0) < start_layer {
            return Err("pipeline node 的 --prefill-devices/--prefill-layer-ends 配置无效".into());
        }
        let contexts = devices.iter().map(|&device| root_context.for_device(device).map_err(|error| format!("ROCm pipeline device {device} 初始化失败: {error}"))).collect::<Result<Vec<_>, _>>()?;
        let mut layers = Vec::with_capacity(end_layer - start_layer);
        for layer in start_layer..end_layer {
            let placement = layer_ends.iter().position(|&end| layer <= end).ok_or_else(|| format!("L{layer} 没有 pipeline device"))?;
            let backend = &contexts[placement];
            backend.activate()?;
            let resident = if layer < cfg.dense_layer_count {
                PrefillLayer::Dense(crate::runtime::glm52::load_prepare_dense_prefill_layer(backend, cfg, mla, weights, layer, false).map_err(|error| format!("准备 ROCm pipeline L{layer}: {error:?}"))?)
            } else {
                PrefillLayer::Moe(crate::runtime::glm52::load_prepare_moe_prefill_layer(backend, cfg, mla, weights, layer, false).map_err(|error| format!("准备 ROCm pipeline L{layer}: {error:?}"))?)
            };
            layers.push((placement, resident));
        }
        let mut experts = contexts
            .iter()
            .map(|_| -> Result<RocmPrefillExperts, Box<dyn std::error::Error>> {
                if weights.source_is_gguf() { Ok(RocmPrefillExperts::gguf(weights.gguf_source()?)) } else { Ok(RocmPrefillExperts::ct(weights.ct_source()?)) }
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (offset, (placement, resident)) in layers.iter().enumerate() {
            if !matches!(resident, PrefillLayer::Moe(_)) {
                continue;
            }
            let layer = start_layer + offset;
            let backend = &contexts[*placement];
            backend.activate()?;
            experts[*placement].preload_layer(backend, layer, cfg.expert_count).map_err(|error| format!("常驻 ROCm pipeline L{layer} experts: {error:?}"))?;
        }
        let output_head = if end_layer == cfg.layer_count {
            let placement = layers.last().ok_or("pipeline node 没有层")?.0;
            let backend = &contexts[placement];
            backend.activate()?;
            let final_norm = weights.final_norm()?;
            let lm_head = weights.lm_head_bf16_bytes()?;
            Some(prepare_glm52_output_head(backend, cfg, &final_norm, LinearWeight::Bf16Bytes(&lm_head)).map_err(|error| format!("准备 ROCm pipeline output head: {error:?}"))?)
        } else {
            None
        };
        eprintln!("[pipeline-resident] layers={start_layer}..{end_layer} devices={devices:?} experts=resident");
        Ok(Self {
            cfg: cfg.clone(),
            mla: mla.clone(),
            contexts,
            start_layer,
            end_layer,
            layers,
            experts,
            rope: RopeTable::precompute(args.max_seq_len, mla.qk_rope_head_dim, mla.rope_theta),
            output_head,
            sessions: HashMap::new(),
            max_seq_len: args.max_seq_len,
        })
    }

    fn execute(&mut self, message: PipelineMessage) -> Result<Option<PipelineMessage>, String> {
        let execution_hash = message.execution_hash.clone();
        if matches!(message.kind, PipelineMessageKind::Token | PipelineMessageKind::Error) {
            return Err("非入口 pipeline 节点不能接收 token/error".to_owned());
        }
        if message.layer != self.start_layer {
            return Err(format!("pipeline hidden 输入层={}，本节点 start-layer={}", message.layer, self.start_layer));
        }
        if message.kind == PipelineMessageKind::Clear {
            self.sessions.remove(&execution_hash);
            return Ok((self.end_layer < self.cfg.layer_count).then_some(PipelineMessage { layer: self.end_layer, ..message }));
        }
        if message.hidden_size != self.cfg.hidden_size || message.rows == 0 {
            return Err(format!("pipeline hidden shape={}x{}，期望 Nx{}", message.rows, message.hidden_size, self.cfg.hidden_size));
        }
        if message.kind == PipelineMessageKind::Prefill && message.position != 0 {
            return Err("新 prefill position 必须为 0".to_owned());
        }
        if message.kind == PipelineMessageKind::Decode && message.rows != 1 {
            return Err("decode hidden 必须只有一行".to_owned());
        }
        if message.position.saturating_add(message.rows) > self.max_seq_len {
            return Err(format!("pipeline position={} rows={} 超过 max_seq_len={}", message.position, message.rows, self.max_seq_len));
        }
        if message.kind == PipelineMessageKind::Prefill {
            self.sessions.remove(&execution_hash);
            let caches = (0..self.contexts.len()).map(|_| RocmKvCache::with_capacity(self.cfg.layer_count, self.max_seq_len)).collect();
            let dsa = (0..self.contexts.len())
                .map(|_| RocmDsaState::new(self.cfg.layer_count, self.max_seq_len, self.cfg.index_head_dim, self.cfg.index_top_k))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| format!("创建 pipeline DSA: {error}"))?;
            self.sessions.insert(execution_hash.clone(), Glm52PipelineSession { caches, dsa });
        }
        let session = self.sessions.get_mut(&execution_hash).ok_or_else(|| format!("execution_hash={execution_hash} 没有 KV cache，请先 prefill"))?;
        let data = message.payload.chunks_exact(2).map(|bytes| half::bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32()).collect::<Vec<_>>();
        let first_placement = self.layers.first().ok_or("pipeline node 没有层")?.0;
        let mut hidden = self.contexts[first_placement].tensor_from_f32(data, message.rows, self.cfg.hidden_size).map_err(|error| format!("上传 pipeline hidden: {error}"))?;
        for (offset, (placement, resident)) in self.layers.iter().enumerate() {
            let layer = self.start_layer + offset;
            let backend = &self.contexts[*placement];
            backend.activate().map_err(|error| format!("激活 L{layer} device: {error}"))?;
            hidden = backend.tensor_on_device(hidden).and_then(|hidden| backend.tensor_as_f32(hidden)).map_err(|error| format!("迁移 L{layer} hidden: {error:?}"))?;
            let dsa = &mut session.dsa[*placement];
            let cache = &mut session.caches[*placement];
            hidden = match resident {
                PrefillLayer::Dense(resident) => glm52_dense_prefill_layer(backend, &self.cfg, &self.mla, resident, layer, Some(dsa), &hidden, &self.rope, Some(cache), message.position),
                PrefillLayer::Moe(resident) => glm52_moe_prefill_layer(backend, &self.cfg, &self.mla, resident, layer, &mut self.experts[*placement], None, Some(dsa), &hidden, &self.rope, Some(cache), message.position),
            }
            .map_err(|error| format!("ROCm pipeline L{layer}: {error:?}"))?;
            hidden = backend.tensor_as_bf16(hidden).map_err(|error| format!("ROCm pipeline L{layer} 转 BF16: {error:?}"))?;
        }
        let last_placement = self.layers.last().ok_or("pipeline node 没有层")?.0;
        let backend = &self.contexts[last_placement];
        if let Some(output_head) = self.output_head.as_ref() {
            let last = backend.select_row(&hidden, message.rows - 1).map_err(|error| format!("pipeline 选择末行: {error:?}"))?;
            let token_id = glm52_token_output(backend, &self.cfg, output_head, &last).map_err(|error| format!("pipeline token output: {error:?}"))?.token_id;
            return Ok(Some(PipelineMessage {
                execution_hash,
                kind: PipelineMessageKind::Token,
                layer: self.end_layer,
                position: message.position + message.rows,
                rows: 0,
                hidden_size: 0,
                token_id: Some(token_id),
                error: None,
                payload: Vec::new(),
            }));
        }
        let bits = backend.tensor_to_bf16_bits(&hidden).map_err(|error| format!("下载 pipeline hidden: {error:?}"))?;
        let mut payload = Vec::with_capacity(bits.len() * 2);
        for value in bits {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        Ok(Some(PipelineMessage { execution_hash, kind: message.kind, layer: self.end_layer, position: message.position, rows: message.rows, hidden_size: self.cfg.hidden_size, token_id: None, error: None, payload }))
    }
}

pub(super) fn run_glm52_pipeline_node(root_context: &RocmContext, args: &Args, cfg: &Glm52Config, mla: &MlaSpec, weights: &Glm52Weights, start_layer: usize) -> Result<(), Box<dyn std::error::Error>> {
    let engine = Glm52PipelineNode::load(root_context, args, cfg, mla, weights, start_layer)?;
    let secret_key = args.pipeline_iroh.as_ref().and_then(|iroh| iroh.secret_key.clone()).ok_or("pipeline node 要求 YAML 配置固定 iroh.secret_key")?;
    let node_id = secret_key.public();
    let parse_target = |ticket: Option<&String>| ticket.map(|ticket| iroh_tickets::endpoint::EndpointTicket::from_str(ticket)).transpose().map(|ticket| ticket.map(|ticket| ticket.endpoint_addr().clone()));
    let next = parse_target(args.next_node.as_ref())?;
    let decode = parse_target(args.decode_node.as_ref())?;
    if engine.end_layer < cfg.layer_count && next.is_none() {
        return Err("非末端 pipeline node 必须设置 --next-node TICKET".into());
    }
    if engine.end_layer == cfg.layer_count && decode.is_none() {
        return Err("末端 pipeline node 必须设置 --decode-node TICKET".into());
    }
    let runtime = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    runtime.block_on(async move {
        let endpoint = iroh::Endpoint::builder(iroh::endpoint::presets::N0).secret_key(secret_key).alpns(vec![PIPELINE_ALPN.to_vec()]).bind().await?;
        tokio::time::timeout(std::time::Duration::from_secs(15), endpoint.online()).await.map_err(|_| std::io::Error::new(std::io::ErrorKind::TimedOut, "GLM pipeline 连接 iroh 官方 relay 超时"))?;
        let ticket = iroh_tickets::endpoint::EndpointTicket::new(endpoint.addr()).to_string();
        eprintln!("[pipeline-node] id={node_id} layers={}..{} ticket={ticket}", engine.start_layer, engine.end_layer);
        serve_glm52_pipeline(endpoint, next, decode, engine).await
    })
}

async fn serve_glm52_pipeline(endpoint: iroh::Endpoint, next: Option<iroh::EndpointAddr>, decode: Option<iroh::EndpointAddr>, mut engine: Glm52PipelineNode) -> Result<(), Box<dyn std::error::Error>> {
    while let Some(accepting) = endpoint.accept().await {
        let result = async {
            let connection = accepting.await?;
            let mut recv = connection.accept_uni().await?;
            let message = read_pipeline_message(&mut recv).await?;
            let execution_hash = message.execution_hash.clone();
            let outgoing = match engine.execute(message) {
                Ok(message) => message,
                Err(error) => Some(PipelineMessage { execution_hash, kind: PipelineMessageKind::Error, layer: engine.end_layer, position: 0, rows: 0, hidden_size: 0, token_id: None, error: Some(error), payload: Vec::new() }),
            };
            if let Some(message) = outgoing {
                let target = match message.kind {
                    PipelineMessageKind::Prefill | PipelineMessageKind::Decode | PipelineMessageKind::Clear => next.as_ref(),
                    PipelineMessageKind::Token | PipelineMessageKind::Error => decode.as_ref(),
                }
                .ok_or("pipeline 消息没有目标节点")?;
                let connection = endpoint.connect(target.clone(), PIPELINE_ALPN).await?;
                let mut send = connection.open_uni().await?;
                write_pipeline_message(&mut send, &message).await?;
                send.finish()?;
            }
            Ok::<(), Box<dyn std::error::Error>>(())
        }
        .await;
        if let Err(error) = result {
            eprintln!("[pipeline-node] 消息失败: {error}");
        }
    }
    Ok(())
}
