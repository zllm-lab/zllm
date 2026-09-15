//! H3 网站业务入口：账户任务、余额扣费与用户资产。

use std::{collections::HashMap, path::PathBuf};

use axum::{
    Json,
    extract::{Multipart, Path as AxumPath, Query, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use base64::Engine;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio_postgres::{Client, Row};

use super::{ServerConfig, ServerState, account, h3, image};

const IMAGE_PRICE_MILLI: i64 = 100;
const VIDEO_SECOND_PRICE_MILLI: i64 = 200;
pub(super) const HD_SECOND_PRICE_MILLI: i64 = 500;
// 2K、33帧批次约5分钟；冷启动估算留出加载余量，实际队列优先采用已完成任务计时。
pub(super) const HD_ESTIMATED_SECONDS_PER_VIDEO_SECOND: i64 = 240;
const MAX_IMAGE_UPLOAD_BYTES: usize = 20 * 1024 * 1024;
const MAX_MEDIA_UPLOAD_BYTES: usize = 75 * 1024 * 1024;

#[derive(Deserialize)]
pub(super) struct GenerationRequest {
    #[serde(rename = "type")]
    kind: String,
    prompt: String,
    #[serde(default)]
    references: HashMap<String, String>,
    output: GenerationOutput,
    purpose: Option<String>,
}

#[derive(Deserialize)]
pub(super) struct GenerationOutput {
    aspect_ratio: String,
    resolution: String,
    duration_seconds: Option<u64>,
}

pub(super) async fn initialize(config: &ServerConfig) -> Result<(), String> {
    let Some(postgres) = config.account_postgres.as_deref() else {
        return Ok(());
    };
    let client = account::connect(postgres).await?;
    client
        .batch_execute(
            "CREATE TABLE IF NOT EXISTS h3_studio_tasks (
                 id TEXT PRIMARY KEY,
                 account_id BIGINT NOT NULL REFERENCES h3_accounts(id) ON DELETE CASCADE,
                 task_kind TEXT NOT NULL,
                 modality TEXT NOT NULL,
                 title TEXT NOT NULL,
                 purpose TEXT,
                 price_milli BIGINT NOT NULL CHECK (price_milli >= 0),
                 duration_seconds INTEGER,
                 resolution TEXT,
                 request_json TEXT NOT NULL,
                 status TEXT NOT NULL DEFAULT 'queued',
                 refunded_at TIMESTAMPTZ,
                 settled_price_milli BIGINT CHECK (settled_price_milli >= 0),
                 settlement_progress_milli INTEGER CHECK (settlement_progress_milli BETWEEN 0 AND 1000),
                 created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                 updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
             );
             ALTER TABLE h3_studio_tasks ADD COLUMN IF NOT EXISTS settled_price_milli BIGINT CHECK (settled_price_milli >= 0);
             ALTER TABLE h3_studio_tasks ADD COLUMN IF NOT EXISTS settlement_progress_milli INTEGER CHECK (settlement_progress_milli BETWEEN 0 AND 1000);
             CREATE INDEX IF NOT EXISTS idx_h3_studio_tasks_account_created ON h3_studio_tasks(account_id, created_at DESC);
             CREATE TABLE IF NOT EXISTS h3_ledger (
                 id BIGSERIAL PRIMARY KEY,
                 account_id BIGINT NOT NULL REFERENCES h3_accounts(id) ON DELETE CASCADE,
                 task_id TEXT NOT NULL REFERENCES h3_studio_tasks(id) ON DELETE CASCADE,
                 kind TEXT NOT NULL,
                 amount_milli BIGINT NOT NULL,
                 created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                 UNIQUE(task_id, kind)
             );
             CREATE TABLE IF NOT EXISTS h3_assets (
                 id TEXT PRIMARY KEY,
                 account_id BIGINT NOT NULL REFERENCES h3_accounts(id) ON DELETE CASCADE,
                 task_id TEXT REFERENCES h3_studio_tasks(id) ON DELETE SET NULL,
                 artifact_id TEXT,
                 kind TEXT NOT NULL,
                 role TEXT,
                 title TEXT NOT NULL,
                 file_name TEXT NOT NULL,
                 content_type TEXT NOT NULL,
                 bytes BIGINT NOT NULL CHECK (bytes >= 0),
                 file_path TEXT,
                 duration_seconds INTEGER,
                 resolution TEXT,
                 deleted_at TIMESTAMPTZ,
                 created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                 UNIQUE(task_id, artifact_id)
             );
             ALTER TABLE h3_assets ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ;
             CREATE INDEX IF NOT EXISTS idx_h3_assets_account_created ON h3_assets(account_id, created_at DESC);",
        )
        .await
        .map_err(|error| format!("初始化 H3 任务与资产表失败: {error}"))?;
    Ok(())
}

pub(super) async fn create_generation(State(state): State<ServerState>, headers: HeaderMap, Json(input): Json<GenerationRequest>) -> Response {
    let (mut client, account) = match account::require_account(&state, &headers).await {
        Ok(value) => value,
        Err(response) => return response,
    };
    let task_id = format!("task_{}", state.request_id().trim_start_matches("req_"));
    let title = title(&input.prompt);
    let prepared = match prepare_request(&state, &client, &account, &input).await {
        Ok(prepared) => prepared,
        Err(response) => return response,
    };
    if let Err(response) = reserve(&mut client, &account, &task_id, &title, &input, &prepared).await {
        return response;
    }
    let dispatched = state.scheduler.dispatch_task(task_id.clone(), prepared.model.to_owned(), prepared.task_kind.to_owned(), prepared.request.clone(), prepared.metadata.clone()).await;
    if let Err(error) = dispatched {
        let _ = settle(&mut client, &task_id, 0).await;
        return error_response(StatusCode::SERVICE_UNAVAILABLE, "generation_unavailable", error.to_string());
    }
    generation_response(&state, &client, &account, &task_id).await
}

pub(super) async fn get_generation(State(state): State<ServerState>, headers: HeaderMap, AxumPath(task_id): AxumPath<String>) -> Response {
    let (client, account) = match account::require_account(&state, &headers).await {
        Ok(value) => value,
        Err(response) => return response,
    };
    generation_response(&state, &client, &account, &task_id).await
}

pub(super) async fn list_generations(State(state): State<ServerState>, headers: HeaderMap, Query(query): Query<HashMap<String, String>>) -> Response {
    let (client, account) = match account::require_account(&state, &headers).await {
        Ok(value) => value,
        Err(response) => return response,
    };
    let limit = query.get("limit").and_then(|value| value.parse::<i64>().ok()).unwrap_or(30).clamp(1, 100);
    let rows = match client.query("SELECT id FROM h3_studio_tasks WHERE account_id = $1 ORDER BY created_at DESC LIMIT $2", &[&account.id, &limit]).await {
        Ok(rows) => rows,
        Err(error) => return server_error(format!("读取用户任务失败: {error}")),
    };
    let mut items = Vec::with_capacity(rows.len());
    for row in rows {
        if let Ok(value) = generation_value(&state, &client, &account, row.get("id")).await {
            items.push(value);
        }
    }
    Json(json!({"items": items})).into_response()
}

pub(super) async fn cancel_generation(State(state): State<ServerState>, headers: HeaderMap, AxumPath(task_id): AxumPath<String>) -> Response {
    let (mut client, account) = match account::require_account(&state, &headers).await {
        Ok(value) => value,
        Err(response) => return response,
    };
    let persisted_status = match client.query_opt("SELECT status FROM h3_studio_tasks WHERE id = $1 AND account_id = $2", &[&task_id, &account.id]).await {
        Ok(Some(row)) => row.get::<_, String>("status"),
        Ok(None) => return error_response(StatusCode::NOT_FOUND, "task_not_found", "任务不存在"),
        Err(error) => return server_error(format!("读取取消任务失败: {error}")),
    };
    if matches!(persisted_status.as_str(), "succeeded" | "failed" | "cancelled") {
        return generation_response(&state, &client, &account, &task_id).await;
    }
    let Some(task) = state.scheduler.cancel_task(&task_id).await else {
        return error_response(StatusCode::NOT_FOUND, "task_not_found", "任务不存在");
    };
    if let Err(error) = client.execute("UPDATE h3_studio_tasks SET status = $1, updated_at = NOW() WHERE id = $2", &[&task.status, &task_id]).await {
        return server_error(format!("更新取消状态失败: {error}"));
    }
    if task.status == "cancelled" {
        let progress_milli = billable_progress_milli(&task.task_kind, task.progress.as_ref());
        if let Err(error) = settle(&mut client, &task_id, progress_milli).await {
            return server_error(error);
        }
    }
    generation_response(&state, &client, &account, &task_id).await
}

pub(super) async fn create_hd(State(state): State<ServerState>, headers: HeaderMap, AxumPath(task_id): AxumPath<String>) -> Response {
    let (mut client, account) = match account::require_account(&state, &headers).await {
        Ok(value) => value,
        Err(response) => return response,
    };
    // 先同步源任务及资产；请求只接受任务 ID，输入路径和时长均由服务端确定。
    let source = match generation_value(&state, &client, &account, &task_id).await {
        Ok(source) => source,
        Err(response) => return response,
    };
    if source["status"] != "succeeded" || source["type"] != "video" || source["resolution"] == "2K" {
        return error_response(StatusCode::BAD_REQUEST, "invalid_hd_source", "请选择已完成的普通视频生成 2K 视频");
    }
    let duration = match source["duration_seconds"].as_i64().filter(|value| matches!(value, 5 | 10 | 15)) {
        Some(value) => value,
        None => return error_response(StatusCode::BAD_REQUEST, "invalid_hd_duration", "原视频时长必须是 5、10 或 15 秒"),
    };
    let purpose = format!("hd:{task_id}");
    let previous = match client.query("SELECT id, status FROM h3_studio_tasks WHERE account_id = $1 AND purpose = $2 AND task_kind = 'video_super_resolution' ORDER BY created_at DESC", &[&account.id, &purpose]).await {
        Ok(rows) => rows,
        Err(error) => return server_error(format!("读取高清任务失败: {error}")),
    };
    if let Some(row) = previous.first().filter(|row| !matches!(row.get::<_, String>("status").as_str(), "failed" | "cancelled")) {
        return generation_response(&state, &client, &account, &row.get::<_, String>("id")).await;
    }
    if !state.scheduler.models().await.iter().any(|model| model == "SeedVR2-7B") {
        return error_response(StatusCode::SERVICE_UNAVAILABLE, "hd_node_unavailable", "2K 视频节点暂未就绪，请稍后再试");
    }
    let asset_id = source["output"]["asset_id"].as_str().unwrap_or_default();
    let asset = match client.query_opt("SELECT task_id, artifact_id, bytes FROM h3_assets WHERE id = $1 AND account_id = $2 AND kind = 'video' AND content_type = 'video/mp4' AND deleted_at IS NULL", &[&asset_id, &account.id]).await {
        Ok(Some(row)) => row,
        Ok(None) => return error_response(StatusCode::NOT_FOUND, "source_not_found", "原视频文件不存在"),
        Err(error) => return server_error(format!("读取原视频资产失败: {error}")),
    };
    if !(1..=50 * 1024 * 1024).contains(&asset.get::<_, i64>("bytes")) {
        return error_response(StatusCode::PAYLOAD_TOO_LARGE, "source_too_large", "原视频文件不能超过 50 MB");
    }
    let Some(path) = state.scheduler.artifact_path(&asset.get::<_, String>("task_id"), &asset.get::<_, String>("artifact_id")) else {
        return error_response(StatusCode::NOT_FOUND, "source_not_found", "原视频文件不存在");
    };
    let probe = match tokio::time::timeout(
        std::time::Duration::from_secs(30),
        tokio::process::Command::new("ffprobe").args(["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "json"]).arg(&path).kill_on_drop(true).output(),
    )
    .await
    {
        Ok(Ok(output)) if output.status.success() => serde_json::from_slice::<Value>(&output.stdout).ok(),
        _ => None,
    };
    let dimensions = probe.as_ref().and_then(|value| Some((value["streams"][0]["width"].as_u64()?, value["streams"][0]["height"].as_u64()?)));
    let (width, height) = match dimensions.and_then(|(width, height)| hd_dimensions(width, height)) {
        Some(value) => value,
        None => return error_response(StatusCode::BAD_REQUEST, "invalid_source_video", "无法读取原视频尺寸"),
    };
    let bytes = match tokio::fs::read(&path).await {
        Ok(bytes) if !bytes.is_empty() && bytes.len() <= 50 * 1024 * 1024 => bytes,
        _ => return error_response(StatusCode::BAD_REQUEST, "source_unavailable", "原视频文件不可读取或过大"),
    };
    // 同一源任务、同一次重试使用相同 ID；并发点击由任务主键保证不会重复扣费。
    let id = format!("task_hd_{}", &blake3::hash(format!("{task_id}:{}", previous.len()).as_bytes()).to_hex()[..24]);
    let prepared = PreparedGeneration {
        model: "SeedVR2-7B",
        task_kind: "video_super_resolution",
        modality: "video",
        request: json!({"model": "SeedVR2-7B", "input_video_data": format!("data:video/mp4;base64,{}", base64::engine::general_purpose::STANDARD.encode(bytes)), "width": width, "height": height, "frames": duration * 24, "seed": 666}),
        metadata: json!({"width": width, "height": height, "frames": duration * 24, "fps": 24, "duration": duration, "source_task_id": task_id, "resolution": "2K"}),
        price_milli: duration * HD_SECOND_PRICE_MILLI,
        duration_seconds: Some(duration as i32),
        resolution: "2K".to_owned(),
    };
    let title = format!("2K · {}", source["title"].as_str().unwrap_or("视频"));
    let input = GenerationRequest {
        kind: "video".to_owned(),
        prompt: title.clone(),
        references: HashMap::new(),
        output: GenerationOutput { aspect_ratio: String::new(), resolution: "2K".to_owned(), duration_seconds: Some(duration as u64) },
        purpose: Some(purpose),
    };
    if let Err(response) = reserve(&mut client, &account, &id, &title, &input, &prepared).await {
        if owns_task(&client, account.id, &id).await {
            return generation_response(&state, &client, &account, &id).await;
        }
        return response;
    }
    if let Err(error) = state.scheduler.dispatch_task(id.clone(), prepared.model.to_owned(), prepared.task_kind.to_owned(), prepared.request, prepared.metadata).await {
        let _ = client.execute("UPDATE h3_studio_tasks SET status = 'failed' WHERE id = $1", &[&id]).await;
        let _ = settle(&mut client, &id, 0).await;
        return error_response(StatusCode::SERVICE_UNAVAILABLE, "hd_unavailable", error.to_string());
    }
    generation_response(&state, &client, &account, &id).await
}

fn hd_dimensions(width: u64, height: u64) -> Option<(u64, u64)> {
    if width == 0 || height == 0 || width > 16384 || height > 16384 {
        return None;
    }
    // 沿用已确定的 1344×768 → 2688×1536 画布面积；其他比例保持构图并对齐 VAE。
    let scale = (4_128_768.0 / (width * height) as f64).sqrt();
    Some((((width as f64 * scale / 16.0).round() as u64).max(4) * 16, ((height as f64 * scale / 16.0).round() as u64).max(4) * 16))
}

pub(super) async fn upload_asset(State(state): State<ServerState>, headers: HeaderMap, mut multipart: Multipart) -> Response {
    let (client, account) = match account::require_account(&state, &headers).await {
        Ok(value) => value,
        Err(response) => return response,
    };
    let mut bytes = None;
    let mut file_name = None;
    let mut declared_content_type = None;
    let mut role = None;
    while let Ok(Some(field)) = multipart.next_field().await {
        match field.name() {
            Some("file") => {
                file_name = field.file_name().map(safe_file_name);
                declared_content_type = field.content_type().map(ToString::to_string);
                bytes = match field.bytes().await {
                    Ok(value) if value.len() <= MAX_MEDIA_UPLOAD_BYTES => Some(value),
                    Ok(_) => return error_response(StatusCode::PAYLOAD_TOO_LARGE, "asset_too_large", "参考素材不能超过 75 MB"),
                    Err(error) => return error_response(StatusCode::BAD_REQUEST, "invalid_upload", format!("读取参考素材失败: {error}")),
                };
            }
            Some("role") => role = field.text().await.ok(),
            _ => {}
        }
    }
    let Some(bytes) = bytes else {
        return error_response(StatusCode::BAD_REQUEST, "file_required", "请选择图片、视频或音频");
    };
    let (kind, content_type, extension) = match upload_format(declared_content_type.as_deref(), &bytes) {
        Some(value) => value,
        None => return error_response(StatusCode::BAD_REQUEST, "invalid_media", "只支持常用图片、MP4/WebM/MOV 视频及 MP3/WAV/M4A/AAC/OGG/FLAC 音频"),
    };
    if kind == "image" && bytes.len() > MAX_IMAGE_UPLOAD_BYTES {
        return error_response(StatusCode::PAYLOAD_TOO_LARGE, "asset_too_large", "单张图片不能超过 20 MB");
    }
    let role = role.filter(|role| matches!(role.as_str(), "start" | "end" | "person" | "background" | "video" | "audio"));
    let asset_id = format!("asset_{}", state.request_id().trim_start_matches("req_"));
    let directory = state.config.scheduler.artifact_dir.join("studio-assets").join(account.id.to_string());
    if let Err(error) = tokio::fs::create_dir_all(&directory).await {
        return server_error(format!("创建资产目录失败: {error}"));
    }
    let path = directory.join(format!("{asset_id}.{extension}"));
    if let Err(error) = tokio::fs::write(&path, &bytes).await {
        return server_error(format!("保存上传资产失败: {error}"));
    }
    let name = file_name.unwrap_or_else(|| format!("参考素材.{extension}"));
    let path_text = path.to_string_lossy().into_owned();
    let stored = client
        .execute(
            "INSERT INTO h3_assets(id, account_id, kind, role, title, file_name, content_type, bytes, file_path)
             VALUES($1, $2, $3, $4, $5, $5, $6, $7, $8)",
            &[&asset_id, &account.id, &kind, &role, &name, &content_type, &(bytes.len() as i64), &path_text],
        )
        .await;
    if let Err(error) = stored {
        let _ = tokio::fs::remove_file(&path).await;
        return server_error(format!("登记上传资产失败: {error}"));
    }
    Json(json!({"id": asset_id, "kind": kind, "url": format!("/api/assets/{asset_id}/content"), "name": name, "role": role, "content_type": content_type})).into_response()
}

pub(super) async fn list_assets(State(state): State<ServerState>, headers: HeaderMap, Query(query): Query<HashMap<String, String>>) -> Response {
    let (client, account) = match account::require_account(&state, &headers).await {
        Ok(value) => value,
        Err(response) => return response,
    };
    let limit = query.get("limit").and_then(|value| value.parse::<i64>().ok()).unwrap_or(100).clamp(1, 200);
    let rows = match client
        .query(
            "SELECT id, task_id, kind, role, title, file_name, content_type, bytes, duration_seconds, resolution,
                    EXTRACT(EPOCH FROM created_at)::BIGINT AS created_at
             FROM h3_assets WHERE account_id = $1 AND deleted_at IS NULL ORDER BY created_at DESC LIMIT $2",
            &[&account.id, &limit],
        )
        .await
    {
        Ok(rows) => rows,
        Err(error) => return server_error(format!("读取资产失败: {error}")),
    };
    Json(json!({"items": rows.iter().map(asset_value).collect::<Vec<_>>()})).into_response()
}

pub(super) async fn asset_content(State(state): State<ServerState>, headers: HeaderMap, AxumPath(asset_id): AxumPath<String>) -> Response {
    let (client, account) = match account::require_account(&state, &headers).await {
        Ok(value) => value,
        Err(response) => return response,
    };
    let row = match client.query_opt("SELECT task_id, artifact_id, file_path, file_name, content_type FROM h3_assets WHERE id = $1 AND account_id = $2 AND deleted_at IS NULL", &[&asset_id, &account.id]).await {
        Ok(Some(row)) => row,
        Ok(None) => return error_response(StatusCode::NOT_FOUND, "asset_not_found", "资产不存在"),
        Err(error) => return server_error(format!("读取资产失败: {error}")),
    };
    let path = if let Some(path) = row.get::<_, Option<String>>("file_path") {
        PathBuf::from(path)
    } else {
        let task_id = row.get::<_, Option<String>>("task_id").unwrap_or_default();
        let artifact_id = row.get::<_, Option<String>>("artifact_id").unwrap_or_default();
        match state.scheduler.artifact_path(&task_id, &artifact_id) {
            Some(path) => path,
            None => return error_response(StatusCode::NOT_FOUND, "asset_not_found", "资产文件不存在"),
        }
    };
    h3::file_response(&path, row.get("content_type"), row.get("file_name"), &headers).await
}

pub(super) async fn delete_asset(State(state): State<ServerState>, headers: HeaderMap, AxumPath(asset_id): AxumPath<String>) -> Response {
    let (client, account) = match account::require_account(&state, &headers).await {
        Ok(value) => value,
        Err(response) => return response,
    };
    let row = match client.query_opt("UPDATE h3_assets SET deleted_at = NOW() WHERE id = $1 AND account_id = $2 AND deleted_at IS NULL RETURNING task_id, artifact_id, file_path", &[&asset_id, &account.id]).await {
        Ok(Some(row)) => row,
        Ok(None) => return error_response(StatusCode::NOT_FOUND, "asset_not_found", "资产不存在"),
        Err(error) => return server_error(format!("删除资产记录失败: {error}")),
    };
    let path = row.get::<_, Option<String>>("file_path").map(PathBuf::from).or_else(|| {
        let task_id = row.get::<_, Option<String>>("task_id")?;
        let artifact_id = row.get::<_, Option<String>>("artifact_id")?;
        state.scheduler.artifact_path(&task_id, &artifact_id)
    });
    if let Some(path) = path {
        let _ = tokio::fs::remove_file(path).await;
    }
    Json(json!({"id": asset_id, "deleted": true})).into_response()
}

struct PreparedGeneration {
    model: &'static str,
    task_kind: &'static str,
    modality: &'static str,
    request: Value,
    metadata: Value,
    price_milli: i64,
    duration_seconds: Option<i32>,
    resolution: String,
}

async fn prepare_request(state: &ServerState, client: &Client, account: &account::Account, input: &GenerationRequest) -> Result<PreparedGeneration, Response> {
    let prompt = input.prompt.trim();
    if prompt.is_empty() || prompt.chars().count() > 2000 {
        return Err(error_response(StatusCode::BAD_REQUEST, "invalid_prompt", "提示词需要 1–2000 个字符"));
    }
    let references = reference_data_urls(state, client, account.id, &input.references).await?;
    match input.kind.as_str() {
        "video" => {
            let duration = input.output.duration_seconds.filter(|duration| matches!(duration, 5 | 10 | 15)).ok_or_else(|| error_response(StatusCode::BAD_REQUEST, "invalid_duration", "视频时长只能选 5、10 或 15 秒"))?;
            let resolution = input.output.resolution.to_uppercase();
            if resolution == "2K" {
                return Err(error_response(StatusCode::BAD_REQUEST, "hd_is_separate", "请先生成普通视频，完成后再单独转为高清"));
            }
            let resolution = match resolution.as_str() {
                "720P" | "768P" => "768P",
                "1080P" => "1080P",
                _ => return Err(error_response(StatusCode::BAD_REQUEST, "invalid_resolution", "请选择 768P 或 1080P")),
            };
            let mut content = vec![json!({"type": "text", "text": prompt})];
            let has_keyframe = references.contains_key("start") || references.contains_key("end");
            let has_reference = ["person", "background", "video", "audio"].into_iter().any(|role| references.contains_key(role));
            if has_keyframe && has_reference {
                return Err(error_response(StatusCode::BAD_REQUEST, "reference_conflict", "首尾帧与人物、背景、视频或音频参考暂不能同时使用，请保留其中一组"));
            }
            for (role, native_role) in [("start", "first_frame"), ("end", "last_frame"), ("person", "reference_image"), ("background", "reference_image")] {
                if let Some(url) = references.get(role) {
                    content.push(json!({"type": "image_url", "role": native_role, "image_url": {"url": url}}));
                }
            }
            for (role, kind, native_role) in [("video", "video_url", "reference_video"), ("audio", "audio_url", "reference_audio")] {
                if let Some(url) = references.get(role) {
                    let mut item = json!({"type": kind, "role": native_role});
                    item.as_object_mut().expect("参考媒体请求必须是对象").insert(kind.to_owned(), json!({"url": url}));
                    content.push(item);
                }
            }
            let request = json!({"model": "MiniMax-H3", "resolution": resolution, "duration": duration, "ratio": input.output.aspect_ratio, "content": content});
            let metadata = h3::validate_request(&request).map_err(|message| error_response(StatusCode::BAD_REQUEST, "invalid_video", message))?;
            Ok(PreparedGeneration {
                model: "MiniMax-H3",
                task_kind: "video_generation",
                modality: "video",
                request,
                metadata,
                price_milli: i64::try_from(duration).unwrap_or(0) * VIDEO_SECOND_PRICE_MILLI,
                duration_seconds: Some(duration as i32),
                resolution: resolution.to_owned(),
            })
        }
        "image" => {
            let (width, height) = match input.output.aspect_ratio.as_str() {
                "16:9" => (512, 288),
                "9:16" => (288, 512),
                "1:1" => (512, 512),
                _ => return Err(error_response(StatusCode::BAD_REQUEST, "invalid_ratio", "图片画幅仅支持 16:9、9:16 或 1:1")),
            };
            let request = json!({
                "model": "FLUX.2-klein-4B",
                "prompt": prompt,
                "width": width,
                "height": height,
                "input_images": (["start", "end", "person", "background"].into_iter().filter_map(|role| references.get(role)).collect::<Vec<_>>())
            });
            let metadata = image::validate_request(&request).map_err(|message| error_response(StatusCode::BAD_REQUEST, "invalid_image", message))?;
            Ok(PreparedGeneration { model: "FLUX.2-klein-4B", task_kind: "image_generation", modality: "image", request, metadata, price_milli: IMAGE_PRICE_MILLI, duration_seconds: None, resolution: format!("{width}×{height}") })
        }
        _ => Err(error_response(StatusCode::BAD_REQUEST, "invalid_type", "只支持视频或视频参考图生成")),
    }
}

async fn reference_data_urls(state: &ServerState, client: &Client, account_id: i64, references: &HashMap<String, String>) -> Result<HashMap<String, String>, Response> {
    let mut result = HashMap::new();
    for (role, asset_id) in references {
        let expected_kind = match role.as_str() {
            "start" | "end" | "person" | "background" => "image",
            "video" => "video",
            "audio" => "audio",
            _ => return Err(error_response(StatusCode::BAD_REQUEST, "invalid_reference", "参考素材类型不正确")),
        };
        let row = client
            .query_opt("SELECT task_id, artifact_id, file_path, content_type, bytes, kind FROM h3_assets WHERE id = $1 AND account_id = $2", &[asset_id, &account_id])
            .await
            .map_err(|error| server_error(format!("读取参考素材失败: {error}")))?
            .filter(|row| row.get::<_, String>("kind") == expected_kind)
            .ok_or_else(|| error_response(StatusCode::BAD_REQUEST, "reference_not_found", format!("{expected_kind} 参考素材 {asset_id} 不存在")))?;
        let bytes = row.get::<_, i64>("bytes");
        let maximum = if expected_kind == "image" { MAX_IMAGE_UPLOAD_BYTES } else { MAX_MEDIA_UPLOAD_BYTES };
        if bytes < 0 || bytes as usize > maximum {
            return Err(error_response(StatusCode::BAD_REQUEST, "reference_too_large", "参考素材大小超出限制"));
        }
        let path = if let Some(path) = row.get::<_, Option<String>>("file_path") {
            PathBuf::from(path)
        } else {
            let task_id = row.get::<_, Option<String>>("task_id").unwrap_or_default();
            let artifact_id = row.get::<_, Option<String>>("artifact_id").unwrap_or_default();
            state.scheduler.artifact_path(&task_id, &artifact_id).ok_or_else(|| error_response(StatusCode::BAD_REQUEST, "reference_not_found", "参考素材文件不存在"))?
        };
        let bytes = tokio::fs::read(path).await.map_err(|error| server_error(format!("读取参考素材文件失败: {error}")))?;
        let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
        result.insert(role.clone(), format!("data:{};base64,{encoded}", row.get::<_, String>("content_type")));
    }
    Ok(result)
}

async fn reserve(client: &mut Client, account: &account::Account, task_id: &str, title: &str, input: &GenerationRequest, prepared: &PreparedGeneration) -> Result<(), Response> {
    let transaction = client.transaction().await.map_err(|error| server_error(format!("开始扣费事务失败: {error}")))?;
    let updated =
        transaction.execute("UPDATE h3_accounts SET balance_milli = balance_milli - $1 WHERE id = $2 AND balance_milli >= $1", &[&prepared.price_milli, &account.id]).await.map_err(|error| server_error(format!("扣减余额失败: {error}")))?;
    if updated == 0 {
        return Err(error_response(StatusCode::PAYMENT_REQUIRED, "insufficient_balance", format!("余额不足，本次需要 {:.1}", prepared.price_milli as f64 / 1000.0)));
    }
    let request_json = serde_json::to_string(&prepared.request).map_err(|error| server_error(format!("序列化任务失败: {error}")))?;
    transaction
        .execute(
            "INSERT INTO h3_studio_tasks(id, account_id, task_kind, modality, title, purpose, price_milli, duration_seconds, resolution, request_json)
             VALUES($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)",
            &[&task_id, &account.id, &prepared.task_kind, &prepared.modality, &title, &input.purpose, &prepared.price_milli, &prepared.duration_seconds, &prepared.resolution, &request_json],
        )
        .await
        .map_err(|error| server_error(format!("创建任务记录失败: {error}")))?;
    transaction
        .execute("INSERT INTO h3_ledger(account_id, task_id, kind, amount_milli) VALUES($1, $2, 'charge', $3)", &[&account.id, &task_id, &(-prepared.price_milli)])
        .await
        .map_err(|error| server_error(format!("写入扣费账本失败: {error}")))?;
    transaction.commit().await.map_err(|error| server_error(format!("提交扣费事务失败: {error}")))
}

struct Settlement {
    charged_milli: i64,
    progress_milli: i32,
}

async fn settle(client: &mut Client, task_id: &str, progress_milli: i32) -> Result<Settlement, String> {
    let transaction = client.transaction().await.map_err(|error| format!("开始退款事务失败: {error}"))?;
    let row = transaction.query_opt("SELECT account_id, price_milli FROM h3_studio_tasks WHERE id = $1 AND refunded_at IS NULL FOR UPDATE", &[&task_id]).await.map_err(|error| format!("读取退款任务失败: {error}"))?;
    let Some(row) = row else {
        let row = transaction.query_one("SELECT COALESCE(settled_price_milli, 0), COALESCE(settlement_progress_milli, 0) FROM h3_studio_tasks WHERE id = $1", &[&task_id]).await.map_err(|error| format!("读取已结算任务失败: {error}"))?;
        let charged_milli: i64 = row.get(0);
        let progress_milli: i32 = row.get(1);
        transaction.commit().await.map_err(|error| format!("提交空退款事务失败: {error}"))?;
        return Ok(Settlement { charged_milli, progress_milli });
    };
    let account_id: i64 = row.get("account_id");
    let price_milli: i64 = row.get("price_milli");
    let progress_milli = progress_milli.clamp(0, 1000);
    let charged_milli = retained_price_milli(price_milli, progress_milli);
    let refunded_milli = price_milli.saturating_sub(charged_milli);
    transaction.execute("UPDATE h3_accounts SET balance_milli = balance_milli + $1 WHERE id = $2", &[&refunded_milli, &account_id]).await.map_err(|error| format!("退回余额失败: {error}"))?;
    transaction
        .execute("UPDATE h3_studio_tasks SET refunded_at = NOW(), settled_price_milli = $1, settlement_progress_milli = $2, updated_at = NOW() WHERE id = $3", &[&charged_milli, &progress_milli, &task_id])
        .await
        .map_err(|error| format!("标记退款失败: {error}"))?;
    transaction
        .execute("INSERT INTO h3_ledger(account_id, task_id, kind, amount_milli) VALUES($1, $2, 'refund', $3) ON CONFLICT(task_id, kind) DO NOTHING", &[&account_id, &task_id, &refunded_milli])
        .await
        .map_err(|error| format!("写入退款账本失败: {error}"))?;
    transaction.commit().await.map_err(|error| format!("提交退款事务失败: {error}"))?;
    Ok(Settlement { charged_milli, progress_milli })
}

async fn generation_response(state: &ServerState, client: &Client, account: &account::Account, task_id: &str) -> Response {
    match generation_value(state, client, account, task_id).await {
        Ok(value) => Json(value).into_response(),
        Err(response) => response,
    }
}

async fn generation_value(state: &ServerState, client: &Client, account: &account::Account, task_id: &str) -> Result<Value, Response> {
    let row = client
        .query_opt(
            "SELECT id, task_kind, modality, title, purpose, price_milli, duration_seconds, resolution, status,
                    refunded_at IS NOT NULL AS refunded, settled_price_milli, settlement_progress_milli,
                    EXTRACT(EPOCH FROM created_at)::BIGINT AS created_at,
                    EXTRACT(EPOCH FROM updated_at)::BIGINT AS updated_at
             FROM h3_studio_tasks WHERE id = $1 AND account_id = $2",
            &[&task_id, &account.id],
        )
        .await
        .map_err(|error| server_error(format!("读取任务失败: {error}")))?
        .ok_or_else(|| error_response(StatusCode::NOT_FOUND, "task_not_found", "任务不存在"))?;
    let mut status: String = row.get("status");
    let mut progress = None;
    let mut task_error = None;
    let mut queue = None;
    let mut previews = Vec::new();
    let task_kind = row.get::<_, String>("task_kind");
    let price_milli = row.get::<_, i64>("price_milli");
    let mut current_progress_milli = row.get::<_, Option<i32>>("settlement_progress_milli").unwrap_or(0);
    let mut settled_price_milli = row.get::<_, Option<i64>>("settled_price_milli");
    let mut settlement_progress_milli = row.get::<_, Option<i32>>("settlement_progress_milli");
    if let Some(task) = state.scheduler.task(task_id).await {
        status = task.status.clone();
        progress = task.progress.as_ref().map(|value| {
            current_progress_milli = billable_progress_milli(&task_kind, Some(value));
            let percent = current_progress_milli as f64 / 10.0;
            json!({"percent": percent, "stage": phase_label(&value.phase), "detail": phase_detail(&value.phase), "eta_seconds": value.phase_eta_seconds})
        });
        task_error = task.error.clone();
        previews = task
            .previews
            .iter()
            .map(|preview| {
                json!({
                    "index": preview.index,
                    "timestamp_seconds": preview.timestamp_seconds,
                    "url": format!("data:{};base64,{}", preview.content_type, preview.data_base64)
                })
            })
            .collect();
        if status == "queued" {
            queue = state.scheduler.task_queue_info(task_id).await.map(|queue| json!(queue));
        }
        if status == "succeeded" {
            current_progress_milli = 1000;
            settled_price_milli = Some(price_milli);
            settlement_progress_milli = Some(1000);
            client
                .execute("UPDATE h3_studio_tasks SET status = $1, settled_price_milli = price_milli, settlement_progress_milli = 1000, updated_at = NOW() WHERE id = $2", &[&status, &task_id])
                .await
                .map_err(|error| server_error(format!("同步任务状态失败: {error}")))?;
        } else {
            client.execute("UPDATE h3_studio_tasks SET status = $1, updated_at = NOW() WHERE id = $2", &[&status, &task_id]).await.map_err(|error| server_error(format!("同步任务状态失败: {error}")))?;
        }
        if status == "succeeded" {
            sync_outputs(client, account.id, &row, &task).await?;
        }
    }
    if matches!(status.as_str(), "failed" | "cancelled") && !row.get::<_, bool>("refunded") {
        let progress_milli = if status == "cancelled" { current_progress_milli } else { 0 };
        let mut settlement_client = account::configured_client(state).await?;
        let settlement = settle(&mut settlement_client, task_id, progress_milli).await.map_err(server_error)?;
        settled_price_milli = Some(settlement.charged_milli);
        settlement_progress_milli = Some(settlement.progress_milli);
    }
    let output = client
        .query_opt("SELECT id, kind FROM h3_assets WHERE account_id = $1 AND task_id = $2 ORDER BY created_at LIMIT 1", &[&account.id, &task_id])
        .await
        .map_err(|error| server_error(format!("读取任务产物失败: {error}")))?
        .map(|asset| json!({"asset_id": asset.get::<_, String>("id"), "url": format!("/api/assets/{}/content", asset.get::<_, String>("id")), "kind": asset.get::<_, String>("kind")}));
    let charged_milli = settled_price_milli.or_else(|| match status.as_str() {
        "succeeded" => Some(price_milli),
        "failed" | "cancelled" => Some(0),
        _ => None,
    });
    Ok(json!({
        "id": row.get::<_, String>("id"),
        "status": status,
        "type": row.get::<_, String>("modality"),
        "title": row.get::<_, String>("title"),
        "purpose": row.get::<_, Option<String>>("purpose"),
        "price": price_milli as f64 / 1000.0,
        "billing": {
            "reserved": price_milli as f64 / 1000.0,
            "charged": charged_milli.map(|value| value as f64 / 1000.0),
            "refunded": charged_milli.map(|value| price_milli.saturating_sub(value) as f64 / 1000.0),
            "progress_percent": settlement_progress_milli.unwrap_or(current_progress_milli) as f64 / 10.0
        },
        "duration_seconds": row.get::<_, Option<i32>>("duration_seconds"),
        "resolution": row.get::<_, Option<String>>("resolution"),
        "created_at": row.get::<_, i64>("created_at"),
        "updated_at": row.get::<_, i64>("updated_at"),
        "progress": progress,
        "queue": queue,
        "previews": previews,
        "error": task_error.map(|message| json!({"message": message})),
        "output": output,
        "hd": {
            "price_per_second": HD_SECOND_PRICE_MILLI as f64 / 1000.0,
            "estimated_seconds": row.get::<_, Option<i32>>("duration_seconds").map(i64::from).unwrap_or(0) * HD_ESTIMATED_SECONDS_PER_VIDEO_SECOND,
            "available": status == "succeeded" && row.get::<_, String>("modality") == "video" && row.get::<_, Option<String>>("resolution").as_deref() != Some("2K") && state.scheduler.models().await.iter().any(|model| model == "SeedVR2-7B")
        }
    }))
}

async fn sync_outputs(client: &Client, account_id: i64, row: &Row, task: &super::scheduler::TaskView) -> Result<(), Response> {
    for output in &task.outputs {
        let asset_id = format!("asset_{}", &blake3::hash(format!("{}:{}", task.id, output.id).as_bytes()).to_hex()[..24]);
        let kind = if output.content_type.starts_with("video/") { "video" } else { "image" };
        client
            .execute(
                "INSERT INTO h3_assets(id, account_id, task_id, artifact_id, kind, role, title, file_name, content_type, bytes, duration_seconds, resolution)
                 VALUES($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                 ON CONFLICT(task_id, artifact_id) DO NOTHING",
                &[
                    &asset_id,
                    &account_id,
                    &task.id,
                    &output.id,
                    &kind,
                    &row.get::<_, Option<String>>("purpose"),
                    &row.get::<_, String>("title"),
                    &output.file_name,
                    &output.content_type,
                    &(output.bytes as i64),
                    &row.get::<_, Option<i32>>("duration_seconds"),
                    &row.get::<_, Option<String>>("resolution"),
                ],
            )
            .await
            .map_err(|error| server_error(format!("登记生成资产失败: {error}")))?;
    }
    Ok(())
}

async fn owns_task(client: &Client, account_id: i64, task_id: &str) -> bool {
    client.query_opt("SELECT 1 FROM h3_studio_tasks WHERE id = $1 AND account_id = $2", &[&task_id, &account_id]).await.ok().flatten().is_some()
}

fn asset_value(row: &Row) -> Value {
    let id = row.get::<_, String>("id");
    json!({
        "id": id,
        "kind": row.get::<_, String>("kind"),
        "task_id": row.get::<_, Option<String>>("task_id"),
        "role": row.get::<_, Option<String>>("role"),
        "title": row.get::<_, String>("title"),
        "name": row.get::<_, String>("file_name"),
        "content_type": row.get::<_, String>("content_type"),
        "bytes": row.get::<_, i64>("bytes"),
        "duration": row.get::<_, Option<i32>>("duration_seconds"),
        "resolution": row.get::<_, Option<String>>("resolution"),
        "created_at": row.get::<_, i64>("created_at"),
        "url": format!("/api/assets/{id}/content")
    })
}

fn title(prompt: &str) -> String {
    let value = prompt.trim().chars().take(28).collect::<String>();
    if value.is_empty() { "未命名创作".to_owned() } else { value }
}

fn safe_file_name(value: &str) -> String {
    let name = value.rsplit(['/', '\\']).next().unwrap_or("参考图");
    name.chars().filter(|character| !character.is_control()).take(120).collect::<String>()
}

fn upload_format(declared: Option<&str>, bytes: &[u8]) -> Option<(&'static str, &'static str, &'static str)> {
    match ::image::guess_format(bytes).ok() {
        Some(::image::ImageFormat::Jpeg) => return Some(("image", "image/jpeg", "jpg")),
        Some(::image::ImageFormat::Png) => return Some(("image", "image/png", "png")),
        Some(::image::ImageFormat::WebP) => return Some(("image", "image/webp", "webp")),
        _ => {}
    }
    match declared?.split(';').next()?.trim().to_ascii_lowercase().as_str() {
        "video/mp4" => Some(("video", "video/mp4", "mp4")),
        "video/webm" => Some(("video", "video/webm", "webm")),
        "video/quicktime" => Some(("video", "video/quicktime", "mov")),
        "audio/mpeg" => Some(("audio", "audio/mpeg", "mp3")),
        "audio/wav" | "audio/x-wav" => Some(("audio", "audio/wav", "wav")),
        "audio/mp4" | "audio/x-m4a" => Some(("audio", "audio/mp4", "m4a")),
        "audio/aac" => Some(("audio", "audio/aac", "aac")),
        "audio/ogg" => Some(("audio", "audio/ogg", "ogg")),
        "audio/flac" => Some(("audio", "audio/flac", "flac")),
        _ => None,
    }
}

fn retained_price_milli(price_milli: i64, progress_milli: i32) -> i64 {
    price_milli.saturating_mul(i64::from(progress_milli.clamp(0, 1000))).saturating_add(999) / 1000
}

fn billable_progress_milli(task_kind: &str, progress: Option<&super::scheduler::TaskProgress>) -> i32 {
    let Some(progress) = progress else { return 0 };
    if progress.phase == "model_loading" {
        return 0;
    }
    let phase = |start: i32, end: i32| {
        if progress.total == 0 {
            return start;
        }
        let completed = progress.completed.min(progress.total) as u128;
        let span = (end - start) as u128;
        start + (span * completed / progress.total as u128) as i32
    };
    match task_kind {
        "video_super_resolution" => phase(0, 1000),
        "video_generation" => match progress.phase.as_str() {
            "conditioning" => phase(10, 100),
            "denoise" => phase(100, 700),
            "video_vae" => phase(700, 900),
            "previews" => phase(900, 940),
            "audio_vae" => phase(940, 980),
            "muxing" => phase(980, 995),
            "completed" => 1000,
            _ => 10,
        },
        "image_generation" => match progress.phase.as_str() {
            "generation" => phase(10, 1000),
            _ => 10,
        },
        _ => phase(10, 1000),
    }
}

fn phase_label(phase: &str) -> &'static str {
    match phase {
        "model_loading" => "正在准备生成",
        "vae_encode" => "正在读取视频画面",
        "vae_decode" => "正在生成 2K 视频",
        "denoise" => "正在生成画面",
        "video_vae" => "正在还原视频",
        "audio_vae" => "正在生成声音",
        "muxing" => "正在合成视频",
        "generation" => "正在生成参考图",
        "previews" => "正在生成预览",
        "completed" => "生成完成",
        _ => "正在处理",
    }
}

fn phase_detail(phase: &str) -> &'static str {
    match phase {
        "denoise" => "构建镜头运动与画面细节",
        "video_vae" => "将生成结果转换为连续画面",
        "audio_vae" => "生成并同步音轨",
        "muxing" => "写入可播放的视频文件",
        "generation" => "正在快速生成图片",
        "previews" => "真实视频帧正在逐张返回",
        _ => "任务正在节点上执行",
    }
}

fn error_response(status: StatusCode, code: &'static str, message: impl Into<String>) -> Response {
    (status, Json(json!({"error": code, "message": message.into()}))).into_response()
}

fn server_error(message: impl Into<String>) -> Response {
    eprintln!("{}", message.into());
    error_response(StatusCode::INTERNAL_SERVER_ERROR, "studio_server_error", "服务暂时不可用")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 价格使用整数千分单位() {
        assert_eq!(IMAGE_PRICE_MILLI, 100);
        assert_eq!(15 * VIDEO_SECOND_PRICE_MILLI, 3000);
        assert_eq!(5 * HD_SECOND_PRICE_MILLI, 2500);
        assert_eq!(10 * HD_SECOND_PRICE_MILLI, 5000);
        assert_eq!(15 * HD_SECOND_PRICE_MILLI, 7500);
    }

    #[test]
    fn 高清画布保持横竖比例并拒绝非法尺寸() {
        assert_eq!(hd_dimensions(1344, 768), Some((2688, 1536)));
        assert_eq!(hd_dimensions(768, 1344), Some((1536, 2688)));
        assert_eq!(hd_dimensions(0, 768), None);
        assert_eq!(hd_dimensions(u64::MAX, 768), None);
    }

    #[test]
    fn 文件名不允许路径穿越() {
        assert_eq!(safe_file_name("../../portrait.png"), "portrait.png");
    }

    #[test]
    fn 上传格式区分图片视频和音频() {
        assert_eq!(upload_format(Some("video/mp4"), b"not decoded here").map(|value| value.0), Some("video"));
        assert_eq!(upload_format(Some("audio/mpeg"), b"not decoded here").map(|value| value.0), Some("audio"));
        assert_eq!(upload_format(Some("application/octet-stream"), b"unknown"), None);
    }

    fn progress(phase: &str, completed: usize, total: usize) -> super::super::scheduler::TaskProgress {
        super::super::scheduler::TaskProgress { phase: phase.to_owned(), completed, total, elapsed_seconds: 0.0, phase_eta_seconds: None, preview: None }
    }

    #[test]
    fn 视频计费进度跨阶段单调递增() {
        let values = [
            billable_progress_milli("video_generation", None),
            billable_progress_milli("video_generation", Some(&progress("conditioning", 0, 1))),
            billable_progress_milli("video_generation", Some(&progress("conditioning", 1, 1))),
            billable_progress_milli("video_generation", Some(&progress("denoise", 2, 4))),
            billable_progress_milli("video_generation", Some(&progress("video_vae", 0, 1))),
            billable_progress_milli("video_generation", Some(&progress("video_vae", 1, 1))),
            billable_progress_milli("video_generation", Some(&progress("previews", 8, 8))),
            billable_progress_milli("video_generation", Some(&progress("audio_vae", 1, 1))),
            billable_progress_milli("video_generation", Some(&progress("muxing", 0, 1))),
            billable_progress_milli("video_generation", Some(&progress("completed", 1, 1))),
        ];
        assert_eq!(values, [0, 10, 100, 400, 700, 900, 940, 980, 980, 1000]);
        assert!(values.windows(2).all(|pair| pair[0] <= pair[1]));
    }

    #[test]
    fn 取消结算向上取最小计费单位() {
        assert_eq!(retained_price_milli(3000, 0), 0);
        assert_eq!(retained_price_milli(3000, 400), 1200);
        assert_eq!(retained_price_milli(100, 10), 1);
        assert_eq!(retained_price_milli(3000, 1000), 3000);
    }
}
