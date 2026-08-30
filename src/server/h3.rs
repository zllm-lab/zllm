use std::{collections::HashMap, io::SeekFrom};

use axum::{
    Json,
    body::Body,
    extract::{Path as AxumPath, Query, State},
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncSeekExt};
use tokio_util::io::ReaderStream;

use super::scheduler::TaskView;
use super::{ServerState, authorize};

pub(super) async fn create_video_generation(State(state): State<ServerState>, mut headers: HeaderMap, Json(request): Json<Value>) -> Response {
    let request_id = h3_request_id(&state, "task");
    if let Err(error) = authorize(&state.config, &mut headers) {
        return error.into_response(&request_id);
    }
    let metadata = match validate_request(&request) {
        Ok(metadata) => metadata,
        Err(message) => return h3_error(StatusCode::BAD_REQUEST, message),
    };
    match state.scheduler.dispatch_task(request_id.clone(), "MiniMax-H3".to_owned(), "video_generation".to_owned(), request, metadata).await {
        Ok(_) => Json(json!({"task_id": request_id})).into_response(),
        Err(error) => h3_error(StatusCode::SERVICE_UNAVAILABLE, error.to_string()),
    }
}

pub(super) async fn query_video_generation(State(state): State<ServerState>, mut headers: HeaderMap, AxumPath(task_id): AxumPath<String>) -> Response {
    let request_id = h3_request_id(&state, "req");
    if let Err(error) = authorize(&state.config, &mut headers) {
        return error.into_response(&request_id);
    }
    match state.scheduler.task(&task_id).await.filter(|task| task.task_kind == "video_generation") {
        Some(task) => Json(json!({"task": h3_task(task)})).into_response(),
        None => h3_error(StatusCode::NOT_FOUND, format!("任务 {task_id} 不存在")),
    }
}

pub(super) async fn list_video_generation(State(state): State<ServerState>, mut headers: HeaderMap, Query(query): Query<HashMap<String, String>>) -> Response {
    let request_id = h3_request_id(&state, "req");
    if let Err(error) = authorize(&state.config, &mut headers) {
        return error.into_response(&request_id);
    }
    let mut tasks = state.scheduler.tasks(Some("video_generation")).await;
    if let Some(status) = query.get("filter.status") {
        tasks.retain(|task| task.status == status.as_str());
    }
    if let Some(model) = query.get("filter.model") {
        tasks.retain(|task| task.model == model.as_str());
    }
    tasks.sort_by_key(|task| std::cmp::Reverse(task.created_at));
    let total = tasks.len();
    let page = query.get("page_num").and_then(|value| value.parse::<usize>().ok()).unwrap_or(1).max(1);
    let page_size = query.get("page_size").and_then(|value| value.parse::<usize>().ok()).unwrap_or(20).clamp(1, 100);
    let items = tasks.into_iter().skip((page - 1).saturating_mul(page_size)).take(page_size).map(h3_task).collect::<Vec<_>>();
    Json(json!({"items": items, "total": total})).into_response()
}

pub(super) async fn delete_video_generation(State(state): State<ServerState>, mut headers: HeaderMap, AxumPath(task_id): AxumPath<String>) -> Response {
    let request_id = h3_request_id(&state, "req");
    if let Err(error) = authorize(&state.config, &mut headers) {
        return error.into_response(&request_id);
    }
    let Some(task) = state.scheduler.task(&task_id).await else {
        return h3_error(StatusCode::NOT_FOUND, format!("任务 {task_id} 不存在"));
    };
    if matches!(task.status.as_str(), "queued" | "running" | "uploading") {
        state.scheduler.cancel_task(&task_id).await;
    }
    state.scheduler.remove_task(&task_id).await;
    Json(json!({"task_id": task_id, "action": "deleted", "status": "deleted"})).into_response()
}

pub(super) async fn download_artifact(State(state): State<ServerState>, mut headers: HeaderMap, AxumPath((task_id, artifact_id)): AxumPath<(String, String)>) -> Response {
    let request_id = h3_request_id(&state, "req");
    if let Err(error) = authorize(&state.config, &mut headers) {
        return error.into_response(&request_id);
    }
    let Some(task) = state.scheduler.task(&task_id).await else {
        return h3_error(StatusCode::NOT_FOUND, "任务不存在");
    };
    let Some(artifact) = task.outputs.iter().find(|artifact| artifact.id == artifact_id) else {
        return h3_error(StatusCode::NOT_FOUND, "产物不存在");
    };
    let Some(path) = state.scheduler.artifact_path(&task_id, &artifact_id) else {
        return h3_error(StatusCode::BAD_REQUEST, "产物 ID 非法");
    };
    let metadata = match tokio::fs::metadata(&path).await {
        Ok(metadata) if metadata.is_file() => metadata,
        _ => return h3_error(StatusCode::NOT_FOUND, "产物文件不存在"),
    };
    let total = metadata.len();
    let (status, start, end) = match headers.get(header::RANGE).and_then(|value| value.to_str().ok()) {
        Some(range) => match byte_range(range, total) {
            Some((start, end)) => (StatusCode::PARTIAL_CONTENT, start, end),
            None => {
                return Response::builder().status(StatusCode::RANGE_NOT_SATISFIABLE).header(header::CONTENT_RANGE, format!("bytes */{total}")).body(Body::empty()).expect("416 response 构建失败");
            }
        },
        None => (StatusCode::OK, 0, total.saturating_sub(1)),
    };
    let mut file = match tokio::fs::File::open(&path).await {
        Ok(file) => file,
        Err(error) => return h3_error(StatusCode::INTERNAL_SERVER_ERROR, format!("打开产物失败: {error}")),
    };
    if let Err(error) = file.seek(SeekFrom::Start(start)).await {
        return h3_error(StatusCode::INTERNAL_SERVER_ERROR, format!("定位产物失败: {error}"));
    }
    let length = end.saturating_sub(start).saturating_add(1);
    let stream = ReaderStream::new(file.take(length));
    let mut builder = Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, artifact.content_type.as_str())
        .header(header::CONTENT_LENGTH, length.to_string())
        .header(header::ACCEPT_RANGES, "bytes")
        .header(header::CONTENT_DISPOSITION, format!("inline; filename=\"{}\"", artifact.file_name.chars().map(|character| if matches!(character, '"' | '\r' | '\n') { '_' } else { character }).collect::<String>()));
    if status == StatusCode::PARTIAL_CONTENT {
        builder = builder.header(header::CONTENT_RANGE, format!("bytes {start}-{end}/{total}"));
    }
    builder.body(Body::from_stream(stream)).expect("artifact response 构建失败")
}

fn validate_request(request: &Value) -> Result<Value, String> {
    let object = request.as_object().ok_or("请求必须是 JSON 对象")?;
    if object.get("model").and_then(Value::as_str) != Some("MiniMax-H3") {
        return Err("model 必须为 MiniMax-H3".to_owned());
    }
    if object.get("callback_url").is_some() {
        return Err("当前本地 H3 服务尚未支持 callback_url".to_owned());
    }
    let resolution = object.get("resolution").and_then(Value::as_str).filter(|value| matches!(*value, "512P" | "768P" | "1080P" | "2K")).ok_or("resolution 必须为 512P、768P、1080P 或 2K")?;
    let duration = object.get("duration").and_then(Value::as_u64).filter(|value| (4..=15).contains(value)).ok_or("duration 必须是 4 到 15 的整数")?;
    let ratio = object.get("ratio").and_then(Value::as_str).unwrap_or("adaptive");
    if !matches!(ratio, "adaptive" | "21:9" | "16:9" | "4:3" | "1:1" | "3:4" | "9:16") {
        return Err("ratio 无效".to_owned());
    }
    let content = object.get("content").and_then(Value::as_array).filter(|items| !items.is_empty()).ok_or("content 必须是非空数组")?;
    let mut text_count = 0usize;
    let mut image_count = 0usize;
    let mut reference_count = 0usize;
    let mut keyframe_count = 0usize;
    for (index, item) in content.iter().enumerate() {
        let kind = item.get("type").and_then(Value::as_str).ok_or_else(|| format!("content.{index}.type 缺失"))?;
        let role = item.get("role").and_then(Value::as_str);
        match kind {
            "text" => {
                let text = item.get("text").and_then(Value::as_str).filter(|text| !text.trim().is_empty()).ok_or_else(|| format!("content.{index}.text 不能为空"))?;
                if text.chars().count() > 7000 {
                    return Err(format!("content.{index}.text 超过 7000 字符"));
                }
                text_count += 1;
            }
            "image_url" => {
                require_url(item, "image_url", index)?;
                image_count += 1;
                match role {
                    None | Some("first_frame") | Some("last_frame") => keyframe_count += 1,
                    Some("reference_image") => reference_count += 1,
                    _ => return Err(format!("content.{index}.role 与 image_url 不兼容")),
                }
            }
            "video_url" => {
                require_url(item, "video_url", index)?;
                if role != Some("reference_video") {
                    return Err(format!("content.{index}.role 必须是 reference_video"));
                }
                reference_count += 1;
            }
            "audio_url" => {
                require_url(item, "audio_url", index)?;
                if role != Some("reference_audio") {
                    return Err(format!("content.{index}.role 必须是 reference_audio"));
                }
                reference_count += 1;
            }
            _ => return Err(format!("content.{index}.type 不支持 {kind}")),
        }
    }
    if text_count == 0 {
        return Err("content 必须包含至少一个非空 text".to_owned());
    }
    if keyframe_count > 0 && reference_count > 0 {
        return Err("first/last frame 与 reference 输入不能混用".to_owned());
    }
    if keyframe_count > 2 || image_count > 9 {
        return Err("首尾帧最多 2 张，参考图片最多 9 张".to_owned());
    }
    if content.iter().all(|item| item.get("type").and_then(Value::as_str) == Some("text")) && ratio == "adaptive" {
        return Err("纯文本生成必须指定非 adaptive 的 ratio".to_owned());
    }
    Ok(json!({
        "resolution": resolution,
        "duration": duration,
        "ratio": ratio,
        "input_image_count": image_count,
    }))
}

fn require_url(item: &Value, field: &str, index: usize) -> Result<(), String> {
    item.get(field).and_then(Value::as_object).and_then(|value| value.get("url")).and_then(Value::as_str).filter(|url| !url.trim().is_empty()).map(|_| ()).ok_or_else(|| format!("content.{index}.{field}.url 不能为空"))
}

fn h3_task(task: TaskView) -> Value {
    let metadata = task.metadata.as_object();
    let duration = metadata.and_then(|value| value.get("duration")).and_then(Value::as_u64).unwrap_or(0);
    let image_count = metadata.and_then(|value| value.get("input_image_count")).and_then(Value::as_u64).unwrap_or(0);
    let succeeded = task.status == "succeeded";
    let mut value = json!({
        "id": task.id,
        "model": task.model,
        "status": task.status.clone(),
        "created_at": task.created_at,
        "updated_at": task.updated_at,
        "resolution": metadata.and_then(|value| value.get("resolution")).cloned().unwrap_or(Value::Null),
        "duration": duration,
        "ratio": metadata.and_then(|value| value.get("ratio")).cloned().unwrap_or(Value::Null),
        "usage": {
            "total_seconds": if succeeded { duration } else { 0 },
            "input_seconds": 0,
            "output_seconds": if succeeded { duration } else { 0 },
            "input_image_count": image_count,
        },
        "task_type": "generation",
        "modality": "video",
    });
    let object = value.as_object_mut().expect("H3 task 必须是对象");
    if let Some(progress) = task.progress {
        object.insert("progress".to_owned(), json!(progress));
    }
    if let Some(error) = task.error {
        object.insert("error".to_owned(), json!({"code": "node_error", "message": error}));
    }
    if let Some(video) = task.outputs.iter().find(|output| output.content_type.starts_with("video/")) {
        object.insert("content".to_owned(), json!({"url": video.url}));
    }
    value
}

fn byte_range(value: &str, total: u64) -> Option<(u64, u64)> {
    let value = value.strip_prefix("bytes=")?;
    if value.contains(',') || total == 0 {
        return None;
    }
    let (start, end) = value.split_once('-')?;
    if start.is_empty() {
        let suffix = end.parse::<u64>().ok()?.min(total);
        return (suffix > 0).then_some((total - suffix, total - 1));
    }
    let start = start.parse::<u64>().ok()?;
    let end = if end.is_empty() { total - 1 } else { end.parse::<u64>().ok()?.min(total - 1) };
    (start <= end && start < total).then_some((start, end))
}

fn h3_error(status: StatusCode, message: impl Into<String>) -> Response {
    (status, Json(json!({"error": {"code": status.as_u16().to_string(), "message": message.into()}}))).into_response()
}

fn h3_request_id(state: &ServerState, prefix: &str) -> String {
    let sequence = state.request_sequence.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    format!("{prefix}_{:016x}", sequence)
}
