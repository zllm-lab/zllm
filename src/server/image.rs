//! 图片生成任务 HTTP 入口；执行与产物生命周期复用通用 scheduler/node 协议。

use std::collections::HashMap;

use axum::{
    Json,
    extract::{Path as AxumPath, Query, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use serde_json::{Value, json};

use super::{ServerState, authorize, scheduler::TaskView};

pub(super) async fn create_image_generation(State(state): State<ServerState>, mut headers: HeaderMap, Json(request): Json<Value>) -> Response {
    let request_id = task_id(&state);
    if let Err(error) = authorize(&state.config, &mut headers) {
        return error.into_response(&request_id);
    }
    let metadata = match validate_request(&request) {
        Ok(metadata) => metadata,
        Err(message) => return image_error(StatusCode::BAD_REQUEST, message),
    };
    match state.scheduler.dispatch_task(request_id.clone(), "FLUX.2-klein-4B".to_owned(), "image_generation".to_owned(), request, metadata).await {
        Ok(_) => Json(json!({"task_id": request_id})).into_response(),
        Err(error) => image_error(StatusCode::SERVICE_UNAVAILABLE, error.to_string()),
    }
}

pub(super) async fn query_image_generation(State(state): State<ServerState>, mut headers: HeaderMap, AxumPath(task_id): AxumPath<String>) -> Response {
    let request_id = state.request_id();
    if let Err(error) = authorize(&state.config, &mut headers) {
        return error.into_response(&request_id);
    }
    match state.scheduler.task(&task_id).await.filter(|task| task.task_kind == "image_generation") {
        Some(task) => Json(json!({"task": image_task(task)})).into_response(),
        None => image_error(StatusCode::NOT_FOUND, format!("任务 {task_id} 不存在")),
    }
}

pub(super) async fn list_image_generation(State(state): State<ServerState>, mut headers: HeaderMap, Query(query): Query<HashMap<String, String>>) -> Response {
    let request_id = state.request_id();
    if let Err(error) = authorize(&state.config, &mut headers) {
        return error.into_response(&request_id);
    }
    let mut tasks = state.scheduler.tasks(Some("image_generation")).await;
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
    let items = tasks.into_iter().skip((page - 1).saturating_mul(page_size)).take(page_size).map(image_task).collect::<Vec<_>>();
    Json(json!({"items": items, "total": total})).into_response()
}

pub(super) async fn delete_image_generation(State(state): State<ServerState>, mut headers: HeaderMap, AxumPath(task_id): AxumPath<String>) -> Response {
    let request_id = state.request_id();
    if let Err(error) = authorize(&state.config, &mut headers) {
        return error.into_response(&request_id);
    }
    let Some(task) = state.scheduler.task(&task_id).await.filter(|task| task.task_kind == "image_generation") else {
        return image_error(StatusCode::NOT_FOUND, format!("任务 {task_id} 不存在"));
    };
    if matches!(task.status.as_str(), "queued" | "running" | "uploading") {
        state.scheduler.cancel_task(&task_id).await;
    }
    state.scheduler.remove_task(&task_id).await;
    Json(json!({"task_id": task_id, "action": "deleted", "status": "deleted"})).into_response()
}

pub(super) fn validate_request(request: &Value) -> Result<Value, String> {
    let object = request.as_object().ok_or("请求必须是 JSON 对象")?;
    for field in object.keys() {
        if !matches!(field.as_str(), "model" | "prompt" | "width" | "height" | "steps" | "seed" | "input_image" | "input_images") {
            return Err(format!("不支持字段 {field}"));
        }
    }
    if object.get("model").and_then(Value::as_str) != Some("FLUX.2-klein-4B") {
        return Err("model 必须为 FLUX.2-klein-4B".to_owned());
    }
    let prompt = object.get("prompt").and_then(Value::as_str).filter(|value| !value.trim().is_empty()).ok_or("prompt 不能为空")?;
    if prompt.chars().count() > 7000 {
        return Err("prompt 超过 7000 字符".to_owned());
    }
    let width = dimension(object, "width")?;
    let height = dimension(object, "height")?;
    let steps = object.get("steps").map(|value| value.as_u64().filter(|value| (1..=50).contains(value)).ok_or("steps 必须是 1..=50 的整数")).transpose()?.unwrap_or(4);
    if object.get("seed").is_some_and(|value| value.as_u64().is_none()) {
        return Err("seed 必须是非负整数".to_owned());
    }
    let input_image = object.get("input_image").map(|value| value.as_str().filter(|url| !url.trim().is_empty()).ok_or("input_image 不能为空")).transpose()?;
    let input_images = object
        .get("input_images")
        .map(|value| value.as_array().filter(|images| images.len() <= 4).ok_or("input_images 必须是最多 4 项的数组"))
        .transpose()?
        .map(|images| images.iter().enumerate().map(|(index, value)| value.as_str().filter(|url| !url.trim().is_empty()).ok_or_else(|| format!("input_images.{index} 不能为空"))).collect::<Result<Vec<_>, _>>())
        .transpose()?
        .unwrap_or_default();
    let input_image_count = input_images.len() + usize::from(input_image.is_some());
    if input_image_count > 4 {
        return Err("input_image 与 input_images 合计最多 4 张".to_owned());
    }
    let mode = if input_image_count > 0 { "image_to_image" } else { "text_to_image" };
    Ok(json!({"width": width, "height": height, "steps": steps, "input_image_count": input_image_count, "mode": mode}))
}

fn dimension(object: &serde_json::Map<String, Value>, field: &str) -> Result<u64, String> {
    let value = object.get(field).map(|value| value.as_u64().ok_or_else(|| format!("{field} 必须是整数"))).transpose()?.unwrap_or(512);
    if value == 0 || value > 2048 || value % 16 != 0 {
        return Err(format!("{field} 必须是 16 的倍数且位于 16..=2048"));
    }
    Ok(value)
}

fn image_task(task: TaskView) -> Value {
    let metadata = task.metadata.as_object();
    let mut value = json!({
        "id": task.id,
        "model": task.model,
        "status": task.status,
        "created_at": task.created_at,
        "updated_at": task.updated_at,
        "width": metadata.and_then(|value| value.get("width")).cloned().unwrap_or(Value::Null),
        "height": metadata.and_then(|value| value.get("height")).cloned().unwrap_or(Value::Null),
        "steps": metadata.and_then(|value| value.get("steps")).cloned().unwrap_or(Value::Null),
        "input_image_count": metadata.and_then(|value| value.get("input_image_count")).cloned().unwrap_or(Value::Null),
        "mode": metadata.and_then(|value| value.get("mode")).cloned().unwrap_or(Value::Null),
        "task_type": "generation",
        "modality": "image",
    });
    let object = value.as_object_mut().expect("image task 必须是对象");
    if let Some(progress) = task.progress {
        object.insert("progress".to_owned(), json!(progress));
    }
    if let Some(error) = task.error {
        object.insert("error".to_owned(), json!({"code": "node_error", "message": error}));
    }
    if let Some(image) = task.outputs.iter().find(|output| output.content_type.starts_with("image/")) {
        object.insert("content".to_owned(), json!({"url": image.url}));
    }
    value
}

fn task_id(state: &ServerState) -> String {
    let id = state.request_id();
    format!("task_{}", id.trim_start_matches("req_"))
}

fn image_error(status: StatusCode, message: impl Into<String>) -> Response {
    (status, Json(json!({"error": {"code": status.as_u16().to_string(), "message": message.into()}}))).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_defaults_match_distilled_model() {
        let metadata = validate_request(&json!({"model": "FLUX.2-klein-4B", "prompt": "fox"})).unwrap();
        assert_eq!(metadata, json!({"width": 512, "height": 512, "steps": 4, "input_image_count": 0, "mode": "text_to_image"}));
    }

    #[test]
    fn request_rejects_unaligned_dimensions_and_unknown_fields() {
        assert!(validate_request(&json!({"model": "FLUX.2-klein-4B", "prompt": "fox", "width": 513})).is_err());
        assert!(validate_request(&json!({"model": "FLUX.2-klein-4B", "prompt": "fox", "guidance": 1.0})).is_err());
        assert!(validate_request(&json!({"model": "FLUX.2-klein-4B", "prompt": "fox", "input_images": ["1", "2", "3", "4", "5"]})).is_err());
        assert!(validate_request(&json!({"model": "FLUX.2-klein-4B", "prompt": "fox", "input_image": "1", "input_images": ["2", "3", "4", "5"]})).is_err());
        assert_eq!(validate_request(&json!({"model": "FLUX.2-klein-4B", "prompt": "fox", "input_image": "1"})).unwrap()["mode"], "image_to_image");
    }
}
