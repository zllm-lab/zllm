//! 视频超分任务入口，复用 scheduler/node 的任务与产物生命周期。

use super::{ServerState, authorize};
use crate::runtime::seedvr2::SeedVr2Request;
use axum::{
    Json,
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use serde_json::{Value, json};

pub(super) async fn create(State(state): State<ServerState>, mut headers: HeaderMap, Json(request): Json<Value>) -> Response {
    let id = state.request_id();
    if let Err(e) = authorize(&state.config, &mut headers) {
        return e.into_response(&id);
    }
    let parsed = serde_json::from_value::<SeedVr2Request>(request.clone()).map_err(|e| e.to_string()).and_then(|r| r.validate().map(|_| r));
    let parsed = match parsed {
        Ok(r) => r,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({"error":e}))).into_response(),
    };
    let metadata = json!({"width":parsed.width,"height":parsed.height,"frames":parsed.frames,"fps":24,"seed":parsed.seed});
    match state.scheduler.dispatch_task(id.clone(), parsed.model, "video_super_resolution".to_owned(), request, metadata).await {
        Ok(_) => Json(json!({"task_id":id})).into_response(),
        Err(e) => (StatusCode::SERVICE_UNAVAILABLE, Json(json!({"error":e.to_string()}))).into_response(),
    }
}

pub(super) async fn query(State(state): State<ServerState>, mut headers: HeaderMap, Path(id): Path<String>) -> Response {
    let request_id = state.request_id();
    if let Err(e) = authorize(&state.config, &mut headers) {
        return e.into_response(&request_id);
    }
    match state.scheduler.task(&id).await.filter(|t| t.task_kind == "video_super_resolution") {
        Some(task) => Json(json!({"task":task})).into_response(),
        None => (StatusCode::NOT_FOUND, Json(json!({"error":"超分任务不存在"}))).into_response(),
    }
}

pub(super) async fn cancel(State(state): State<ServerState>, mut headers: HeaderMap, Path(id): Path<String>) -> Response {
    let request_id = state.request_id();
    if let Err(e) = authorize(&state.config, &mut headers) {
        return e.into_response(&request_id);
    }
    if state.scheduler.task(&id).await.is_none_or(|t| t.task_kind != "video_super_resolution") {
        return (StatusCode::NOT_FOUND, Json(json!({"error":"超分任务不存在"}))).into_response();
    }
    match state.scheduler.cancel_task(&id).await {
        Some(task) => Json(json!({"task":task})).into_response(),
        None => (StatusCode::NOT_FOUND, Json(json!({"error":"超分任务不存在"}))).into_response(),
    }
}
