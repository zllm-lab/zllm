//! H3 网站账户：邮箱密码、PostgreSQL 会话和浏览器 Cookie。

use std::{
    collections::{HashMap, VecDeque},
    sync::{Mutex, OnceLock},
    time::{Duration, Instant},
};

use axum::{
    Json,
    extract::State,
    http::{HeaderMap, HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
};
use serde::Deserialize;
use serde_json::json;
use tokio_postgres::{Client, NoTls, Row};

use super::{ServerConfig, ServerState};

const SESSION_SECONDS: i64 = 30 * 24 * 60 * 60;
const LOGIN_WINDOW: Duration = Duration::from_secs(15 * 60);
const LOGIN_LIMIT: usize = 10;
static LOGIN_ATTEMPTS: OnceLock<Mutex<HashMap<String, VecDeque<Instant>>>> = OnceLock::new();

#[derive(Deserialize)]
pub(super) struct Credentials {
    email: String,
    password: String,
}

#[derive(Deserialize)]
pub(super) struct WechatLoginRequest {
    code: String,
}

#[derive(Deserialize)]
struct Code2SessionResponse {
    openid: Option<String>,
    errcode: Option<i64>,
    errmsg: Option<String>,
}

#[derive(Clone, Debug)]
pub(super) struct Account {
    pub id: i64,
    pub email: String,
    pub balance_milli: i64,
}

pub(super) async fn initialize(config: &ServerConfig) -> Result<(), String> {
    let Some(postgres) = config.account_postgres.as_deref() else {
        return Ok(());
    };
    let client = connect(postgres).await?;
    client
        .batch_execute(
            "CREATE EXTENSION IF NOT EXISTS pgcrypto;
             CREATE TABLE IF NOT EXISTS h3_accounts (
                 id BIGSERIAL PRIMARY KEY,
                 email TEXT NOT NULL UNIQUE,
                 password_hash TEXT NOT NULL,
                 balance_milli BIGINT NOT NULL DEFAULT 0 CHECK (balance_milli >= 0),
                 created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
             );
             ALTER TABLE h3_accounts ADD COLUMN IF NOT EXISTS balance_milli BIGINT NOT NULL DEFAULT 0;
             ALTER TABLE h3_accounts ADD COLUMN IF NOT EXISTS wechat_openid TEXT;
             CREATE UNIQUE INDEX IF NOT EXISTS idx_h3_accounts_wechat_openid
                 ON h3_accounts(wechat_openid) WHERE wechat_openid IS NOT NULL;
             CREATE TABLE IF NOT EXISTS h3_sessions (
                 token_hash BYTEA PRIMARY KEY,
                 account_id BIGINT NOT NULL REFERENCES h3_accounts(id) ON DELETE CASCADE,
                 expires_at TIMESTAMPTZ NOT NULL
             );
             CREATE INDEX IF NOT EXISTS idx_h3_sessions_expires_at ON h3_sessions(expires_at);",
        )
        .await
        .map_err(|error| format!("初始化 H3 账户表失败: {error}"))?;
    Ok(())
}

pub(super) async fn register(State(state): State<ServerState>, Json(credentials): Json<Credentials>) -> Response {
    let (email, password) = match validate(credentials) {
        Ok(value) => value,
        Err(response) => return response,
    };
    let client = match configured_client(&state).await {
        Ok(client) => client,
        Err(response) => return response,
    };
    let row = match client
        .query_opt(
            "INSERT INTO h3_accounts(email, password_hash)
             VALUES($1, crypt($2, gen_salt('bf', 12)))
             ON CONFLICT(email) DO NOTHING RETURNING id, email",
            &[&email, &password],
        )
        .await
    {
        Ok(row) => row,
        Err(error) => return server_error(format!("创建账户失败: {error}")),
    };
    let Some(row) = row else {
        return json_error(StatusCode::CONFLICT, "email_exists", "这个邮箱已经注册，请直接登录");
    };
    issue_session(&state, &client, &row, StatusCode::CREATED).await
}

pub(super) async fn login(State(state): State<ServerState>, headers: HeaderMap, Json(credentials): Json<Credentials>) -> Response {
    let client_address = client_address(&headers);
    if !login_permitted(&client_address) {
        return json_error(StatusCode::TOO_MANY_REQUESTS, "too_many_attempts", "尝试次数过多，请稍后再试");
    }
    let (email, password) = match validate(credentials) {
        Ok(value) => value,
        Err(response) => return response,
    };
    let client = match configured_client(&state).await {
        Ok(client) => client,
        Err(response) => return response,
    };
    let row = match client.query_opt("SELECT id, email FROM h3_accounts WHERE email = $1 AND password_hash = crypt($2, password_hash)", &[&email, &password]).await {
        Ok(row) => row,
        Err(error) => return server_error(format!("查询账户失败: {error}")),
    };
    let Some(row) = row else {
        record_login_failure(&client_address);
        return json_error(StatusCode::UNAUTHORIZED, "invalid_credentials", "邮箱或密码不正确");
    };
    clear_login_failures(&client_address);
    issue_session(&state, &client, &row, StatusCode::OK).await
}

/// 微信小程序一键登录：wx.login 的 code → jscode2session 换 openid → 查/建账户 → 发 h3_session。
pub(super) async fn wechat_login(State(state): State<ServerState>, Json(request): Json<WechatLoginRequest>) -> Response {
    let code = request.code.trim();
    if code.is_empty() || code.len() > 128 {
        return json_error(StatusCode::BAD_REQUEST, "invalid_code", "微信登录 code 无效");
    }
    let (Some(appid), Some(secret)) = (state.config.wechat_appid.as_deref(), state.config.wechat_secret.as_deref()) else {
        return json_error(StatusCode::SERVICE_UNAVAILABLE, "wechat_not_configured", "微信登录尚未配置");
    };
    let url = format!("https://api.weixin.qq.com/sns/jscode2session?appid={appid}&secret={secret}&js_code={code}&grant_type=authorization_code");
    let http = match reqwest::Client::builder().timeout(Duration::from_secs(10)).build() {
        Ok(client) => client,
        Err(error) => return server_error(format!("创建 HTTP 客户端失败: {error}")),
    };
    let session = match http.get(&url).send().await {
        Ok(response) => match response.json::<Code2SessionResponse>().await {
            Ok(parsed) => parsed,
            Err(error) => return server_error(format!("解析微信登录响应失败: {error}")),
        },
        Err(error) => return server_error(format!("请求微信登录服务失败: {error}")),
    };
    let Some(openid) = session.openid.filter(|value| !value.is_empty()) else {
        let message = session.errmsg.unwrap_or_else(|| format!("errcode {}", session.errcode.unwrap_or(-1)));
        return json_error(StatusCode::UNAUTHORIZED, "wechat_code_rejected", format!("微信登录失败: {message}"));
    };

    let client = match configured_client(&state).await {
        Ok(client) => client,
        Err(response) => return response,
    };
    let existing = match client.query_opt("SELECT id, email FROM h3_accounts WHERE wechat_openid = $1", &[&openid]).await {
        Ok(row) => row,
        Err(error) => return server_error(format!("查询微信账户失败: {error}")),
    };
    let row = match existing {
        Some(row) => row,
        None => {
            // 占位邮箱与随机密码仅满足表约束，用户不可见；展示侧按 @wechat.local 识别微信账户。
            let placeholder_email = format!("wx_{openid}@wechat.local");
            let placeholder_password = session_token().unwrap_or_else(|_| openid.clone());
            let inserted = match client
                .query_opt(
                    "INSERT INTO h3_accounts(email, password_hash, wechat_openid)
                     VALUES($1, crypt($2, gen_salt('bf', 12)), $3)
                     ON CONFLICT(wechat_openid) WHERE wechat_openid IS NOT NULL
                     DO NOTHING RETURNING id, email",
                    &[&placeholder_email, &placeholder_password, &openid],
                )
                .await
            {
                Ok(row) => row,
                Err(error) => return server_error(format!("创建微信账户失败: {error}")),
            };
            match inserted {
                Some(row) => row,
                None => match client.query_opt("SELECT id, email FROM h3_accounts WHERE wechat_openid = $1", &[&openid]).await {
                    Ok(Some(row)) => row,
                    Ok(None) => return server_error("创建微信账户失败: 并发冲突后未找到账户".to_owned()),
                    Err(error) => return server_error(format!("查询微信账户失败: {error}")),
                },
            }
        }
    };
    issue_session(&state, &client, &row, StatusCode::OK).await
}

pub(super) async fn me(State(state): State<ServerState>, headers: HeaderMap) -> Response {
    let client = match configured_client(&state).await {
        Ok(client) => client,
        Err(response) => return response,
    };
    match current_account(&client, &headers).await {
        Ok(Some(account)) => (
            StatusCode::OK,
            Json(json!({
                "authenticated": true,
                "user": {"email": account.email},
                "balance": account.balance_milli as f64 / 1000.0,
                "pricing": {"image": 0.1, "video_per_second": 0.2, "hd_per_second": super::studio::HD_SECOND_PRICE_MILLI as f64 / 1000.0, "cancellation": "server_progress"}
            })),
        )
            .into_response(),
        Ok(None) => (StatusCode::UNAUTHORIZED, Json(json!({"authenticated": false}))).into_response(),
        Err(error) => server_error(error),
    }
}

pub(super) async fn logout(State(state): State<ServerState>, headers: HeaderMap) -> Response {
    let client = match configured_client(&state).await {
        Ok(client) => client,
        Err(response) => return response,
    };
    if let Some(token) = cookie(&headers, "h3_session") {
        if let Err(error) = client.execute("DELETE FROM h3_sessions WHERE token_hash = $1", &[&token_hash(&token)]).await {
            return server_error(format!("退出登录失败: {error}"));
        }
    }
    let mut response = (StatusCode::OK, Json(json!({"ok": true}))).into_response();
    response.headers_mut().insert(header::SET_COOKIE, HeaderValue::from_static("h3_session=; Path=/; Max-Age=0; HttpOnly; Secure; SameSite=Lax"));
    response
}

fn validate(credentials: Credentials) -> Result<(String, String), Response> {
    let email = credentials.email.trim().to_lowercase();
    let valid_email =
        email.len() <= 254 && !email.chars().any(char::is_whitespace) && email.split_once('@').is_some_and(|(local, domain)| !local.is_empty() && domain.split_once('.').is_some_and(|(left, right)| !left.is_empty() && !right.is_empty()));
    if !valid_email {
        return Err(json_error(StatusCode::BAD_REQUEST, "invalid_email", "请输入正确的邮箱地址"));
    }
    if !(8..=128).contains(&credentials.password.chars().count()) {
        return Err(json_error(StatusCode::BAD_REQUEST, "invalid_password", "密码需要 8–128 位"));
    }
    Ok((email, credentials.password))
}

pub(super) async fn configured_client(state: &ServerState) -> Result<Client, Response> {
    let Some(postgres) = state.config.account_postgres.as_deref() else {
        return Err(json_error(StatusCode::SERVICE_UNAVAILABLE, "account_not_configured", "账户服务尚未配置"));
    };
    connect(postgres).await.map_err(server_error)
}

pub(super) async fn connect(postgres: &str) -> Result<Client, String> {
    let (client, connection) = tokio_postgres::connect(postgres, NoTls).await.map_err(|error| format!("连接 H3 账户库失败: {error}"))?;
    tokio::spawn(async move {
        if let Err(error) = connection.await {
            eprintln!("H3 账户库连接中断: {error}");
        }
    });
    Ok(client)
}

async fn issue_session(_state: &ServerState, client: &Client, account: &Row, status: StatusCode) -> Response {
    let token = match session_token() {
        Ok(token) => token,
        Err(error) => return server_error(error),
    };
    if let Err(error) = client.execute("DELETE FROM h3_sessions WHERE expires_at <= NOW()", &[]).await {
        return server_error(format!("清理过期会话失败: {error}"));
    }
    if let Err(error) = client.execute("INSERT INTO h3_sessions(token_hash, account_id, expires_at) VALUES($1, $2, NOW() + ($3::bigint * INTERVAL '1 second'))", &[&token_hash(&token), &account.get::<_, i64>("id"), &SESSION_SECONDS]).await {
        return server_error(format!("创建登录会话失败: {error}"));
    }
    let email = account.get::<_, String>("email");
    // body 同时返回 session：微信小程序 wx.request 会过滤 Set-Cookie 响应头，
    // 小程序端只能从 body 取 token 后自行携带 Cookie 头（浏览器端不受影响）。
    let mut response = (status, Json(json!({"authenticated": true, "user": {"email": email}, "session": token}))).into_response();
    let cookie = format!("h3_session={token}; Path=/; Max-Age={SESSION_SECONDS}; HttpOnly; Secure; SameSite=Lax");
    response.headers_mut().insert(header::SET_COOKIE, HeaderValue::from_str(&cookie).expect("会话 token 只含十六进制字符"));
    response
}

pub(super) async fn current_account(client: &Client, headers: &HeaderMap) -> Result<Option<Account>, String> {
    let Some(token) = cookie(headers, "h3_session") else {
        return Ok(None);
    };
    client
        .query_opt(
            "SELECT h3_accounts.id, h3_accounts.email, h3_accounts.balance_milli
             FROM h3_sessions JOIN h3_accounts ON h3_accounts.id = h3_sessions.account_id
             WHERE h3_sessions.token_hash = $1 AND h3_sessions.expires_at > NOW()",
            &[&token_hash(&token)],
        )
        .await
        .map(|row| row.map(|row| Account { id: row.get("id"), email: row.get("email"), balance_milli: row.get("balance_milli") }))
        .map_err(|error| format!("查询登录会话失败: {error}"))
}

pub(super) async fn require_account(state: &ServerState, headers: &HeaderMap) -> Result<(Client, Account), Response> {
    let client = configured_client(state).await?;
    match current_account(&client, headers).await {
        Ok(Some(account)) => Ok((client, account)),
        Ok(None) => Err(json_error(StatusCode::UNAUTHORIZED, "login_required", "请先登录")),
        Err(error) => Err(server_error(error)),
    }
}

fn session_token() -> Result<String, String> {
    let mut bytes = [0u8; 32];
    getrandom::fill(&mut bytes).map_err(|error| format!("生成登录会话随机数失败: {error}"))?;
    let mut token = String::with_capacity(64);
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for byte in bytes {
        token.push(HEX[(byte >> 4) as usize] as char);
        token.push(HEX[(byte & 0x0f) as usize] as char);
    }
    Ok(token)
}

fn token_hash(token: &str) -> Vec<u8> {
    blake3::hash(token.as_bytes()).as_bytes().to_vec()
}

fn cookie(headers: &HeaderMap, name: &str) -> Option<String> {
    headers.get(header::COOKIE)?.to_str().ok()?.split(';').filter_map(|part| part.trim().split_once('=')).find_map(|(key, value)| (key == name && !value.is_empty()).then(|| value.to_owned()))
}

fn client_address(headers: &HeaderMap) -> String {
    headers.get("x-forwarded-for").and_then(|value| value.to_str().ok()).and_then(|value| value.split(',').next()).map(str::trim).filter(|value| !value.is_empty()).unwrap_or("unknown").to_owned()
}

fn login_entries() -> std::sync::MutexGuard<'static, HashMap<String, VecDeque<Instant>>> {
    let attempts = LOGIN_ATTEMPTS.get_or_init(Default::default);
    attempts.lock().expect("登录限流 mutex 已损坏")
}

fn login_permitted(address: &str) -> bool {
    let now = Instant::now();
    let mut attempts = login_entries();
    let entries = attempts.entry(address.to_owned()).or_default();
    while entries.front().is_some_and(|instant| now.duration_since(*instant) > LOGIN_WINDOW) {
        entries.pop_front();
    }
    entries.len() < LOGIN_LIMIT
}

fn record_login_failure(address: &str) {
    login_entries().entry(address.to_owned()).or_default().push_back(Instant::now());
}

fn clear_login_failures(address: &str) {
    login_entries().remove(address);
}

fn json_error(status: StatusCode, code: &'static str, message: impl Into<String>) -> Response {
    (status, Json(json!({"error": code, "message": message.into()}))).into_response()
}

fn server_error(message: impl Into<String>) -> Response {
    let message = message.into();
    eprintln!("{message}");
    json_error(StatusCode::INTERNAL_SERVER_ERROR, "account_server_error", "账户服务暂时不可用")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 邮箱密码校验保持最小规则() {
        assert!(validate(Credentials { email: " User@Example.com ".to_owned(), password: "12345678".to_owned() }).is_ok());
        assert!(validate(Credentials { email: "missing-at".to_owned(), password: "12345678".to_owned() }).is_err());
        assert!(validate(Credentials { email: "a@example.com".to_owned(), password: "short".to_owned() }).is_err());
    }

    #[test]
    fn 会话cookie只读取精确名称() {
        let mut headers = HeaderMap::new();
        headers.insert(header::COOKIE, HeaderValue::from_static("other=1; h3_session=abc123; h3_session_x=bad"));
        assert_eq!(cookie(&headers, "h3_session").as_deref(), Some("abc123"));
    }

    #[test]
    fn 会话token不可预测且长度固定() {
        let first = session_token().unwrap();
        let second = session_token().unwrap();
        assert_eq!(first.len(), 64);
        assert_ne!(first, second);
    }
}
