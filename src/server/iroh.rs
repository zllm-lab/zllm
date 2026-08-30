//! iroh 服务共享的节点身份加载。

use std::str::FromStr;

use iroh::SecretKey;

#[derive(Clone, Debug, Default)]
pub struct IrohConfig {
    pub secret_key: Option<SecretKey>,
    pub bind_addr: Option<String>,
    pub expected_peer: Option<String>,
}

pub fn parse_secret(value: &str) -> Result<SecretKey, String> {
    if value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        let mut bytes = [0u8; 32];
        for (index, byte) in bytes.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16).map_err(|error| format!("iroh secret_key 十六进制解析失败: {error}"))?;
        }
        return Ok(SecretKey::from_bytes(&bytes));
    }
    SecretKey::from_str(value).map_err(|error| format!("iroh secret_key 必须是 64 位十六进制或 iroh base32hex: {error}"))
}

pub fn validate_peer(connection: &iroh::endpoint::Connection, expected: Option<&str>, role: &str) -> Result<(), String> {
    let peer = connection.remote_id().to_string();
    if let Some(expected) = expected
        && expected != peer
    {
        return Err(format!("{role} peer id 不匹配: expected={expected} actual={peer}"));
    }
    eprintln!("[{role}-connected] peer={peer}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_hex_secret_is_accepted() {
        let bytes = [7u8; 32];
        let value = bytes.iter().map(|byte| format!("{byte:02x}")).collect::<String>();
        let parsed = parse_secret(&value).unwrap();
        assert_eq!(parsed.to_bytes(), bytes);
    }
}
