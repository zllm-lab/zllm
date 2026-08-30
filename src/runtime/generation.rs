//! 单模型 CLI 的标准 token 生成生命周期。

use std::time::{Duration, Instant};

#[derive(Clone, Copy)]
pub struct GenerationLimits<'a> {
    pub prompt_tokens: usize,
    pub max_tokens: usize,
    pub max_sequence_length: usize,
    pub eos_tokens: &'a [u32],
}

#[allow(dead_code)]
pub struct GenerationStats {
    pub generated_tokens: usize,
    pub decode_rounds: usize,
    pub elapsed: Duration,
}

#[allow(dead_code)]
impl GenerationStats {
    pub fn tokens_per_second(&self) -> f64 {
        self.generated_tokens as f64 / self.elapsed.as_secs_f64().max(f64::MIN_POSITIVE)
    }

    #[allow(dead_code)]
    pub fn decode_tokens_per_second(&self) -> f64 {
        self.decode_rounds as f64 / self.elapsed.as_secs_f64().max(f64::MIN_POSITIVE)
    }
}

/// `state` 只保存模型当前生成状态；模型前向、设备操作与输出策略由调用方闭包拥有。
pub fn run_generation<S, E>(
    state: &mut S,
    limits: GenerationLimits<'_>,
    mut select_token: impl FnMut(&mut S, usize) -> Result<u32, E>,
    mut emit_token: impl FnMut(u32, usize) -> Result<(), E>,
    mut advance: impl FnMut(&mut S, u32, usize, usize) -> Result<(), E>,
) -> Result<GenerationStats, E> {
    let started = Instant::now();
    let mut generated_tokens = 0;
    let mut decode_rounds = 0;
    for step in 0..limits.max_tokens {
        let token = select_token(state, step)?;
        if limits.eos_tokens.contains(&token) {
            break;
        }
        emit_token(token, step)?;
        generated_tokens += 1;
        if step + 1 == limits.max_tokens {
            break;
        }
        let Some(position) = limits.prompt_tokens.checked_add(step) else {
            break;
        };
        if position >= limits.max_sequence_length {
            break;
        }
        advance(state, token, position, step)?;
        decode_rounds += 1;
    }
    Ok(GenerationStats { generated_tokens, decode_rounds, elapsed: started.elapsed() })
}

#[allow(dead_code)]
pub fn write_token(detokenizer: &crate::tokenizer::Detokenizer, token: u32, skip_special_tokens: bool) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;

    std::io::stdout().lock().write_all(&detokenizer.decode_bytes(&[token], skip_special_tokens)?)?;
    std::io::stdout().flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eos_stops_before_emit_or_decode() {
        let mut state = vec![7_u32, 9, 2];
        let mut emitted = Vec::new();
        let mut positions = Vec::new();
        let stats = run_generation(
            &mut state,
            GenerationLimits { prompt_tokens: 4, max_tokens: 5, max_sequence_length: 16, eos_tokens: &[9] },
            |state, _| Ok::<_, ()>(state.remove(0)),
            |token, _| {
                emitted.push(token);
                Ok(())
            },
            |_, _, position, _| {
                positions.push(position);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(emitted, vec![7]);
        assert_eq!(positions, vec![4]);
        assert_eq!(stats.generated_tokens, 1);
        assert_eq!(stats.decode_rounds, 1);
    }

    #[test]
    fn sequence_limit_keeps_last_visible_token_without_advancing() {
        let mut state = ();
        let mut emitted = Vec::new();
        let stats = run_generation(
            &mut state,
            GenerationLimits { prompt_tokens: 8, max_tokens: 4, max_sequence_length: 8, eos_tokens: &[] },
            |_, step| Ok::<_, ()>(step as u32),
            |token, _| {
                emitted.push(token);
                Ok(())
            },
            |_, _, _, _| panic!("达到序列上限后不应 decode"),
        )
        .unwrap();
        assert_eq!(emitted, vec![0]);
        assert_eq!(stats.decode_rounds, 0);
    }
}
