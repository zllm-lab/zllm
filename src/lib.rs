//! zLLM:跨平台、跨模型、高性能 LLM 推理引擎。

pub mod artifact;
pub mod attention;
pub mod backend;
pub mod config;
pub mod diffusion;
#[cfg(any(target_os = "macos", feature = "with-cuda", all(target_os = "linux", feature = "with-rocm")))]
pub mod embedded;
pub mod kernel;
pub mod kv_cache;
pub mod model_spec;
pub mod moe;
pub mod norm;
pub mod runtime;
pub mod server;
pub mod speculative;
pub mod tokenizer;
pub mod vae;
pub mod vision;
pub mod weight;

#[cfg(target_os = "macos")]
pub use backend::metal;
#[cfg(any(target_os = "macos", feature = "with-cuda", all(target_os = "linux", feature = "with-rocm")))]
pub use embedded::{Cancellation, Engine, GenerationEvent, GenerationResult, KvResourceReport, SessionConfig};
