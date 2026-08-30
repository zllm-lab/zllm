# zLLM

**高性能、跨模型、跨硬件的大语言模型推理引擎。**  
**A high-performance LLM inference engine across models and hardware.**

[![Crates.io](https://img.shields.io/crates/v/zllm.svg)](https://crates.io/crates/zllm)
[![Documentation](https://docs.rs/zllm/badge.svg)](https://docs.rs/zllm)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

[中文](#中文) · [English](#english) · [官网 / Website](https://zhuai.tech) · [GitHub](https://github.com/zllm-lab/zllm)

---

## 中文

zLLM 是一个 Rust 原生的推理引擎，目标是在同一套架构下支持不同模型、权重格式与计算后端。它既可以作为独立进程运行，也可以直接嵌入 Rust 应用，不依赖 Python、PyTorch、Conda 或常驻模型服务。

### 核心能力

- **Rust 原生**：模型加载、tokenize、推理、采样与缓存生命周期均由同一运行时管理。
- **可嵌入**：通过 `zllm::Engine` 在应用进程内加载模型并流式生成文本。
- **多后端**：覆盖 CPU、Apple Metal、NVIDIA CUDA、AMD ROCm 与 Vulkan 路径。
- **多模型**：包含 Gemma 4、Qwen 3.6/3.8、GLM、DeepSeek、Mistral、MiniCPM、Ornith 等模型运行时。
- **多种权重格式**：支持 Safetensors、GGUF/GGML 及多种低比特量化格式。
- **服务接口**：提供 OpenAI Chat Completions、Responses 与 Anthropic Messages 兼容入口。
- **长会话**：支持 KV cache、持久化会话状态、前缀复用与分布式调度。

具体模型和后端能力仍在持续演进；未在真实设备完成验证的组合不应视为生产就绪。

### 安装

```bash
cargo add zllm
```

或在 `Cargo.toml` 中指定版本：

```toml
[dependencies]
zllm = "0.8"
serde_json = "1"
```

可选后端 feature：

```toml
zllm = { version = "0.8", features = ["with-cuda"] }
```

- `with-cuda`：NVIDIA CUDA
- `with-rocm`：AMD ROCm
- `with-vulkan`：Vulkan
- Apple Metal 在 macOS 目标上直接可用

### 嵌入 Rust 应用

```rust
use std::io::{self, Write};
use serde_json::json;
use zllm::Engine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut engine = Engine::from_config("config/embedded-gemma4-metal.yaml")?;
    let cancellation = engine.cancellation();
    let result = engine.generate(
        &json!({
            "model": "gemma4",
            "messages": [{"role": "user", "content": "用一句话介绍 zLLM"}],
            "max_completion_tokens": 128
        }),
        &cancellation,
        |_token, text| {
            print!("{text}");
            io::stdout().flush().is_ok()
        },
    )?;
    println!("\nfinish={} tokens={}", result.finish_reason, result.completion_tokens);
    Ok(())
}
```

`Engine` 直接持有模型、设备资源与 KV cache，并不是远程 HTTP 客户端包装。返回 `false` 或调用 `cancellation.cancel()` 可以停止生成。

### 运行示例

```bash
cargo run --release --example gemma4_embedded -- \
  config/embedded-gemma4-metal.yaml "你好，介绍一下你自己" 256

cargo run --release --example qwen36_embedded -- \
  config/embedded-qwen36-metal.yaml "你好，介绍一下你自己" 256
```

运行前请把 YAML 中的 `weights_directory` 改为本机模型路径。

### 构建检查

```bash
cargo fmt --all --check
cargo check --lib
```

### 许可证

本项目使用 [Apache License 2.0](LICENSE)。

---

## English

zLLM is a Rust-native inference engine designed to support different models, weight formats, and compute backends within one architecture. It can run as a standalone process or embed directly into a Rust application without requiring Python, PyTorch, Conda, or a resident model server.

### Highlights

- **Rust-native runtime:** model loading, tokenization, inference, sampling, and cache lifecycles are managed in one runtime.
- **Embeddable:** load a model in-process and stream generated text through `zllm::Engine`.
- **Multiple backends:** CPU, Apple Metal, NVIDIA CUDA, AMD ROCm, and Vulkan paths.
- **Multiple model families:** runtimes for Gemma 4, Qwen 3.6/3.8, GLM, DeepSeek, Mistral, MiniCPM, Ornith, and others.
- **Flexible weight formats:** Safetensors, GGUF/GGML, and multiple low-bit quantization formats.
- **Service APIs:** OpenAI Chat Completions, Responses, and Anthropic Messages compatible endpoints.
- **Long-running sessions:** KV cache, persistent session state, prefix reuse, and distributed scheduling.

Model and backend coverage continues to evolve. A combination that has not been validated on real hardware should not be considered production-ready.

### Installation

```bash
cargo add zllm
```

Or add it to `Cargo.toml`:

```toml
[dependencies]
zllm = "0.8"
serde_json = "1"
```

Optional backend features:

```toml
zllm = { version = "0.8", features = ["with-cuda"] }
```

- `with-cuda`: NVIDIA CUDA
- `with-rocm`: AMD ROCm
- `with-vulkan`: Vulkan
- Apple Metal is available directly on macOS targets

### Embed zLLM in Rust

```rust
use std::io::{self, Write};
use serde_json::json;
use zllm::Engine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut engine = Engine::from_config("config/embedded-gemma4-metal.yaml")?;
    let cancellation = engine.cancellation();
    let result = engine.generate(
        &json!({
            "model": "gemma4",
            "messages": [{"role": "user", "content": "Introduce zLLM in one sentence."}],
            "max_completion_tokens": 128
        }),
        &cancellation,
        |_token, text| {
            print!("{text}");
            io::stdout().flush().is_ok()
        },
    )?;
    println!("\nfinish={} tokens={}", result.finish_reason, result.completion_tokens);
    Ok(())
}
```

`Engine` owns the model, device resources, and KV cache directly; it is not an HTTP client wrapper. Return `false` from the callback or call `cancellation.cancel()` to stop generation.

### Run the examples

```bash
cargo run --release --example gemma4_embedded -- \
  config/embedded-gemma4-metal.yaml "Hello, introduce yourself." 256

cargo run --release --example qwen36_embedded -- \
  config/embedded-qwen36-metal.yaml "Hello, introduce yourself." 256
```

Before running an example, update `weights_directory` in its YAML file to point to your local model weights.

### Validation

```bash
cargo fmt --all --check
cargo check --lib
```

### License

Licensed under the [Apache License 2.0](LICENSE).
