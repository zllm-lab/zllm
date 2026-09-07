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

### 客户端与后端能力矩阵

零配置对话客户端——`<客户端> <模型路径>` 直接启动，不写 YAML：

| 客户端 | 平台 | 支持模型 | 权重形态 | KV cache | 上下文策略 |
| --- | --- | --- | --- | --- | --- |
| `zllm-metal` | macOS (Metal) | Gemma4、Qwen3.6/3.8、Ornith、Mistral、K2-Horizon MoVA 36B-A4B、MiniCPM5 | GGUF、MLX safetensors | Q8g64（Gemma4 为 f16） | 按统一内存预算自动推导 |
| `zllm-fedora-cuda` / `zllm-windows-cuda` | NVIDIA CUDA | Gemma4（12B/E4B）、Qwen3.6/3.8、Ornith、Mistral | GGUF、MLX safetensors | Q8g64（Gemma4 为 f16） | 8192 起步、加载失败减半；Qwen3.6/3.8 按空闲显存自动 CPU+CUDA 分层 |

MiniCPM5 无 CUDA 执行路径；CUDA 上的 Qwen3.6/3.8 暂不支持 MTP。需要多卡、MTP、专家缓存、持久化 KV 等更多控制时，使用 `zllm-rt-<platform> --config <yaml>`。

模型 × 后端（`zllm-server` / `zllm-rt-*` 服务入口，以配置校验矩阵为准）：

| 模型 | CPU | Metal | CUDA | ROCm | 备注 |
| --- | --- | --- | --- | --- | --- |
| Gemma4 | — | ✅ MTP、视觉 | ✅ 12B/E4B | — | |
| Qwen3.6 / Qwen3.8 | — | ✅ MTP、视觉 | ✅ CPU+CUDA 混合分层 | ✅ 混合分层 | CUDA 无 MTP |
| Ornith | — | ✅ | ✅ lazy experts | ✅ 多卡分层 | |
| Mistral | — | ✅ | ✅ | — | |
| K2-Horizon MoVA 36B-A4B | — | ✅ | — | — | 分组 RMSNorm、MoVA、MoE；GGUF IQ3_XS 已在 Apple M5 真机验证 |
| MiniCPM5 | ✅ | ✅ | — | — | |
| GLM-5.2 / GLM-5.3 | — | — | — | ✅ | 分布式；GGUF prefill/decode/MTP 双机 16 卡路径已真机验证 |
| GLM-5.3-Flash | — | — | — | ✅ | |
| DeepSeek-V4 | — | — | — | ✅ | |
| MiniMax-H3 | — | — | — | ✅ 1/2/4/8 卡 | |

CPU 主要承担跨后端 oracle、单元测试与 Qwen 混合分层的前缀层执行；服务级 CPU 入口目前仅 MiniCPM5。

Qwen3.8-Flash-Next 使用独立的 `qwen4exp` 架构，支持 CUDA QSA 稀疏注意力、按需专家上传与 shared MTP。GLM-5.3 ROCm 支持多 GPU 分层、算子并行与动态 MTP。

### 实测性能

以下结果来自无风扇 Apple M5 24GB 与 NVIDIA RTX 3060 12GB 真机。吞吐会随提示长度、温度、量化格式和 MTP 接受率变化；表中数据用于说明已验证路径，不代表所有工作负载的固定值。

| 模型与权重 | 硬件 / 后端 | Prefill | Decode | 测试说明 |
| --- | --- | ---: | ---: | --- |
| GLM-5.3 UD-IQ4_XS GGUF | 双机 16× AMD GPU / ROCm | **1600+ tok/s** | **26.912 tok/s** | 最新 50K prefill；decode 为独立的 1,024-token 动态 MTP5 测试（三轮中位数） |
| Qwen 3.8 27B UD-Q3_K_XL | Apple M5 24GB / Metal | 58 tokens / 1.278s | 8.33 tok/s | 短提示 fused prefill；普通 decode |
| Qwen 3.8 27B Q4_K_M | 双路 E5-2696 v4 + RTX 3060 12GB / CPU+CUDA | — | **2.68 tok/s** | CPU 前缀层 + CUDA 后缀层；40 个 CPU decode 线程 |
| Gemma 4 12B IQ4_NL GGUF | Apple M5 24GB / Metal | 337.4 tok/s | 14.5–15.2 tok/s | 约 5.6K token 长提示，64-token 回复 |
| Gemma 4 E4B MLX 4-bit | Apple M5 24GB / Metal | 32.2–38.8 tok/s | 40.3–40.7 tok/s | 22-token 短提示；MTP 路径 |
| Gemma 4 12B Q4_0 GGUF | RTX 3060 12GB / CUDA | 345 tok/s | 15.5 tok/s；MTP 34.2 tok/s | 4,518-token prefill；256-token MTP 测试 |
| MiniCPM5-1B Q4_K_M GGUF | Apple M5 24GB / Metal | — | 约 165 tok/s | HTTP 生成路径；短 KV 直接 attention |

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

### 终端对话

`zllm-metal` 是 macOS Metal 的零配置终端 chatbot：给出模型路径即可进入多轮对话，自动识别权重架构并按统一内存规划上下文。

```bash
cargo run --release --bin zllm-metal -- <模型路径(GGUF 或权重目录)>
```

可选参数：`--image-max-tokens N`（Qwen 视觉预算）。需要多卡、MTP、持久化 KV 等控制时使用 `zllm-rt-metal --config <yaml>`。

### 构建检查

```bash
cargo check --lib
cargo test --lib
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

### Client & backend capability matrix

Zero-config chat clients — start directly with `<client> <model-path>`, no YAML required:

| Client | Platform | Models | Weight formats | KV cache | Context strategy |
| --- | --- | --- | --- | --- | --- |
| `zllm-metal` | macOS (Metal) | Gemma4, Qwen3.6/3.8, Ornith, Mistral, K2-Horizon MoVA 36B-A4B, MiniCPM5 | GGUF, MLX safetensors | Q8g64 (f16 for Gemma4) | Derived from unified-memory budget |
| `zllm-fedora-cuda` / `zllm-windows-cuda` | NVIDIA CUDA | Gemma4 (12B/E4B), Qwen3.6/3.8, Ornith, Mistral | GGUF, MLX safetensors | Q8g64 (f16 for Gemma4) | 8192 start, halved on load failure; Qwen3.6/3.8 auto CPU+CUDA layering by free VRAM |

MiniCPM5 has no CUDA execution path; MTP is not available for Qwen3.6/3.8 on CUDA. For multi-GPU, MTP, expert-cache, or persistent-KV control, use `zllm-rt-<platform> --config <yaml>`.

Models × backends (`zllm-server` / `zllm-rt-*` service entries, per the config validation matrix):

| Model | CPU | Metal | CUDA | ROCm | Notes |
| --- | --- | --- | --- | --- | --- |
| Gemma4 | — | ✅ MTP, vision | ✅ 12B/E4B | — | |
| Qwen3.6 / Qwen3.8 | — | ✅ MTP, vision | ✅ CPU+CUDA hybrid layering | ✅ hybrid layering | No MTP on CUDA |
| Ornith | — | ✅ | ✅ lazy experts | ✅ multi-GPU layering | |
| Mistral | — | ✅ | ✅ | — | |
| K2-Horizon MoVA 36B-A4B | — | ✅ | — | — | Grouped RMSNorm, MoVA, and MoE; GGUF IQ3_XS validated on Apple M5 |
| MiniCPM5 | ✅ | ✅ | — | — | |
| GLM-5.2 / GLM-5.3 | — | — | — | ✅ | Distributed; GGUF prefill/decode/MTP path validated on a two-node 16-GPU deployment |
| GLM-5.3-Flash | — | — | — | ✅ | |
| DeepSeek-V4 | — | — | — | ✅ | |
| MiniMax-H3 | — | — | — | ✅ 1/2/4/8 GPUs | |

CPU mainly serves as the cross-backend oracle, unit-test baseline, and CPU prefix layers of the Qwen hybrid path; the service-level CPU entry currently supports MiniCPM5 only.

Qwen3.8-Flash-Next uses the separate `qwen4exp` architecture with CUDA QSA sparse attention, on-demand expert uploads, and shared MTP. GLM-5.3 on ROCm supports multi-GPU layer placement, operator parallelism, and dynamic MTP.

### Measured performance

The following results were measured on a fanless Apple M5 with 24 GB unified memory and an NVIDIA RTX 3060 with 12 GB VRAM. Throughput varies with prompt length, temperature, quantization, and MTP acceptance rate; these numbers describe validated paths rather than guaranteed performance.

| Model and weights | Hardware / backend | Prefill | Decode | Workload |
| --- | --- | ---: | ---: | --- |
| GLM-5.3 UD-IQ4_XS GGUF | Two nodes, 16× AMD GPUs / ROCm | **1600+ tok/s** | **26.912 tok/s** | Latest 50K prefill; decode is a separate 1,024-token dynamic-MTP5 run (three-run median) |
| Qwen 3.8 27B UD-Q3_K_XL | Apple M5 24GB / Metal | 58 tokens / 1.278s | 8.33 tok/s | Short-prompt fused prefill; standard decode |
| Qwen 3.8 27B Q4_K_M | Dual E5-2696 v4 + RTX 3060 12GB / CPU+CUDA | — | **2.68 tok/s** | CPU prefix layers + CUDA suffix layers; 40 CPU decode threads |
| Gemma 4 12B IQ4_NL GGUF | Apple M5 24GB / Metal | 337.4 tok/s | 14.5–15.2 tok/s | About 5.6K prompt tokens, 64-token response |
| Gemma 4 E4B MLX 4-bit | Apple M5 24GB / Metal | 32.2–38.8 tok/s | 40.3–40.7 tok/s | 22-token short prompt with MTP |
| Gemma 4 12B Q4_0 GGUF | RTX 3060 12GB / CUDA | 345 tok/s | 15.5 tok/s; 34.2 tok/s with MTP | 4,518-token prefill; 256-token MTP run |
| MiniCPM5-1B Q4_K_M GGUF | Apple M5 24GB / Metal | — | About 165 tok/s | HTTP generation path with direct short-KV attention |

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
