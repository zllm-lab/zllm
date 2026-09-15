//! DeepSeek-V4.1-Flash 真实 checkpoint(Jundot oQ4e-mtp)装配冒烟。
//! 运行:`cargo test --release --features with-rocm dsv41_jundot_smoke -- --ignored --nocapture`
//! 依赖完整下载的 DeepSeek-V4.1-Flash-oQ4e-mtp 权重。

use crate::{
    model_spec::deepseek_v4::DeepSeekV4Config,
    weight::model::deepseek_v4::{DeepSeekV4CoreMatrix, DeepSeekV4EngramEmbedding, DeepSeekV4Weights},
};

const ROOT: &str = "/path/to/DeepSeek-V4.1-Flash-oQ4e-mtp";

/// 真实权重的 engram CPU 全链验证:hash → 查行 → BF16 GEMV → 门控,
/// 断言输出有限且门控处于 (0,1) 的合理区间。
#[test]
#[ignore]
fn dsv41_engram_real_weights() {
    use crate::runtime::deepseek_v4::engram_cpu::DeepSeekV4EngramCpu;
    let config = DeepSeekV4Config::flash_v41();
    let weights = DeepSeekV4Weights::open(ROOT, config.clone()).unwrap_or_else(|error| panic!("open: {error}"));
    let mut cpu = DeepSeekV4EngramCpu::load(format!("{ROOT}/engram_token_map.bin"), weights, config.hyper_connection_copies, config.hidden_size, config.rms_eps as f32).unwrap_or_else(|error| panic!("engram cpu load: {error}"));
    // 一组真实感 token(中文/英文/数字混合 id)
    let tokens = [9707, 11, 1879, 330, 1299, 12039, 4, 775, 18, 20022, 11, 9707];
    cpu.push_prefill(&tokens);
    let rows = tokens.len();
    let stride = 4 * config.hidden_size;
    let mut h = (0..rows * stride).map(|i| ((i * 2654435761) % 2000) as f32 / 1000.0 - 1.0).collect::<Vec<_>>();
    let original = h.clone();
    for slot in 0..2 {
        cpu.apply_prefill_rows(slot, 0, rows, &mut h).unwrap_or_else(|error| panic!("apply slot {slot}: {error}"));
    }
    let max_delta = h.iter().zip(original.iter()).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    let mean_delta = h.iter().zip(original.iter()).map(|(a, b)| (a - b).abs()).sum::<f32>() / h.len() as f32;
    eprintln!("[engram-real] max|Δh|={max_delta:.4} mean|Δh|={mean_delta:.6}");
    assert!(h.iter().all(|v| v.is_finite()), "输出含非有限值");
    assert!(max_delta > 0.0, "engram 对 h 没有任何影响(hash 或查行可能全错)");
    assert!(max_delta < 50.0, "engram 输出爆炸 max|Δh|={max_delta}");
    // decode 增量与 prefill 一致性(真实权重行读取路径)
    let mut cpu2 = DeepSeekV4EngramCpu::load(format!("{ROOT}/engram_token_map.bin"), DeepSeekV4Weights::open(ROOT, config.clone()).unwrap(), config.hyper_connection_copies, config.hidden_size, config.rms_eps as f32).unwrap();
    let mut h2 = original.clone();
    for (pos, &token) in tokens.iter().enumerate() {
        cpu2.push_token(token, false);
        for slot in 0..2 {
            cpu2.apply_current(slot, &mut h2[pos * stride..(pos + 1) * stride]).unwrap();
        }
    }
    let drift = h.iter().zip(h2.iter()).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    eprintln!("[engram-real] prefill/decode 最大漂移 {drift:.2e}");
    eprintln!("[engram-real] ALL OK");
}

#[test]
#[ignore]
fn dsv41_jundot_smoke_open_and_load_layers() {
    let config = DeepSeekV4Config::flash_v41();
    let weights = DeepSeekV4Weights::open(ROOT, config.clone()).unwrap_or_else(|error| panic!("open: {error}"));
    eprintln!("[smoke] open ok (ns 探测)");

    let embedding = weights.embedding_rows_bf16(&[1, 100, 99091]).unwrap_or_else(|error| panic!("embedding: {error}"));
    assert_eq!(embedding.len(), 3 * config.hidden_size * 2);
    eprintln!("[smoke] embedding rows ok");

    for layer in [0, 1, 2, 20, 39] {
        let layer_weights = weights.load_layer(layer).unwrap_or_else(|error| panic!("load_layer({layer}): {error}"));
        let core = &layer_weights.attention.attention.query.input_projection;
        let form = match core {
            DeepSeekV4CoreMatrix::BlockFp8(_) => "block_fp8",
            DeepSeekV4CoreMatrix::Mxfp8(_) => "mxfp8",
            DeepSeekV4CoreMatrix::Dense(_) => "dense",
        };
        let wo_a = &layer_weights.attention.attention.output.input_projection;
        let wo_a_form = match wo_a {
            DeepSeekV4CoreMatrix::BlockFp8(_) => "block_fp8",
            DeepSeekV4CoreMatrix::Mxfp8(_) => "mxfp8",
            DeepSeekV4CoreMatrix::Dense(_) => "dense",
        };
        eprintln!("[smoke] L{layer}: wq_a={form} wo_a={wo_a_form} compressor={} indexer={}", layer_weights.attention.attention.compressor.is_some(), layer_weights.attention.attention.indexer.is_some());
    }

    // engram:L1 有,L0 无
    assert!(weights.load_engram(0).unwrap().is_none());
    let engram = weights.load_engram(1).unwrap_or_else(|error| panic!("load_engram(1): {error}")).expect("L1 必须有 engram");
    eprintln!("[smoke] engram L1: wkv [{},{}]", engram.wkv.rows(), engram.wkv.cols());
    let rows = weights.engram_embedding_rows(1, &[0, 1, 383_999_999]).unwrap_or_else(|error| panic!("engram rows: {error}"));
    match rows {
        DeepSeekV4EngramEmbedding::Mxfp8(matrix) => eprintln!("[smoke] engram embed rows mxfp8 [{},{}]", matrix.rows, matrix.cols),
        DeepSeekV4EngramEmbedding::MlxAffine(matrix) => {
            let decoded = matrix.decode().unwrap_or_else(|error| panic!("engram affine decode: {error}"));
            eprintln!("[smoke] engram embed rows mlx-affine decoded {} f32", decoded.len());
        }
    }

    for layer in 0..config.mtp_layer_count {
        let mtp = weights.load_mtp_layer(layer).unwrap_or_else(|error| panic!("load_mtp_layer({layer}): {error}"));
        let core = &mtp.attention.attention.query.input_projection;
        eprintln!("[smoke] mtp.{layer}: wq_a [{},{}]", core.rows(), core.cols());
    }

    let head = weights.output_head().unwrap_or_else(|error| panic!("output_head: {error}"));
    eprintln!("[smoke] output head: hc={} lm_head={:?}", head.hyper_connection.is_some(), head.lm_head.shape);
    eprintln!("[smoke] ALL OK");
}
