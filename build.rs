use std::{
    env, fs,
    path::{Path, PathBuf},
};

fn copy_qnn_headers(source: &Path, destination: &Path, core_minor: &str) {
    fs::create_dir_all(destination).expect("创建 QNN 临时 include 目录");
    for entry in fs::read_dir(source).expect("读取 QNN include 目录") {
        let entry = entry.expect("读取 QNN include 项");
        let target = destination.join(entry.file_name());
        if entry.file_type().expect("读取 QNN include 类型").is_dir() {
            copy_qnn_headers(&entry.path(), &target, core_minor);
        } else {
            let bytes = fs::read(entry.path()).expect("读取 QNN header");
            if entry.file_name() == "QnnCommon.h" {
                let text = String::from_utf8(bytes).expect("QnnCommon.h 不是 UTF-8");
                let patched = text.lines().map(|line| if line.trim_start().starts_with("#define QNN_API_VERSION_MINOR ") { format!("#define QNN_API_VERSION_MINOR {core_minor}") } else { line.to_owned() }).collect::<Vec<_>>().join("\n");
                fs::write(target, patched).expect("写入兼容 QNN header");
            } else {
                fs::write(target, bytes).expect("复制 QNN header");
            }
        }
    }
}

fn main() {
    println!("cargo:rerun-if-changed=src/backend/qnn/qnn_linear.cpp");
    println!("cargo:rerun-if-env-changed=QNN_SDK_ROOT");
    println!("cargo:rerun-if-env-changed=ZLLM_QNN_CORE_API_MINOR");
    if env::var_os("CARGO_FEATURE_WITH_QNN").is_none() {
        return;
    }
    let root = PathBuf::from(env::var_os("QNN_SDK_ROOT").expect("with-qnn 需要 QNN_SDK_ROOT"));
    // QAIRT 2.3x 起 header 移入 include/QNN 子目录;旧 SDK 平铺在 include/ 下。
    let source = root.join("include/QNN");
    let source = if source.is_dir() { source } else { root.join("include") };
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR 缺失")).join("qnn-include");
    // QAIRT 2.38 的 core API 是 2.28;运行时我们推送同版本 HTP 库,直接按 2.28 编译。
    // 若要对接手机系统自带旧 QAIRT(≤2.29,core 2.22),用该环境变量降级。
    let minor = env::var("ZLLM_QNN_CORE_API_MINOR").unwrap_or_else(|_| "28".to_owned());
    copy_qnn_headers(&source, &out, &minor);
    cc::Build::new().cpp(true).std("c++17").include(&out).file("src/backend/qnn/qnn_linear.cpp").flag_if_supported("-fno-exceptions").compile("zllm_qnn");
    // 设备侧 HTP 引擎(trace 解释器 / decode / prefill / ASR):源码使用 C++ 异常,
    // 与上面的 no-exceptions FFI shim 分开编译。引擎只依赖 QNN header,架构差异
    // (V73/V75/V79)全部由运行期配置与部署产物承载,不进入编译期。
    let engine = [
        "src/backend/qnn/htp_trace.cpp",
        "src/runtime/minicpm5/qnn_decode.cpp",
        "src/runtime/minicpm5/qnn_prefill.cpp",
        "src/runtime/sensevoice/qnn_asr.cpp",
    ];
    for file in &engine {
        println!("cargo:rerun-if-changed={file}");
    }
    let mut build = cc::Build::new();
    build.cpp(true).std("c++17").include(out).include("src/backend/qnn").files(engine);
    build.compile("zllm_qnn_htp");
}
