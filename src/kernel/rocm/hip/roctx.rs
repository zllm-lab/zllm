//! ROCm profiler marker。只在显式 stage event trace 中调用，不进入生产热路径。

use libloading::Library;
use std::{ffi::CString, os::raw::c_char, sync::OnceLock};

type RoctxRangePush = unsafe extern "C" fn(*const c_char) -> i32;
type RoctxRangePop = unsafe extern "C" fn() -> i32;

struct RoctxRuntime {
    _library: Library,
    push: RoctxRangePush,
    pop: RoctxRangePop,
}

fn runtime() -> Result<&'static RoctxRuntime, String> {
    static RUNTIME: OnceLock<Result<RoctxRuntime, String>> = OnceLock::new();
    RUNTIME
        .get_or_init(|| {
            let mut last_error = None;
            // 多版本 ROCm alternatives 可能只改可执行文件，未把对应 lib 写入
            // ld.so cache；/opt/rocm/lib 是官方安装与版本 symlink 的共同入口。
            for name in ["/opt/rocm/lib/librocprofiler-sdk-roctx.so", "/opt/rocm/lib/libroctx64.so", "librocprofiler-sdk-roctx.so", "libroctx64.so"] {
                let library = match unsafe { Library::new(name) } {
                    Ok(library) => library,
                    Err(error) => {
                        last_error = Some(format!("{name}: {error}"));
                        continue;
                    }
                };
                let push = unsafe { *library.get::<RoctxRangePush>(b"roctxRangePushA\0").map_err(|error| format!("加载 roctxRangePushA: {error}"))? };
                let pop = unsafe { *library.get::<RoctxRangePop>(b"roctxRangePop\0").map_err(|error| format!("加载 roctxRangePop: {error}"))? };
                return Ok(RoctxRuntime { _library: library, push, pop });
            }
            Err(format!("未找到 ROCm ROCTx runtime: {}", last_error.unwrap_or_else(|| "没有候选库".to_owned())))
        })
        .as_ref()
        .map_err(Clone::clone)
}

pub(crate) fn roctx_stage_work_begin(label: &str) -> Result<(), String> {
    let label = CString::new(label).map_err(|_| "ROCTx stage label 含 NUL".to_owned())?;
    let runtime = runtime()?;
    let level = unsafe { (runtime.push)(label.as_ptr()) };
    if level < 0 {
        return Err(format!("roctxRangePushA 返回 {level}"));
    }
    Ok(())
}

pub(crate) fn roctx_stage_work_end() -> Result<(), String> {
    let level = unsafe { (runtime()?.pop)() };
    if level < 0 {
        return Err(format!("roctxRangePop 返回 {level}"));
    }
    Ok(())
}
