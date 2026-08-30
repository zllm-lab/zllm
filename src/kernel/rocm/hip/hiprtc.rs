use super::*;

pub(super) fn resolve_library(names: &[&str]) -> Option<Library> {
    // 优先从 rocm_root/lib 整套同源加载:系统可能同时存在多套 ROCm
    // (如 ldconfig 注册旧版 hip 而 /opt/rocm 是新版),混载会引发跨版本
    // 符号解析失败(例如新版库引用旧 libhsa 缺失的符号)。
    let lib_dir = std::path::Path::new(&options().rocm_root).join("lib");
    for name in names {
        let path = lib_dir.join(name);
        if let Ok(lib) = unsafe { UnixLibrary::open(Some(path.as_os_str()), RTLD_NOW | RTLD_GLOBAL) } {
            return Some(lib.into());
        }
    }
    for name in names {
        if let Ok(lib) = unsafe { UnixLibrary::open(Some(name), RTLD_NOW | RTLD_GLOBAL) } {
            return Some(lib.into());
        }
    }
    None
}

pub(super) fn raise_rocm_open_file_limit() -> Result<(), String> {
    #[repr(C)]
    struct Rlimit {
        current: u64,
        maximum: u64,
    }
    unsafe extern "C" {
        fn getrlimit(resource: i32, limit: *mut Rlimit) -> i32;
        fn setrlimit(resource: i32, limit: *const Rlimit) -> i32;
    }

    const RLIMIT_NOFILE: i32 = 7;
    const TARGET: u64 = 65_536;
    let mut limit = Rlimit { current: 0, maximum: 0 };
    if unsafe { getrlimit(RLIMIT_NOFILE, &mut limit) } != 0 {
        return Err(format!("读取 ROCm open-file limit 失败: {}", std::io::Error::last_os_error(),));
    }
    let target = limit.maximum.min(TARGET);
    if limit.current < target {
        limit.current = target;
        if unsafe { setrlimit(RLIMIT_NOFILE, &limit) } != 0 {
            return Err(format!("提升 ROCm open-file limit 到 {target} 失败: {}", std::io::Error::last_os_error(),));
        }
    }
    Ok(())
}

pub(super) fn hip_runtime_version_tag() -> u32 {
    let Ok(runtime) = RocmRuntime::open() else { return 0 };
    type HipGetVersion = unsafe extern "C" fn(*mut i32) -> i32;
    let Ok(query): Result<Symbol<HipGetVersion>, _> = runtime.symbol(&runtime.hip, b"hipGetVersion\0") else { return 0 };
    let mut version = 0i32;
    unsafe { query(&mut version) };
    version as u32
}

/// bf16/f16/scale 换算 helper 的公共前导。此前在 ct_common/bf16_gemv/moe/
/// paged_mla/compressed_sparse/elementwise 各自复制并带不同前缀,统一为
/// zllm_* 命名后由各 module 源码头部拼接,消除逐行重复。
pub(super) const DEVICE_CONVERSIONS_PREAMBLE: &str = r#"
__device__ __forceinline__ float zllm_bf16_to_f32(unsigned short value)
{
    union { unsigned int bits; float number; } converted;
    converted.bits = ((unsigned int)value) << 16;
    return converted.number;
}

__device__ __forceinline__ unsigned short zllm_f32_to_bf16(float value)
{
    union { unsigned int bits; float number; } converted;
    converted.number = value;
    const unsigned int rounding = 0x7fffu + ((converted.bits >> 16) & 1u);
    return (unsigned short)((converted.bits + rounding) >> 16);
}

__device__ __forceinline__ float zllm_f16_to_f32(unsigned short bits)
{
    const unsigned int sign = ((unsigned int)bits & 0x8000u) << 16;
    unsigned int exponent = ((unsigned int)bits >> 10) & 0x1fu;
    unsigned int mantissa = (unsigned int)bits & 0x03ffu;
    unsigned int value;
    if (exponent == 0) {
        if (mantissa == 0) {
            value = sign;
        } else {
            exponent = 113;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                --exponent;
            }
            value = sign | (exponent << 23) | ((mantissa & 0x03ffu) << 13);
        }
    } else if (exponent == 31) {
        value = sign | 0x7f800000u | (mantissa << 13);
    } else {
        value = sign | ((exponent + 112) << 23) | (mantissa << 13);
    }
    union { unsigned int bits; float number; } converted;
    converted.bits = value;
    return converted.number;
}

__device__ __forceinline__ float zllm_scale(
    const unsigned char* scales,
    unsigned long long index,
    unsigned int scale_dtype)
{
    if (scale_dtype == 0) {
        return zllm_bf16_to_f32(((const unsigned short*)scales)[index]);
    }
    if (scale_dtype == 1) {
        return zllm_f16_to_f32(((const unsigned short*)scales)[index]);
    }
    return ((const float*)scales)[index];
}
"#;

pub(super) fn compile_hip_source(source: &str, name: &str) -> Result<Vec<u8>, String> {
    let rocm_root = options().rocm_root.clone();
    let version_tag = hip_runtime_version_tag().to_le_bytes();
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in source.bytes().chain(name.bytes()).chain(rocm_root.bytes()).chain(version_tag) {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    let cache_root = options().hiprtc_cache_dir.clone();
    let cache_path = cache_root.join(format!("{hash:016x}.co"));
    if let Ok(code) = std::fs::read(&cache_path) {
        if !code.is_empty() {
            return Ok(code);
        }
    }
    let hiprtc = resolve_library(HIPRTC_LIBRARIES).ok_or_else(|| "未找到 libhiprtc.so".to_owned())?;
    let create_program: Symbol<HiprtcCreateProgram> = unsafe { hiprtc.get(b"hiprtcCreateProgram\0") }.map_err(|error| error.to_string())?;
    let compile_program: Symbol<HiprtcCompileProgram> = unsafe { hiprtc.get(b"hiprtcCompileProgram\0") }.map_err(|error| error.to_string())?;
    let get_log_size: Symbol<HiprtcGetProgramLogSize> = unsafe { hiprtc.get(b"hiprtcGetProgramLogSize\0") }.map_err(|error| error.to_string())?;
    let get_log: Symbol<HiprtcGetProgramLog> = unsafe { hiprtc.get(b"hiprtcGetProgramLog\0") }.map_err(|error| error.to_string())?;
    let get_code_size: Symbol<HiprtcGetCodeSize> = unsafe { hiprtc.get(b"hiprtcGetCodeSize\0") }.map_err(|error| error.to_string())?;
    let get_code: Symbol<HiprtcGetCode> = unsafe { hiprtc.get(b"hiprtcGetCode\0") }.map_err(|error| error.to_string())?;
    let destroy_program: Symbol<HiprtcDestroyProgram> = unsafe { hiprtc.get(b"hiprtcDestroyProgram\0") }.map_err(|error| error.to_string())?;

    let source = CString::new(source).map_err(|error| error.to_string())?;
    let name = CString::new(name).map_err(|error| error.to_string())?;
    let mut program: HiprtcProgram = ptr::null_mut();
    let create_status = unsafe { create_program(&mut program, source.as_ptr(), name.as_ptr(), 0, ptr::null(), ptr::null()) };
    if create_status != HIPRTC_SUCCESS {
        return Err(format!("hiprtcCreateProgram 失败: code={create_status}"));
    }
    let option_values = [CString::new("--std=c++17").unwrap(), CString::new(format!("-I{rocm_root}/include")).unwrap()];
    let options = option_values.iter().map(|option| option.as_ptr()).collect::<Vec<_>>();
    let compile_status = unsafe { compile_program(program, options.len() as i32, options.as_ptr()) };
    if compile_status != HIPRTC_SUCCESS {
        let mut log_size = 0;
        let _ = unsafe { get_log_size(program, &mut log_size) };
        let mut log = vec![0u8; log_size.max(1)];
        let _ = unsafe { get_log(program, log.as_mut_ptr().cast()) };
        let _ = unsafe { destroy_program(&mut program) };
        return Err(format!("hiprtcCompileProgram 失败: code={compile_status}: {}", String::from_utf8_lossy(&log)));
    }
    let mut code_size = 0;
    let size_status = unsafe { get_code_size(program, &mut code_size) };
    if size_status != HIPRTC_SUCCESS || code_size == 0 {
        let _ = unsafe { destroy_program(&mut program) };
        return Err(format!("hiprtcGetCodeSize 失败: code={size_status} size={code_size}"));
    }
    let mut code = vec![0u8; code_size];
    let code_status = unsafe { get_code(program, code.as_mut_ptr().cast()) };
    let destroy_status = unsafe { destroy_program(&mut program) };
    if code_status != HIPRTC_SUCCESS || destroy_status != HIPRTC_SUCCESS {
        return Err(format!("HIPRTC code/destroy 失败: code={code_status} destroy={destroy_status}"));
    }
    if std::fs::create_dir_all(&cache_root).is_ok() {
        let temporary = cache_path.with_extension(format!("{}.tmp", std::process::id()));
        if std::fs::write(&temporary, &code).is_ok() {
            let _ = std::fs::rename(&temporary, &cache_path);
        }
    }
    Ok(code)
}
