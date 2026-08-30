//! 可选的运行时 BLAS。系统没有 OpenBLAS 时由调用方回退内置 kernel。

#[cfg(target_os = "linux")]
mod linux {
    use std::{
        ffi::{c_char, c_int, c_void},
        sync::OnceLock,
    };

    const ROW_MAJOR: c_int = 101;
    const NO_TRANS: c_int = 111;
    const TRANS: c_int = 112;
    const RTLD_NOW: c_int = 2;

    type Sgemm = unsafe extern "C" fn(c_int, c_int, c_int, c_int, c_int, c_int, f32, *const f32, c_int, *const f32, c_int, f32, *mut f32, c_int);
    type SetThreads = unsafe extern "C" fn(c_int);

    #[link(name = "dl")]
    unsafe extern "C" {
        fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }

    static SGEMM: OnceLock<Option<Sgemm>> = OnceLock::new();

    fn load() -> Option<Sgemm> {
        *SGEMM.get_or_init(|| unsafe {
            let mut handle = std::ptr::null_mut();
            for library in [b"libopenblaso.so.0\0".as_ptr(), b"libopenblas.so.0\0".as_ptr(), b"libopenblas.so\0".as_ptr()] {
                handle = dlopen(library.cast(), RTLD_NOW);
                if !handle.is_null() {
                    break;
                }
            }
            if handle.is_null() {
                return None;
            }
            let symbol = dlsym(handle, b"cblas_sgemm\0".as_ptr().cast());
            if symbol.is_null() {
                return None;
            }
            let set_threads = dlsym(handle, b"openblas_set_num_threads\0".as_ptr().cast());
            if !set_threads.is_null() {
                let set_threads: SetThreads = std::mem::transmute(set_threads);
                set_threads(1);
            }
            // Fedora 的 libopenblaso 使用 OpenMP；只限制 OpenBLAS 线程数不足以阻止
            // 每个 Rayon worker 各自创建一支 OpenMP team。
            let omp_set_dynamic = dlsym(handle, b"omp_set_dynamic\0".as_ptr().cast());
            if !omp_set_dynamic.is_null() {
                let omp_set_dynamic: SetThreads = std::mem::transmute(omp_set_dynamic);
                omp_set_dynamic(0);
            }
            let omp_set_threads = dlsym(handle, b"omp_set_num_threads\0".as_ptr().cast());
            if !omp_set_threads.is_null() {
                let omp_set_threads: SetThreads = std::mem::transmute(omp_set_threads);
                omp_set_threads(1);
            }
            Some(std::mem::transmute(symbol))
        })
    }

    pub fn available() -> bool {
        load().is_some()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sgemm_nt(m: usize, n: usize, k: usize, alpha: f32, left: &[f32], right: &[f32], out: &mut [f32]) -> bool {
        let Some(sgemm) = load() else { return false };
        unsafe {
            sgemm(ROW_MAJOR, NO_TRANS, TRANS, m as c_int, n as c_int, k as c_int, alpha, left.as_ptr(), k as c_int, right.as_ptr(), k as c_int, 0.0, out.as_mut_ptr(), n as c_int);
        }
        true
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sgemm_nn(m: usize, n: usize, k: usize, left: &[f32], right: &[f32], out: &mut [f32]) -> bool {
        let Some(sgemm) = load() else { return false };
        unsafe {
            sgemm(ROW_MAJOR, NO_TRANS, NO_TRANS, m as c_int, n as c_int, k as c_int, 1.0, left.as_ptr(), k as c_int, right.as_ptr(), n as c_int, 0.0, out.as_mut_ptr(), n as c_int);
        }
        true
    }
}

#[cfg(target_os = "linux")]
pub use linux::{available, sgemm_nn, sgemm_nt};

#[cfg(not(target_os = "linux"))]
pub fn available() -> bool {
    false
}

#[cfg(not(target_os = "linux"))]
#[allow(clippy::too_many_arguments)]
pub fn sgemm_nt(_m: usize, _n: usize, _k: usize, _alpha: f32, _left: &[f32], _right: &[f32], _out: &mut [f32]) -> bool {
    false
}

#[cfg(not(target_os = "linux"))]
#[allow(clippy::too_many_arguments)]
pub fn sgemm_nn(_m: usize, _n: usize, _k: usize, _left: &[f32], _right: &[f32], _out: &mut [f32]) -> bool {
    false
}
