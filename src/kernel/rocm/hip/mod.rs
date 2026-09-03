//! ROCm runtime FFI 封装。

use crate::attention::rope::RotaryLayout;
use crate::moe::Activation;
use libloading::{
    Library, Symbol,
    os::unix::{Library as UnixLibrary, RTLD_GLOBAL, RTLD_NOW},
};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::ffi::{CString, c_char, c_void};
use std::ptr;
use std::sync::{Mutex, OnceLock};

thread_local! {
    /// 每个执行线程可在不同 device 上选择独立 stream；未绑定时保持 legacy
    /// null stream，现有模型与 stage scheduler 的提交顺序完全不变。
    static ACTIVE_COMPUTE_STREAMS: RefCell<HashMap<i32, usize>> = RefCell::new(HashMap::new());
    static ACTIVE_DEVICE: Cell<i32> = const { Cell::new(-1) };
    // kernel launch 是 decode 热路径，不能为每次取流都查询 HashMap；跨卡
    // 映射仍保留在上面，切换 current device 时只刷新一次这个快照。
    static ACTIVE_COMPUTE_STREAM: Cell<usize> = const { Cell::new(0) };
}

pub(crate) fn activate_compute_stream(device_id: i32, stream: usize) -> Result<(), String> {
    set_device(device_id)?;
    ACTIVE_COMPUTE_STREAMS.with(|streams| {
        let mut streams = streams.borrow_mut();
        if stream == 0 {
            streams.remove(&device_id);
        } else {
            streams.insert(device_id, stream);
        }
    });
    ACTIVE_COMPUTE_STREAM.set(stream);
    Ok(())
}

pub(super) fn remember_active_device(device_id: i32) {
    ACTIVE_DEVICE.set(device_id);
    ACTIVE_COMPUTE_STREAM.set(compute_stream_for(device_id) as usize);
}

pub(crate) fn active_compute_stream() -> *mut c_void {
    ACTIVE_COMPUTE_STREAM.get() as *mut c_void
}

pub(crate) fn compute_stream_for(device_id: i32) -> *mut c_void {
    ACTIVE_COMPUTE_STREAMS.with(|streams| streams.borrow().get(&device_id).copied().unwrap_or_default() as *mut c_void)
}

/// cooperative peer 使用独立队列，避免相邻两个 stage 在同一对物理卡上形成
/// 双向 stream wait 环，同时允许本卡 stage 与替邻卡执行的 MoE 重叠。
pub(crate) fn activate_cooperative_peer_stream(owner_device_id: i32, peer_device_id: i32) -> Result<(), String> {
    let owner_stream = compute_stream_for(owner_device_id) as usize;
    if owner_stream != 0 && initialized_background_stage_stream(owner_device_id) != Some(owner_stream) {
        return Err(format!("ROCm cooperative peer 不支持 owner 自定义 stream: device={owner_device_id} stream={owner_stream:#x}"));
    }
    activate_compute_stream(peer_device_id, cooperative_peer_stream(peer_device_id)?)
}

pub(super) fn compute_workspace_key(device_id: i32) -> (i32, usize) {
    let stream = if ACTIVE_DEVICE.get() == device_id { ACTIVE_COMPUTE_STREAM.get() } else { compute_stream_for(device_id) as usize };
    (device_id, stream)
}

mod device_buffer;
mod device_profile;
#[allow(dead_code)] // 动态 HIP 表覆盖诊断与 graph API，调用点按已启用能力逐步接入。
mod ffi;
mod graph;
mod hiprtc;
mod options;
mod roctx;

pub(crate) use peer_copy::try_peer_copy_kernel_ordered;

pub(crate) use graph::{StaticGraphRecorder, StaticHipGraph, kernel_launch_trampoline};

mod attention;
mod attn_res;
mod audio;
mod block_fp8;
mod compressed_sparse;
mod gated_delta_net;
mod hyper_connection;
mod kda;
mod linear;
mod moe;
mod peer_copy;
mod radix_topk;
mod tensor;

pub use attention::*;
pub(crate) use attn_res::*;
pub use audio::*;
pub use block_fp8::*;
pub use compressed_sparse::*;
pub(crate) use gated_delta_net::*;
pub use hyper_connection::*;
pub(crate) use kda::*;
pub use linear::*;
pub use moe::*;
pub use radix_topk::*;
pub use tensor::*;

pub use device_buffer::*;
pub(crate) use device_profile::*;
pub use ffi::*;
use hiprtc::*;
pub use options::*;
pub(crate) use roctx::*;
