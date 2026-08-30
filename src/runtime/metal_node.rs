//! Metal 文本节点的资源探测与能力声明。

use std::sync::Arc;

use super::node::{DeviceDescriptor, SessionDescriptor};

pub fn text_capabilities(ctx: &crate::backend::metal::MetalContext, descriptor: SessionDescriptor<'_>) -> crate::runtime::session::NodeCapabilities {
    let accelerator = ctx.device.name();
    let unified_memory = ctx.device.has_unified_memory();
    let system_memory_bytes = crate::backend::metal::system_memory_bytes();
    super::node::text_capabilities(
        DeviceDescriptor {
            backend: "metal",
            accelerator: accelerator.clone(),
            compute_units: crate::backend::metal::gpu_compute_units(&accelerator),
            compute_unit_kind: "gpu_core",
            memory_kind: if unified_memory { "unified" } else { "dedicated" },
            unified_memory,
            system_memory_bytes,
            accelerator_memory_bytes: if unified_memory { system_memory_bytes } else { None },
            recommended_working_set_bytes: Some(ctx.device.recommended_max_working_set_size()),
        },
        descriptor,
    )
}

pub fn startup_info(ctx: &crate::backend::metal::MetalContext, capabilities: &crate::runtime::session::NodeCapabilities) -> (crate::runtime::session::NodeCapabilities, Arc<dyn Fn() -> u64 + Send + Sync>) {
    let device = ctx.device.clone();
    (capabilities.clone(), Arc::new(move || device.current_allocated_size()))
}
