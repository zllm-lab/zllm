use crate::attention::gated_delta_net::{GatedDeltaNetHeadLayout, GatedDeltaNetInputs, GatedDeltaNetKernel, GatedDeltaNetSpec, GatedDeltaNetStorage, GatedDeltaNetWeightsRef};

use super::{BackendError, VulkanContext, VulkanTensor, VulkanWeight, binding, compute, u32_bytes};

pub struct VulkanGatedDeltaNetStorage {
    conv: wgpu::Buffer,
    recurrent: wgpu::Buffer,
    bytes: usize,
}

impl GatedDeltaNetStorage for VulkanGatedDeltaNetStorage {
    fn allocated_bytes(&self) -> usize {
        self.bytes
    }
}

impl GatedDeltaNetKernel for VulkanContext {
    type GatedDeltaNetStorage = VulkanGatedDeltaNetStorage;

    fn allocate_gated_delta_net_storage(&self, spec: &GatedDeltaNetSpec) -> Result<Self::GatedDeltaNetStorage, BackendError> {
        let conv_bytes = spec.conv_state_elements().checked_mul(size_of::<f32>()).ok_or_else(|| compute("Vulkan DeltaNet conv state 大小溢出"))?;
        let recurrent_bytes = spec.recurrent_elements().checked_mul(size_of::<f32>()).ok_or_else(|| compute("Vulkan DeltaNet recurrent state 大小溢出"))?;
        let create = |label, size| self.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: size as u64, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
        Ok(VulkanGatedDeltaNetStorage { conv: create("Vulkan DeltaNet conv state", conv_bytes), recurrent: create("Vulkan DeltaNet recurrent state", recurrent_bytes), bytes: conv_bytes + recurrent_bytes })
    }

    fn gated_delta_net_fused(
        &self,
        storage: &mut Self::GatedDeltaNetStorage,
        inputs: GatedDeltaNetInputs<'_, Self::Tensor>,
        weights: GatedDeltaNetWeightsRef<'_, Self::Weight>,
        spec: &GatedDeltaNetSpec,
    ) -> Result<Self::Tensor, BackendError> {
        self.gated_delta_net_fused_layout(storage, inputs, weights, GatedDeltaNetHeadLayout::Tiled, spec)
    }

    fn gated_delta_net_fused_layout(
        &self,
        storage: &mut Self::GatedDeltaNetStorage,
        inputs: GatedDeltaNetInputs<'_, Self::Tensor>,
        weights: GatedDeltaNetWeightsRef<'_, Self::Weight>,
        head_layout: GatedDeltaNetHeadLayout,
        spec: &GatedDeltaNetSpec,
    ) -> Result<Self::Tensor, BackendError> {
        spec.validate().map_err(compute)?;
        let GatedDeltaNetInputs { qkv, z, alpha, beta } = inputs;
        let GatedDeltaNetWeightsRef { conv, a_log, dt_bias, norm } = weights;
        let conv = f32_weight(conv, spec.conv_dim(), spec.conv_kernel, "conv")?;
        let a_log = f32_weight(a_log, 1, spec.value_heads, "a_log")?;
        let dt_bias = f32_weight(dt_bias, 1, spec.value_heads, "dt_bias")?;
        let norm = f32_weight(norm, 1, spec.value_head_dim, "norm")?;
        if qkv.rows == 0 || qkv.cols != spec.conv_dim() || z.rows != qkv.rows || z.cols != spec.value_dim() || alpha.rows != qkv.rows || alpha.cols != spec.value_heads || beta.rows != qkv.rows || beta.cols != spec.value_heads {
            return Err(compute(format!("Vulkan DeltaNet input shape 不一致: qkv=[{},{}] z=[{},{}] alpha=[{},{}] beta=[{},{}]", qkv.rows, qkv.cols, z.rows, z.cols, alpha.rows, alpha.cols, beta.rows, beta.cols)));
        }
        if spec.key_head_dim > 128 || spec.value_head_dim > 128 {
            return Err(compute(format!("Vulkan DeltaNet head dim 超过 128: key={} value={}", spec.key_head_dim, spec.value_head_dim)));
        }
        let mixed = self.output_buffer(qkv.rows, spec.conv_dim(), "Vulkan DeltaNet mixed")?;
        let core = self.output_buffer(qkv.rows, spec.value_dim(), "Vulkan DeltaNet core")?;
        let output = self.output_buffer(qkv.rows, spec.value_dim(), "Vulkan DeltaNet output")?;
        let params = [
            qkv.rows as u32,
            spec.key_dim() as u32,
            spec.value_dim() as u32,
            spec.conv_dim() as u32,
            spec.conv_kernel as u32,
            spec.key_heads as u32,
            spec.value_heads as u32,
            spec.key_head_dim as u32,
            spec.value_head_dim as u32,
            matches!(head_layout, GatedDeltaNetHeadLayout::Grouped) as u32,
            spec.rms_eps.to_bits(),
            0,
        ];
        let params_buffer = self.uniform_buffer("Vulkan DeltaNet 参数", u32_bytes(&params));
        let mut encoder = self.device.create_command_encoder(&Default::default());
        dispatch(self, &mut encoder, &self.gdn_conv_pipeline, &[&qkv.buffer, conv, &storage.conv, &mixed, &params_buffer], spec.conv_dim().div_ceil(256) as u32, 1);
        dispatch(self, &mut encoder, &self.gdn_norm_qk_pipeline, &[&mixed, &params_buffer], spec.key_heads as u32, qkv.rows as u32);
        dispatch(self, &mut encoder, &self.gdn_recurrent_pipeline, &[&mixed, &alpha.buffer, &beta.buffer, a_log, dt_bias, &storage.recurrent, &core, &params_buffer], spec.value_heads as u32, 1);
        dispatch(self, &mut encoder, &self.gdn_norm_gate_pipeline, &[&core, &z.buffer, norm, &output, &params_buffer], spec.value_heads as u32, qkv.rows as u32);
        self.queue.submit([encoder.finish()]);
        Ok(VulkanTensor { buffer: output, rows: qkv.rows, cols: spec.value_dim() })
    }
}

fn f32_weight<'a>(weight: &'a VulkanWeight, rows: usize, cols: usize, name: &str) -> Result<&'a wgpu::Buffer, BackendError> {
    match weight {
        VulkanWeight::F32 { buffer, rows: actual_rows, cols: actual_cols } if (*actual_rows, *actual_cols) == (rows, cols) => Ok(buffer),
        _ => Err(compute(format!("Vulkan DeltaNet {name} 需要 F32 [{rows},{cols}] 权重"))),
    }
}

fn dispatch(context: &VulkanContext, encoder: &mut wgpu::CommandEncoder, pipeline: &wgpu::ComputePipeline, buffers: &[&wgpu::Buffer], x: u32, y: u32) {
    let entries = buffers.iter().enumerate().map(|(index, buffer)| binding(index as u32, buffer)).collect::<Vec<_>>();
    let layout = pipeline.get_bind_group_layout(0);
    let bind_group = context.bind_group("Vulkan DeltaNet", &layout, &entries);
    let mut pass = encoder.begin_compute_pass(&Default::default());
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(x, y, 1);
}

pub const CONV_SHADER: &str = r#"
struct Params { rows:u32,key_dim:u32,value_dim:u32,conv_dim:u32,conv_kernel:u32,key_heads:u32,value_heads:u32,key_head_dim:u32,value_head_dim:u32,grouped:u32,rms_eps:f32,_pad:u32 }
@group(0) @binding(0) var<storage,read> qkv:array<f32>;
@group(0) @binding(1) var<storage,read> weight:array<f32>;
@group(0) @binding(2) var<storage,read_write> state:array<f32>;
@group(0) @binding(3) var<storage,read_write> mixed:array<f32>;
@group(0) @binding(4) var<uniform> p:Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
  let channel=id.x; if channel>=p.conv_dim{return;}
  let begin=channel*p.conv_kernel;
  for(var token=0u;token<p.rows;token+=1u){
    for(var slot=0u;slot+1u<p.conv_kernel;slot+=1u){state[begin+slot]=state[begin+slot+1u];}
    state[begin+p.conv_kernel-1u]=qkv[token*p.conv_dim+channel];
    var sum=0.0; for(var slot=0u;slot<p.conv_kernel;slot+=1u){sum+=state[begin+slot]*weight[begin+slot];}
    mixed[token*p.conv_dim+channel]=sum/(1.0+exp(-sum));
  }
}
"#;

pub const NORM_QK_SHADER: &str = r#"
struct Params { rows:u32,key_dim:u32,value_dim:u32,conv_dim:u32,conv_kernel:u32,key_heads:u32,value_heads:u32,key_head_dim:u32,value_head_dim:u32,grouped:u32,rms_eps:f32,_pad:u32 }
@group(0) @binding(0) var<storage,read_write> mixed:array<f32>;
@group(0) @binding(1) var<uniform> p:Params;
var<workgroup> qsum:array<f32,128>; var<workgroup> ksum:array<f32,128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_id) l:vec3<u32>){
  let i=l.x;let base=g.y*p.conv_dim+g.x*p.key_head_dim;
  if i<p.key_head_dim{let q=mixed[base+i];let k=mixed[base+p.key_dim+i];qsum[i]=q*q;ksum[i]=k*k;}else{qsum[i]=0.0;ksum[i]=0.0;} workgroupBarrier();
  var s=64u;loop{if i<s{qsum[i]+=qsum[i+s];ksum[i]+=ksum[i+s];}workgroupBarrier();if s==1u{break;}s/=2u;}
  if i<p.key_head_dim{mixed[base+i]*=inverseSqrt(max(qsum[0],1e-12))/sqrt(f32(p.key_head_dim));mixed[base+p.key_dim+i]*=inverseSqrt(max(ksum[0],1e-12));}
}
"#;

pub const RECURRENT_SHADER: &str = r#"
struct Params { rows:u32,key_dim:u32,value_dim:u32,conv_dim:u32,conv_kernel:u32,key_heads:u32,value_heads:u32,key_head_dim:u32,value_head_dim:u32,grouped:u32,rms_eps:f32,_pad:u32 }
@group(0) @binding(0) var<storage,read> mixed:array<f32>;
@group(0) @binding(1) var<storage,read> alpha:array<f32>;
@group(0) @binding(2) var<storage,read> beta:array<f32>;
@group(0) @binding(3) var<storage,read> a_log:array<f32>;
@group(0) @binding(4) var<storage,read> dt_bias:array<f32>;
@group(0) @binding(5) var<storage,read_write> state:array<f32>;
@group(0) @binding(6) var<storage,read_write> core:array<f32>;
@group(0) @binding(7) var<uniform> p:Params;
fn softplus(x:f32)->f32{if x>20.0{return x;}if x < -20.0{return exp(x);}return log(1.0+exp(x));}
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_id) l:vec3<u32>){
 let vh=g.x;let vc=l.x;if vc>=p.value_head_dim{return;}let kh=select(vh%p.key_heads,vh/(p.value_heads/p.key_heads),p.grouped!=0u);
 let stride=p.key_head_dim*p.value_head_dim;let state_base=vh*stride;
 for(var token=0u;token<p.rows;token+=1u){
  let decay=exp(-exp(a_log[vh])*softplus(alpha[token*p.value_heads+vh]+dt_bias[vh]));let b=1.0/(1.0+exp(-beta[token*p.value_heads+vh]));
  let qbase=token*p.conv_dim+kh*p.key_head_dim;let kbase=qbase+p.key_dim;let vbase=token*p.conv_dim+2u*p.key_dim+vh*p.value_head_dim;
  var memory=0.0;for(var kc=0u;kc<p.key_head_dim;kc+=1u){let si=state_base+kc*p.value_head_dim+vc;state[si]*=decay;memory+=state[si]*mixed[kbase+kc];}
  let delta=(mixed[vbase+vc]-memory)*b;var out=0.0;for(var kc=0u;kc<p.key_head_dim;kc+=1u){let si=state_base+kc*p.value_head_dim+vc;state[si]+=mixed[kbase+kc]*delta;out+=state[si]*mixed[qbase+kc];}
  core[token*p.value_dim+vh*p.value_head_dim+vc]=out;
 }
}
"#;

pub const NORM_GATE_SHADER: &str = r#"
struct Params { rows:u32,key_dim:u32,value_dim:u32,conv_dim:u32,conv_kernel:u32,key_heads:u32,value_heads:u32,key_head_dim:u32,value_head_dim:u32,grouped:u32,rms_eps:f32,_pad:u32 }
@group(0) @binding(0) var<storage,read> core:array<f32>;
@group(0) @binding(1) var<storage,read> z:array<f32>;
@group(0) @binding(2) var<storage,read> weight:array<f32>;
@group(0) @binding(3) var<storage,read_write> output:array<f32>;
@group(0) @binding(4) var<uniform> p:Params;
var<workgroup> sum:array<f32,128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_id) l:vec3<u32>){
 let vh=g.x;let token=g.y;let vc=l.x;let base=token*p.value_dim+vh*p.value_head_dim;
 if vc<p.value_head_dim{let x=core[base+vc];sum[vc]=x*x;}else{sum[vc]=0.0;}workgroupBarrier();var s=64u;loop{if vc<s{sum[vc]+=sum[vc+s];}workgroupBarrier();if s==1u{break;}s/=2u;}
 if vc<p.value_head_dim{let gate=z[base+vc];output[base+vc]=core[base+vc]*inverseSqrt(sum[0]/f32(p.value_head_dim)+p.rms_eps)*weight[vc]*gate/(1.0+exp(-gate));}
}
"#;
