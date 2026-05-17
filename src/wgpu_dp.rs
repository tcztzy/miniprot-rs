//! GPU-accelerated DP via wgpu compute shaders.
//!
//! Cross-platform: uses Metal on macOS, Vulkan on Linux, DX12 on Windows.
//! One GPU thread per DP call — batched dispatch amortizes overhead.

use std::sync::Mutex;
use wgpu::util::DeviceExt;

// Re-use metal_dp types for compatibility across backends.
pub(crate) use crate::metal_dp::{DpParams, DpResult};

struct GpuState {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

static GPU: std::sync::OnceLock<Option<Mutex<GpuState>>> = std::sync::OnceLock::new();

async fn init_gpu() -> Option<GpuState> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await;
    let adapter = match adapter {
        Some(a) => a,
        None => {
            eprintln!("wgpu: no adapter found");
            return None;
        }
    };
    eprintln!(
        "wgpu: adapter={:?}, backend={:?}",
        adapter.get_info().name,
        adapter.get_info().backend,
    );
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("miniprot dp device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            },
            None,
        )
        .await
        .ok()?;

    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("dp shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("dp.wgsl").into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("dp bind group layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("dp pipeline layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("dp compute pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    Some(GpuState {
        device,
        queue,
        pipeline,
        bind_group_layout,
    })
}

fn ensure_gpu() -> &'static Option<Mutex<GpuState>> {
    GPU.get_or_init(|| match pollster::block_on(init_gpu()) {
        Some(state) => Some(Mutex::new(state)),
        None => {
            eprintln!("wgpu: failed to initialize GPU adapter/device");
            None
        }
    })
}

pub fn available() -> bool {
    ensure_gpu().is_some()
}

fn create_bind_group(
    state: &GpuState,
    nas_buf: &wgpu::Buffer,
    aas_buf: &wgpu::Buffer,
    params_buf: &wgpu::Buffer,
    results_buf: &wgpu::Buffer,
    matrix_buf: &wgpu::Buffer,
) -> wgpu::BindGroup {
    state.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("dp bind group"),
        layout: &state.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: nas_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: aas_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: results_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: matrix_buf.as_entire_binding(),
            },
        ],
    })
}

fn create_storage_buffer<T: bytemuck::Pod>(
    device: &wgpu::Device,
    data: &[T],
    label: &str,
) -> wgpu::Buffer {
    let bytes = bytemuck::cast_slice(data);
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    })
}

fn create_storage_buffer_uninit(device: &wgpu::Device, size: u64, label: &str) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Run batched DP on GPU with default BLOSUM62 matrix.
pub fn batch_dp(nas_buf: &[u8], aas_buf: &[u8], params: &[DpParams]) -> Option<Vec<DpResult>> {
    batch_dp_with_matrix(nas_buf, aas_buf, params, &crate::tables::BLOSUM62)
}

/// Run batched DP with custom scoring matrix (22x22, row-major i8).
pub fn batch_dp_with_matrix(
    nas_buf: &[u8],
    aas_buf: &[u8],
    params: &[DpParams],
    matrix: &[[i8; 22]; 22],
) -> Option<Vec<DpResult>> {
    let gpu = ensure_gpu().as_ref()?;
    let state = gpu.lock().ok()?;
    let n = params.len() as u32;
    if n == 0 {
        return Some(Vec::new());
    }

    // Widen u8→u32, i8→i32 for WGSL (no u8/i8 support)
    let nas_u32: Vec<u32> = nas_buf.iter().map(|&b| b as u32).collect();
    let aas_u32: Vec<u32> = aas_buf.iter().map(|&b| b as u32).collect();
    let matrix_i32: Vec<i32> = {
        let flat: [i8; 484] = unsafe { std::mem::transmute(*matrix) };
        flat.iter().map(|&b| b as i32).collect()
    };

    let nas_gpu = create_storage_buffer(&state.device, &nas_u32, "nas");
    let aas_gpu = create_storage_buffer(&state.device, &aas_u32, "aas");
    let params_gpu = create_storage_buffer(&state.device, params, "params");
    let results_gpu = create_storage_buffer_uninit(
        &state.device,
        (n as u64) * std::mem::size_of::<DpResult>() as u64,
        "results",
    );
    let matrix_gpu = create_storage_buffer(&state.device, &matrix_i32, "matrix");

    let bind_group = create_bind_group(
        &state,
        &nas_gpu,
        &aas_gpu,
        &params_gpu,
        &results_gpu,
        &matrix_gpu,
    );

    // Staging buffer for readback
    let staging = state.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dp staging"),
        size: (n as u64) * std::mem::size_of::<DpResult>() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Encode compute pass
    let mut encoder = state
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dp encoder"),
        });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("dp compute pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&state.pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups((n + 63) / 64, 1, 1);
    }
    encoder.copy_buffer_to_buffer(
        &results_gpu,
        0,
        &staging,
        0,
        (n as u64) * std::mem::size_of::<DpResult>() as u64,
    );

    // Submit and wait
    state.queue.submit(Some(encoder.finish()));

    // Map staging buffer and read results
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).ok();
    });
    state.device.poll(wgpu::Maintain::Wait);
    rx.recv().ok()?.ok()?;

    let data = slice.get_mapped_range();
    let results: Vec<DpResult> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();

    Some(results)
}

/// Measure kernel-only time (returns None if GPU unavailable).
pub fn bench_dispatch_only(
    nas_buf: &[u8],
    aas_buf: &[u8],
    params: &[DpParams],
    matrix: &[[i8; 22]; 22],
) -> Option<(std::time::Duration, std::time::Duration)> {
    let gpu = ensure_gpu().as_ref()?;
    let state = gpu.lock().ok()?;
    let n = params.len() as u32;
    if n == 0 {
        return Some((std::time::Duration::ZERO, std::time::Duration::ZERO));
    }

    let nas_u32: Vec<u32> = nas_buf.iter().map(|&b| b as u32).collect();
    let aas_u32: Vec<u32> = aas_buf.iter().map(|&b| b as u32).collect();
    let matrix_i32: Vec<i32> = {
        let flat: [i8; 484] = unsafe { std::mem::transmute(*matrix) };
        flat.iter().map(|&b| b as i32).collect()
    };
    let nas_gpu = create_storage_buffer(&state.device, &nas_u32, "nas");
    let aas_gpu = create_storage_buffer(&state.device, &aas_u32, "aas");
    let params_gpu = create_storage_buffer(&state.device, params, "params");
    let results_gpu = create_storage_buffer_uninit(
        &state.device,
        (n as u64) * std::mem::size_of::<DpResult>() as u64,
        "results",
    );
    let matrix_gpu = create_storage_buffer(&state.device, &matrix_i32, "matrix");

    let bind_group = create_bind_group(
        &state,
        &nas_gpu,
        &aas_gpu,
        &params_gpu,
        &results_gpu,
        &matrix_gpu,
    );

    let do_dispatch = || {
        let mut encoder = state
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("dp encoder"),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("dp compute pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&state.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups((n + 63) / 64, 1, 1);
        }
        state.queue.submit(Some(encoder.finish()));
        state.device.poll(wgpu::Maintain::Wait);
    };

    // Warmup
    do_dispatch();
    let warmup_start = std::time::Instant::now();
    do_dispatch();
    let warmup = warmup_start.elapsed();

    let start = std::time::Instant::now();
    do_dispatch();
    let elapsed = start.elapsed();

    Some((warmup, elapsed))
}
