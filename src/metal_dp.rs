//! GPU-accelerated DP via Metal compute shaders on Apple Silicon.
//!
//! On Apple Silicon (unified memory), we pass pointers directly — zero copy.
//! One GPU thread per DP call, scalar DP. Batch size 256+ to amortize ~86ms
//! dispatch overhead.

#[cfg(target_os = "macos")]
mod imp {
    use std::sync::Mutex;
    use metal::*;
    use objc::rc::autoreleasepool;

    /// Must match dp.metal DpParams exactly (48 bytes, 4-byte aligned).
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
    pub struct DpParams {
        pub nas_offset: u32,
        pub aas_offset: u32,
        pub nl: u32,
        pub al: u32,
        pub go: i32,
        pub ge: i32,
        pub io: i32,
        pub fs: i32,
        pub goe: i32,
        pub end_bonus: i32,
        pub flag: i32,
        pub slen: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
    pub struct DpResult {
        pub score: i32,
        pub nt_len: i32,
        pub aa_len: i32,
    }

    struct GpuState {
        device: Device,
        pipeline: ComputePipelineState,
        cmd_queue: CommandQueue,
    }

    static GPU: std::sync::OnceLock<Option<Mutex<GpuState>>> = std::sync::OnceLock::new();

    fn ensure_gpu() -> &'static Option<Mutex<GpuState>> {
        GPU.get_or_init(|| {
            let device = Device::system_default()?;
            let cmd_queue = device.new_command_queue();
            let source = include_str!("dp.metal");
            let library = device
                .new_library_with_source(source, &CompileOptions::new())
                .map_err(|e| eprintln!("Metal shader compile error: {e}"))
                .ok()?;
            let kernel = library
                .get_function("dp_batch", None)
                .map_err(|e| eprintln!("Metal kernel not found: {e}"))
                .ok()?;
            let pipeline = device
                .new_compute_pipeline_state_with_function(&kernel)
                .map_err(|e| eprintln!("Metal pipeline error: {e:?}"))
                .ok()?;
            Some(Mutex::new(GpuState {
                device,
                pipeline,
                cmd_queue,
            }))
        })
    }

    pub fn available() -> bool {
        ensure_gpu().is_some()
    }

    /// Run batched DP with default BLOSUM62 matrix.
    pub fn batch_dp(
        nas_buf: &[u8],
        aas_buf: &[u8],
        params: &[DpParams],
    ) -> Option<Vec<DpResult>> {
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
        let n = params.len() as u64;
        if n == 0 {
            return Some(Vec::new());
        }

        let flat_matrix: [i8; 484] = unsafe { std::mem::transmute(*matrix) };

        autoreleasepool(|| {
            let opts = MTLResourceOptions::StorageModeShared;
            let nas_gpu = state.device.new_buffer_with_data(
                nas_buf.as_ptr() as *const std::ffi::c_void,
                nas_buf.len() as u64,
                opts,
            );
            let aas_gpu = state.device.new_buffer_with_data(
                aas_buf.as_ptr() as *const std::ffi::c_void,
                aas_buf.len() as u64,
                opts,
            );
            let params_gpu = state.device.new_buffer_with_data(
                params.as_ptr() as *const std::ffi::c_void,
                params.len() as u64 * std::mem::size_of::<DpParams>() as u64,
                opts,
            );
            let results_gpu = state.device.new_buffer(
                n * std::mem::size_of::<DpResult>() as u64,
                opts,
            );
            let matrix_gpu = state.device.new_buffer_with_data(
                flat_matrix.as_ptr() as *const std::ffi::c_void,
                std::mem::size_of_val(&flat_matrix) as u64,
                opts,
            );

            let cmd_buf = state.cmd_queue.new_command_buffer();
            let enc = cmd_buf.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&state.pipeline);
            enc.set_buffer(0, Some(&nas_gpu), 0);
            enc.set_buffer(1, Some(&aas_gpu), 0);
            enc.set_buffer(2, Some(&params_gpu), 0);
            enc.set_buffer(3, Some(&results_gpu), 0);
            enc.set_buffer(4, Some(&matrix_gpu), 0);

            let tg = state
                .pipeline
                .max_total_threads_per_threadgroup()
                .min(64);
            enc.dispatch_threads(MTLSize::new(n, 1, 1), MTLSize::new(tg, 1, 1));
            enc.end_encoding();

            cmd_buf.commit();
            cmd_buf.wait_until_completed();

            let ptr = results_gpu.contents() as *const DpResult;
            Some(unsafe { std::slice::from_raw_parts(ptr, n as usize) }.to_vec())
        })
    }

    /// Measure kernel-only time: pre-allocate buffers, warmup, then timed dispatch.
    pub fn bench_dispatch_only(
        nas_buf: &[u8],
        aas_buf: &[u8],
        params: &[DpParams],
        matrix: &[[i8; 22]; 22],
    ) -> Option<(std::time::Duration, std::time::Duration)> {
        let gpu = ensure_gpu().as_ref()?;
        let state = gpu.lock().ok()?;
        let n = params.len() as u64;
        if n == 0 {
            return Some((std::time::Duration::ZERO, std::time::Duration::ZERO));
        }

        let flat_matrix: [i8; 484] = unsafe { std::mem::transmute(*matrix) };

        autoreleasepool(|| {
            let opts = MTLResourceOptions::StorageModeShared;
            let nas_gpu = state.device.new_buffer_with_data(
                nas_buf.as_ptr() as *const std::ffi::c_void,
                nas_buf.len() as u64,
                opts,
            );
            let aas_gpu = state.device.new_buffer_with_data(
                aas_buf.as_ptr() as *const std::ffi::c_void,
                aas_buf.len() as u64,
                opts,
            );
            let params_gpu = state.device.new_buffer_with_data(
                params.as_ptr() as *const std::ffi::c_void,
                params.len() as u64 * std::mem::size_of::<DpParams>() as u64,
                opts,
            );
            let results_gpu = state.device.new_buffer(
                n * std::mem::size_of::<DpResult>() as u64,
                opts,
            );
            let matrix_gpu = state.device.new_buffer_with_data(
                flat_matrix.as_ptr() as *const std::ffi::c_void,
                std::mem::size_of_val(&flat_matrix) as u64,
                opts,
            );

            let do_dispatch = || {
                let cb = state.cmd_queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&state.pipeline);
                enc.set_buffer(0, Some(&nas_gpu), 0);
                enc.set_buffer(1, Some(&aas_gpu), 0);
                enc.set_buffer(2, Some(&params_gpu), 0);
                enc.set_buffer(3, Some(&results_gpu), 0);
                enc.set_buffer(4, Some(&matrix_gpu), 0);
                let tg = state
                    .pipeline
                    .max_total_threads_per_threadgroup()
                    .min(64);
                enc.dispatch_threads(MTLSize::new(n, 1, 1), MTLSize::new(tg, 1, 1));
                enc.end_encoding();
                cb.commit();
                cb.wait_until_completed();
            };

            // Warmup
            do_dispatch();
            let warmup_start = std::time::Instant::now();
            do_dispatch();
            let warmup = warmup_start.elapsed();

            // Timed
            let start = std::time::Instant::now();
            do_dispatch();
            let elapsed = start.elapsed();

            Some((warmup, elapsed))
        })
    }
}

#[cfg(target_os = "macos")]
pub(crate) use imp::*;

#[cfg(not(target_os = "macos"))]
mod imp {
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
    pub struct DpParams {
        pub nas_offset: u32, pub aas_offset: u32, pub nl: u32, pub al: u32,
        pub go: i32, pub ge: i32, pub io: i32, pub fs: i32,
        pub goe: i32, pub end_bonus: i32, pub flag: i32, pub slen: u32,
    }
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
    pub struct DpResult { pub score: i32, pub nt_len: i32, pub aa_len: i32 }
    pub fn available() -> bool { false }
    pub fn batch_dp(_nas: &[u8], _aas: &[u8], _params: &[DpParams]) -> Option<Vec<DpResult>> { None }
    pub fn batch_dp_with_matrix(_nas: &[u8], _aas: &[u8], _params: &[DpParams], _matrix: &[[i8; 22]; 22]) -> Option<Vec<DpResult>> { None }
    pub fn bench_dispatch_only(_nas: &[u8], _aas: &[u8], _params: &[DpParams], _matrix: &[[i8; 22]; 22]) -> Option<(std::time::Duration, std::time::Duration)> { None }
}
#[cfg(not(target_os = "macos"))]
pub(crate) use imp::*;
