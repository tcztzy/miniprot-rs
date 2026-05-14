//! GPU-accelerated DP via Metal compute shaders on Apple Silicon.
//!
//! On Apple Silicon (unified memory), we pass pointers directly — no copies.
//! The Metal shader runs the scalar DP algorithm; each GPU thread processes
//! one DP call. Batch size must be 100+ to amortize dispatch overhead (~50 μs).

#[cfg(target_os = "macos")]
mod imp {
    use std::sync::Mutex;
    use metal::*;
    use objc::rc::autoreleasepool;

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
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
        pub _pad: [u32; 3],
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
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
                .map_err(|e| eprintln!("Metal shader error: {e}")).ok()?;
            let kernel = library.get_function("dp_batch", None)
                .map_err(|e| eprintln!("Metal kernel: {e}")).ok()?;
            let pipeline = device
                .new_compute_pipeline_state_with_function(&kernel)
                .map_err(|e| eprintln!("Metal pipeline: {e:?}")).ok()?;
            Some(Mutex::new(GpuState { device, pipeline, cmd_queue }))
        })
    }

    pub fn available() -> bool {
        ensure_gpu().is_some()
    }

    /// Run batched DP on GPU. All input slices must remain alive until the
    /// returned `Vec<DpResult>` is dropped (the Metal buffers reference them).
    pub fn batch_dp(
        nas_buf: &[u8],
        aas_buf: &[u8],
        params: &[DpParams],
    ) -> Option<Vec<DpResult>> {
        let gpu = ensure_gpu().as_ref()?;
        let state = gpu.lock().ok()?;
        let n = params.len() as u64;
        if n == 0 {
            return Some(Vec::new());
        }

        autoreleasepool(|| {
            // Use StorageModeShared on Apple Silicon — zero copy with unified memory.
            let opts = MTLResourceOptions::StorageModeShared;
            let nas_gpu = state.device.new_buffer_with_data(
                nas_buf.as_ptr() as *const std::ffi::c_void,
                nas_buf.len() as u64, opts,
            );
            let aas_gpu = state.device.new_buffer_with_data(
                aas_buf.as_ptr() as *const std::ffi::c_void,
                aas_buf.len() as u64, opts,
            );
            let params_gpu = state.device.new_buffer_with_data(
                params.as_ptr() as *const std::ffi::c_void,
                (params.len() * std::mem::size_of::<DpParams>()) as u64, opts,
            );
            let results_gpu = state.device.new_buffer(
                n * std::mem::size_of::<DpResult>() as u64, opts,
            );

            let cmd_buf = state.cmd_queue.new_command_buffer();
            let enc = cmd_buf.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&state.pipeline);
            enc.set_buffer(0, Some(&nas_gpu), 0);
            enc.set_buffer(1, Some(&aas_gpu), 0);
            enc.set_buffer(2, Some(&params_gpu), 0);
            enc.set_buffer(3, Some(&results_gpu), 0);

            let tg = state.pipeline.max_total_threads_per_threadgroup().min(256);
            enc.dispatch_threads(MTLSize::new(n, 1, 1), MTLSize::new(tg, 1, 1));
            enc.end_encoding();

            cmd_buf.commit();
            cmd_buf.wait_until_completed();

            let ptr = results_gpu.contents() as *const DpResult;
            Some(unsafe { std::slice::from_raw_parts(ptr, n as usize) }.to_vec())
        })
    }

    /// Measure kernel-only time by pre-allocating and reusing buffers.
    pub fn bench_dispatch_only(
        nas_buf: &[u8],
        aas_buf: &[u8],
        params: &[DpParams],
    ) -> Option<std::time::Duration> {
        let gpu = ensure_gpu().as_ref()?;
        let state = gpu.lock().ok()?;
        let n = params.len() as u64;
        if n == 0 { return Some(std::time::Duration::ZERO); }

        autoreleasepool(|| {
            let opts = MTLResourceOptions::StorageModeShared;
            let nas_gpu = state.device.new_buffer_with_data(
                nas_buf.as_ptr() as *const std::ffi::c_void, nas_buf.len() as u64, opts,
            );
            let aas_gpu = state.device.new_buffer_with_data(
                aas_buf.as_ptr() as *const std::ffi::c_void, aas_buf.len() as u64, opts,
            );
            let params_gpu = state.device.new_buffer_with_data(
                params.as_ptr() as *const std::ffi::c_void,
                (params.len() * std::mem::size_of::<DpParams>()) as u64, opts,
            );
            let results_gpu = state.device.new_buffer(
                n * std::mem::size_of::<DpResult>() as u64, opts,
            );

            // Warmup: one dispatch
            {
                let cb = state.cmd_queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&state.pipeline);
                enc.set_buffer(0, Some(&nas_gpu), 0);
                enc.set_buffer(1, Some(&aas_gpu), 0);
                enc.set_buffer(2, Some(&params_gpu), 0);
                enc.set_buffer(3, Some(&results_gpu), 0);
                let tg = state.pipeline.max_total_threads_per_threadgroup().min(256);
                enc.dispatch_threads(MTLSize::new(n, 1, 1), MTLSize::new(tg, 1, 1));
                enc.end_encoding();
                cb.commit();
                cb.wait_until_completed();
            }

            // Timed dispatch (reusing the same buffers)
            let start = std::time::Instant::now();
            let cb = state.cmd_queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&state.pipeline);
            enc.set_buffer(0, Some(&nas_gpu), 0);
            enc.set_buffer(1, Some(&aas_gpu), 0);
            enc.set_buffer(2, Some(&params_gpu), 0);
            enc.set_buffer(3, Some(&results_gpu), 0);
            let tg = state.pipeline.max_total_threads_per_threadgroup().min(256);
            enc.dispatch_threads(MTLSize::new(n, 1, 1), MTLSize::new(tg, 1, 1));
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            Some(start.elapsed())
        })
    }
}

#[cfg(target_os = "macos")]
pub(crate) use imp::*;

#[cfg(not(target_os = "macos"))]
mod imp {
    #[repr(C)] #[derive(Clone, Copy, Debug, Default)]
    pub struct DpParams { pub nas_offset: u32, pub aas_offset: u32, pub nl: u32, pub al: u32, pub go: i32, pub ge: i32, pub io: i32, pub fs: i32, pub goe: i32, pub end_bonus: i32, pub flag: i32, pub slen: u32, pub _pad: [u32; 3] }
    #[repr(C)] #[derive(Clone, Copy, Debug, Default)]
    pub struct DpResult { pub score: i32, pub nt_len: i32, pub aa_len: i32 }
    pub fn available() -> bool { false }
    pub fn batch_dp(_nas: &[u8], _aas: &[u8], _params: &[DpParams]) -> Option<Vec<DpResult>> { None }
}
#[cfg(not(target_os = "macos"))]
pub(crate) use imp::*;

#[cfg(test)]
mod bench_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn metal_dispatch_overhead() {
        if !available() {
            eprintln!("Metal not available");
            return;
        }

        // Generate 100 DP calls for extension-like sizes
        let mut seed: u64 = 12345;
        let mut rand = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); seed };
        let bases = [b'A', b'C', b'G', b'T'];
        let aa_list = b"ACDEFGHIKLMNPQRSTVWY";
        let tables = crate::tables::make_tables(1).expect("tables");

        let mut nas_buf = Vec::new();
        let mut aas_buf = Vec::new();
        let mut params = Vec::new();

        for _ in 0..100 {
            let nl = 3000; // extension-like size
            let al = 50;
            let nas_offset = nas_buf.len() as u32;
            let aas_offset = aas_buf.len() as u32;

            let mut ns_raw = Vec::with_capacity(nl);
            for _ in 0..nl { ns_raw.push(bases[(rand() as usize) % 4]); }
            let mut nas = vec![21u8; nl];
            let mut codon = 0u8; let mut l = 0i32;
            for (i, &byte) in ns_raw.iter().enumerate() {
                let c = tables.nt4[byte as usize];
                if c < 4 { codon = ((codon << 2) | c) & 0x3f; l += 1; if l >= 3 { nas[i] = tables.codon[codon as usize]; } }
                else { codon = 0; l = 0; }
            }
            nas_buf.extend_from_slice(&nas);
            let aa: Vec<u8> = (0..al).map(|_| tables.aa20[aa_list[(rand() as usize) % 20] as usize]).collect();
            aas_buf.extend_from_slice(&aa);

            params.push(DpParams {
                nas_offset, aas_offset,
                nl: nl as u32, al: al as u32,
                go: 11, ge: 1, io: 29, fs: 23, goe: 12,
                end_bonus: 5, flag: 2, slen: 7, _pad: [0; 3],
            });
        }

        // Warmup
        let _ = batch_dp(&nas_buf, &aas_buf, &params);

        // Timed runs
        let n_runs = 10;
        let start = Instant::now();
        for _ in 0..n_runs {
            let _ = batch_dp(&nas_buf, &aas_buf, &params);
        }
        let elapsed = start.elapsed();
        let per_dispatch = elapsed / n_runs;
        let per_call = elapsed / (n_runs * 100);
        eprintln!(
            "Metal batch (100 calls × {} runs): {:?} total, {:?}/dispatch, {:?}/call",
            n_runs, elapsed, per_dispatch, per_call,
        );

        // CPU comparison
        let cpu_start = Instant::now();
        for i in 0..params.len() {
            let p = &params[i];
            let ns = &nas_buf[p.nas_offset as usize..(p.nas_offset + p.nl) as usize];
            let aa = &aas_buf[p.aas_offset as usize..(p.aas_offset + p.al) as usize];
            let opt = crate::align::NsOpt {
                flag: 2, go: 11, ge: 1, io: 29, fs: 23, xdrop: 100, end_bonus: 5,
                sp: [8, 15, 21, 30, 4, 4], sp_null_bonus: -7, ie_coef: 0.5,
                sc: &crate::tables::BLOSUM62, tables: &tables,
            };
            let _ = crate::neon_dp::global_gs16b(ns, aa, &opt, None);
        }
        let cpu_elapsed = cpu_start.elapsed();
        let cpu_per_call = cpu_elapsed / 100;
        eprintln!("CPU NEON (100 calls): {:?} total, {:?}/call", cpu_elapsed, cpu_per_call);
        eprintln!("GPU vs CPU per call: {:?} vs {:?}", per_call, cpu_per_call);
    }
}
