//! GPU-accelerated DP via Metal compute shaders on Apple Silicon.
//! Batches multiple independent DP calls into one GPU dispatch.

#[cfg(target_os = "macos")]
mod imp {
    use std::sync::OnceLock;
    use metal::*;
    use objc::rc::autoreleasepool;

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
    pub struct DpParams {
        pub nas_offset: u32,
        pub aas_offset: u32,
        pub donor_offset: u32,
        pub acceptor_offset: u32,
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
        pub _pad: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default)]
    pub struct DpResult {
        pub score: i32,
        pub nt_len: i32,
        pub aa_len: i32,
        pub _pad: i32,
    }

    struct MetalState {
        device: Device,
        pipeline: ComputePipelineState,
    }

    static METAL_STATE: OnceLock<Option<MetalState>> = OnceLock::new();

    fn state() -> &'static Option<MetalState> {
        METAL_STATE.get_or_init(|| {
            let device = Device::system_default()?;
            let source = include_str!("dp.metal");
            let library = match device.new_library_with_source(source, &CompileOptions::new()) {
                Ok(lib) => lib,
                Err(e) => {
                    eprintln!("Metal shader compile error: {e:?}");
                    return None;
                }
            };
            let kernel = match library.get_function("dp_batch", None) {
                Ok(k) => k,
                Err(e) => {
                    eprintln!("Metal kernel not found: {e}");
                    return None;
                }
            };
            let pipeline = match device.new_compute_pipeline_state_with_function(&kernel) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("Metal pipeline error: {e:?}");
                    return None;
                }
            };
            Some(MetalState { device, pipeline })
        })
    }

    pub fn available() -> bool {
        state().is_some()
    }

    pub fn batch_dp(
        nas_buf: &[u8],
        aas_buf: &[u8],
        donor_buf: &[i16],
        acceptor_buf: &[i16],
        params: &[DpParams],
    ) -> Option<Vec<DpResult>> {
        let st = state().as_ref()?;
        let n = params.len() as u64;
        if n == 0 {
            return Some(Vec::new());
        }

        autoreleasepool(|| {
            let nas_gpu = st.device.new_buffer_with_data(
                nas_buf.as_ptr() as *const std::ffi::c_void,
                nas_buf.len() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let aas_gpu = st.device.new_buffer_with_data(
                aas_buf.as_ptr() as *const std::ffi::c_void,
                aas_buf.len() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let params_gpu = st.device.new_buffer_with_data(
                params.as_ptr() as *const std::ffi::c_void,
                std::mem::size_of_val(params) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let results_gpu = st.device.new_buffer(
                n * std::mem::size_of::<DpResult>() as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let cmd_queue = st.device.new_command_queue();
            let cmd_buffer = cmd_queue.new_command_buffer();
            let encoder = cmd_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&st.pipeline);
            encoder.set_buffer(0, Some(&nas_gpu), 0);
            encoder.set_buffer(1, Some(&aas_gpu), 0);
            encoder.set_buffer(2, Some(&params_gpu), 0);
            encoder.set_buffer(3, Some(&results_gpu), 0);

            let tg = st.pipeline.max_total_threads_per_threadgroup().min(256);
            encoder.dispatch_threads(MTLSize::new(n, 1, 1), MTLSize::new(tg, 1, 1));
            encoder.end_encoding();

            cmd_buffer.commit();
            cmd_buffer.wait_until_completed();

            let ptr = results_gpu.contents() as *const DpResult;
            let results: Vec<DpResult> =
                unsafe { std::slice::from_raw_parts(ptr, n as usize) }.to_vec();
            Some(results)
        })
    }
}

#[cfg(target_os = "macos")]
pub(crate) use imp::*;

#[cfg(not(target_os = "macos"))]
mod imp {
    use std::time::Duration;
    #[repr(C)] #[derive(Clone, Copy, Debug, Default)]
    pub struct DpParams { pub nas_offset: u32, pub aas_offset: u32, pub donor_offset: u32, pub acceptor_offset: u32, pub nl: u32, pub al: u32, pub go: i32, pub ge: i32, pub io: i32, pub fs: i32, pub goe: i32, pub end_bonus: i32, pub flag: i32, pub slen: u32, pub _pad: u32 }
    #[repr(C)] #[derive(Clone, Copy, Debug, Default)]
    pub struct DpResult { pub score: i32, pub nt_len: i32, pub aa_len: i32, pub _pad: i32 }
    pub fn available() -> bool { false }
    pub fn batch_dp(_nas: &[u8], _aas: &[u8], _donor: &[i16], _acceptor: &[i16], _params: &[DpParams]) -> Option<Vec<DpResult>> { None }
}
#[cfg(not(target_os = "macos"))]
pub(crate) use imp::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::{make_tables, BLOSUM62};
    use crate::align::NsOpt;
    use crate::neon_dp;

    fn prep_nas(ns: &[u8], tables: &crate::tables::Tables) -> Vec<u8> {
        let mut nas = vec![21u8; ns.len()];
        let mut codon = 0u8;
        let mut len = 0i32;
        for (i, &byte) in ns.iter().enumerate() {
            let c = tables.nt4[byte as usize];
            if c < 4 {
                codon = ((codon << 2) | c) & 0x3f;
                len += 1;
                if len >= 3 { nas[i] = tables.codon[codon as usize]; }
            } else { codon = 0; len = 0; }
        }
        nas
    }

    #[test]
    fn metal_dp_matches_neon() {
        if !super::available() {
            eprintln!("Metal not available, skipping");
            return;
        }

        let tables = make_tables(1).expect("tables");
        let sc = &BLOSUM62;

        // Simple LCG random generator
        let mut seed: u64 = 99;
        let mut rand = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            seed
        };
        let num_cases = 50;
        let mut nas_buf = Vec::new();
        let mut aas_buf = Vec::new();
        let mut donor_buf = Vec::new();
        let mut acceptor_buf = Vec::new();
        let mut params = Vec::with_capacity(num_cases);

        let bases = [b'A', b'C', b'G', b'T'];
        let aa_list = b"ACDEFGHIKLMNPQRSTVWY";

        // Start with a single trivial test case
        {
            // One codon (ACG = Threonine) vs one amino acid (T = 16)
            let ns_raw = b"ACG".to_vec();
            let nas = prep_nas(&ns_raw, &tables);
            let al = 1;
            let nl = 3;
            let aas = vec![tables.aa20[b'T' as usize]];
            let donor = vec![30i16; nl + 1];
            let acceptor = vec![30i16; nl + 1];

            let metal_params = vec![super::DpParams {
                nas_offset: 0, aas_offset: 0, donor_offset: 0, acceptor_offset: 0,
                nl: 3, al: 1, go: 11, ge: 1, io: 29, fs: 23, goe: 12,
                end_bonus: 5, flag: 0, slen: 1, _pad: 0,
            }];
            let metal_r = super::batch_dp(&nas, &aas, &donor, &acceptor, &metal_params).expect("batch");

            let opt = NsOpt {
                flag: 0, go: 11, ge: 1, io: 29, fs: 23,
                xdrop: 100, end_bonus: 5,
                sp: [8, 15, 21, 30, 4, 4],
                sp_null_bonus: -7, ie_coef: 0.5,
                sc, tables: &tables,
            };
            let cpu_r = neon_dp::global_gs16b(&ns_raw, b"T", &opt, None);
            eprintln!("Test 1 codon vs 1 aa: CPU={} GPU={} nas={:?} aas={:?}", cpu_r.score, metal_r[0].score, nas, aas);
        }

        for _ in 0..num_cases {
            let nl = (rand() as usize % 190) + 10;
            let al = (rand() as usize % 97) + 3;
            if al > 256 { continue; }

            let nas_offset = nas_buf.len() as u32;
            let aas_offset = aas_buf.len() as u32;
            let donor_offset = donor_buf.len() as u32;
            let acceptor_offset = acceptor_buf.len() as u32;

            // Generate raw nt bytes
            let mut ns_raw = Vec::with_capacity(nl);
            for _ in 0..nl { ns_raw.push(bases[(rand() as usize) % 4]); }
            // Translate to amino acid codes (what Metal expects)
            let nas = prep_nas(&ns_raw, &tables);
            nas_buf.extend_from_slice(&nas);
            // Convert aa chars to numeric codes
            let aa: Vec<u8> = (0..al).map(|_| {
                let b = aa_list[(rand() as usize) % 20];
                tables.aa20[b as usize]
            }).collect();
            aas_buf.extend_from_slice(&aa);
            for _ in 0..=nl { donor_buf.push(30i16); acceptor_buf.push(30i16); }

            params.push(super::DpParams {
                nas_offset, aas_offset, donor_offset, acceptor_offset,
                nl: nl as u32, al: al as u32,
                go: 11, ge: 1, io: 29, fs: 23, goe: 12,
                end_bonus: 5, flag: 0,
                slen: (al as u32).div_ceil(8), _pad: 0,
            });
        }

        // Run GPU batch
        let gpu_results = super::batch_dp(&nas_buf, &aas_buf, &donor_buf, &acceptor_buf, &params)
            .expect("batch_dp should succeed");

        // Compare with NEON CPU
        let mut mismatches = 0;
        for i in 0..params.len() {
            let p = &params[i];
            let ns = &nas_buf[p.nas_offset as usize..(p.nas_offset + p.nl) as usize];
            let aa = &aas_buf[p.aas_offset as usize..(p.aas_offset + p.al) as usize];

            let opt = NsOpt {
                flag: p.flag, go: p.go, ge: p.ge, io: p.io, fs: p.fs,
                xdrop: 100, end_bonus: p.end_bonus,
                sp: [8, 15, 21, 30, 4, 4],
                sp_null_bonus: -7, ie_coef: 0.5,
                sc, tables: &tables,
            };
            let cpu_result = neon_dp::global_gs16b(ns, aa, &opt, None);

            if cpu_result.score != gpu_results[i].score && mismatches < 3 {
                eprintln!(
                    "Mismatch {}: nl={} al={} NEON={} GPU={}",
                    i, p.nl, p.al, cpu_result.score, gpu_results[i].score,
                );
            }
            if cpu_result.score != gpu_results[i].score {
                mismatches += 1;
            }
        }
        if mismatches > 0 {
            panic!("{} mismatches out of {}", mismatches, params.len());
        }
        eprintln!("All {} calls match between GPU and CPU", params.len());
    }
}
