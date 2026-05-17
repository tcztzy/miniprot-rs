//! CUDA DP backend for NVIDIA GPUs.
//!
//! Built only with `--features cuda`. The CUDA C++ translation mirrors the
//! optimized Metal scalar kernel: one GPU thread per DP call, pointer-rotated
//! DP rows, and per-thread amino-acid/score-row caches.

use crate::metal_dp::{DpParams, DpResult};

#[cfg(feature = "cuda")]
#[allow(dead_code)]
mod imp {
    use super::{DpParams, DpResult};
    use std::{ffi::c_void, ptr::NonNull, sync::Mutex, time::Duration};

    static CUDA_LOCK: Mutex<()> = Mutex::new(());

    unsafe extern "C" {
        fn miniprot_cuda_available() -> i32;
        fn miniprot_cuda_batch_dp(
            nas: *const u8,
            nas_len: usize,
            aas: *const u8,
            aas_len: usize,
            params: *const DpParams,
            n: usize,
            matrix: *const i8,
            results: *mut DpResult,
        ) -> i32;
        fn miniprot_cuda_bench_dispatch_only(
            nas: *const u8,
            nas_len: usize,
            aas: *const u8,
            aas_len: usize,
            params: *const DpParams,
            n: usize,
            matrix: *const i8,
            warmup_ms: *mut f32,
            timed_ms: *mut f32,
        ) -> i32;
        fn miniprot_cuda_prepared_batch_create(
            nas: *const u8,
            nas_len: usize,
            aas: *const u8,
            aas_len: usize,
            params: *const DpParams,
            n: usize,
            matrix: *const i8,
            out: *mut *mut c_void,
        ) -> i32;
        fn miniprot_cuda_prepared_batch_run(batch: *mut c_void, results: *mut DpResult) -> i32;
        fn miniprot_cuda_prepared_batch_destroy(batch: *mut c_void);
    }

    pub fn available() -> bool {
        unsafe { miniprot_cuda_available() != 0 }
    }

    pub fn batch_dp(nas_buf: &[u8], aas_buf: &[u8], params: &[DpParams]) -> Option<Vec<DpResult>> {
        batch_dp_with_matrix(nas_buf, aas_buf, params, &crate::tables::BLOSUM62)
    }

    pub fn batch_dp_with_matrix(
        nas_buf: &[u8],
        aas_buf: &[u8],
        params: &[DpParams],
        matrix: &[[i8; 22]; 22],
    ) -> Option<Vec<DpResult>> {
        if params.is_empty() {
            return Some(Vec::new());
        }
        let flat_matrix = matrix.as_flattened();
        let mut results = vec![DpResult::default(); params.len()];
        let _guard = CUDA_LOCK.lock().ok()?;
        let code = unsafe {
            miniprot_cuda_batch_dp(
                nas_buf.as_ptr(),
                nas_buf.len(),
                aas_buf.as_ptr(),
                aas_buf.len(),
                params.as_ptr(),
                params.len(),
                flat_matrix.as_ptr(),
                results.as_mut_ptr(),
            )
        };
        (code == 0).then_some(results)
    }

    pub fn bench_dispatch_only(
        nas_buf: &[u8],
        aas_buf: &[u8],
        params: &[DpParams],
        matrix: &[[i8; 22]; 22],
    ) -> Option<(Duration, Duration)> {
        if params.is_empty() {
            return Some((Duration::ZERO, Duration::ZERO));
        }
        let flat_matrix = matrix.as_flattened();
        let mut warmup_ms = 0.0f32;
        let mut timed_ms = 0.0f32;
        let _guard = CUDA_LOCK.lock().ok()?;
        let code = unsafe {
            miniprot_cuda_bench_dispatch_only(
                nas_buf.as_ptr(),
                nas_buf.len(),
                aas_buf.as_ptr(),
                aas_buf.len(),
                params.as_ptr(),
                params.len(),
                flat_matrix.as_ptr(),
                &mut warmup_ms,
                &mut timed_ms,
            )
        };
        (code == 0).then(|| {
            (
                Duration::from_secs_f64(warmup_ms as f64 / 1000.0),
                Duration::from_secs_f64(timed_ms as f64 / 1000.0),
            )
        })
    }

    pub struct PreparedBatch {
        raw: NonNull<c_void>,
        len: usize,
    }

    impl PreparedBatch {
        pub fn new(
            nas_buf: &[u8],
            aas_buf: &[u8],
            params: &[DpParams],
            matrix: &[[i8; 22]; 22],
        ) -> Option<Self> {
            if params.is_empty() {
                return None;
            }
            let flat_matrix = matrix.as_flattened();
            let mut raw = std::ptr::null_mut();
            let _guard = CUDA_LOCK.lock().ok()?;
            let code = unsafe {
                miniprot_cuda_prepared_batch_create(
                    nas_buf.as_ptr(),
                    nas_buf.len(),
                    aas_buf.as_ptr(),
                    aas_buf.len(),
                    params.as_ptr(),
                    params.len(),
                    flat_matrix.as_ptr(),
                    &mut raw,
                )
            };
            if code != 0 {
                return None;
            }
            Some(Self {
                raw: NonNull::new(raw)?,
                len: params.len(),
            })
        }

        pub fn run(&self) -> Option<Vec<DpResult>> {
            let mut results = vec![DpResult::default(); self.len];
            let _guard = CUDA_LOCK.lock().ok()?;
            let code = unsafe {
                miniprot_cuda_prepared_batch_run(self.raw.as_ptr(), results.as_mut_ptr())
            };
            (code == 0).then_some(results)
        }
    }

    impl Drop for PreparedBatch {
        fn drop(&mut self) {
            let _guard = CUDA_LOCK.lock().ok();
            unsafe { miniprot_cuda_prepared_batch_destroy(self.raw.as_ptr()) };
        }
    }
}

#[cfg(not(feature = "cuda"))]
#[allow(dead_code)]
mod imp {
    use super::{DpParams, DpResult};

    pub fn available() -> bool {
        false
    }

    pub fn batch_dp(
        _nas_buf: &[u8],
        _aas_buf: &[u8],
        _params: &[DpParams],
    ) -> Option<Vec<DpResult>> {
        None
    }

    pub fn batch_dp_with_matrix(
        _nas_buf: &[u8],
        _aas_buf: &[u8],
        _params: &[DpParams],
        _matrix: &[[i8; 22]; 22],
    ) -> Option<Vec<DpResult>> {
        None
    }

    pub fn bench_dispatch_only(
        _nas_buf: &[u8],
        _aas_buf: &[u8],
        _params: &[DpParams],
        _matrix: &[[i8; 22]; 22],
    ) -> Option<(std::time::Duration, std::time::Duration)> {
        None
    }

    pub struct PreparedBatch;

    impl PreparedBatch {
        pub fn new(
            _nas_buf: &[u8],
            _aas_buf: &[u8],
            _params: &[DpParams],
            _matrix: &[[i8; 22]; 22],
        ) -> Option<Self> {
            None
        }

        pub fn run(&self) -> Option<Vec<DpResult>> {
            None
        }
    }
}

#[allow(unused_imports)]
pub(crate) use imp::*;
