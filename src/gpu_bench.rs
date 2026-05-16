//! GPU DP benchmark harness — compares Metal, wgpu, and CPU NEON backends.
//!
//! Run with: `cargo test --release gpu_bench -- --nocapture`

use crate::metal_dp::{self, DpParams, DpResult};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Test data generation
// ---------------------------------------------------------------------------

struct BenchData {
    nas_buf: Vec<u8>,
    aas_buf: Vec<u8>,
    params: Vec<DpParams>,
    /// Raw nucleotide sequences (for CPU DP which does its own translation)
    ns_raw: Vec<Vec<u8>>,
    /// Raw amino acid sequences (for CPU DP which does its own encoding)
    aa_raw: Vec<Vec<u8>>,
}

/// Generate synthetic DP workloads.
/// Uses only A bases for nucleotides (avoids all splice dinucleotide patterns:
/// GT, GC, AT, AG, AC). Random amino acids. GPU shaders lack splice signal
/// handling, so splice-free input guarantees CPU==GPU for correctness validation.
fn generate_workload(
    n_calls: usize,
    nl: usize,
    al: usize,
    is_ext: bool,
    rng: &mut impl FnMut() -> u64,
) -> BenchData {
    // Single base type avoids all splice patterns.
    let nt_base = b'A';
    let aa_chars = b"ACDEFGHIKLMNPQRSTVWY";
    let tables = crate::tables::make_tables(1).expect("tables");

    let mut nas_buf = Vec::new();
    let mut aas_buf = Vec::new();
    let mut params = Vec::with_capacity(n_calls);
    let mut ns_raw = Vec::with_capacity(n_calls);
    let mut aa_raw = Vec::with_capacity(n_calls);

    for _ in 0..n_calls {
        let nas_offset = nas_buf.len() as u32;
        let aas_offset = aas_buf.len() as u32;

        // Generate raw nucleotide sequence (single base, no splice) and translate
        let ns = vec![nt_base; nl];
        let mut nas = vec![21u8; nl]; // AA_AMBI
        let mut codon = 0u8;
        let mut l = 0i32;
        for (i, &byte) in ns.iter().enumerate() {
            let c = tables.nt4[byte as usize];
            if c < 4 {
                codon = ((codon << 2) | c) & 0x3f;
                l += 1;
                if l >= 3 {
                    nas[i] = tables.codon[codon as usize];
                }
            } else {
                codon = 0;
                l = 0;
            }
        }
        nas_buf.extend_from_slice(&nas);
        ns_raw.push(ns);

        // Generate raw amino acid sequence
        let aa: Vec<u8> = (0..al).map(|_| aa_chars[(rng() as usize) % 20]).collect();
        let aas: Vec<u8> = aa.iter().map(|&byte| tables.aa20[byte as usize]).collect();
        aas_buf.extend_from_slice(&aas);
        aa_raw.push(aa);

        params.push(DpParams {
            nas_offset,
            aas_offset,
            nl: nl as u32,
            al: al as u32,
            go: 11,
            ge: 1,
            io: 29,
            fs: 23,
            goe: 12,
            end_bonus: 5,
            flag: if is_ext { 2 } else { 0 },
            slen: al.div_ceil(8) as u32,
        });
    }

    BenchData {
        nas_buf,
        aas_buf,
        params,
        ns_raw,
        aa_raw,
    }
}

// ---------------------------------------------------------------------------
// CPU NEON baseline
// ---------------------------------------------------------------------------

/// Run scalar DP (reference implementation, matches GPU algorithm exactly).
fn run_cpu_scalar(data: &BenchData, is_ext: bool) -> (Vec<DpResult>, std::time::Duration) {
    let tables = crate::tables::make_tables(1).expect("tables");
    let flag = if is_ext { 2i32 } else { 0i32 };
    let start = Instant::now();
    let results: Vec<DpResult> = data
        .params
        .iter()
        .enumerate()
        .map(|(i, _p)| {
            let ns = &data.ns_raw[i];
            let aa = &data.aa_raw[i];
            let opt = crate::align::NsOpt {
                flag,
                go: 11,
                ge: 1,
                io: 29,
                fs: 23,
                xdrop: 100,
                end_bonus: 5,
                sp: [8, 15, 21, 30, 4, 4],
                sp_null_bonus: -7,
                ie_coef: 0.5,
                sc: &crate::tables::BLOSUM62,
                tables: &tables,
            };
            let rst = crate::scalar_dp::global(ns, aa, &opt, None);
            DpResult {
                score: rst.score,
                nt_len: rst.nt_len,
                aa_len: rst.aa_len,
            }
        })
        .collect();
    let elapsed = start.elapsed();
    (results, elapsed)
}

/// Run NEON SIMD DP (production baseline, includes splice handling).
fn run_cpu_neon(data: &BenchData, is_ext: bool) -> (Vec<DpResult>, std::time::Duration) {
    let tables = crate::tables::make_tables(1).expect("tables");
    let flag = if is_ext { 2i32 } else { 0i32 };
    let start = Instant::now();
    let results: Vec<DpResult> = data
        .params
        .iter()
        .enumerate()
        .map(|(i, _p)| {
            let ns = &data.ns_raw[i];
            let aa = &data.aa_raw[i];
            let opt = crate::align::NsOpt {
                flag,
                go: 11,
                ge: 1,
                io: 29,
                fs: 23,
                xdrop: 100,
                end_bonus: 5,
                sp: [8, 15, 21, 30, 4, 4],
                sp_null_bonus: -7,
                ie_coef: 0.5,
                sc: &crate::tables::BLOSUM62,
                tables: &tables,
            };
            let rst = crate::neon_dp::global_gs16b(ns, aa, &opt, None);
            DpResult {
                score: rst.score,
                nt_len: rst.nt_len,
                aa_len: rst.aa_len,
            }
        })
        .collect();
    let elapsed = start.elapsed();
    (results, elapsed)
}

// ---------------------------------------------------------------------------
// Benchmark runner
// ---------------------------------------------------------------------------

fn format_dur(d: std::time::Duration) -> String {
    let secs = d.as_secs_f64();
    if secs >= 1.0 {
        format!("{secs:.2}s")
    } else if secs >= 0.001 {
        format!("{:.2}ms", secs * 1000.0)
    } else {
        format!("{:.0}us", d.as_micros())
    }
}

struct BenchResult {
    name: String,
    n_calls: usize,
    total: std::time::Duration,
    per_call: std::time::Duration,
    results: Vec<DpResult>,
}

fn run_metal_batch(data: &BenchData, _n_repeats: usize) -> Option<BenchResult> {
    if !metal_dp::available() {
        return None;
    }
    let start = Instant::now();
    let results = metal_dp::batch_dp(&data.nas_buf, &data.aas_buf, &data.params)?;
    let total = start.elapsed();
    let n = data.params.len();
    Some(BenchResult {
        name: "Metal".into(),
        n_calls: n,
        total,
        per_call: total / n as u32,
        results,
    })
}

fn run_wgpu_batch(data: &BenchData, _n_repeats: usize) -> Option<BenchResult> {
    if !crate::wgpu_dp::available() {
        return None;
    }
    let start = Instant::now();
    let results = crate::wgpu_dp::batch_dp(&data.nas_buf, &data.aas_buf, &data.params)?;
    let total = start.elapsed();
    let n = data.params.len();
    Some(BenchResult {
        name: "wgpu".into(),
        n_calls: n,
        total,
        per_call: total / n as u32,
        results,
    })
}

fn run_cuda_batch(data: &BenchData, _n_repeats: usize) -> Option<BenchResult> {
    if !crate::cuda_dp::available() {
        return None;
    }
    let start = Instant::now();
    let results = crate::cuda_dp::batch_dp(&data.nas_buf, &data.aas_buf, &data.params)?;
    let total = start.elapsed();
    let n = data.params.len();
    Some(BenchResult {
        name: "CUDA".into(),
        n_calls: n,
        total,
        per_call: total / n as u32,
        results,
    })
}

fn check_correctness(
    cpu: &[DpResult],
    gpu: &[DpResult],
    tolerance: i32,
) -> (usize, usize, Vec<String>) {
    let mut ok = 0;
    let mut diff = 0;
    let mut diffs = Vec::new();
    for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        let sc_diff = (c.score - g.score).abs();
        if sc_diff <= tolerance && c.nt_len == g.nt_len && c.aa_len == g.aa_len {
            ok += 1;
        } else {
            diff += 1;
            if diffs.len() < 5 {
                diffs.push(format!(
                    "  [#{i}] CPU: sc={} nl={} al={} | GPU: sc={} nl={} al={} | Δsc={}",
                    c.score, c.nt_len, c.aa_len, g.score, g.nt_len, g.aa_len, sc_diff
                ));
            }
        }
    }
    (ok, diff, diffs)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn bench_batch_size_sweep() {
    eprintln!("\n=== Batch Size Sweep (nl=3000, al=50, ext=false) ===");
    let mut seed: u64 = 12345;
    let mut rng = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        seed
    };

    let batch_sizes = [1, 4, 16, 64, 256, 512, 1024, 2048, 4096, 8192];

    for &bs in &batch_sizes {
        eprintln!("\n--- Batch size: {bs} ---");
        let data = generate_workload(bs, 3000, 50, false, &mut rng);

        // CPU scalar (reference, matches GPU algorithm)
        let (scalar_results, scalar_time) = run_cpu_scalar(&data, false);
        eprintln!(
            "  CPU scalar: {} total, {}/call",
            format_dur(scalar_time),
            format_dur(scalar_time / bs as u32)
        );

        // CPU NEON SIMD (production, includes splice handling)
        let (_neon_res, neon_time) = run_cpu_neon(&data, false);
        eprintln!(
            "  CPU NEON:   {} total, {}/call",
            format_dur(neon_time),
            format_dur(neon_time / bs as u32)
        );

        // Metal (verified against scalar, not NEON)
        if let Some(br) = run_metal_batch(&data, 1) {
            let (ok, _, _) = check_correctness(&scalar_results, &br.results, 2);
            let _vs_neon = br.total.as_secs_f64() / neon_time.as_secs_f64().max(1e-9);
            eprintln!(
                "  Metal:      {} total, {}/call  [match:{ok}/{bs}]  {:.1}x CPU scalar,  {:.1}x CPU NEON",
                format_dur(br.total),
                format_dur(br.per_call),
                scalar_time.as_secs_f64() / br.total.as_secs_f64().max(1e-9),
                neon_time.as_secs_f64() / br.total.as_secs_f64().max(1e-9),
            );
        } else {
            eprintln!("  Metal:      n/a");
        }

        // wgpu
        if let Some(br) = run_wgpu_batch(&data, 1) {
            let (ok, _, _) = check_correctness(&scalar_results, &br.results, 2);
            eprintln!(
                "  wgpu:       {} total, {}/call  [match:{ok}/{bs}]  {:.1}x CPU scalar,  {:.1}x CPU NEON",
                format_dur(br.total),
                format_dur(br.per_call),
                scalar_time.as_secs_f64() / br.total.as_secs_f64().max(1e-9),
                neon_time.as_secs_f64() / br.total.as_secs_f64().max(1e-9),
            );
        } else {
            eprintln!("  wgpu:       n/a");
        }

        if let Some(br) = run_cuda_batch(&data, 1) {
            let (ok, _, _) = check_correctness(&scalar_results, &br.results, 2);
            eprintln!(
                "  CUDA:       {} total, {}/call  [match:{ok}/{bs}]  {:.1}x CPU scalar,  {:.1}x CPU NEON",
                format_dur(br.total),
                format_dur(br.per_call),
                scalar_time.as_secs_f64() / br.total.as_secs_f64().max(1e-9),
                neon_time.as_secs_f64() / br.total.as_secs_f64().max(1e-9),
            );
        } else {
            eprintln!("  CUDA:       n/a");
        }
    }
}

#[test]
fn bench_matrix_size_sweep() {
    eprintln!("\n=== Matrix Size Sweep (batch=64, ext=false) ===");
    let mut seed: u64 = 67890;
    let mut rng = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        seed
    };

    let configs = [
        (1000, 10, "1k×10"),
        (3000, 50, "3k×50"),
        (10000, 50, "10k×50"),
        (30000, 50, "30k×50"),
        (3000, 200, "3k×200"),
    ];

    for &(nl, al, label) in &configs {
        eprintln!("\n--- Matrix: {label} ---");
        let data = generate_workload(64, nl, al, false, &mut rng);

        let (cpu_results, cpu_time) = run_cpu_scalar(&data, false);
        eprintln!(
            "  CPU scalar:{} total, {}/call",
            format_dur(cpu_time),
            format_dur(cpu_time / 64)
        );

        if let Some(br) = run_metal_batch(&data, 1) {
            let (ok, diff, _) = check_correctness(&cpu_results, &br.results, 2);
            let speedup = cpu_time.as_secs_f64() / br.total.as_secs_f64();
            eprintln!(
                "  Metal:     {} total, {}/call  [{ok}/{diff} match, {speedup:.1}x vs CPU]",
                format_dur(br.total),
                format_dur(br.per_call),
            );
        } else {
            eprintln!("  Metal:     not available");
        }

        if let Some(br) = run_wgpu_batch(&data, 1) {
            let (ok, diff, _) = check_correctness(&cpu_results, &br.results, 2);
            let speedup = cpu_time.as_secs_f64() / br.total.as_secs_f64();
            eprintln!(
                "  wgpu:      {} total, {}/call  [{ok}/{diff} match, {speedup:.1}x vs CPU]",
                format_dur(br.total),
                format_dur(br.per_call),
            );
        } else {
            eprintln!("  wgpu:      not available");
        }

        if let Some(br) = run_cuda_batch(&data, 1) {
            let (ok, diff, _) = check_correctness(&cpu_results, &br.results, 2);
            let speedup = cpu_time.as_secs_f64() / br.total.as_secs_f64();
            eprintln!(
                "  CUDA:      {} total, {}/call  [{ok}/{diff} match, {speedup:.1}x vs CPU]",
                format_dur(br.total),
                format_dur(br.per_call),
            );
        } else {
            eprintln!("  CUDA:      not available");
        }
    }
}

#[test]
fn bench_extension_mode() {
    eprintln!("\n=== Extension Mode (batch=64, nl=3000, al=50, ext=true) ===");
    let mut seed: u64 = 99999;
    let mut rng = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        seed
    };

    let data = generate_workload(64, 3000, 50, true, &mut rng);

    let (cpu_results, cpu_time) = run_cpu_scalar(&data, true);
    eprintln!(
        "  CPU scalar:{} total, {}/call",
        format_dur(cpu_time),
        format_dur(cpu_time / 64)
    );

    if let Some(br) = run_metal_batch(&data, 1) {
        let (ok, diff, diffs) = check_correctness(&cpu_results, &br.results, 5);
        let speedup = cpu_time.as_secs_f64() / br.total.as_secs_f64();
        eprintln!(
            "  Metal:     {} total, {}/call  [{ok}/{diff} match, {speedup:.1}x vs CPU]",
            format_dur(br.total),
            format_dur(br.per_call),
        );
        for d in &diffs {
            eprintln!("{d}");
        }
    } else {
        eprintln!("  Metal:     not available");
    }

    if let Some(br) = run_wgpu_batch(&data, 1) {
        let (ok, diff, diffs) = check_correctness(&cpu_results, &br.results, 5);
        let speedup = cpu_time.as_secs_f64() / br.total.as_secs_f64();
        eprintln!(
            "  wgpu:      {} total, {}/call  [{ok}/{diff} match, {speedup:.1}x vs CPU]",
            format_dur(br.total),
            format_dur(br.per_call),
        );
        for d in &diffs {
            eprintln!("{d}");
        }
    } else {
        eprintln!("  wgpu:      not available");
    }

    if let Some(br) = run_cuda_batch(&data, 1) {
        let (ok, diff, diffs) = check_correctness(&cpu_results, &br.results, 5);
        let speedup = cpu_time.as_secs_f64() / br.total.as_secs_f64();
        eprintln!(
            "  CUDA:      {} total, {}/call  [{ok}/{diff} match, {speedup:.1}x vs CPU]",
            format_dur(br.total),
            format_dur(br.per_call),
        );
        for d in &diffs {
            eprintln!("{d}");
        }
    } else {
        eprintln!("  CUDA:      not available");
    }
}

#[test]
fn bench_kernel_only() {
    eprintln!("\n=== Kernel-Only Time (nl=3000, al=50) ===");
    let mut seed: u64 = 11111;
    let mut rng = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        seed
    };

    for &bs in &[256, 1024, 4096, 8192] {
        eprintln!("\n--- Batch size: {bs} ---");
        let data = generate_workload(bs, 3000, 50, false, &mut rng);

        if let Some((warmup, timed)) = metal_dp::bench_dispatch_only(
            &data.nas_buf,
            &data.aas_buf,
            &data.params,
            &crate::tables::BLOSUM62,
        ) {
            eprintln!(
                "  Metal kernel: warmup={}, timed={}, {}/call",
                format_dur(warmup),
                format_dur(timed),
                format_dur(timed / bs as u32)
            );
        } else {
            eprintln!("  Metal kernel: not available");
        }

        if bs <= 4096 {
            if let Some((warmup, timed)) = crate::wgpu_dp::bench_dispatch_only(
                &data.nas_buf,
                &data.aas_buf,
                &data.params,
                &crate::tables::BLOSUM62,
            ) {
                eprintln!(
                    "  wgpu kernel:  warmup={}, timed={}, {}/call",
                    format_dur(warmup),
                    format_dur(timed),
                    format_dur(timed / bs as u32)
                );
            } else {
                eprintln!("  wgpu kernel:  not available");
            }
        }

        if let Some((warmup, timed)) = crate::cuda_dp::bench_dispatch_only(
            &data.nas_buf,
            &data.aas_buf,
            &data.params,
            &crate::tables::BLOSUM62,
        ) {
            eprintln!(
                "  CUDA kernel:  warmup={}, timed={}, {}/call",
                format_dur(warmup),
                format_dur(timed),
                format_dur(timed / bs as u32)
            );
        } else {
            eprintln!("  CUDA kernel:  not available");
        }
    }
}

#[test]
fn bench_cuda_repeated_batch() {
    if !crate::cuda_dp::available() {
        eprintln!("CUDA repeated batch: not available");
        return;
    }

    eprintln!("\n=== CUDA Repeated Batch (batch=8192, nl=3000, al=50) ===");
    let mut seed: u64 = 22222;
    let mut rng = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        seed
    };
    let data = generate_workload(8192, 3000, 50, false, &mut rng);
    let (scalar_results, scalar_time) = run_cpu_scalar(&data, false);
    eprintln!(
        "  CPU scalar: {} total, {}/call",
        format_dur(scalar_time),
        format_dur(scalar_time / 8192)
    );

    let mut times = Vec::new();
    for iter in 0..6 {
        let start = Instant::now();
        let results = crate::cuda_dp::batch_dp(&data.nas_buf, &data.aas_buf, &data.params)
            .expect("CUDA batch");
        let total = start.elapsed();
        let (ok, _, _) = check_correctness(&scalar_results, &results, 2);
        eprintln!(
            "  iter {iter}: {} total, {}/call [match:{ok}/8192]",
            format_dur(total),
            format_dur(total / 8192)
        );
        times.push(total);
    }
    let steady = &times[1..];
    let best = steady.iter().copied().min().unwrap();
    let avg = steady.iter().sum::<std::time::Duration>() / steady.len() as u32;
    eprintln!(
        "  steady best: {}, avg: {}",
        format_dur(best),
        format_dur(avg)
    );
}

// ---------------------------------------------------------------------------
// Code complexity / maintainability metrics
// ---------------------------------------------------------------------------

#[test]
fn bench_metal_vs_scalar_large() {
    eprintln!("\n=== Metal vs CPU (nl=3000, al=50, large batches) ===");
    let mut seed: u64 = 77777;
    let mut rng = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        seed
    };

    let batch_sizes = [64, 256, 512, 1024];

    for &bs in &batch_sizes {
        eprintln!("\n--- Batch size: {bs} ---");
        let data = generate_workload(bs, 3000, 50, false, &mut rng);

        let (scalar_results, scalar_time) = run_cpu_scalar(&data, false);
        let (_neon_res, neon_time) = run_cpu_neon(&data, false);
        eprintln!(
            "  CPU scalar: {} total, {}/call",
            format_dur(scalar_time),
            format_dur(scalar_time / bs as u32)
        );
        eprintln!(
            "  CPU NEON:   {} total, {}/call",
            format_dur(neon_time),
            format_dur(neon_time / bs as u32)
        );

        if let Some(br) = run_metal_batch(&data, 1) {
            let (ok, _, _) = check_correctness(&scalar_results, &br.results, 2);
            eprintln!(
                "  Metal:      {} total, {}/call  [ok:{ok}/{bs}]  {:.1}x CPU scalar  {:.1}x CPU NEON",
                format_dur(br.total),
                format_dur(br.per_call),
                scalar_time.as_secs_f64() / br.total.as_secs_f64().max(1e-9),
                neon_time.as_secs_f64() / br.total.as_secs_f64().max(1e-9),
            );
        } else {
            eprintln!("  Metal:      n/a");
        }

        if let Some(br) = run_cuda_batch(&data, 1) {
            let (ok, _, _) = check_correctness(&scalar_results, &br.results, 2);
            eprintln!(
                "  CUDA:       {} total, {}/call  [ok:{ok}/{bs}]  {:.1}x CPU scalar  {:.1}x CPU NEON",
                format_dur(br.total),
                format_dur(br.per_call),
                scalar_time.as_secs_f64() / br.total.as_secs_f64().max(1e-9),
                neon_time.as_secs_f64() / br.total.as_secs_f64().max(1e-9),
            );
        } else {
            eprintln!("  CUDA:       n/a");
        }
    }
}

#[test]
fn report_code_metrics() {
    eprintln!("\n=== Code Metrics ===");

    // Count lines in each backend file + shader
    let metal_host = include_str!("metal_dp.rs");
    let metal_shader = include_str!("dp.metal");
    let wgpu_host = include_str!("wgpu_dp.rs");
    let wgpu_shader = include_str!("dp.wgsl");
    let cuda_host = include_str!("cuda_dp.rs");
    let cuda_kernel = include_str!("cuda_dp.cu");

    let metal_host_loc = metal_host.lines().count();
    let metal_shader_loc = metal_shader.lines().count();
    let wgpu_host_loc = wgpu_host.lines().count();
    let wgpu_shader_loc = wgpu_shader.lines().count();
    let cuda_host_loc = cuda_host.lines().count();
    let cuda_kernel_loc = cuda_kernel.lines().count();

    // Count unsafe blocks
    let metal_unsafe = metal_host.matches("unsafe").count();
    let wgpu_unsafe = wgpu_host.matches("unsafe").count();
    let cuda_unsafe = cuda_host.matches("unsafe").count();

    // Count API interaction points (simplified: count lines with common patterns)
    let metal_api_calls = metal_host
        .lines()
        .filter(|l| {
            l.contains("new_buffer")
                || l.contains("new_command")
                || l.contains("set_buffer")
                || l.contains("set_compute")
                || l.contains("dispatch")
                || l.contains("commit")
                || l.contains("wait_until")
                || l.contains("end_encoding")
                || l.contains("new_library")
                || l.contains("new_compute_pipeline")
        })
        .count();
    let wgpu_api_calls = wgpu_host
        .lines()
        .filter(|l| {
            l.contains("create_buffer")
                || l.contains("create_bind_group")
                || l.contains("create_compute_pipeline")
                || l.contains("create_command_encoder")
                || l.contains("create_shader_module")
                || l.contains("begin_compute_pass")
                || l.contains("set_pipeline")
                || l.contains("set_bind_group")
                || l.contains("dispatch_workgroups")
                || l.contains("submit")
                || l.contains("request_device")
                || l.contains("request_adapter")
        })
        .count();

    eprintln!("  Backend       | Host LOC | Shader LOC | Unsafe | API calls");
    eprintln!("  --------------+----------+------------+--------+----------");
    eprintln!(
        "  Metal (raw)   | {:>8} | {:>10} | {:>6} | {:>8}",
        metal_host_loc, metal_shader_loc, metal_unsafe, metal_api_calls
    );
    eprintln!(
        "  wgpu (cross)  | {:>8} | {:>10} | {:>6} | {:>8}",
        wgpu_host_loc, wgpu_shader_loc, wgpu_unsafe, wgpu_api_calls
    );
    eprintln!(
        "  CUDA (raw)    | {:>8} | {:>10} | {:>6} | {:>8}",
        cuda_host_loc, cuda_kernel_loc, cuda_unsafe, "FFI"
    );

    // Dependency count
    eprintln!();
    eprintln!("  Dependencies added:");
    eprintln!("    Metal:  metal, objc (2 crates)");
    eprintln!("    wgpu:   wgpu, bytemuck, pollster (3 crates)");
    eprintln!("    CUDA:   no Rust crates; requires CUDA toolkit at build time");

    // Platform support
    eprintln!();
    eprintln!("  Platform support:");
    eprintln!("    Metal:  macOS only (ARM + x86_64), no Linux/Windows");
    eprintln!("    wgpu:   macOS (Metal), Linux (Vulkan), Windows (DX12/Vulkan)");
    eprintln!("    CUDA:   NVIDIA GPUs, opt-in with --features cuda");
}
