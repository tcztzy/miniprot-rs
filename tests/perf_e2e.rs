mod support;

use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

fn perf_lock() -> MutexGuard<'static, ()> {
    static PERF_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    PERF_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|err| err.into_inner())
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_i32(name: &str, default: i32) -> i32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_f64(name: &str) -> Option<f64> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
}

fn median(samples: &[Duration]) -> Duration {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    sorted[sorted.len() / 2]
}

fn mean(samples: &[Duration]) -> Duration {
    let total = samples.iter().map(Duration::as_secs_f64).sum::<f64>();
    Duration::from_secs_f64(total / samples.len() as f64)
}

#[derive(Debug)]
struct TimingStats {
    min: Duration,
    median: Duration,
    mean: Duration,
    samples: Vec<Duration>,
}

fn summarize(samples: Vec<Duration>) -> TimingStats {
    let min = *samples.iter().min().expect("non-empty timing samples");
    let median = median(&samples);
    let mean = mean(&samples);
    TimingStats {
        min,
        median,
        mean,
        samples,
    }
}

fn run_timed(mut cmd: Command) -> Duration {
    cmd.stdout(Stdio::null());
    cmd.stderr(Stdio::null());
    let start = Instant::now();
    let status = cmd.status().expect("failed to run benchmark command");
    assert!(status.success(), "benchmark command failed");
    start.elapsed()
}

fn run_capture(mut cmd: Command) -> Vec<u8> {
    cmd.stderr(Stdio::null());
    let output = cmd.output().expect("failed to run command");
    assert!(output.status.success(), "command failed");
    output.stdout
}

fn benchmark_pair<FC, FR>(
    warmups: usize,
    samples: usize,
    mut make_c_cmd: FC,
    mut make_rust_cmd: FR,
) -> (TimingStats, TimingStats)
where
    FC: FnMut() -> Command,
    FR: FnMut() -> Command,
{
    for i in 0..warmups {
        if (i & 1) == 0 {
            let _ = run_timed(make_c_cmd());
            let _ = run_timed(make_rust_cmd());
        } else {
            let _ = run_timed(make_rust_cmd());
            let _ = run_timed(make_c_cmd());
        }
    }

    let mut c_timings = Vec::with_capacity(samples);
    let mut rust_timings = Vec::with_capacity(samples);
    for i in 0..samples {
        if (i & 1) == 0 {
            c_timings.push(run_timed(make_c_cmd()));
            rust_timings.push(run_timed(make_rust_cmd()));
        } else {
            rust_timings.push(run_timed(make_rust_cmd()));
            c_timings.push(run_timed(make_c_cmd()));
        }
    }
    (summarize(c_timings), summarize(rust_timings))
}

fn fmt_ms(value: Duration) -> String {
    format!("{:.3}", value.as_secs_f64() * 1e3)
}

fn print_comparison(label: &str, c: &TimingStats, rust: &TimingStats) -> f64 {
    let ratio = rust.median.as_secs_f64() / c.median.as_secs_f64();
    println!(
        "{label}\n  C    min/median/mean: {} / {} / {} ms\n  Rust min/median/mean: {} / {} / {} ms\n  Rust/C median ratio: {:.3}x\n  samples: {:?}\n",
        fmt_ms(c.min),
        fmt_ms(c.median),
        fmt_ms(c.mean),
        fmt_ms(rust.min),
        fmt_ms(rust.median),
        fmt_ms(rust.mean),
        ratio,
        c.samples
            .iter()
            .zip(&rust.samples)
            .map(|(c_sample, rust_sample)| format!(
                "{}/{}",
                fmt_ms(*c_sample),
                fmt_ms(*rust_sample)
            ))
            .collect::<Vec<_>>(),
    );
    ratio
}

fn maybe_assert_ratio(label: &str, ratio: f64) {
    if let Some(max_ratio) = env_f64("MINIPROT_PERF_MAX_RUST_OVER_C") {
        assert!(
            ratio <= max_ratio,
            "{label}: Rust/C median ratio {ratio:.3} exceeds configured limit {max_ratio:.3}"
        );
    }
}

fn base_cmd(binary: &Path, threads: i32) -> Command {
    let mut cmd = Command::new(binary);
    cmd.arg("-t").arg(threads.to_string());
    cmd
}

#[test]
#[ignore = "manual end-to-end performance benchmark"]
fn compare_index_build_perf() {
    let _lock = perf_lock();
    support::ensure_c_oracle();
    let rust_bin = support::rust_release_bin();
    let (reference, _) = support::fixture_paths();
    let threads = env_i32("MINIPROT_PERF_THREADS", 1);
    let warmups = env_usize("MINIPROT_PERF_WARMUP", 1);
    let samples = env_usize("MINIPROT_PERF_SAMPLES", 5);
    let tmp = support::tempdir();
    let c_index = tmp.path().join("c-build.mpi");
    let rust_index = tmp.path().join("rust-build.mpi");

    let (c_stats, rust_stats) = benchmark_pair(
        warmups,
        samples,
        || {
            let _ = std::fs::remove_file(&c_index);
            let mut cmd = base_cmd(&support::c_oracle(), threads);
            cmd.arg("-d").arg(&c_index).arg(&reference);
            cmd
        },
        || {
            let _ = std::fs::remove_file(&rust_index);
            let mut cmd = base_cmd(&rust_bin, threads);
            cmd.arg("-d").arg(&rust_index).arg(&reference);
            cmd
        },
    );

    let ratio = print_comparison(
        "index build on miniprot/test/DPP3-hs.gen.fa.gz",
        &c_stats,
        &rust_stats,
    );
    maybe_assert_ratio("index build", ratio);
}

#[test]
#[ignore = "manual end-to-end performance benchmark"]
fn compare_no_align_map_perf() {
    let _lock = perf_lock();
    support::ensure_c_oracle();
    let rust_bin = support::rust_release_bin();
    let (reference, query) = support::fixture_paths();
    let threads = env_i32("MINIPROT_PERF_THREADS", 1);
    let warmups = env_usize("MINIPROT_PERF_WARMUP", 1);
    let samples = env_usize("MINIPROT_PERF_SAMPLES", 5);
    let tmp = support::tempdir();
    let index = tmp.path().join("shared.mpi");

    let status = Command::new(support::c_oracle())
        .arg("-d")
        .arg(&index)
        .arg(&reference)
        .status()
        .expect("failed to build shared C index");
    assert!(status.success(), "failed to build shared C index");

    let expected = run_capture({
        let mut cmd = base_cmd(&support::c_oracle(), threads);
        cmd.arg("-A").arg(&index).arg(&query);
        cmd
    });
    let actual = run_capture({
        let mut cmd = base_cmd(&rust_bin, threads);
        cmd.arg("-A").arg(&index).arg(&query);
        cmd
    });
    assert_eq!(actual, expected, "Rust and C no-align outputs diverged");

    let (c_stats, rust_stats) = benchmark_pair(
        warmups,
        samples,
        || {
            let mut cmd = base_cmd(&support::c_oracle(), threads);
            cmd.arg("-A").arg(&index).arg(&query);
            cmd
        },
        || {
            let mut cmd = base_cmd(&rust_bin, threads);
            cmd.arg("-A").arg(&index).arg(&query);
            cmd
        },
    );

    let ratio = print_comparison("no-align map on shared C-built .mpi", &c_stats, &rust_stats);
    maybe_assert_ratio("no-align map", ratio);
}

#[test]
#[ignore = "manual end-to-end performance benchmark"]
fn compare_default_map_perf() {
    let _lock = perf_lock();
    support::ensure_c_oracle();
    let rust_bin = support::rust_release_bin();
    let (reference, query) = support::fixture_paths();
    let threads = env_i32("MINIPROT_PERF_THREADS", 1);
    let warmups = env_usize("MINIPROT_PERF_WARMUP", 1);
    let samples = env_usize("MINIPROT_PERF_SAMPLES", 5);
    let tmp = support::tempdir();
    let index = tmp.path().join("shared.mpi");

    let status = Command::new(support::c_oracle())
        .arg("-d")
        .arg(&index)
        .arg(&reference)
        .status()
        .expect("failed to build shared C index");
    assert!(status.success(), "failed to build shared C index");

    let expected = run_capture({
        let mut cmd = base_cmd(&support::c_oracle(), threads);
        cmd.arg(&index).arg(&query);
        cmd
    });
    let actual = run_capture({
        let mut cmd = base_cmd(&rust_bin, threads);
        cmd.arg(&index).arg(&query);
        cmd
    });
    assert_eq!(actual, expected, "Rust and C default outputs diverged");

    let (c_stats, rust_stats) = benchmark_pair(
        warmups,
        samples,
        || {
            let mut cmd = base_cmd(&support::c_oracle(), threads);
            cmd.arg(&index).arg(&query);
            cmd
        },
        || {
            let mut cmd = base_cmd(&rust_bin, threads);
            cmd.arg(&index).arg(&query);
            cmd
        },
    );

    let ratio = print_comparison("default map on shared C-built .mpi", &c_stats, &rust_stats);
    maybe_assert_ratio("default map", ratio);
}
