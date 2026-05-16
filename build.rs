use std::{env, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=src/cuda_dp.cu");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_THREADS");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");

    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let cuda_home = env::var_os("CUDA_HOME")
        .or_else(|| env::var_os("CUDA_PATH"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/usr/local/cuda"));
    let nvcc = cuda_home.join("bin/nvcc");
    let arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_90".to_string());
    let cuda_threads = env::var("CUDA_THREADS").unwrap_or_else(|_| "32".to_string());
    let src = manifest_dir.join("src/cuda_dp.cu");
    let obj = out_dir.join("cuda_dp.o");
    let lib = out_dir.join("libcuda_dp.a");

    let mut nvcc_cmd = Command::new(&nvcc);
    nvcc_cmd
        .args([
            "-O3",
            "--use_fast_math",
            "-lineinfo",
            "-Xcompiler",
            "-fPIC",
            "-c",
            src.to_str().unwrap(),
            "-o",
            obj.to_str().unwrap(),
            "-arch",
            &arch,
        ])
        .arg(format!("-DCUDA_THREADS={cuda_threads}"));
    let status = nvcc_cmd
        .status()
        .unwrap_or_else(|err| panic!("failed to run {}: {err}", nvcc.display()));
    assert!(status.success(), "nvcc failed with status {status}");

    let status = Command::new("ar")
        .args(["crus", lib.to_str().unwrap(), obj.to_str().unwrap()])
        .status()
        .expect("failed to run ar");
    assert!(status.success(), "ar failed with status {status}");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=cuda_dp");
    println!(
        "cargo:rustc-link-search=native={}",
        cuda_home.join("targets/x86_64-linux/lib").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        cuda_home.join("lib64").display()
    );
    println!("cargo:rustc-link-lib=dylib=cudart");
}
