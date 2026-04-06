use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use tempfile::TempDir;

#[allow(dead_code)]
static ORACLE_READY: OnceLock<()> = OnceLock::new();
#[allow(dead_code)]
static RUST_RELEASE_READY: OnceLock<PathBuf> = OnceLock::new();

pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

pub fn upstream_root() -> PathBuf {
    repo_root().join("miniprot")
}

pub fn fixture_path(name: &str) -> PathBuf {
    upstream_root().join("test").join(name)
}

#[allow(dead_code)]
pub fn fixture_paths() -> (PathBuf, PathBuf) {
    (
        fixture_path("DPP3-hs.gen.fa.gz"),
        fixture_path("DPP3-mm.pep.fa.gz"),
    )
}

fn existing_c_oracle() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("MINIPROT_C_ORACLE") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Some(path);
        }
    }
    let path = upstream_root().join("miniprot");
    path.is_file().then_some(path)
}

#[allow(dead_code)]
pub fn ensure_c_oracle() {
    ORACLE_READY.get_or_init(|| {
        if existing_c_oracle().is_some() {
            return;
        }
        let upstream = upstream_root();
        assert!(
            upstream.is_dir(),
            "missing upstream miniprot checkout at {}; run `git submodule update --init --recursive` or set MINIPROT_C_ORACLE",
            upstream.display()
        );
        let status = Command::new("make")
            .current_dir(&upstream)
            .status()
            .expect("failed to run make for C oracle");
        assert!(status.success(), "building C oracle failed");
    });
}

#[allow(dead_code)]
pub fn c_oracle() -> PathBuf {
    ensure_c_oracle();
    existing_c_oracle().unwrap_or_else(|| upstream_root().join("miniprot"))
}

#[allow(dead_code)]
pub fn rust_release_bin() -> PathBuf {
    RUST_RELEASE_READY
        .get_or_init(|| {
            let status = Command::new("cargo")
                .args(["build", "--release", "--bin", "miniprot"])
                .current_dir(repo_root())
                .status()
                .expect("failed to build release Rust binary");
            assert!(status.success(), "building release Rust binary failed");
            let path = repo_root().join(format!(
                "target/release/miniprot{}",
                std::env::consts::EXE_SUFFIX
            ));
            assert!(path.exists(), "release Rust binary is missing");
            path
        })
        .clone()
}

pub fn tempdir() -> TempDir {
    tempfile::tempdir().expect("failed to create tempdir")
}
