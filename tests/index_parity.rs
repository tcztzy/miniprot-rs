mod support;

use assert_cmd::Command;

#[test]
fn mpi_dump_matches_c_oracle() {
    support::ensure_c_oracle();
    let tmp = support::tempdir();
    let rust_idx = tmp.path().join("rust.mpi");
    let c_idx = tmp.path().join("c.mpi");
    let fasta = support::fixture_path("DPP3-hs.gen.fa.gz");

    Command::cargo_bin("miniprot")
        .expect("missing cargo binary")
        .args(["-d", rust_idx.to_str().unwrap(), fasta.to_str().unwrap()])
        .assert()
        .success();

    let status = std::process::Command::new(support::c_oracle())
        .args(["-d", c_idx.to_str().unwrap(), fasta.to_str().unwrap()])
        .status()
        .expect("failed to run C oracle");
    assert!(status.success());

    let rust_bytes = std::fs::read(rust_idx).expect("failed to read rust mpi");
    let c_bytes = std::fs::read(c_idx).expect("failed to read c mpi");
    assert_eq!(rust_bytes, c_bytes);
}

#[test]
fn rust_round_trips_c_index_without_changes() {
    support::ensure_c_oracle();
    let tmp = support::tempdir();
    let c_idx = tmp.path().join("c.mpi");
    let rust_idx = tmp.path().join("roundtrip.mpi");
    let fasta = support::fixture_path("DPP3-hs.gen.fa.gz");

    let status = std::process::Command::new(support::c_oracle())
        .args(["-d", c_idx.to_str().unwrap(), fasta.to_str().unwrap()])
        .status()
        .expect("failed to run C oracle");
    assert!(status.success());

    Command::cargo_bin("miniprot")
        .expect("missing cargo binary")
        .args(["-d", rust_idx.to_str().unwrap(), c_idx.to_str().unwrap()])
        .assert()
        .success();

    let rust_bytes = std::fs::read(rust_idx).expect("failed to read rust mpi");
    let c_bytes = std::fs::read(c_idx).expect("failed to read c mpi");
    assert_eq!(rust_bytes, c_bytes);
}
