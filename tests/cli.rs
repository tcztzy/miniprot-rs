mod support;

use std::fs;

use assert_cmd::Command;

#[test]
fn help_exits_successfully() {
    let output = Command::cargo_bin("miniprot")
        .expect("missing cargo binary")
        .arg("--help")
        .output()
        .expect("run help");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Usage:"));
    assert!(stdout.contains("miniprot"));
}

#[test]
fn missing_reference_is_reported() {
    let output = Command::cargo_bin("miniprot")
        .expect("missing cargo binary")
        .output()
        .expect("run without args");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("missing reference argument"));
}

#[test]
fn missing_query_is_reported() {
    let fasta = support::fixture_path("DPP3-hs.gen.fa.gz");
    let output = Command::cargo_bin("miniprot")
        .expect("missing cargo binary")
        .arg(fasta)
        .output()
        .expect("run without query");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("missing query argument"));
}

#[test]
fn deprecated_s_warns_but_succeeds() {
    let (fasta, query) = support::fixture_paths();
    let output = Command::cargo_bin("miniprot")
        .expect("missing cargo binary")
        .args(["-s", "-A"])
        .arg(&fasta)
        .arg(&query)
        .output()
        .expect("run deprecated -s");
    assert!(output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Option '-s' is deprecated."));
}

#[test]
fn invalid_splice_model_is_rejected() {
    let (fasta, query) = support::fixture_paths();
    let output = Command::cargo_bin("miniprot")
        .expect("missing cargo binary")
        .args(["-j", "9", "-A"])
        .arg(&fasta)
        .arg(&query)
        .output()
        .expect("run invalid -j");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("option -j should be between 0 and 2"));
}

#[test]
fn dump_index_without_query_succeeds() {
    let tmp = support::tempdir();
    let index = tmp.path().join("out.mpi");
    let fasta = support::fixture_path("DPP3-hs.gen.fa.gz");
    Command::cargo_bin("miniprot")
        .expect("missing cargo binary")
        .args(["-d", index.to_str().expect("utf8 index")])
        .arg(&fasta)
        .assert()
        .success();
    assert!(fs::metadata(index).is_ok());
}
