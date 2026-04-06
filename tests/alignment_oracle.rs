mod support;

use std::fs;
use std::io::Write;
use std::process::Stdio;

use assert_cmd::Command;

fn c_stdout(args: &[&str]) -> Vec<u8> {
    support::ensure_c_oracle();
    let output = std::process::Command::new(support::c_oracle())
        .args(args)
        .output()
        .expect("failed to run C oracle");
    assert!(output.status.success());
    output.stdout
}

fn c_stdout_with_stdin(args: &[&str], stdin: &[u8]) -> Vec<u8> {
    support::ensure_c_oracle();
    let mut child = std::process::Command::new(support::c_oracle())
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("failed to spawn C oracle");
    child
        .stdin
        .take()
        .expect("missing C stdin")
        .write_all(stdin)
        .expect("failed to write C stdin");
    let output = child
        .wait_with_output()
        .expect("failed to wait for C oracle");
    assert!(output.status.success());
    output.stdout
}

fn write_spsc_fixture(dir: &std::path::Path) -> std::path::PathBuf {
    let path = dir.join("score.tsv");
    fs::write(
        &path,
        concat!(
            "chr11:66478458-66505490\t270\t+\tD\t14\n",
            "chr11:66478458-66505490\t2972\t+\tA\t14\n",
            "chr11:66478458-66505490\t3062\t+\tD\t14\n",
            "chr11:66478458-66505490\t4334\t+\tA\t14\n",
        ),
    )
    .expect("write spsc fixture");
    path
}

fn gz_fixture_bytes(path: &std::path::Path) -> Vec<u8> {
    fs::read(path).expect("read gz fixture")
}

#[test]
fn default_paf_matches_c_oracle() {
    let (fasta, query) = support::fixture_paths();
    let expected = c_stdout(&[
        fasta.to_str().expect("utf8 fasta"),
        query.to_str().expect("utf8 query"),
    ]);

    Command::cargo_bin("miniprot")
        .expect("missing cargo binary")
        .args([
            fasta.to_str().expect("utf8 fasta"),
            query.to_str().expect("utf8 query"),
        ])
        .assert()
        .success()
        .stdout(expected);
}

#[test]
fn default_matches_between_fasta_and_index_input() {
    let tmp = support::tempdir();
    let (fasta, query) = support::fixture_paths();
    let index = tmp.path().join("rust.mpi");

    Command::cargo_bin("miniprot")
        .expect("missing cargo binary")
        .args([
            "-d",
            index.to_str().expect("utf8 index"),
            fasta.to_str().expect("utf8 fasta"),
        ])
        .assert()
        .success();

    let direct = Command::cargo_bin("miniprot")
        .expect("missing cargo binary")
        .args([
            fasta.to_str().expect("utf8 fasta"),
            query.to_str().expect("utf8 query"),
        ])
        .output()
        .expect("failed direct mapping");
    assert!(direct.status.success());

    let restored = Command::cargo_bin("miniprot")
        .expect("missing cargo binary")
        .args([
            index.to_str().expect("utf8 index"),
            query.to_str().expect("utf8 query"),
        ])
        .output()
        .expect("failed restored mapping");
    assert!(restored.status.success());

    assert_eq!(direct.stdout, restored.stdout);
}

#[test]
fn query_stdin_matches_c_oracle() {
    let (fasta, query) = support::fixture_paths();
    let stdin = gz_fixture_bytes(&query);
    let expected = c_stdout_with_stdin(&[fasta.to_str().expect("utf8 fasta"), "-"], &stdin);

    Command::cargo_bin("miniprot")
        .expect("missing cargo binary")
        .args([fasta.to_str().expect("utf8 fasta"), "-"])
        .write_stdin(stdin)
        .assert()
        .success()
        .stdout(expected);
}

#[test]
fn query_stdin_plain_matches_c_oracle() {
    let (fasta, _) = support::fixture_paths();
    let stdin = b">plain_query\nMST\n";
    let expected = c_stdout_with_stdin(&[fasta.to_str().expect("utf8 fasta"), "-"], stdin);

    Command::cargo_bin("miniprot")
        .expect("missing cargo binary")
        .args([fasta.to_str().expect("utf8 fasta"), "-"])
        .write_stdin(stdin)
        .assert()
        .success()
        .stdout(expected);
}

#[test]
fn gff_matches_c_oracle() {
    let (fasta, query) = support::fixture_paths();
    let expected = c_stdout(&[
        "--gff",
        fasta.to_str().expect("utf8 fasta"),
        query.to_str().expect("utf8 query"),
    ]);

    Command::cargo_bin("miniprot")
        .expect("missing cargo binary")
        .args([
            "--gff",
            fasta.to_str().expect("utf8 fasta"),
            query.to_str().expect("utf8 query"),
        ])
        .assert()
        .success()
        .stdout(expected);
}

#[test]
fn gff_only_matches_c_oracle() {
    let (fasta, query) = support::fixture_paths();
    let expected = c_stdout(&[
        "--gff-only",
        fasta.to_str().expect("utf8 fasta"),
        query.to_str().expect("utf8 query"),
    ]);

    Command::cargo_bin("miniprot")
        .expect("missing cargo binary")
        .args([
            "--gff-only",
            fasta.to_str().expect("utf8 fasta"),
            query.to_str().expect("utf8 query"),
        ])
        .assert()
        .success()
        .stdout(expected);
}

#[test]
fn gtf_matches_c_oracle() {
    let (fasta, query) = support::fixture_paths();
    let expected = c_stdout(&[
        "--gtf",
        fasta.to_str().expect("utf8 fasta"),
        query.to_str().expect("utf8 query"),
    ]);

    Command::cargo_bin("miniprot")
        .expect("missing cargo binary")
        .args([
            "--gtf",
            fasta.to_str().expect("utf8 fasta"),
            query.to_str().expect("utf8 query"),
        ])
        .assert()
        .success()
        .stdout(expected);
}

#[test]
fn no_cs_matches_c_oracle() {
    let (fasta, query) = support::fixture_paths();
    let expected = c_stdout(&[
        "--no-cs",
        fasta.to_str().expect("utf8 fasta"),
        query.to_str().expect("utf8 query"),
    ]);

    Command::cargo_bin("miniprot")
        .expect("missing cargo binary")
        .args([
            "--no-cs",
            fasta.to_str().expect("utf8 fasta"),
            query.to_str().expect("utf8 query"),
        ])
        .assert()
        .success()
        .stdout(expected);
}

#[test]
fn residue_alignment_matches_c_oracle() {
    let (fasta, query) = support::fixture_paths();
    let expected = c_stdout(&[
        "--aln",
        fasta.to_str().expect("utf8 fasta"),
        query.to_str().expect("utf8 query"),
    ]);

    Command::cargo_bin("miniprot")
        .expect("missing cargo binary")
        .args([
            "--aln",
            fasta.to_str().expect("utf8 fasta"),
            query.to_str().expect("utf8 query"),
        ])
        .assert()
        .success()
        .stdout(expected);
}

#[test]
fn spsc_gff_matches_c_oracle() {
    let tmp = support::tempdir();
    let spsc = write_spsc_fixture(tmp.path());
    let (fasta, query) = support::fixture_paths();
    let expected = c_stdout(&[
        "--gff",
        "--spsc",
        spsc.to_str().expect("utf8 spsc"),
        fasta.to_str().expect("utf8 fasta"),
        query.to_str().expect("utf8 query"),
    ]);

    Command::cargo_bin("miniprot")
        .expect("missing cargo binary")
        .args([
            "--gff",
            "--spsc",
            spsc.to_str().expect("utf8 spsc"),
            fasta.to_str().expect("utf8 fasta"),
            query.to_str().expect("utf8 query"),
        ])
        .assert()
        .success()
        .stdout(expected);
}

#[test]
fn spsc_stdin_matches_c_oracle() {
    let tmp = support::tempdir();
    let spsc = write_spsc_fixture(tmp.path());
    let stdin = fs::read(&spsc).expect("read spsc fixture");
    let (fasta, query) = support::fixture_paths();
    let expected = c_stdout_with_stdin(
        &[
            "--gff",
            "--spsc",
            "-",
            fasta.to_str().expect("utf8 fasta"),
            query.to_str().expect("utf8 query"),
        ],
        &stdin,
    );

    Command::cargo_bin("miniprot")
        .expect("missing cargo binary")
        .args([
            "--gff",
            "--spsc",
            "-",
            fasta.to_str().expect("utf8 fasta"),
            query.to_str().expect("utf8 query"),
        ])
        .write_stdin(stdin)
        .assert()
        .success()
        .stdout(expected);
}

#[test]
fn translated_output_matches_c_oracle() {
    let (fasta, query) = support::fixture_paths();
    let expected = c_stdout(&[
        "--trans",
        fasta.to_str().expect("utf8 fasta"),
        query.to_str().expect("utf8 query"),
    ]);

    Command::cargo_bin("miniprot")
        .expect("missing cargo binary")
        .args([
            "--trans",
            fasta.to_str().expect("utf8 fasta"),
            query.to_str().expect("utf8 query"),
        ])
        .assert()
        .success()
        .stdout(expected);
}

#[test]
fn reference_stdin_matches_direct_rust_output() {
    let (fasta, query) = support::fixture_paths();
    let direct = Command::cargo_bin("miniprot")
        .expect("missing cargo binary")
        .args([
            fasta.to_str().expect("utf8 fasta"),
            query.to_str().expect("utf8 query"),
        ])
        .output()
        .expect("failed direct mapping");
    assert!(direct.status.success());

    Command::cargo_bin("miniprot")
        .expect("missing cargo binary")
        .args(["-", query.to_str().expect("utf8 query")])
        .write_stdin(gz_fixture_bytes(&fasta))
        .assert()
        .success()
        .stdout(direct.stdout);
}
