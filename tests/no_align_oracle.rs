mod support;

use assert_cmd::Command;

#[test]
fn no_align_paf_matches_c_oracle() {
    support::ensure_c_oracle();
    let (fasta, query) = support::fixture_paths();

    let expected = std::process::Command::new(support::c_oracle())
        .args([
            "-A",
            fasta.to_str().expect("utf8 fasta"),
            query.to_str().expect("utf8 query"),
        ])
        .output()
        .expect("failed to run C oracle");
    assert!(expected.status.success());

    Command::cargo_bin("miniprot")
        .expect("missing cargo binary")
        .args([
            "-A",
            fasta.to_str().expect("utf8 fasta"),
            query.to_str().expect("utf8 query"),
        ])
        .assert()
        .success()
        .stdout(expected.stdout);
}

#[test]
fn no_align_matches_between_fasta_and_index_input() {
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
            "-A",
            fasta.to_str().expect("utf8 fasta"),
            query.to_str().expect("utf8 query"),
        ])
        .output()
        .expect("failed direct mapping");
    assert!(direct.status.success());

    let restored = Command::cargo_bin("miniprot")
        .expect("missing cargo binary")
        .args([
            "-A",
            index.to_str().expect("utf8 index"),
            query.to_str().expect("utf8 query"),
        ])
        .output()
        .expect("failed restored mapping");
    assert!(restored.status.success());

    assert_eq!(direct.stdout, restored.stdout);
}
