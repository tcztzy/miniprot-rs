use assert_cmd::Command;
use miniprot::MP_VERSION;

#[test]
fn version_reports_current_release() {
    let mut cmd = Command::cargo_bin("miniprot").expect("missing cargo binary");
    cmd.arg("--version")
        .assert()
        .success()
        .stdout(format!("miniprot {MP_VERSION}\n"));
}
