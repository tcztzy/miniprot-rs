use super::Error;

#[test]
fn error_display_and_sources_match_variant() {
    let io = Error::from(std::io::Error::other("disk"));
    assert_eq!(io.to_string(), "disk");
    assert!(std::error::Error::source(&io).is_some());

    let invalid_arg = Error::InvalidArgument("bad arg".to_owned());
    assert_eq!(invalid_arg.to_string(), "bad arg");
    assert!(std::error::Error::source(&invalid_arg).is_none());

    let invalid_data = Error::InvalidData("bad data".to_owned());
    assert_eq!(invalid_data.to_string(), "bad data");

    let unsupported = Error::Unsupported("nope".to_owned());
    assert_eq!(unsupported.to_string(), "nope");

    let table = Error::InvalidTranslationTable(27);
    assert_eq!(table.to_string(), "failed to find translation table 27");
}
