use std::io::BufReader;
use std::path::Path;

use noodles::fasta::record::Definition as NoodlesDefinition;

use crate::io::{fasta_head_name, open_niffler_reader};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QueryRecord {
    pub name: String,
    pub seq: String,
}

pub fn read_queries_path<P: AsRef<Path>>(path: P) -> crate::Result<Vec<QueryRecord>> {
    let mut records = Vec::new();
    for_each_fasta_record_path(path, "query", |name, seq| {
        let seq = std::str::from_utf8(seq)
            .map_err(|_| crate::Error::InvalidData("non-utf8 query sequence".to_owned()))?
            .to_owned();
        records.push(QueryRecord { name, seq });
        Ok(())
    })?;
    Ok(records)
}

pub(crate) fn for_each_fasta_record_path<P, F>(
    path: P,
    record_kind: &str,
    mut f: F,
) -> crate::Result<()>
where
    P: AsRef<Path>,
    F: FnMut(String, &[u8]) -> crate::Result<()>,
{
    let path = path.as_ref();
    let reader = open_niffler_reader(path, "FASTA")?;
    let mut reader = noodles::fasta::io::Reader::new(BufReader::new(reader));
    let mut definition = String::new();
    let mut sequence = Vec::new();

    loop {
        definition.clear();
        let bytes_read = reader.read_definition(&mut definition).map_err(|err| {
            if err.kind() == std::io::ErrorKind::InvalidData && err.to_string().contains("UTF-8") {
                crate::Error::InvalidData(format!("non-utf8 {record_kind} name"))
            } else {
                crate::Error::InvalidData(format!(
                    "failed to read FASTA record from {}: {err}",
                    path.display()
                ))
            }
        })?;
        if bytes_read == 0 {
            break;
        }

        sequence.clear();
        reader.read_sequence(&mut sequence).map_err(|err| {
            crate::Error::InvalidData(format!(
                "failed to read FASTA record from {}: {err}",
                path.display()
            ))
        })?;

        let definition: NoodlesDefinition = definition.parse().map_err(|err| {
            crate::Error::InvalidData(format!(
                "failed to read FASTA record from {}: {err}",
                path.display()
            ))
        })?;
        let name = fasta_head_name(definition.name(), record_kind)?;
        f(name, &sequence)?;
    }

    Ok(())
}
