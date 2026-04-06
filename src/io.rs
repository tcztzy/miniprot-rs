use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::Path;

use zerocopy::IntoBytes;

#[inline]
pub fn fasta_head_name(head: &[u8], record_kind: &str) -> crate::Result<String> {
    Ok(std::str::from_utf8(head)
        .map_err(|_| crate::Error::InvalidData(format!("non-utf8 {record_kind} name")))?
        .split_ascii_whitespace()
        .next()
        .unwrap_or_default()
        .to_owned())
}

pub fn open_niffler_reader<P: AsRef<Path>>(
    path: P,
    stream_kind: &str,
) -> crate::Result<Box<dyn Read>> {
    let path = path.as_ref();
    let source: Box<dyn Read> = if path == Path::new("-") {
        Box::new(std::io::stdin())
    } else {
        Box::new(BufReader::new(File::open(path)?))
    };
    let (reader, _) = niffler::get_reader(source).map_err(|err| {
        crate::Error::InvalidData(format!(
            "failed to open {stream_kind} stream from {}: {err}",
            path.display()
        ))
    })?;
    Ok(reader)
}

pub fn open_seq_reader<P: AsRef<Path>>(path: P) -> crate::Result<Box<dyn Read>> {
    open_niffler_reader(path, "sequence")
}

#[inline]
pub fn write_i64_slice<W: Write>(writer: &mut W, slice: &[i64]) -> crate::Result<()> {
    writer.write_all(slice.as_bytes()).map_err(Into::into)
}

#[inline]
pub fn read_i64_slice<R: Read>(reader: &mut R, slice: &mut [i64]) -> crate::Result<()> {
    reader.read_exact(slice.as_mut_bytes()).map_err(Into::into)
}

#[inline]
pub fn write_u32_slice<W: Write>(writer: &mut W, slice: &[u32]) -> crate::Result<()> {
    writer.write_all(slice.as_bytes()).map_err(Into::into)
}

#[inline]
pub fn read_u32_slice<R: Read>(reader: &mut R, slice: &mut [u32]) -> crate::Result<()> {
    reader.read_exact(slice.as_mut_bytes()).map_err(Into::into)
}
