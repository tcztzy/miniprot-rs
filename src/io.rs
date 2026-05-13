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
pub fn write_usize_as_i64_slice<W: Write>(writer: &mut W, slice: &[usize]) -> crate::Result<()> {
    #[cfg(all(target_pointer_width = "64", target_endian = "little"))]
    {
        if slice.iter().any(|&offset| offset > i64::MAX as usize) {
            return Err(crate::Error::InvalidData(
                "index exceeds MPI limits".to_owned(),
            ));
        }
        let bytes =
            unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<u8>(), size_of_val(slice)) };
        writer.write_all(bytes).map_err(Into::into)
    }
    #[cfg(not(all(target_pointer_width = "64", target_endian = "little")))]
    {
        let raw = slice
            .iter()
            .copied()
            .map(i64::try_from)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| crate::Error::InvalidData("index exceeds MPI limits".to_owned()))?;
        write_i64_slice(writer, &raw)
    }
}

#[inline]
pub fn read_i64_slice<R: Read>(reader: &mut R, slice: &mut [i64]) -> crate::Result<()> {
    reader.read_exact(slice.as_mut_bytes()).map_err(Into::into)
}

#[inline]
pub fn read_i64_as_usize_vec<R: Read>(reader: &mut R, len: usize) -> crate::Result<Vec<usize>> {
    #[cfg(all(target_pointer_width = "64", target_endian = "little"))]
    {
        let mut out = vec![0usize; len];
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), size_of_val(&out[..]))
        };
        reader.read_exact(bytes)?;
        if out.iter().any(|&offset| offset > i64::MAX as usize) {
            return Err(crate::Error::InvalidData(
                "invalid k-mer bucket offset".to_owned(),
            ));
        }
        Ok(out)
    }
    #[cfg(not(all(target_pointer_width = "64", target_endian = "little")))]
    {
        let mut raw = vec![0i64; len];
        read_i64_slice(reader, &mut raw)?;
        raw.into_iter()
            .map(|offset| {
                usize::try_from(offset).map_err(|_| {
                    crate::Error::InvalidData("invalid k-mer bucket offset".to_owned())
                })
            })
            .collect()
    }
}

#[inline]
pub fn write_u32_slice<W: Write>(writer: &mut W, slice: &[u32]) -> crate::Result<()> {
    writer.write_all(slice.as_bytes()).map_err(Into::into)
}

#[inline]
pub fn read_u32_slice<R: Read>(reader: &mut R, slice: &mut [u32]) -> crate::Result<()> {
    reader.read_exact(slice.as_mut_bytes()).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::read_i64_as_usize_vec;

    #[test]
    fn negative_i64_offsets_are_rejected() {
        let storage = (-1i64).to_le_bytes();
        let mut bytes = storage.as_slice();
        let err = read_i64_as_usize_vec(&mut bytes, 1).unwrap_err();
        assert_eq!(err.to_string(), "invalid k-mer bucket offset");
    }
}
