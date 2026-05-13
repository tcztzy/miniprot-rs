use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::Path;

use crate::io::{read_i64_as_usize_vec, read_u32_slice, write_u32_slice, write_usize_as_i64_slice};
use crate::seqdb::NtDb;
use crate::sketch::{collect_nt_sketches, collect_nt_sketches_flat};
use crate::tables::{Tables, make_tables};
use crate::types::{IndexOptions, MP_IDX_MAGIC, VirtualId};

#[derive(Clone, Debug)]
pub struct Index {
    pub opt: IndexOptions,
    pub n_block: u32,
    pub nt: NtDb,
    pub n_kb: usize,
    pub ki: Vec<usize>,
    pub bo: Vec<u32>,
    pub kb: Vec<u32>,
    pub tables: Tables,
}

impl Index {
    pub fn load<P: AsRef<Path>>(path: P, io: &IndexOptions, _threads: i32) -> crate::Result<Self> {
        if is_index_file(path.as_ref())? {
            Self::restore(path)
        } else {
            Self::build(path, io, _threads)
        }
    }

    pub fn build<P: AsRef<Path>>(path: P, io: &IndexOptions, threads: i32) -> crate::Result<Self> {
        let tables = make_tables(io.trans_code as i32)?;
        let nt = NtDb::read_path(path, &tables)?;
        let (bo, n_block) = idx_boff(&nt, io.bbit);
        let mut idx = Self {
            opt: *io,
            n_block,
            nt,
            n_kb: 0,
            ki: Vec::new(),
            bo,
            kb: Vec::new(),
            tables,
        };
        if threads <= 1 {
            let sketches = collect_nt_sketches_flat(&idx.nt, io, &idx.tables, &idx.bo);
            idx.build_bidx_inner(sketches.iter().copied());
        } else {
            let sketches = collect_nt_sketches(&idx.nt, io, &idx.tables, &idx.bo, threads);
            idx.build_bidx_inner(sketches.iter().flatten().copied());
        }
        Ok(idx)
    }

    pub fn dump<P: AsRef<Path>>(&self, path: P) -> crate::Result<()> {
        let mut fp = File::create(path)?;
        self.dump_writer(&mut fp)
    }

    pub fn set_spsc_path(
        &mut self,
        path: &str,
        mo: &mut crate::types::MapOptions,
        keep_io: bool,
    ) -> crate::Result<()> {
        if !keep_io {
            mo.io += 10;
            mo.io_end += 10;
        }
        let max_sc = ((mo.io + 1) / 2 - 1)
            .min(mo.io - mo.go)
            .min(mo.sp_max_bonus);
        self.nt.read_spsc_path(path, max_sc)
    }

    pub fn dump_writer<W: Write>(&self, writer: &mut W) -> crate::Result<()> {
        writer.write_all(MP_IDX_MAGIC)?;
        writer.write_all(self.opt.as_bytes())?;
        writer.write_all(&usize_to_i64(self.n_kb)?.to_le_bytes())?;
        self.nt.dump(writer)?;
        write_usize_as_i64_slice(writer, &self.ki)?;
        write_u32_slice(writer, &self.kb)?;
        Ok(())
    }

    pub fn restore<P: AsRef<Path>>(path: P) -> crate::Result<Self> {
        // Large buffer (1 MiB) avoids excessive read() syscalls on the 7+ GiB index.
        let mut reader = BufReader::with_capacity(1 << 20, File::open(path)?);
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != MP_IDX_MAGIC {
            return Err(crate::Error::InvalidData("invalid MPI magic".to_owned()));
        }
        let opt = IndexOptions::from_reader(&mut reader)?;
        let tables = make_tables(opt.trans_code as i32)?;
        let mut buf8 = [0u8; 8];
        reader.read_exact(&mut buf8)?;
        let n_kb = i64_to_usize(i64::from_le_bytes(buf8), "invalid k-mer pair count")?;
        let nt = NtDb::restore(&mut reader)?;
        let ki_len = opt.n_bucket();
        let ki = read_i64_as_usize_vec(&mut reader, ki_len)?;
        let mut kb = vec![0u32; n_kb];
        read_u32_slice(&mut reader, &mut kb)?;
        let (bo, n_block) = idx_boff(&nt, opt.bbit);
        Ok(Self {
            opt,
            n_block,
            nt,
            n_kb,
            ki,
            bo,
            kb,
            tables,
        })
    }

    pub fn block2pos(&self, b: u32) -> Option<VirtualId> {
        block2pos_core(&self.bo, b).and_then(VirtualId::from_index)
    }

    fn build_bidx_inner(&mut self, iter: impl Iterator<Item = u64> + Clone) {
        self.ki = vec![0usize; self.opt.n_bucket()];
        for entry in iter.clone() {
            self.ki[(entry >> 32) as usize] += 1;
        }
        let mut next_start = 0usize;
        for start in &mut self.ki {
            let count = *start;
            *start = next_start;
            next_start += count;
        }
        self.n_kb = next_start;
        self.kb = vec![0u32; self.n_kb];
        for entry in iter {
            let idx = (entry >> 32) as usize;
            let pos = &mut self.ki[idx];
            self.kb[*pos] = entry as u32;
            *pos += 1;
        }
        if self.ki.len() > 1 {
            let last = self.ki.len() - 1;
            self.ki.copy_within(..last, 1);
        }
        if let Some(first) = self.ki.first_mut() {
            *first = 0;
        }
    }
}

pub fn idx_boff(db: &NtDb, bbit: i32) -> (Vec<u32>, u32) {
    let mut boff = 0u32;
    let mut bo = Vec::with_capacity(db.contigs.len() * 2 + 1);
    for ctg in &db.contigs {
        let block_span = ((ctg.len + ((1i64 << bbit) - 1)) >> bbit) as u32;
        bo.push(boff);
        boff += block_span;
        bo.push(boff);
        boff += block_span;
    }
    bo.push(boff);
    (bo, boff)
}

fn block2pos_core(offsets: &[u32], b: u32) -> Option<usize> {
    let &last = offsets.last()?;
    if b >= last {
        return None;
    }
    let idx = offsets.partition_point(|&offset| offset <= b);
    idx.checked_sub(1)
}

fn is_index_file(path: &Path) -> crate::Result<bool> {
    if path == Path::new("-") {
        return Ok(false);
    }
    let mut file = File::open(path)?;
    let mut magic = [0u8; 4];
    if file.read_exact(&mut magic).is_err() {
        return Ok(false);
    }
    Ok(&magic == MP_IDX_MAGIC)
}

#[inline]
fn i64_to_usize(value: i64, err: &'static str) -> crate::Result<usize> {
    usize::try_from(value).map_err(|_| crate::Error::InvalidData(err.to_owned()))
}

#[inline]
fn usize_to_i64(value: usize) -> crate::Result<i64> {
    i64::try_from(value)
        .map_err(|_| crate::Error::InvalidData("index exceeds MPI limits".to_owned()))
}
