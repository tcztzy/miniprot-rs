use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};

use crate::fastx::for_each_fasta_record_path;
use crate::io::{open_seq_reader, read_i64_slice, write_i64_slice};
use crate::tables::{NS_SPSC_OFFSET, Tables};
use crate::{ContigId, VirtualId};

#[inline]
fn packed_nt(seq: &[u8], pos: i64) -> u8 {
    (seq[(pos >> 1) as usize] >> ((pos & 1) * 4)) & 0x0f
}

#[derive(Clone, Debug)]
pub struct Contig {
    pub off: i64,
    pub len: i64,
    pub name: String,
}

#[derive(Clone, Debug, Default)]
pub struct SpscTrack {
    pub entries: Vec<u64>,
}

#[derive(Clone, Debug, Default)]
pub struct NtDb {
    pub contigs: Vec<Contig>,
    pub l_name: i32,
    pub l_seq: i64,
    pub seq: Vec<u8>,
    pub names: Vec<u8>,
    pub name_to_id: Option<HashMap<String, ContigId>>,
    pub spsc: Option<Vec<SpscTrack>>,
}

fn spsc_find_intv(a: &[u64], x: i64) -> isize {
    a.partition_point(|&entry| ((entry >> 8) as i64) <= x) as isize - 1
}

impl NtDb {
    pub fn read_path<P: AsRef<std::path::Path>>(path: P, tables: &Tables) -> crate::Result<Self> {
        let mut db = NtDb::default();
        for_each_fasta_record_path(path, "contig", |name, seq| {
            db.push_contig(name, seq, tables);
            Ok(())
        })?;
        db.rebuild_name_blob();
        Ok(db)
    }

    fn push_contig(&mut self, name: String, seq: &[u8], tables: &Tables) {
        let off = self.l_seq;
        self.contigs.push(Contig {
            off,
            len: seq.len() as i64,
            name,
        });
        let needed = ((self.l_seq + seq.len() as i64 + 1) >> 1) as usize;
        if needed > self.seq.len() {
            self.seq.resize(needed, 0);
        }
        for (i, &base) in seq.iter().enumerate() {
            let pos = off + i as i64;
            let nibble = tables.nt4[base as usize];
            let slot = (pos >> 1) as usize;
            self.seq[slot] |= nibble << ((pos & 1) * 4);
        }
        self.l_seq += seq.len() as i64;
    }

    pub fn get(
        &self,
        cid: ContigId,
        st: i64,
        en: i64,
        rev: bool,
        out: &mut [u8],
    ) -> crate::Result<i64> {
        if cid.index() >= self.contigs.len() {
            return Err(crate::Error::InvalidArgument(
                "invalid contig id".to_owned(),
            ));
        }
        let ctg = &self.contigs[cid.index()];
        let en = if en < 0 || en > ctg.len { ctg.len } else { en };
        let s = ctg.off + st;
        let e = ctg.off + en;
        let len = (e - s) as usize;
        let out = &mut out[..len];
        if !rev {
            for (dst, pos) in out.iter_mut().zip(s..e) {
                *dst = packed_nt(&self.seq, pos);
            }
        } else {
            for (dst, pos) in out.iter_mut().zip((s..e).rev()) {
                let base = packed_nt(&self.seq, pos);
                *dst = if base >= 4 { base } else { 3 - base };
            }
        }
        Ok(len as i64)
    }

    #[inline]
    fn resolve_virtual_interval(
        &self,
        vid: VirtualId,
        st: i64,
        en: i64,
        err: &'static str,
    ) -> crate::Result<(ContigId, i64, i64, bool)> {
        let cid = vid.contig();
        if cid.index() >= self.contigs.len() {
            return Err(crate::Error::InvalidArgument(err.to_owned()));
        }
        let ctg_len = self.contigs[cid.index()].len;
        if st < 0 || en < 0 || st > ctg_len {
            return Err(crate::Error::InvalidArgument(err.to_owned()));
        }
        let en = en.min(ctg_len);
        let rev = vid.is_rev();
        Ok(if !rev {
            (cid, st, en, rev)
        } else {
            (cid, ctg_len - en, ctg_len - st, rev)
        })
    }

    pub fn get_by_v(&self, vid: VirtualId, st: i64, en: i64, out: &mut [u8]) -> crate::Result<i64> {
        let (cid, st, en, rev) =
            self.resolve_virtual_interval(vid, st, en, "invalid virtual interval")?;
        self.get(cid, st, en, rev, out)
    }

    pub fn dump<W: Write>(&self, writer: &mut W) -> crate::Result<()> {
        let x0 = self.contigs.len() as i32;
        let x1 = self.names.len() as i32;
        writer.write_all(&x0.to_le_bytes())?;
        writer.write_all(&x1.to_le_bytes())?;
        writer.write_all(&self.l_seq.to_le_bytes())?;
        let lengths: Vec<i64> = self.contigs.iter().map(|ctg| ctg.len).collect();
        write_i64_slice(writer, &lengths)?;
        writer.write_all(&self.seq[..((self.l_seq + 1) >> 1) as usize])?;
        writer.write_all(&self.names)?;
        Ok(())
    }

    pub fn restore<R: Read>(reader: &mut R) -> crate::Result<Self> {
        let mut i32_buf = [0u8; 4];
        let mut i64_buf = [0u8; 8];

        reader.read_exact(&mut i32_buf)?;
        let n_ctg = i32::from_le_bytes(i32_buf) as usize;
        reader.read_exact(&mut i32_buf)?;
        let l_name = i32::from_le_bytes(i32_buf);
        reader.read_exact(&mut i64_buf)?;
        let l_seq = i64::from_le_bytes(i64_buf);

        let mut lengths = vec![0i64; n_ctg];
        read_i64_slice(reader, &mut lengths)?;
        let mut off = 0i64;
        let mut contigs: Vec<_> = lengths
            .into_iter()
            .map(|len| {
                let contig = Contig {
                    off,
                    len,
                    name: String::new(),
                };
                off += len;
                contig
            })
            .collect();

        let packed_len = ((l_seq + 1) >> 1) as usize;
        let mut seq = vec![0u8; packed_len];
        let mut names = vec![0u8; l_name as usize];
        reader.read_exact(&mut seq)?;
        reader.read_exact(&mut names)?;
        if !names.is_empty()
            && (!names.ends_with(&[0]) || names.iter().filter(|&&byte| byte == 0).count() != n_ctg)
        {
            return Err(crate::Error::InvalidData(
                "unterminated name blob".to_owned(),
            ));
        }

        for (ctg, name) in contigs.iter_mut().zip(names.split(|&byte| byte == 0)) {
            ctg.name = String::from_utf8(name.to_vec())
                .map_err(|_| crate::Error::InvalidData("non-utf8 restored name".to_owned()))?;
        }

        Ok(Self {
            contigs,
            l_name,
            l_seq,
            seq,
            names,
            name_to_id: None,
            spsc: None,
        })
    }

    pub fn index_names(&mut self) {
        self.name_to_id = Some(
            self.contigs
                .iter()
                .enumerate()
                .map(|(i, ctg)| (ctg.name.clone(), ContigId::new(i)))
                .collect(),
        );
    }

    pub fn has_spsc(&self) -> bool {
        self.spsc.is_some()
    }

    pub fn read_spsc_path(&mut self, path: &str, mut max_sc: i32) -> crate::Result<()> {
        max_sc = max_sc.min(63);
        self.index_names();
        let Some(name_to_id) = self.name_to_id.as_ref() else {
            return Err(crate::Error::InvalidData(
                "sequence name index is not available".to_owned(),
            ));
        };
        let mut tracks = vec![SpscTrack::default(); self.contigs.len() * 2];
        let reader = open_seq_reader(path)?;
        let reader = BufReader::new(reader);
        for line in reader.lines() {
            let line = line?;
            let mut fields = line.split('\t');
            let (Some(name), Some(pos_str), Some(strand_str), Some(type_str), Some(score_str)) = (
                fields.next(),
                fields.next(),
                fields.next(),
                fields.next(),
                fields.next(),
            ) else {
                continue;
            };
            let Some(&cid) = name_to_id.get(name) else {
                continue;
            };
            let Ok(mut pos) = pos_str.parse::<i64>() else {
                continue;
            };
            let strand = match strand_str.as_bytes().first().copied() {
                Some(b'+') => 1,
                Some(b'-') => -1,
                _ => 0,
            };
            let type_code = match type_str.as_bytes().first().copied() {
                Some(b'D') => 0u64,
                Some(b'A') => 1u64,
                _ => continue,
            };
            let Ok(mut score) = score_str.parse::<i32>() else {
                continue;
            };
            if strand == 0 || pos < 0 {
                continue;
            }
            score = score.clamp(-max_sc, max_sc);
            let ctg_len = self.contigs[cid.index()].len;
            if strand < 0 {
                pos = ctg_len - pos;
            }
            if pos <= 0 || pos >= ctg_len {
                continue;
            }
            let vid = if strand < 0 {
                VirtualId::reverse(cid)
            } else {
                VirtualId::forward(cid)
            };
            let idx = vid.to_index();
            tracks[idx].entries.push(
                ((pos as u64) << 8) | (((score + NS_SPSC_OFFSET as i32) as u64) << 1) | type_code,
            );
        }
        for track in &mut tracks {
            track.entries.sort_unstable();
        }
        self.spsc = Some(tracks);
        Ok(())
    }

    pub fn spsc_get(
        &self,
        cid: ContigId,
        st0: i64,
        en0: i64,
        rev: bool,
        out: &mut [u8],
    ) -> crate::Result<i64> {
        let Some(tracks) = self.spsc.as_ref() else {
            return Err(crate::Error::InvalidArgument(
                "splice-score tracks are not loaded".to_owned(),
            ));
        };
        if cid.index() >= self.contigs.len() {
            return Err(crate::Error::InvalidArgument(
                "invalid contig id".to_owned(),
            ));
        }
        let ctg = &self.contigs[cid.index()];
        let en0 = if en0 < 0 || en0 > ctg.len {
            ctg.len
        } else {
            en0
        };
        let (st, en) = if !rev {
            (st0, en0)
        } else {
            (ctg.len - en0, ctg.len - st0)
        };
        let len = (en - st) as usize;
        out[..len].fill(0xff);
        let track = &tracks[if rev {
            VirtualId::reverse(cid).to_index()
        } else {
            VirtualId::forward(cid).to_index()
        }];
        if !track.entries.is_empty() {
            let l = spsc_find_intv(&track.entries, st);
            let r = spsc_find_intv(&track.entries, en);
            let start = (l + 1).max(0) as usize;
            let end = r.max(-1) as usize;
            if start <= end {
                for entry in &track.entries[start..=end] {
                    let x = ((entry >> 8) as i64 - st) as usize;
                    let score = *entry as u8;
                    if x == len {
                        continue;
                    }
                    if out[x] == 0xff || out[x] < score {
                        out[x] = score;
                    }
                }
            }
        }
        Ok(en - st)
    }

    pub fn spsc_get_by_v(
        &self,
        vid: VirtualId,
        st: i64,
        en: i64,
        out: &mut [u8],
    ) -> crate::Result<i64> {
        let (cid, st, en, rev) =
            self.resolve_virtual_interval(vid, st, en, "invalid virtual splice-score interval")?;
        self.spsc_get(cid, st, en, rev, out)
    }

    fn rebuild_name_blob(&mut self) {
        self.names.clear();
        for ctg in &self.contigs {
            self.names.extend_from_slice(ctg.name.as_bytes());
            self.names.push(0);
        }
        self.l_name = self.names.len() as i32;
    }
}
