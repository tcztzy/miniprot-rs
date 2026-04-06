use crate::seqdb::NtDb;
use crate::tables::Tables;
use crate::types::{IndexOptions, MP_BITS_PER_AA, VirtualId};

fn hash32_mask(mut key: u32, mask: u32) -> u32 {
    key = (key.wrapping_add(!(key << 15))) & mask;
    key ^= key >> 10;
    key = (key.wrapping_add(key << 3)) & mask;
    key ^= key >> 6;
    key = (key.wrapping_add(!(key << 11))) & mask;
    key ^= key >> 16;
    key
}

pub fn sketch_prot(tables: &Tables, seq: &[u8], kmer: i32, mod_bit: i32) -> Vec<u64> {
    let mut out = Vec::with_capacity(seq.len() / 4);
    let mask_k = (1u32 << (kmer as usize * MP_BITS_PER_AA)) - 1;
    let mask_mod = (1u32 << mod_bit) - 1;
    let mut x = 0u32;
    let mut l = 0i32;
    for (i, &byte) in seq.iter().enumerate() {
        let c = tables.aa13[byte as usize];
        if c < 14 {
            x = ((x << MP_BITS_PER_AA) | c as u32) & mask_k;
            l += 1;
            if l >= kmer {
                let y = hash32_mask(x, mask_k);
                if (y & mask_mod) == 0 {
                    out.push(((y >> mod_bit) as u64) << 32 | i as u64);
                }
            }
        } else {
            x = 0;
            l = 0;
        }
    }
    out
}

#[inline(always)]
fn packed_nt(seq: &[u8], pos: i64) -> u8 {
    (seq[(pos >> 1) as usize] >> ((pos & 1) * 4)) & 0x0f
}

#[allow(clippy::too_many_arguments)]
fn sketch_clean_orf<F>(
    tables: &Tables,
    st: i64,
    en: i64,
    kmer: i32,
    mod_bit: i32,
    bbit: i32,
    boff: i64,
    nt4_at: F,
    out: &mut Vec<u64>,
) where
    F: Fn(usize) -> u8 + Copy,
{
    let mask_k = (1u32 << (kmer as usize * MP_BITS_PER_AA)) - 1;
    let mask_mod = (1u32 << mod_bit) - 1;
    let mut x = 0u32;
    let mut l = 0i32;
    for offset in 0..((en - st) / 3) as usize {
        let pos = st as usize + offset * 3;
        let codon = (nt4_at(pos) << 4) | (nt4_at(pos + 1) << 2) | nt4_at(pos + 2);
        x = ((x << MP_BITS_PER_AA) | tables.codon13[codon as usize] as u32) & mask_k;
        l += 1;
        if l >= kmer {
            let y = hash32_mask(x, mask_k);
            if (y & mask_mod) == 0 {
                out.push(
                    ((y >> mod_bit) as u64) << 32 | (((((pos as i64) + 2) >> bbit) + boff) as u64),
                );
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn flush_orf<F>(
    tables: &Tables,
    min_aa_len: i32,
    kmer: i32,
    mod_bit: i32,
    bbit: i32,
    boff: i64,
    q: usize,
    k: &mut [i64; 3],
    e: &mut [i64; 3],
    nt4_at: F,
    out: &mut Vec<u64>,
) where
    F: Fn(usize) -> u8 + Copy,
{
    if k[q] >= min_aa_len as i64 {
        sketch_clean_orf(
            tables,
            e[q] + 1 - k[q] * 3,
            e[q] + 1,
            kmer,
            mod_bit,
            bbit,
            boff,
            nt4_at,
            out,
        );
    }
    k[q] = 0;
    e[q] = -1;
}

#[allow(clippy::too_many_arguments)]
fn sketch_nt4_core<F>(
    tables: &Tables,
    len: i64,
    min_aa_len: i32,
    kmer: i32,
    mod_bit: i32,
    bbit: i32,
    boff: i64,
    nt4_at: F,
) -> Vec<u64>
where
    F: Fn(usize) -> u8 + Copy,
{
    let mut out = Vec::with_capacity((len >> 2).max(0) as usize);
    let mut e = [-1i64; 3];
    let mut k = [0i64; 3];
    let mut p = 1usize;
    let mut codon = 0u8;
    let mut l = 0i64;
    let flush_all = |k: &mut [i64; 3], e: &mut [i64; 3], out: &mut Vec<u64>| {
        for q in 0..3 {
            flush_orf(
                tables, min_aa_len, kmer, mod_bit, bbit, boff, q, k, e, nt4_at, out,
            );
        }
    };

    for i in 0..len as usize {
        let base = nt4_at(i);
        if p == 3 {
            p = 0;
        }
        if base < 4 {
            codon = ((codon << 2) | base) & 0x3f;
            l += 1;
            if l >= 3 {
                let aa = tables.codon[codon as usize];
                if aa >= 20 {
                    flush_orf(
                        tables, min_aa_len, kmer, mod_bit, bbit, boff, p, &mut k, &mut e, nt4_at,
                        &mut out,
                    );
                } else {
                    e[p] = i as i64;
                    k[p] += 1;
                }
            }
        } else {
            flush_all(&mut k, &mut e, &mut out);
            l = 0;
            codon = 0;
        }
        p += 1;
    }

    flush_all(&mut k, &mut e, &mut out);
    if out.len() <= 1 {
        return out;
    }
    out.sort_unstable();
    out.dedup();
    out
}

#[allow(clippy::too_many_arguments)]
pub fn sketch_nt4(
    tables: &Tables,
    seq: &[u8],
    len: i64,
    min_aa_len: i32,
    kmer: i32,
    mod_bit: i32,
    bbit: i32,
    boff: i64,
) -> Vec<u64> {
    sketch_nt4_core(tables, len, min_aa_len, kmer, mod_bit, bbit, boff, |i| {
        seq[i]
    })
}

#[allow(clippy::too_many_arguments)]
fn sketch_nt4_packed_db(
    db: &NtDb,
    vid: VirtualId,
    tables: &Tables,
    min_aa_len: i32,
    kmer: i32,
    mod_bit: i32,
    bbit: i32,
    boff: i64,
) -> Vec<u64> {
    let ctg = &db.contigs[vid.contig().index()];
    let off = ctg.off;
    let len = ctg.len;
    let rev = vid.is_rev();
    let end = off + len - 1;
    sketch_nt4_core(tables, len, min_aa_len, kmer, mod_bit, bbit, boff, |i| {
        let pos = if rev { end - i as i64 } else { off + i as i64 };
        let base = packed_nt(&db.seq, pos);
        if rev && base < 4 { 3 - base } else { base }
    })
}

pub fn collect_nt_sketches(
    db: &NtDb,
    opt: &IndexOptions,
    tables: &Tables,
    bo: &[u32],
    threads: i32,
) -> Vec<Vec<u64>> {
    use rayon::prelude::*;

    let make_sketches = |idx: usize| {
        let vid = VirtualId::from_index(idx).expect("virtual id index should be valid");
        sketch_nt4_packed_db(
            db,
            vid,
            tables,
            opt.min_aa_len,
            opt.kmer,
            opt.mod_bit,
            opt.bbit,
            bo[vid.to_index()] as i64,
        )
    };

    if threads <= 1 {
        (0..db.contigs.len() * 2).map(make_sketches).collect()
    } else {
        (0..db.contigs.len() * 2)
            .into_par_iter()
            .map(make_sketches)
            .collect()
    }
}

pub fn collect_nt_sketches_flat(
    db: &NtDb,
    opt: &IndexOptions,
    tables: &Tables,
    bo: &[u32],
) -> Vec<u64> {
    let mut all = Vec::new();
    for idx in 0..db.contigs.len() * 2 {
        let vid = VirtualId::from_index(idx).expect("virtual id index should be valid");
        let sketches = sketch_nt4_packed_db(
            db,
            vid,
            tables,
            opt.min_aa_len,
            opt.kmer,
            opt.mod_bit,
            opt.bbit,
            bo[vid.to_index()] as i64,
        );
        all.extend(sketches);
    }
    all
}
