use std::path::Path;

use crate::Index;
use crate::align::{AlignBatch, align_batches, align_regs, select_multi_exon};
use crate::chain::chain;
use crate::fastx::{QueryRecord, read_queries_path};
use crate::format::write_output;
use crate::hit::{
    cal_chn_sc_ungap, cal_max_ext, reg_gen_from_block, select_sub, set_parent, sort_reg,
};
use crate::sketch::{sketch_nt4, sketch_prot};
use crate::sort::radix_sort_anchors;
use crate::types::{
    Alignment, Anchor, MP_F_NO_ALIGN, MP_F_NO_PRE_CHAIN, MP_F_NO_SPLICE, MapOptions, lo32,
};

#[inline]
fn kmer_bucket_range(mi: &Index, entry: u64) -> std::ops::Range<usize> {
    let bucket = (entry >> 32) as usize;
    let start = mi.ki[bucket];
    let end = mi.ki.get(bucket + 1).copied().unwrap_or(mi.n_kb);
    start..end
}

fn skip_key(sketches: &[u64], mut i: usize, key: u64) -> usize {
    i += 1;
    while i < sketches.len() && (sketches[i] >> 32) == key {
        i += 1;
    }
    i
}

fn append_anchor_product(targets: &[u64], queries: &[u64], out: &mut Vec<Anchor>) {
    out.reserve(targets.len() * queries.len());
    for &left in targets {
        let prefix = (lo32(left) as u32 as u64) << 32;
        for &right in queries {
            out.push(Anchor::from_raw(prefix | (lo32(right) as u32 as u64)));
        }
    }
}

#[inline]
fn rank_regs(mi: &Index, opt: &MapOptions, pri_ratio: f32, regs: &mut Vec<Alignment>) {
    sort_reg(regs);
    set_parent(opt.mask_level, opt.mask_len, regs, mi.opt.kmer, false);
    select_sub(pri_ratio, mi.opt.kmer * 2, opt.best_n, regs);
}

fn cal_max_occ(mi: &Index, a: &[u64]) -> i32 {
    let mut counts: Vec<_> = a
        .iter()
        .map(|&anchor| kmer_bucket_range(mi, anchor).len() as u64)
        .collect();
    let q25_idx = (a.len() as f64 * 0.25 + 0.499) as usize;
    let q75_idx = (a.len() as f64 * 0.75 + 0.499) as usize;
    counts.select_nth_unstable(q25_idx);
    let q25 = counts[q25_idx];
    counts[q25_idx + 1..=q75_idx].select_nth_unstable(q75_idx - q25_idx - 1);
    let q75 = counts[q75_idx];
    (q75 as f64 + (q75 as f64 - q25 as f64) * 1.5 + 10.0) as i32
}

fn refine_reg(
    mi: &Index,
    opt: &MapOptions,
    query_sketches: &[u64],
    reg: &mut Alignment,
    left_ext: i32,
    right_ext: i32,
) {
    let io = &mi.opt;
    let is_splice = (opt.flag & MP_F_NO_SPLICE) == 0;
    let kmer = opt.kmer2;
    let ctg_len = mi.nt.contigs[reg.vid.contig().index()].len;
    let target_start = reg.vs.saturating_sub(left_ext as i64).min(ctg_len).max(0);
    let target_end = (reg.ve + right_ext as i64).min(ctg_len).max(0);
    if target_start >= target_end {
        reg.invalidate();
        return;
    }
    let mut nt = vec![0u8; (target_end - target_start) as usize];
    let Ok(nt_len) = mi.nt.get_by_v(reg.vid, target_start, target_end, &mut nt) else {
        reg.invalidate();
        return;
    };
    let nt_sketches = sketch_nt4(&mi.tables, &nt, nt_len, io.min_aa_len, kmer, 0, 0, 0);
    let max_ava = opt.max_ava as usize;
    let mut anchors: Vec<Anchor> = Vec::new();
    let mut nt_i = 0usize;
    let mut query_i = 0usize;
    while nt_i < nt_sketches.len() && query_i < query_sketches.len() {
        let nt_key = nt_sketches[nt_i] >> 32;
        let query_key = query_sketches[query_i] >> 32;
        if nt_key < query_key {
            nt_i = skip_key(&nt_sketches, nt_i, nt_key);
            continue;
        }
        if nt_key > query_key {
            query_i = skip_key(query_sketches, query_i, query_key);
            continue;
        }
        let nt_start = nt_i;
        nt_i = skip_key(&nt_sketches, nt_i, nt_key);
        let query_start = query_i;
        query_i = skip_key(query_sketches, query_i, query_key);
        let nt_group = &nt_sketches[nt_start..nt_i];
        let query_group = &query_sketches[query_start..query_i];
        if nt_group.len() * query_group.len() <= max_ava {
            append_anchor_product(nt_group, query_group, &mut anchors);
        }
    }
    radix_sort_anchors(&mut anchors);
    let Some(result) = chain(
        opt.max_intron,
        opt.max_gap,
        opt.bw,
        opt.max_chn_max_skip,
        opt.max_chn_iter,
        opt.min_chn_cnt,
        opt.min_chn_sc,
        opt.chn_coef_log,
        is_splice,
        kmer,
        0,
        anchors,
    ) else {
        reg.invalidate();
        return;
    };
    if result.chains.is_empty() {
        reg.invalidate();
        return;
    }

    let (max_i, &best_chain) = result
        .chains
        .iter()
        .enumerate()
        .max_by_key(|&(_, chain)| chain.score)
        .expect("chain result should contain at least one chain");
    let anchor_offset: usize = result.chains[..max_i].iter().map(|chain| chain.len).sum();
    let anchor_count = best_chain.len;
    let mut anchors = result.anchors[anchor_offset..anchor_offset + anchor_count].to_vec();
    reg.chn_sc = best_chain.score;
    reg.cnt = anchor_count;
    reg.qs = anchors[0].query() - (kmer - 1);
    reg.qe = anchors[anchor_count - 1].query() + 1;
    reg.vs = target_start + i64::from(anchors[0].target()) + 1 - (3 * kmer) as i64;
    reg.ve = target_start + i64::from(anchors[anchor_count - 1].target()) + 1;
    for anchor in &mut anchors {
        *anchor = anchor.with_target((i64::from(anchor.target()) + target_start - reg.vs) as i32);
    }
    reg.anchors = anchors;
    reg.chn_sc_ungap = cal_chn_sc_ungap(&reg.anchors, kmer);
}

pub fn map_protein(mi: &Index, seq: &str, opt: &MapOptions) -> crate::Result<Vec<Alignment>> {
    let is_splice = (opt.flag & MP_F_NO_SPLICE) == 0;
    let mut query_sketches = sketch_prot(&mi.tables, seq.as_bytes(), mi.opt.kmer, mi.opt.mod_bit);
    crate::sort::radix_sort_u64(&mut query_sketches);

    let max_occ = if query_sketches.len() >= 8 {
        cal_max_occ(mi, &query_sketches).min(opt.max_occ)
    } else {
        opt.max_occ
    } as i64;

    let mut anchors: Vec<Anchor> = Vec::new();
    for &entry in &query_sketches {
        let range = kmer_bucket_range(mi, entry);
        if range.len() as i64 <= max_occ {
            anchors.extend(mi.kb[range].iter().map(|&target| {
                Anchor::from_raw(((u64::from(target)) << 32) | (lo32(entry) as u32 as u64))
            }));
        }
    }
    radix_sort_anchors(&mut anchors);

    if (opt.flag & MP_F_NO_PRE_CHAIN) == 0 && is_splice {
        let block_width = 1 << mi.opt.bbit;
        if let Some(result) = chain(
            block_width,
            block_width,
            block_width,
            opt.max_chn_max_skip,
            opt.max_chn_iter,
            2,
            0,
            opt.chn_coef_log,
            true,
            mi.opt.kmer,
            mi.opt.bbit,
            std::mem::take(&mut anchors),
        ) {
            anchors = result.anchors;
            radix_sort_anchors(&mut anchors);
        }
    }

    let Some(result) = chain(
        opt.max_intron,
        opt.max_gap,
        opt.bw,
        opt.max_chn_max_skip,
        opt.max_chn_iter,
        opt.min_chn_cnt,
        opt.min_chn_sc,
        opt.chn_coef_log,
        is_splice,
        mi.opt.kmer,
        mi.opt.bbit,
        anchors,
    ) else {
        return Ok(Vec::new());
    };

    let mut regs = reg_gen_from_block(mi, &result.chains, &result.anchors);
    rank_regs(mi, opt, opt.pri_ratio * opt.pri_ratio, &mut regs);

    let ext = cal_max_ext(None, &regs, 100, opt.max_ext);
    let mut refine_query_sketches = sketch_prot(&mi.tables, seq.as_bytes(), opt.kmer2, 0);
    crate::sort::radix_sort_u64(&mut refine_query_sketches);
    let mut refined: Vec<_> = regs
        .into_iter()
        .zip(ext)
        .filter_map(|(mut reg, ext)| {
            refine_reg(
                mi,
                opt,
                &refine_query_sketches,
                &mut reg,
                ext.left,
                ext.right,
            );
            (reg.cnt > 0).then_some(reg)
        })
        .collect();
    rank_regs(mi, opt, opt.pri_ratio * opt.pri_ratio, &mut refined);
    Ok(refined)
}

pub fn map_file<P: AsRef<Path>>(mi: &Index, path: P, opt: &MapOptions) -> crate::Result<String> {
    map_file_threads(mi, path, opt, 1)
}

pub fn map_file_threads<P: AsRef<Path>>(
    mi: &Index,
    path: P,
    opt: &MapOptions,
    threads: i32,
) -> crate::Result<String> {
    let queries = read_queries_path(path)?;
    map_queries(mi, &queries, opt, threads)
}

fn map_queries(
    mi: &Index,
    queries: &[QueryRecord],
    opt: &MapOptions,
    threads: i32,
) -> crate::Result<String> {
    use rayon::prelude::*;

    let mut out = String::new();
    if (opt.flag & crate::types::MP_F_GFF) != 0 {
        out.push_str("##gff-version 3\n");
    }

    let map_one = |query: &QueryRecord| -> crate::Result<Vec<Alignment>> {
        let mut regs = map_protein(mi, &query.seq, opt)?;
        if (opt.flag & MP_F_NO_ALIGN) == 0 {
            let ext = cal_max_ext(Some(&mi.nt), &regs, 100, opt.max_intron / 2);
            regs = align_regs(
                mi,
                opt,
                query.seq.len() as i32,
                query.seq.as_bytes(),
                regs,
                ext,
            );
            sort_reg(&mut regs);
            select_multi_exon(&mut regs, opt.io);
            rank_regs(mi, opt, opt.pri_ratio, &mut regs);
        }
        Ok(regs)
    };

    let mut next_id = 1i64;
    let mut write_regs = |out: &mut String, query: &QueryRecord, regs: &mut Vec<Alignment>| {
        write_query_regs(out, mi, query, regs, opt, &mut next_id);
    };

    if opt.use_gpu && crate::cuda_dp::available() && (opt.flag & MP_F_NO_ALIGN) == 0 {
        let map_candidates = |query: &QueryRecord| -> crate::Result<Vec<Alignment>> {
            map_protein(mi, &query.seq, opt)
        };
        let process_aligned =
            |query: &QueryRecord, regs: &mut Vec<Alignment>| -> crate::Result<()> {
                sort_reg(regs);
                select_multi_exon(regs, opt.io);
                rank_regs(mi, opt, opt.pri_ratio, regs);
                let _ = query;
                Ok(())
            };

        if threads <= 1 {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .map_err(|err| {
                    crate::Error::InvalidArgument(format!(
                        "failed to build mapping thread pool: {err}"
                    ))
                })?;
            let chunk_size = 1024usize;
            for chunk in queries.chunks(chunk_size) {
                let chunk_regs = chunk
                    .iter()
                    .map(map_candidates)
                    .collect::<crate::Result<Vec<_>>>()?;
                let groups: Vec<_> = chunk
                    .iter()
                    .zip(chunk_regs)
                    .map(|(query, regs)| {
                        let ext = cal_max_ext(Some(&mi.nt), &regs, 100, opt.max_intron / 2);
                        AlignBatch {
                            qlen: query.seq.len() as i32,
                            aa: query.seq.as_bytes(),
                            regs,
                            ext,
                        }
                    })
                    .collect();
                let mut aligned = pool.install(|| align_batches(mi, opt, groups));
                for (query, regs) in chunk.iter().zip(aligned.iter_mut()) {
                    process_aligned(query, regs)?;
                    write_regs(&mut out, query, regs);
                }
            }
        } else {
            let threads = threads as usize;
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|err| {
                    crate::Error::InvalidArgument(format!(
                        "failed to build mapping thread pool: {err}"
                    ))
                })?;
            let chunk_size = (threads * 1024).clamp(1024, 4096);
            for chunk in queries.chunks(chunk_size) {
                let chunk_regs = pool.install(|| {
                    chunk
                        .par_iter()
                        .map(map_candidates)
                        .collect::<crate::Result<Vec<_>>>()
                })?;
                let groups: Vec<_> = chunk
                    .iter()
                    .zip(chunk_regs)
                    .map(|(query, regs)| {
                        let ext = cal_max_ext(Some(&mi.nt), &regs, 100, opt.max_intron / 2);
                        AlignBatch {
                            qlen: query.seq.len() as i32,
                            aa: query.seq.as_bytes(),
                            regs,
                            ext,
                        }
                    })
                    .collect();
                let mut aligned = pool.install(|| align_batches(mi, opt, groups));
                for (query, regs) in chunk.iter().zip(aligned.iter_mut()) {
                    process_aligned(query, regs)?;
                    write_regs(&mut out, query, regs);
                }
            }
        }
        return Ok(out);
    }

    if threads <= 1 {
        for query in queries {
            let mut regs = map_one(query)?;
            write_regs(&mut out, query, &mut regs);
        }
    } else {
        let threads = threads as usize;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|err| {
                crate::Error::InvalidArgument(format!("failed to build mapping thread pool: {err}"))
            })?;
        let chunk_size = (threads * 1024).clamp(1024, 4096);
        for chunk in queries.chunks(chunk_size) {
            let mut chunk_regs = pool.install(|| {
                chunk
                    .par_iter()
                    .map(map_one)
                    .collect::<crate::Result<Vec<_>>>()
            })?;
            for (query, regs) in chunk.iter().zip(chunk_regs.iter_mut()) {
                write_regs(&mut out, query, regs);
            }
        }
    }
    Ok(out)
}

fn write_query_regs(
    out: &mut String,
    mi: &Index,
    query: &QueryRecord,
    regs: &mut [Alignment],
    opt: &MapOptions,
    next_id: &mut i64,
) {
    if regs.is_empty() {
        write_output(out, mi, query, None, opt, 0, 0);
        return;
    }
    let best_sc = regs[0].score();
    let mut n_out = 0i32;
    for (j, reg) in regs.iter().take(opt.out_n as usize).enumerate() {
        let sc = reg.score();
        if sc <= 0 || (sc as f64) < (best_sc as f64) * (opt.out_sim as f64) {
            continue;
        }
        if ((reg.qe - reg.qs) as f64) < (query.seq.len() as f64) * (opt.out_cov as f64) {
            continue;
        }
        write_output(out, mi, query, Some(reg), opt, *next_id, j as i32 + 1);
        *next_id += 1;
        n_out += 1;
    }
    if n_out == 0 {
        write_output(out, mi, query, None, opt, 0, 0);
    }
}
