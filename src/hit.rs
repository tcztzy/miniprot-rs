use crate::index::Index;
use crate::types::{Alignment, Anchor, ChainMeta, Extents, MP_BLOCK_BONUS};

fn cal_chn_sc_ungap_approx(anchors: &[Anchor], kmer: i32) -> i32 {
    let mut score = kmer;
    for pair in anchors.windows(2) {
        let (a0, a1) = (pair[0], pair[1]);
        let dq = a1.query() - a0.query();
        score += dq.min(kmer);
        if a1.target() == a0.target() {
            score += MP_BLOCK_BONUS;
        }
    }
    score
}

pub fn cal_chn_sc_ungap(a: &[Anchor], kmer: i32) -> i32 {
    let mut score = kmer;
    for pair in a.windows(2) {
        let (a0, a1) = (pair[0], pair[1]);
        let dq = a1.query() - a0.query();
        let dr3 = a1.target() - a0.target();
        let dr = dr3 / 3;
        let q = dr3 - dr * 3;
        let dg = dq.min(dr);
        if dq >= dr && q != 0 {
            score -= 1;
        } else {
            score += dg.min(kmer);
        }
    }
    score
}

pub fn reg_gen_from_block(mi: &Index, chains: &[ChainMeta], anchors: &[Anchor]) -> Vec<Alignment> {
    let mut reg = Vec::with_capacity(chains.len());
    let mut anchors = anchors;
    for &chain in chains {
        let n = chain.len;
        let (chain_anchors, rest) = anchors.split_at(n);
        anchors = rest;
        let mut is = 0usize;
        let mut ie = n - 1;
        let ts = mi
            .block2pos(
                u32::try_from(chain_anchors[is].target())
                    .expect("block anchors should be non-negative"),
            )
            .expect("block anchors should map to an index block");
        let te = mi
            .block2pos(
                u32::try_from(chain_anchors[ie].target())
                    .expect("block anchors should be non-negative"),
            )
            .expect("block anchors should map to an index block");
        let vid = if ts == te {
            ts
        } else {
            let js = chain_anchors
                .partition_point(|anchor| anchor.target() < mi.bo[ts.to_index() + 1] as i32);
            let je = chain_anchors
                .partition_point(|anchor| anchor.target() < mi.bo[te.to_index()] as i32);
            if js > n - je {
                ie = js - 1;
                ts
            } else {
                is = je;
                te
            }
        };
        let chn_sc = if ts == te {
            chain.score
        } else {
            ((chain.score as f64) * ((ie - is + 1) as f64) / (n as f64) + 0.499) as i32
        };
        let first_anchor = chain_anchors[is];
        let last_anchor = chain_anchors[ie];
        let bo = i64::from(mi.bo[vid.to_index()]);
        reg.push(Alignment {
            cnt: n,
            chn_sc,
            chn_sc_ungap: cal_chn_sc_ungap_approx(chain_anchors, mi.opt.kmer),
            vid,
            qs: first_anchor.query(),
            qe: last_anchor.query(),
            vs: (i64::from(first_anchor.target()) - bo) << mi.opt.bbit,
            ve: (i64::from(last_anchor.target()) - bo + 1) << mi.opt.bbit,
            anchors: chain_anchors.to_vec(),
            ..Alignment::default()
        });
    }
    reg
}

pub fn sort_reg(regs: &mut Vec<Alignment>) {
    if regs.len() <= 1 {
        return;
    }
    let mut src = std::mem::take(regs);
    let mut order: Vec<_> = src
        .iter()
        .enumerate()
        .filter(|&(_, reg)| reg.cnt > 0)
        .map(|(i, reg)| (reg.score(), reg.hash, i))
        .collect();
    order.sort_unstable_by_key(|item| (item.0, item.1));
    *regs = order
        .into_iter()
        .rev()
        .map(|(_, _, i)| std::mem::take(&mut src[i]))
        .collect();
}

#[inline]
fn overlap_len(s0: i32, e0: i32, s1: i32, e1: i32) -> i32 {
    (e0.min(e1) - s0.max(s1)).max(0)
}

pub fn set_parent(
    mask_level: f32,
    mask_len: i32,
    regs: &mut [Alignment],
    sub_diff: i32,
    hard_mask_level: bool,
) {
    if regs.is_empty() {
        return;
    }
    for (i, reg) in regs.iter_mut().enumerate() {
        reg.id = i;
    }
    let mut cov = Vec::with_capacity(regs.len());
    let mut w = vec![0usize; regs.len()];
    w[0] = 0;
    regs[0].parent = Some(0);
    let mut k = 1usize;
    for i in 1..regs.len() {
        let (si, ei) = (regs[i].qs, regs[i].qe);
        let mut uncov_len = 0i32;
        if !hard_mask_level {
            cov.clear();
            for &wj in &w[..k] {
                let sj = regs[wj].qs.max(si);
                let ej = regs[wj].qe.min(ei);
                if sj >= ej {
                    continue;
                }
                cov.push((sj, ej));
            }
            if !cov.is_empty() {
                cov.sort_unstable();
                let mut x = si;
                for &(start, end) in &cov {
                    if start > x {
                        uncov_len += start - x;
                    }
                    x = x.max(end);
                }
                if ei > x {
                    uncov_len += ei - x;
                }
            }
        }
        let mut parent_found = false;
        for &wj in &w[..k] {
            let (sj, ej) = (regs[wj].qs, regs[wj].qe);
            if ej <= si || sj >= ei {
                continue;
            }
            let min_len = (ej - sj).min(ei - si);
            let max_len = (ej - sj).max(ei - si);
            let ol = overlap_len(si, ei, sj, ej);
            if (ol as f32) / (min_len as f32) - (uncov_len as f32) / (max_len as f32) > mask_level
                && uncov_len <= mask_len
            {
                let sci = regs[i].score();
                regs[i].parent = regs[wj].parent;
                regs[wj].subsc = regs[wj].subsc.max(sci);
                let mut cnt_sub = regs[i].cnt >= regs[wj].cnt;
                let parent_extra = regs[wj]
                    .extra
                    .as_ref()
                    .map(|extra| (extra.dp_max, extra.dp_max2));
                let child_dp_max = regs[i].extra.as_ref().map(|extra| extra.dp_max);
                if let (Some((parent_dp_max, parent_dp_max2_old)), Some(child_dp_max)) =
                    (parent_extra, child_dp_max)
                    && (regs[wj].vid != regs[i].vid
                        || regs[wj].vs != regs[i].vs
                        || regs[wj].ve != regs[i].ve
                        || ol != min_len)
                {
                    let parent_dp_max2 = parent_dp_max2_old.max(child_dp_max);
                    regs[wj].extra.as_mut().expect("parent extra").dp_max2 = parent_dp_max2;
                    if parent_dp_max - child_dp_max <= sub_diff {
                        cnt_sub = true;
                    }
                }
                if cnt_sub {
                    regs[wj].n_sub += 1;
                }
                parent_found = true;
                break;
            }
        }
        if !parent_found {
            w[k] = i;
            k += 1;
            regs[i].parent = Some(i);
            regs[i].n_sub = 0;
        }
    }
}

pub fn sync_regs(regs: &mut [Alignment]) {
    if regs.is_empty() {
        return;
    }
    let max_id = regs.iter().map(|reg| reg.id).max().unwrap_or(0);
    let mut tmp = vec![None; max_id + 1];
    for (i, reg) in regs.iter().enumerate() {
        tmp[reg.id] = Some(i);
    }
    for (i, reg) in regs.iter_mut().enumerate() {
        let old_id = reg.id;
        let old_parent = reg.parent;
        reg.id = i;
        if old_parent == Some(old_id) {
            reg.parent = Some(i);
        } else {
            reg.parent = old_parent.and_then(|parent| tmp.get(parent).copied().flatten());
        }
    }
}

pub fn select_sub(pri_ratio: f32, min_diff: i32, best_n: i32, regs: &mut Vec<Alignment>) {
    if pri_ratio <= 0.0 || regs.is_empty() {
        return;
    }
    let chn_sc_ungap = regs.iter().map(|reg| reg.chn_sc_ungap).max().unwrap_or(-1);
    let mut n_2nd = 0i32;
    let orig_len = regs.len();
    let mut keep_indices = Vec::with_capacity(orig_len);
    for (i, reg) in regs.iter().enumerate() {
        let Some(p) = reg.parent else {
            keep_indices.push(i);
            continue;
        };
        let parent = &regs[p];
        if p == i {
            keep_indices.push(i);
            continue;
        }

        let identical = reg.qs == parent.qs
            && reg.qe == parent.qe
            && reg.vid == parent.vid
            && reg.vs == parent.vs
            && reg.ve == parent.ve;
        if identical || n_2nd >= best_n {
            continue;
        }

        let sci = reg.score();
        let scp = parent.score();
        let keep = sci as f32 >= scp as f32 * pri_ratio
            || sci + min_diff >= scp
            || (reg.extra.is_none()
                && parent.extra.is_none()
                && chn_sc_ungap > 0
                && reg.chn_sc_ungap as f32 >= chn_sc_ungap as f32 * pri_ratio);
        if keep {
            keep_indices.push(i);
            n_2nd += 1;
        }
    }
    if keep_indices.len() != orig_len {
        let kept: Vec<_> = keep_indices
            .iter()
            .map(|&i| std::mem::take(&mut regs[i]))
            .collect();
        *regs = kept;
        sync_regs(regs);
    }
}

pub fn cal_max_ext(
    nt: Option<&crate::seqdb::NtDb>,
    regs: &[Alignment],
    min_ext: i32,
    max_ext: i32,
) -> Vec<Extents> {
    if regs.is_empty() {
        return Vec::new();
    }
    let mut order: Vec<_> = regs
        .iter()
        .enumerate()
        .map(|(i, reg)| {
            let x = if let Some(nt) = nt {
                let ctg = &nt.contigs[reg.vid.contig().index()];
                reg.vs + ctg.off + if reg.vid.is_rev() { ctg.len } else { 0 }
            } else {
                reg.anchors
                    .first()
                    .map_or(0, |anchor| i64::from(anchor.target()))
            };
            (x, i)
        })
        .collect();
    order.sort_unstable_by_key(|item| item.0);

    let mut ext = vec![Extents::default(); regs.len()];
    for (i, &(_, j)) in order.iter().enumerate() {
        let reg = &regs[j];
        let mut left = max_ext;
        let mut right = max_ext;
        if let Some(&(_, prev_idx)) = i.checked_sub(1).and_then(|idx| order.get(idx))
            && regs[prev_idx].vid == reg.vid
            && regs[prev_idx].qe >= reg.qs
        {
            left = (reg.vs - regs[prev_idx].ve) as i32;
            left = left.min(max_ext).max(min_ext);
        }
        if let Some(&(_, next_idx)) = order.get(i + 1)
            && regs[next_idx].vid == reg.vid
            && reg.qe >= regs[next_idx].qs
        {
            right = (regs[next_idx].vs - reg.ve) as i32;
            right = right.min(max_ext).max(min_ext);
        }
        ext[j] = Extents::new(left, right);
    }
    ext
}
