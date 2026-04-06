use crate::types::{Anchor, ChainMeta, MP_BLOCK_BONUS};

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ScoredAnchor {
    pub score: i32,
    pub index: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ChainResult {
    pub anchors: Vec<Anchor>,
    pub chains: Vec<ChainMeta>,
}

fn chain_bk_end(
    max_drop: i32,
    z: &[ScoredAnchor],
    f: &[i32],
    p: &[Option<usize>],
    t: &mut [i32],
    k: usize,
) -> Option<usize> {
    let mut i = Some(z[k].index);
    let mut max_i = i;
    let mut max_s = 0i32;
    if t[z[k].index] != 0 {
        return i;
    }
    let end_i = loop {
        let idx = i.expect("active backtrack index");
        t[idx] = 2;
        let next = p[idx];
        i = next;
        let s = i.map(|i| z[k].score - f[i]).unwrap_or(z[k].score);
        if s > max_s {
            max_s = s;
            max_i = i;
        } else if max_s - s > max_drop {
            break next;
        }
        if i.is_none_or(|idx| t[idx] != 0) {
            break next;
        }
    };
    let mut cur = Some(z[k].index);
    while cur.is_some() && cur != end_i {
        let idx = cur.expect("active cleanup index");
        t[idx] = 0;
        cur = p[idx];
    }
    max_i
}

#[inline(always)]
fn backtrack_chain<F>(
    scored_anchors: &[ScoredAnchor],
    f: &[i32],
    p: &[Option<usize>],
    state: &mut [i32],
    max_drop: i32,
    rank: usize,
    mut keep_anchor: F,
) -> (i32, usize)
where
    F: FnMut(usize),
{
    let item = &scored_anchors[rank];
    let end_idx = chain_bk_end(max_drop, scored_anchors, f, p, state, rank);
    let mut i = Some(item.index);
    let mut len = 0usize;
    while i != end_idx {
        let idx = i.expect("active backtrack index");
        keep_anchor(idx);
        len += 1;
        state[idx] = 1;
        i = p[idx];
    }
    (i.map(|i| item.score - f[i]).unwrap_or(item.score), len)
}

fn chain_backtrack(
    f: &[i32],
    p: &[Option<usize>],
    state: &mut [i32],
    min_cnt: i32,
    min_sc: i32,
    max_drop: i32,
) -> Option<(Vec<ChainMeta>, Vec<usize>)> {
    let mut scored_anchors: Vec<_> = f
        .iter()
        .enumerate()
        .filter(|&(_, &score)| score >= min_sc)
        .map(|(i, &score)| ScoredAnchor { score, index: i })
        .collect();
    if scored_anchors.is_empty() {
        return None;
    }
    scored_anchors.sort_unstable_by_key(|value| value.score);

    state.fill(0);
    let min_len = min_cnt.max(1) as usize;
    let mut kept_anchor_count = 0usize;
    let mut chain_count = 0usize;
    for rank in (0..scored_anchors.len()).rev() {
        let y = scored_anchors[rank].index;
        if state[y] == 0 {
            let (sc, len) = backtrack_chain(&scored_anchors, f, p, state, max_drop, rank, |_| {});
            if sc >= min_sc && len >= min_len {
                kept_anchor_count += len;
                chain_count += 1;
            }
        }
    }

    let mut chains = Vec::with_capacity(chain_count);
    let mut anchor_indices = Vec::with_capacity(kept_anchor_count);
    state.fill(0);
    for rank in (0..scored_anchors.len()).rev() {
        let y = scored_anchors[rank].index;
        if state[y] == 0 {
            let chain_start = anchor_indices.len();
            let (sc, len) = backtrack_chain(&scored_anchors, f, p, state, max_drop, rank, |idx| {
                anchor_indices.push(idx)
            });
            if sc >= min_sc && len >= min_len {
                chains.push(ChainMeta::new(sc, len));
            } else {
                anchor_indices.truncate(chain_start);
            }
        }
    }
    Some((chains, anchor_indices))
}

fn compact_anchors(
    chains: Vec<ChainMeta>,
    anchor_indices: Vec<usize>,
    anchors: Vec<Anchor>,
) -> ChainResult {
    let mut ordered_anchors = Vec::with_capacity(anchor_indices.len());
    let mut chain_order = Vec::with_capacity(chains.len());
    let mut offset = 0usize;
    for (chain_idx, &chain) in chains.iter().enumerate() {
        for &idx in anchor_indices[offset..offset + chain.len].iter().rev() {
            ordered_anchors.push(anchors[idx]);
        }
        chain_order.push((ordered_anchors[offset].target(), offset, chain_idx));
        offset += chain.len;
    }
    chain_order.sort_unstable_by_key(|item| item.0);

    let mut compacted_anchors = Vec::with_capacity(ordered_anchors.len());
    let mut compacted_chains = Vec::with_capacity(chains.len());
    for &(_, start, chain_id) in &chain_order {
        let len = chains[chain_id].len;
        compacted_chains.push(chains[chain_id]);
        compacted_anchors.extend_from_slice(&ordered_anchors[start..start + len]);
    }
    ChainResult {
        anchors: compacted_anchors,
        chains: compacted_chains,
    }
}

#[allow(clippy::too_many_arguments)]
fn compute_sc(
    ai: Anchor,
    aj: Anchor,
    max_dist_x: i32,
    max_dist_y: i32,
    bw: i32,
    chn_coef_log: f32,
    is_spliced: bool,
    bbit: i32,
    kmer: i32,
) -> i32 {
    let dq = ai.query() - aj.query();
    let dq3 = dq * 3;
    if dq <= 0 || dq3 > max_dist_x || dq > max_dist_y {
        return i32::MIN;
    }

    let (dr3, dd, dd_signed) = if bbit > 0 {
        let bs = 1 << bbit;
        let dr3 = (ai.target() - aj.target()) << bbit;
        if dq3 >= dr3 - bs && dq3 <= dr3 + bs {
            (dr3, 0, 0)
        } else if dq3 < dr3 - bs {
            let dd = dr3 - bs - dq3;
            (dr3, dd, -dd)
        } else {
            let dd = dq3 - (dr3 + bs);
            (dr3, dd, dd)
        }
    } else {
        let dr3 = ai.target() - aj.target();
        if dr3 == 0 {
            return i32::MIN;
        }
        let dd_signed = dq3 - dr3;
        (dr3, (dr3 - dq3).abs(), dd_signed)
    };
    if dd > bw {
        return i32::MIN;
    }

    let mut sc = if bbit > 0 {
        kmer.min(dq)
    } else if kmer <= dq && kmer * 3 <= dr3 {
        kmer
    } else {
        let dr = dr3 / 3;
        let q = dr3 - dr * 3;
        let mut sc = dr.min(dq).min(kmer);
        if q != 0 {
            sc -= 1;
        }
        sc
    };
    if dd > 0 {
        let lin_pen = dd as f32 * 0.33334;
        let log_pen = if dd >= 2 {
            chn_coef_log * (((dd + 1) as f32).log2() - 1.0) + 1.0
        } else {
            dd as f32
        };
        if is_spliced {
            if dd_signed < 0 {
                sc -= lin_pen.min(log_pen) as i32;
            } else {
                sc -= (lin_pen + log_pen) as i32;
            }
        } else {
            sc -= (lin_pen + log_pen) as i32;
        }
    }
    if bbit > 0 && ai.target() == aj.target() && dd == 0 {
        sc += MP_BLOCK_BONUS;
    }
    sc
}

#[allow(clippy::too_many_arguments)]
pub fn chain(
    mut max_dist_x: i32,
    mut max_dist_y: i32,
    bw: i32,
    max_skip: i32,
    max_iter: i32,
    min_cnt: i32,
    min_sc: i32,
    chn_coef_log: f32,
    is_spliced: bool,
    kmer: i32,
    bbit: i32,
    a: Vec<Anchor>,
) -> Option<ChainResult> {
    if a.is_empty() {
        return None;
    }
    max_dist_x = max_dist_x.max(bw);
    if !is_spliced {
        max_dist_y = max_dist_y.max(bw);
    }
    let max_drop = if is_spliced { i32::MAX } else { bw };
    let anchor_count = a.len();
    let mut predecessor = vec![None; anchor_count];
    let mut best_score = vec![0i32; anchor_count];
    let mut peak_score = vec![0i32; anchor_count];
    let mut state = vec![0i32; anchor_count];

    let sc_pair = |ai: Anchor, aj: Anchor| -> i32 {
        compute_sc(
            ai,
            aj,
            max_dist_x,
            max_dist_y,
            bw,
            chn_coef_log,
            is_spliced,
            bbit,
            kmer,
        )
    };
    let mut window_start = 0usize;
    let mut best_anchor = None;
    let mut best_anchor_score = 0i32;
    for i in 0..anchor_count {
        let ai = a[i];
        let ai_block = ai.target();
        let mut best_predecessor = None;
        let mut chain_score_here = kmer;
        let mut skipped = 0i32;
        while window_start < i && ((ai_block - a[window_start].target()) << bbit) > max_dist_x {
            window_start += 1;
        }
        if let Some(best_anchor_idx) = best_anchor.filter(|&idx| idx >= window_start) {
            let sc = best_anchor_score + sc_pair(ai, a[best_anchor_idx]);
            if sc > chain_score_here {
                chain_score_here = sc;
                best_predecessor = Some(best_anchor_idx);
            }
        } else {
            best_anchor_score = 0;
            best_anchor = None;
        }
        if i.saturating_sub(window_start) > max_iter as usize {
            window_start = i - max_iter as usize;
        }
        for j in (window_start..i).rev() {
            let sc = sc_pair(ai, a[j]);
            if sc == i32::MIN {
                continue;
            }
            let sc = sc + best_score[j];
            if sc > chain_score_here {
                chain_score_here = sc;
                best_predecessor = Some(j);
                if skipped > 0 {
                    skipped -= 1;
                }
            } else if state[j] == i as i32 {
                skipped += 1;
                if skipped > max_skip {
                    break;
                }
            }
            if let Some(predecessor_idx) = predecessor[j] {
                state[predecessor_idx] = i as i32;
            }
        }
        best_score[i] = chain_score_here;
        predecessor[i] = best_predecessor;
        peak_score[i] = if let Some(best_predecessor_idx) = best_predecessor {
            peak_score[best_predecessor_idx].max(chain_score_here)
        } else {
            chain_score_here
        };
        if best_anchor_score < chain_score_here {
            best_anchor_score = chain_score_here;
            best_anchor = Some(i);
        }
    }

    let (chains, anchor_indices) = chain_backtrack(
        &best_score,
        &predecessor,
        &mut state,
        min_cnt,
        min_sc,
        max_drop,
    )?;
    Some(compact_anchors(chains, anchor_indices, a))
}
