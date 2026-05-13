use std::cell::UnsafeCell;

use crate::types::{Anchor, ChainMeta, MP_BLOCK_BONUS};

/// Fast approximate log2(x) for x >= 2 using float bit manipulation.
/// Identical to C miniprot's `mp_log2`.
#[inline(always)]
fn fast_log2(x: i32) -> f32 {
    let mut bits = (x as f32).to_bits();
    let mut log_2 = ((bits >> 23) & 255) as f32 - 128.0;
    bits &= !(255 << 23);
    bits += 127 << 23;
    let z = f32::from_bits(bits);
    log_2 += (-0.34484843 * z + 2.02466578) * z - 0.67487759;
    log_2
}

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

struct ChainWorkspace {
    predecessor: Vec<isize>,
    best_score: Vec<i32>,
    peak_score: Vec<i32>,
    state: Vec<i32>,
}

impl ChainWorkspace {
    const fn new() -> Self {
        Self {
            predecessor: Vec::new(),
            best_score: Vec::new(),
            peak_score: Vec::new(),
            state: Vec::new(),
        }
    }

    fn resize(&mut self, n: usize) {
        self.predecessor.resize(n, -1);
        self.predecessor[..n].fill(-1);
        self.best_score.resize(n, 0);
        self.best_score[..n].fill(0);
        self.peak_score.resize(n, 0);
        self.peak_score[..n].fill(0);
        self.state.resize(n, 0);
        self.state[..n].fill(0);
    }
}

thread_local! {
    static CHAIN_WORKSPACE: UnsafeCell<ChainWorkspace> =
        UnsafeCell::new(ChainWorkspace::new());
}

fn chain_bk_end(
    max_drop: i32,
    z: &[ScoredAnchor],
    f: &[i32],
    p: &[isize],
    t: &mut [i32],
    k: usize,
) -> isize {
    let mut i = z[k].index as isize;
    let mut max_i = i;
    let mut max_s = 0i32;
    if t[i as usize] != 0 {
        return i;
    }
    let end_i = loop {
        let idx = i as usize;
        t[idx] = 2;
        i = p[idx];
        let s = if i < 0 {
            z[k].score
        } else {
            z[k].score - f[i as usize]
        };
        if s > max_s {
            max_s = s;
            max_i = i;
        } else if max_s - s > max_drop {
            break i;
        }
        if i < 0 || t[i as usize] != 0 {
            break i;
        }
    };
    let mut cur = z[k].index as isize;
    while cur >= 0 && cur != end_i {
        let idx = cur as usize;
        t[idx] = 0;
        cur = p[idx];
    }
    max_i
}

#[inline(always)]
fn backtrack_chain<F>(
    scored_anchors: &[ScoredAnchor],
    f: &[i32],
    p: &[isize],
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
    let mut i = item.index as isize;
    let mut len = 0usize;
    while i != end_idx {
        let idx = i as usize;
        keep_anchor(idx);
        len += 1;
        state[idx] = 1;
        i = p[idx];
    }
    let score = if i < 0 {
        item.score
    } else {
        item.score - f[i as usize]
    };
    (score, len)
}

fn chain_backtrack(
    f: &[i32],
    p: &[isize],
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
            chn_coef_log * (fast_log2(dd + 1) - 1.0) + 1.0
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
#[must_use]
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
    let anchor_count = a.len();

    CHAIN_WORKSPACE.with(|cell| {
        // SAFETY: thread-local storage guarantees exclusive access within this thread.
        let ws = unsafe { &mut *cell.get() };
        ws.resize(anchor_count);
        chain_inner(
            max_dist_x,
            max_dist_y,
            bw,
            max_skip,
            max_iter,
            min_cnt,
            min_sc,
            chn_coef_log,
            is_spliced,
            kmer,
            bbit,
            a,
            ws,
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn chain_inner(
    max_dist_x: i32,
    max_dist_y: i32,
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
    ws: &mut ChainWorkspace,
) -> Option<ChainResult> {
    let max_drop = if is_spliced { i32::MAX } else { bw };
    let anchor_count = a.len();
    let predecessor = &mut ws.predecessor[..anchor_count];
    let best_score = &mut ws.best_score[..anchor_count];
    let peak_score = &mut ws.peak_score[..anchor_count];
    let state = &mut ws.state[..anchor_count];

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
        let mut best_predecessor = -1isize;
        let mut chain_score_here = kmer;
        let mut skipped = 0i32;
        while window_start < i && ((ai_block - a[window_start].target()) << bbit) > max_dist_x {
            window_start += 1;
        }
        if let Some(best_anchor_idx) = best_anchor.filter(|&idx| idx >= window_start) {
            let sc = best_anchor_score + sc_pair(ai, a[best_anchor_idx]);
            if sc > chain_score_here {
                chain_score_here = sc;
                best_predecessor = best_anchor_idx as isize;
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
                best_predecessor = j as isize;
                if skipped > 0 {
                    skipped -= 1;
                }
            } else if state[j] == i as i32 {
                skipped += 1;
                if skipped > max_skip {
                    break;
                }
            }
            if predecessor[j] >= 0 {
                state[predecessor[j] as usize] = i as i32;
            }
        }
        best_score[i] = chain_score_here;
        predecessor[i] = best_predecessor;
        peak_score[i] = if best_predecessor >= 0 {
            peak_score[best_predecessor as usize].max(chain_score_here)
        } else {
            chain_score_here
        };
        if best_anchor_score < chain_score_here {
            best_anchor_score = chain_score_here;
            best_anchor = Some(i);
        }
    }

    let (chains, anchor_indices) =
        chain_backtrack(best_score, predecessor, state, min_cnt, min_sc, max_drop)?;
    Some(compact_anchors(chains, anchor_indices, a))
}
