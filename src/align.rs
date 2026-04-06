use crate::format::build_cs;
use crate::index::Index;
use crate::tables::{
    AA_I2C, Kind, NS_F_CIGAR, NS_F_EXT_LEFT, NS_F_EXT_RIGHT, NS_SPSC_OFFSET, Tables, pack_cigar_op,
    unpack_cigar_op,
};
use crate::types::{
    Alignment, AlignmentExtra, Anchor, Feature, FeatureType, MP_F_NO_CS, MapOptions,
};

const NEG_INF: i32 = i32::MIN / 4;
const AA_STOP: u8 = 20;
const AA_AMBI: u8 = 21;

#[derive(Clone, Copy)]
struct NsOpt<'a> {
    flag: i32,
    go: i32,
    ge: i32,
    io: i32,
    fs: i32,
    xdrop: i32,
    end_bonus: i32,
    sp: [i32; 6],
    sp_null_bonus: i32,
    ie_coef: f32,
    sc: &'a [[i8; 22]; 22],
    tables: &'a Tables,
}

#[derive(Clone, Debug, Default)]
struct NsResult {
    cigar: Vec<u32>,
    nt_len: i32,
    aa_len: i32,
    score: i32,
}

#[derive(Default)]
struct CigarBuilder {
    cigar: Vec<u32>,
}

#[derive(Clone)]
struct PreparedSeq {
    nas: Vec<u8>,
    aas: Vec<u8>,
    donor: Vec<i32>,
    acceptor: Vec<i32>,
}

#[derive(Clone)]
struct TraceMatrix {
    cols: usize,
    trace: Vec<u16>,
}

impl TraceMatrix {
    fn new(nl: usize, al: usize) -> Self {
        Self {
            cols: al + 1,
            trace: vec![0; nl * (al + 1)],
        }
    }

    fn set(&mut self, i: usize, j: usize, value: u16) {
        self.trace[i * self.cols + j] = value;
    }

    fn get(&self, i: usize, j: usize) -> u16 {
        self.trace[i * self.cols + j]
    }
}

impl CigarBuilder {
    fn push(&mut self, op: Kind, len: i32) {
        if len <= 0 {
            return;
        }
        let code = op.code();
        if let Some(last) = self.cigar.last_mut()
            && code == (*last & 0x0f) as u8
            && op != Kind::FrameshiftGap
            && op != Kind::FrameshiftMatch
        {
            *last += (len as u32) << 4;
        } else {
            self.cigar.push(pack_cigar_op(op, len as usize));
        }
    }
}

fn nt_code(tables: &Tables, byte: u8) -> u8 {
    if byte < 5 {
        byte
    } else {
        tables.nt4[byte as usize]
    }
}

fn aa_code(tables: &Tables, byte: u8) -> u8 {
    if byte < AA_I2C.len() as u8 {
        byte
    } else {
        tables.aa20[byte as usize]
    }
}

#[inline]
fn codon_aa(tables: &Tables, n1: u8, n2: u8, n3: u8) -> u8 {
    if n1 > 3 || n2 > 3 || n3 > 3 {
        AA_AMBI
    } else {
        tables.codon[((n1 << 4) | (n2 << 2) | n3) as usize]
    }
}

#[inline]
fn slice_codon_aa(tables: &Tables, nt: &[u8], i: usize) -> u8 {
    codon_aa(tables, nt[i], nt[i + 1], nt[i + 2])
}

fn map_to_ns_opt<'a>(opt: &'a MapOptions, tables: &'a Tables, flag: i32) -> NsOpt<'a> {
    let mut sp = crate::tables::opt_set_sp(opt.sp_model);
    for item in &mut sp {
        *item = (*item as f32 * opt.sp_scale + 0.499) as i32;
    }
    NsOpt {
        flag,
        go: opt.go,
        ge: opt.ge,
        io: opt.io,
        fs: opt.fs,
        xdrop: opt.xdrop,
        end_bonus: opt.end_bonus,
        sp,
        sp_null_bonus: opt.sp_null_bonus,
        ie_coef: opt.ie_coef,
        sc: &opt.mat,
        tables,
    }
}

fn prep_nas_from_nts(nts: &[u8], tables: &Tables) -> Vec<u8> {
    let mut nas = vec![AA_AMBI; nts.len()];
    let mut codon = 0u8;
    let mut l = 0i32;
    for (i, &c) in nts.iter().enumerate() {
        if c < 4 {
            codon = ((codon << 2) | c) & 0x3f;
            l += 1;
            if l >= 3 {
                nas[i] = tables.codon[codon as usize];
            }
        } else {
            codon = 0;
            l = 0;
        }
    }
    nas
}

fn prep_nas(ns: &[u8], opt: &NsOpt<'_>) -> Vec<u8> {
    let nts: Vec<u8> = ns.iter().map(|&byte| nt_code(opt.tables, byte)).collect();
    prep_nas_from_nts(&nts, opt.tables)
}

fn prep_seq(ns: &[u8], aa: &[u8], opt: &NsOpt<'_>, ss: Option<&[u8]>) -> PreparedSeq {
    let aas: Vec<_> = aa.iter().map(|&byte| aa_code(opt.tables, byte)).collect();

    let nts: Vec<u8> = ns.iter().map(|&byte| nt_code(opt.tables, byte)).collect();
    let mut donor = vec![opt.sp[3]; nts.len() + 1];
    let mut acceptor = vec![opt.sp[3]; nts.len() + 1];

    for i in 0..nts.len().saturating_sub(3) {
        let mut splice = Some(3usize);
        if nts[i + 1] == 2 && nts[i + 2] == 3 {
            splice = if i + 3 < nts.len() && (nts[i + 3] == 0 || nts[i + 3] == 2) {
                if nts[i] == 2 { None } else { Some(4) }
            } else {
                Some(0)
            };
        } else if nts[i + 1] == 2 && nts[i + 2] == 1 && nts[i] == 2 {
            splice = Some(1);
        } else if nts[i + 1] == 0 && nts[i + 2] == 3 {
            splice = Some(2);
        }
        donor[i] = splice.map_or(0, |idx| opt.sp[idx]);
    }
    for i in 1..nts.len() {
        let mut splice = Some(3usize);
        let mut pen_y = 0;
        if nts[i - 1] == 0 && nts[i] == 2 {
            splice = if i >= 2 && (nts[i - 2] == 1 || nts[i - 2] == 3) {
                None
            } else {
                Some(0)
            };
            let mut j = i.saturating_sub(4);
            while j < i && j + 3 >= i {
                if nts[j] != 1 && nts[j] != 3 {
                    pen_y += opt.sp[5];
                }
                if j == 0 {
                    break;
                }
                j -= 1;
            }
        } else if nts[i - 1] == 0 && nts[i] == 1 {
            splice = Some(2);
        }
        acceptor[i] = splice.map_or(0, |idx| opt.sp[idx]);
        if matches!(splice, None | Some(0)) {
            acceptor[i] += pen_y;
        }
    }

    if let Some(ss) = ss {
        let max_spsc = (opt.io + 1) / 2 - 1;
        for (i, &value) in ss.iter().take(nts.len()).enumerate().skip(1) {
            let spsc = (((value >> 1) as i8 as i32) - NS_SPSC_OFFSET as i32).min(max_spsc);
            if value == 0xff {
                donor[i - 1] -= opt.sp_null_bonus;
                acceptor[i - 1] -= opt.sp_null_bonus;
            } else if (value & 1) != 0 {
                acceptor[i - 1] -= spsc;
            } else {
                donor[i - 1] -= spsc;
            }
        }
    }

    PreparedSeq {
        nas: prep_nas_from_nts(&nts, opt.tables),
        aas,
        donor,
        acceptor,
    }
}

fn prep_seq_left(ns: &[u8], aa: &[u8], opt: &NsOpt<'_>, ss: Option<&[u8]>) -> PreparedSeq {
    let aas: Vec<_> = aa
        .iter()
        .rev()
        .map(|&byte| aa_code(opt.tables, byte))
        .collect();

    let nts: Vec<u8> = ns
        .iter()
        .rev()
        .map(|&byte| nt_code(opt.tables, byte))
        .collect();
    let mut donor = vec![opt.sp[3]; nts.len() + 1];
    let mut acceptor = vec![opt.sp[3]; nts.len() + 1];

    for i in 0..nts.len().saturating_sub(3) {
        let mut splice = Some(3usize);
        let mut pen_y = 0;
        if nts[i + 1] == 2 && nts[i + 2] == 0 {
            splice = if i + 3 < nts.len() && (nts[i + 3] == 1 || nts[i + 3] == 3) {
                None
            } else {
                Some(0)
            };
            let mut j = i + 5;
            while j < nts.len() && j < i + 8 {
                if nts[j] != 1 && nts[j] != 3 {
                    pen_y += opt.sp[5];
                }
                j += 1;
            }
        } else if nts[i + 1] == 1 && nts[i + 2] == 0 {
            splice = Some(2);
        }
        donor[i] = splice.map_or(0, |idx| opt.sp[idx]);
        if matches!(splice, None | Some(0)) {
            donor[i] += pen_y;
        }
    }
    for i in 1..nts.len() {
        let mut splice = Some(3usize);
        if nts[i - 1] == 3 && nts[i] == 2 {
            splice = if i >= 2 && (nts[i - 2] == 0 || nts[i - 2] == 2) {
                if i + 1 < nts.len() && nts[i + 1] == 2 {
                    None
                } else {
                    Some(4)
                }
            } else {
                Some(0)
            };
        } else if nts[i - 1] == 1 && nts[i] == 2 && i + 1 < nts.len() && nts[i + 1] == 1 {
            splice = Some(1);
        } else if nts[i - 1] == 3 && nts[i] == 0 {
            splice = Some(2);
        }
        acceptor[i] = splice.map_or(0, |idx| opt.sp[idx]);
    }

    if let Some(ss) = ss {
        let max_spsc = (opt.io + 1) / 2 - 1;
        for (i, &value) in ss.iter().take(nts.len()).enumerate() {
            let spsc = (((value >> 1) as i8 as i32) - NS_SPSC_OFFSET as i32).min(max_spsc);
            let idx = nts.len() - i - 1;
            if value == 0xff {
                donor[idx] -= opt.sp_null_bonus;
                acceptor[idx] -= opt.sp_null_bonus;
            } else if (value & 1) != 0 {
                donor[idx] -= spsc;
            } else {
                acceptor[idx] -= spsc;
            }
        }
    }

    let mut nas = prep_nas(ns, opt);
    nas.reverse();
    if nas.len() >= 2 {
        let copy_len = nas.len() - 2;
        nas.copy_within(0..copy_len, 2);
    }
    let prefix_len = nas.len().min(2);
    nas[..prefix_len].fill(AA_AMBI);

    PreparedSeq {
        nas,
        aas,
        donor,
        acceptor,
    }
}

fn score_pair(opt: &NsOpt<'_>, nt_aa: u8, aa_aa: u8) -> i32 {
    opt.sc[nt_aa as usize][aa_aa as usize] as i32
}

fn encode_trace(state: u8, i_ext: bool, d_ext: bool, a_ext: bool, b_ext: bool, c_ext: bool) -> u16 {
    (state as u16)
        | ((i_ext as u16) << 4)
        | ((d_ext as u16) << 5)
        | ((a_ext as u16) << 6)
        | ((b_ext as u16) << 7)
        | ((c_ext as u16) << 8)
}

fn trace_ext(x: u16, state: u8) -> bool {
    match state {
        1 => ((x >> 4) & 1) != 0,
        2 => ((x >> 5) & 1) != 0,
        3 => ((x >> 6) & 1) != 0,
        4 => ((x >> 7) & 1) != 0,
        5 => ((x >> 8) & 1) != 0,
        _ => false,
    }
}

fn fix_tiny_uv(cigar: &mut [u32]) {
    for item in cigar {
        let Some((op, len)) = unpack_cigar_op(*item) else {
            continue;
        };
        if (op == Kind::IntronPhase1 || op == Kind::IntronPhase2) && len < 3 {
            *item = pack_cigar_op(Kind::FrameshiftMatch, len);
        }
    }
}

fn backtrack(nl: i32, al: i32, trace: &TraceMatrix) -> Vec<u32> {
    let mut i = nl - 1;
    let mut j = al - 1;
    let mut last = 0u8;
    let mut cigar = CigarBuilder::default();
    while i >= 2 && j >= 0 {
        let x = trace.get(i as usize, j as usize + 1);
        let state = if last == 0 { (x & 0x0f) as u8 } else { last };
        let ext = trace_ext(x, state);
        match state {
            0 => {
                cigar.push(Kind::Match, 1);
                i -= 3;
                j -= 1;
            }
            1 => {
                cigar.push(Kind::Insertion, 1);
                j -= 1;
            }
            2 => {
                cigar.push(Kind::Deletion, 1);
                i -= 3;
            }
            3 => {
                cigar.push(Kind::Skip, 1);
                i -= 1;
            }
            4 => {
                cigar.push(Kind::IntronPhase1, 1);
                i -= 1;
                if !ext {
                    j -= 1;
                }
            }
            5 => {
                cigar.push(Kind::IntronPhase2, 1);
                i -= 1;
                if !ext {
                    j -= 1;
                }
            }
            6 => {
                cigar.push(Kind::FrameshiftGap, 1);
                i -= 1;
            }
            7 => {
                cigar.push(Kind::FrameshiftGap, 2);
                i -= 2;
            }
            8 => {
                cigar.push(Kind::FrameshiftMatch, 1);
                i -= 1;
                j -= 1;
            }
            9 => {
                cigar.push(Kind::FrameshiftMatch, 2);
                i -= 2;
                j -= 1;
            }
            _ => unreachable!(),
        }
        last = if (1..=5).contains(&state) && ext {
            state
        } else {
            0
        };
    }
    if j > 0 {
        cigar.push(Kind::Insertion, j);
    }
    if i >= 0 {
        let l = ((i + 1) / 3) * 3;
        let t = (i + 1) % 3;
        if l > 0 {
            cigar.push(Kind::Deletion, l);
        }
        if t != 0 {
            cigar.push(Kind::FrameshiftGap, t);
        }
    }
    cigar.cigar.reverse();
    fix_tiny_uv(&mut cigar.cigar);
    cigar.cigar
}

fn scalar_dp(ns: &[u8], aa: &[u8], opt: &NsOpt<'_>, ss: Option<&[u8]>) -> NsResult {
    if aa.is_empty() || ns.is_empty() {
        return NsResult::default();
    }
    let prep = if (opt.flag & NS_F_EXT_LEFT) != 0 {
        prep_seq_left(ns, aa, opt, ss)
    } else {
        prep_seq(ns, aa, opt, ss)
    };
    let nl = prep.nas.len();
    let al = prep.aas.len();
    if al == 0 || nl == 0 {
        return NsResult::default();
    }

    let mut h_prev3 = vec![NEG_INF; al + 1];
    let mut h_prev2 = vec![NEG_INF; al + 1];
    let mut h_prev1 = vec![NEG_INF; al + 1];
    let mut h_cur = vec![NEG_INF; al + 1];
    let mut d_prev3 = vec![NEG_INF; al + 1];
    let mut d_prev2 = vec![NEG_INF; al + 1];
    let mut d_prev1 = vec![NEG_INF; al + 1];
    let mut d_cur = vec![NEG_INF; al + 1];
    let mut a_prev1 = vec![NEG_INF; al + 1];
    let mut b_prev1 = vec![NEG_INF; al + 1];
    let mut c_prev1 = vec![NEG_INF; al + 1];
    let mut a_cur = vec![NEG_INF; al + 1];
    let mut b_cur = vec![NEG_INF; al + 1];
    let mut c_cur = vec![NEG_INF; al + 1];

    h_prev3[0] = 0;
    h_prev2[0] = -opt.fs;
    h_prev1[0] = -opt.fs;

    let need_trace =
        (opt.flag & NS_F_CIGAR) != 0 && (opt.flag & (NS_F_EXT_LEFT | NS_F_EXT_RIGHT)) == 0;
    let mut trace = need_trace.then(|| TraceMatrix::new(nl, al));

    let mut max_sc = i32::MIN;
    let mut max_sc_log = i32::MIN;
    let mut max_i = -1i32;
    let mut h_best = vec![NEG_INF; al + 1];
    let pen_len = (al as i32) * 3;

    for i in 2..nl {
        h_cur.fill(NEG_INF);
        d_cur.fill(NEG_INF);
        a_cur.fill(NEG_INF);
        b_cur.fill(NEG_INF);
        c_cur.fill(NEG_INF);

        let gei = if prep.nas[i] == AA_STOP {
            opt.fs
        } else {
            opt.ge
        };
        let dim1 = prep.donor[i - 1];
        let di = prep.donor[i];
        let dip1 = prep.donor[i + 1];
        let ai = prep.acceptor[i];
        let aim1 = prep.acceptor[i - 1];
        let aim2 = prep.acceptor[i - 2];

        let open_d0 = h_prev3[0] - opt.go;
        let ext_d0 = d_prev3[0];
        let d0 = open_d0.max(ext_d0) - gei;
        d_cur[0] = d0;
        h_cur[0] = d0.max(h_prev1[0] - opt.fs).max(h_prev2[0] - opt.fs);

        let mut i_state = NEG_INF;
        let mut row_max = NEG_INF;
        for j in 0..al {
            let col = j + 1;
            let mut best = h_prev3[j] + score_pair(opt, prep.nas[i], prep.aas[j]);
            let mut state = 0u8;

            let open_i = h_cur[j] - opt.go;
            let ext_i = i_state;
            let mut t = open_i.max(ext_i) - opt.ge;
            let i_ext = ext_i > open_i;
            i_state = t;
            if t > best {
                best = t;
                state = 1;
            }

            let open_d = h_prev3[col] - opt.go;
            let ext_d = d_prev3[col];
            t = open_d.max(ext_d) - gei;
            let d_ext = ext_d > open_d;
            d_cur[col] = t;
            if t > best {
                best = t;
                state = 2;
            }

            let open_a = h_prev1[col] - opt.io - dim1;
            let ext_a = a_prev1[col];
            t = open_a.max(ext_a);
            let a_ext = ext_a > open_a;
            a_cur[col] = t;
            let ta = t - ai;
            if ta > best {
                best = ta;
                state = 3;
            }

            let open_b = h_prev1[j] - opt.io - di;
            let ext_b = b_prev1[col];
            t = open_b.max(ext_b);
            let b_ext = ext_b > open_b;
            b_cur[col] = t;
            let tb = t - aim2;
            if tb > best {
                best = tb;
                state = 4;
            }

            let open_c = h_prev1[j] - opt.io - dip1;
            let ext_c = c_prev1[col];
            t = open_c.max(ext_c);
            let c_ext = ext_c > open_c;
            c_cur[col] = t;
            let tc = t - aim1;
            if tc > best {
                best = tc;
                state = 5;
            }

            t = h_prev1[j] - opt.fs;
            if t > best {
                best = t;
                state = 6;
            }
            t = h_prev2[j] - opt.fs;
            if t > best {
                best = t;
                state = 7;
            }
            t = h_prev1[col] - opt.fs;
            if t > best {
                best = t;
                state = 8;
            }
            t = h_prev2[col] - opt.fs;
            if t > best {
                best = t;
                state = 9;
            }

            h_cur[col] = best;
            row_max = row_max.max(best);

            if let Some(trace) = &mut trace {
                trace.set(
                    i,
                    col,
                    encode_trace(state, i_ext, d_ext, a_ext, b_ext, c_ext),
                );
            }
        }

        if (opt.flag & (NS_F_EXT_LEFT | NS_F_EXT_RIGHT)) != 0 {
            let end_sc = h_cur[al] + opt.end_bonus;
            let tmp_sc = row_max.max(end_sc);
            let len_pen = if i as i32 - pen_len < 2 {
                0
            } else {
                (opt.ie_coef * ((i as i32 - pen_len) as f32).log2() + 0.5) as i32
            };
            let tmp_sc_log = tmp_sc - len_pen;
            if tmp_sc_log > max_sc_log {
                max_sc = tmp_sc;
                max_sc_log = tmp_sc_log;
                max_i = i as i32;
                h_best.copy_from_slice(&h_cur);
            }
            if max_sc_log - tmp_sc_log > opt.xdrop {
                break;
            }
        }

        std::mem::swap(&mut h_prev3, &mut h_prev2);
        std::mem::swap(&mut h_prev2, &mut h_prev1);
        std::mem::swap(&mut h_prev1, &mut h_cur);
        std::mem::swap(&mut d_prev3, &mut d_prev2);
        std::mem::swap(&mut d_prev2, &mut d_prev1);
        std::mem::swap(&mut d_prev1, &mut d_cur);
        std::mem::swap(&mut a_prev1, &mut a_cur);
        std::mem::swap(&mut b_prev1, &mut b_cur);
        std::mem::swap(&mut c_prev1, &mut c_cur);
    }

    if (opt.flag & (NS_F_EXT_LEFT | NS_F_EXT_RIGHT)) != 0 {
        let mut best_aa = 0i32;
        for j in 0..al {
            let mut sc = h_best[j + 1];
            if j == al - 1 {
                sc += opt.end_bonus;
            }
            if sc == max_sc {
                best_aa = j as i32 + 1;
                break;
            }
        }
        return NsResult {
            nt_len: max_i + 1,
            aa_len: best_aa,
            score: max_sc,
            ..NsResult::default()
        };
    }

    let cigar = trace
        .as_ref()
        .map(|trace| backtrack(nl as i32, al as i32, trace))
        .unwrap_or_default();
    NsResult {
        nt_len: nl as i32,
        aa_len: al as i32,
        score: h_prev1[al],
        cigar,
    }
}

fn low31(anchor: Anchor) -> i32 {
    anchor.query_pos()
}

fn filter_seed(a: &mut [Anchor], max_aa_dist: i32, min_cnt: i32, kmer2: i32, trim_back: i32) {
    let mut i = 0usize;
    while i < a.len() {
        let mut j = i + 1;
        while j < a.len() {
            let x0 = a[j - 1].target();
            let y0 = low31(a[j - 1]);
            let x1 = a[j].target();
            let y1 = low31(a[j]);
            if (x1 - x0) % 3 != 0 || x1 - x0 > max_aa_dist * 3 || y1 - y0 > max_aa_dist {
                break;
            }
            j += 1;
        }
        if (j - i) as i32 >= min_cnt {
            let mut k = j as i32 - 2;
            let mut t = low31(a[j - 1]);
            while k >= i as i32 {
                if t - low31(a[k as usize]) >= trim_back {
                    break;
                }
                k -= 1;
            }
            t = low31(a[i]) + 1 - kmer2;
            while i < k.max(0) as usize {
                if low31(a[i]) + 1 - t >= trim_back {
                    break;
                }
                i += 1;
            }
            while i <= k.max(-1) as usize && i < a.len() {
                a[i] = a[i].with_query_flag();
                i += 1;
            }
            i = j - 1;
        }
        i += 1;
    }
}

fn score_ungapped(nseq: &[u8], aseq: &[u8], opt: &MapOptions, tables: &Tables) -> i32 {
    nseq.chunks_exact(3)
        .zip(aseq)
        .map(|(codon, &aa_byte)| {
            let nt_aa = codon_aa(tables, codon[0], codon[1], codon[2]);
            let aa_aa = aa_code(tables, aa_byte);
            opt.mat[nt_aa as usize][aa_aa as usize] as i32
        })
        .sum()
}

fn align_seq(
    opt: &MapOptions,
    tables: &Tables,
    nseq: &[u8],
    aseq: &[u8],
    ss: Option<&[u8]>,
    cigar: &mut CigarBuilder,
) -> i32 {
    if nseq.len() == aseq.len() * 3 && aseq.len() as i32 <= opt.kmer2 {
        cigar.push(Kind::Match, aseq.len() as i32);
        return score_ungapped(nseq, aseq, opt, tables);
    }
    let ns_opt = map_to_ns_opt(opt, tables, NS_F_CIGAR);
    let rst = scalar_dp(nseq, aseq, &ns_opt, ss);
    for item in rst.cigar {
        if let Some((op, len)) = unpack_cigar_op(item) {
            cigar.push(op, len as i32);
        }
    }
    rst.score
}

fn extra_stop(reg: &Alignment, nt: &[u8], as_: i64, ae: i64, tables: &Tables) -> i32 {
    let mut j = reg.ve;
    while j + 2 < ae {
        let i = (j - as_) as usize;
        let aa = slice_codon_aa(tables, nt, i);
        if aa == AA_STOP {
            return (j - reg.ve) as i32;
        }
        j += 3;
    }
    -1
}

fn extra_start(reg: &Alignment, nt: &[u8], as_: i64, ae: i64, tables: &Tables) -> i32 {
    let aa_met = aa_code(tables, b'M');
    let mut j = reg.vs;
    loop {
        if j < as_ || j + 2 >= ae {
            break;
        }
        let i = (j - as_) as usize;
        let aa = slice_codon_aa(tables, nt, i);
        if aa == AA_STOP {
            break;
        }
        if aa == aa_met {
            return (reg.vs - j) as i32;
        }
        if j < as_ + 3 {
            break;
        }
        j -= 3;
    }
    -1
}

fn extra_cal(
    reg: &mut Alignment,
    opt: &MapOptions,
    tables: &Tables,
    nt: &[u8],
    aa: &[u8],
    qlen: i32,
) -> bool {
    let Some(extra) = &mut reg.extra else {
        return false;
    };
    let ns_opt = map_to_ns_opt(opt, tables, 0);
    let has_stop = reg.qe == qlen && extra.dist_stop == 0;
    let mut n_intron = 0usize;
    for &item in &extra.cigar {
        let Some((op, _)) = unpack_cigar_op(item) else {
            continue;
        };
        if op == Kind::Skip || op == Kind::IntronPhase1 || op == Kind::IntronPhase2 {
            n_intron += 1;
        }
    }
    reg.n_exon = n_intron + 1;
    reg.n_feat = reg.n_exon + if has_stop { 1 } else { 0 };
    reg.feat = Vec::with_capacity(reg.n_feat);

    extra.blen = 0;
    extra.n_iden = 0;
    extra.n_plus = 0;
    extra.n_fs = 0;
    extra.n_stop = 0;
    extra.dp_max = 0;

    let mut blen0 = 0;
    let mut n_iden0 = 0;
    let mut score0 = 0;
    let mut n_fs0 = 0;
    let mut n_stop0 = 0;
    let mut phase0 = 0i16;
    let mut vs0 = reg.vs;
    let mut qs0 = reg.qs;
    let mut acceptor0 = [0u8; 2];
    let mut nl = 0usize;
    let mut al = 0usize;

    for &item in &extra.cigar {
        let Some((op, len)) = unpack_cigar_op(item) else {
            continue;
        };
        let len3 = len * 3;
        match op {
            Kind::Match => {
                for (codon_nt, &aa_byte) in nt[nl..nl + len3].chunks_exact(3).zip(&aa[al..al + len])
                {
                    let nt_aa = codon_aa(tables, codon_nt[0], codon_nt[1], codon_nt[2]);
                    let aa_aa = aa_code(tables, aa_byte);
                    let s = score_pair(&ns_opt, nt_aa, aa_aa);
                    extra.n_stop += (nt_aa == AA_STOP) as i32;
                    extra.n_iden += (nt_aa == aa_aa) as i32;
                    extra.n_plus += (s > 0) as i32;
                    extra.dp_max += s;
                }
                nl += len3;
                al += len;
                extra.blen += len3 as i32;
            }
            Kind::Insertion => {
                extra.dp_max -= opt.go + opt.ge * len as i32;
                al += len;
                extra.blen += len3 as i32;
            }
            Kind::Deletion => {
                for codon_nt in nt[nl..nl + len3].chunks_exact(3) {
                    let nt_aa = codon_aa(tables, codon_nt[0], codon_nt[1], codon_nt[2]);
                    extra.n_stop += (nt_aa == AA_STOP) as i32;
                }
                extra.dp_max -= opt.go + opt.ge * len as i32;
                nl += len3;
                extra.blen += len3 as i32;
            }
            Kind::FrameshiftGap => {
                extra.dp_max -= opt.fs;
                nl += len;
                extra.blen += len as i32;
                extra.n_fs += 1;
            }
            Kind::FrameshiftMatch => {
                extra.dp_max -= opt.fs;
                nl += len;
                al += 1;
                extra.blen += 3;
                extra.n_fs += 1;
            }
            Kind::Skip | Kind::IntronPhase1 | Kind::IntronPhase2 => {
                if op == Kind::IntronPhase1 || op == Kind::IntronPhase2 {
                    let nt_aa = if op == Kind::IntronPhase1 {
                        codon_aa(tables, nt[nl], nt[nl + len - 2], nt[nl + len - 1])
                    } else {
                        codon_aa(tables, nt[nl], nt[nl + 1], nt[nl + len - 1])
                    };
                    let aa_aa = aa_code(tables, aa[al]);
                    let s = score_pair(&ns_opt, nt_aa, aa_aa);
                    extra.n_stop += (nt_aa == AA_STOP) as i32;
                    extra.n_iden += (nt_aa == aa_aa) as i32;
                    extra.n_plus += (s > 0) as i32;
                    extra.dp_max += s;
                    extra.blen += 3;
                }

                let mut feat = Feature {
                    feature_type: FeatureType::Cds,
                    vs: vs0,
                    qs: qs0,
                    qe: reg.qs + al as i32,
                    n_fs: extra.n_fs - n_fs0,
                    n_stop: extra.n_stop - n_stop0,
                    phase: phase0,
                    blen: extra.blen - blen0,
                    n_iden: extra.n_iden - n_iden0,
                    score: extra.dp_max - score0,
                    ..Feature::default()
                };
                if !reg.feat.is_empty() {
                    feat.acceptor = acceptor0;
                }
                if op == Kind::Skip {
                    feat.ve = reg.vs + nl as i64;
                    vs0 = reg.vs + nl as i64 + len as i64;
                    phase0 = 0;
                } else if op == Kind::IntronPhase1 {
                    feat.ve = reg.vs + nl as i64 + 1;
                    vs0 = reg.vs + nl as i64 + len as i64 - 2;
                    phase0 = 2;
                } else {
                    feat.ve = reg.vs + nl as i64 + 2;
                    vs0 = reg.vs + nl as i64 + len as i64 - 1;
                    phase0 = 1;
                }
                let donor_pos = (feat.ve - reg.vs) as usize;
                feat.donor[0] = nt
                    .get(donor_pos)
                    .map_or(b'.', |&base| crate::tables::NT_I2C[base as usize]);
                feat.donor[1] = nt
                    .get(donor_pos + 1)
                    .map_or(b'.', |&base| crate::tables::NT_I2C[base as usize]);
                qs0 = feat.qe;
                n_fs0 = extra.n_fs;
                n_stop0 = extra.n_stop;
                score0 = extra.dp_max;
                blen0 = extra.blen;
                n_iden0 = extra.n_iden;
                let acc_pos = (vs0 - reg.vs) as usize;
                acceptor0[0] = acc_pos
                    .checked_sub(2)
                    .and_then(|idx| nt.get(idx))
                    .map_or(b'.', |&base| crate::tables::NT_I2C[base as usize]);
                acceptor0[1] = acc_pos
                    .checked_sub(1)
                    .and_then(|idx| nt.get(idx))
                    .map_or(b'.', |&base| crate::tables::NT_I2C[base as usize]);
                reg.feat.push(feat);
                nl += len;
                al += (op != Kind::Skip) as usize;
            }
            _ => {}
        }
    }

    let mut feat = Feature {
        feature_type: FeatureType::Cds,
        vs: vs0,
        ve: reg.vs + nl as i64,
        qs: qs0,
        qe: reg.qs + al as i32,
        phase: phase0,
        blen: extra.blen - blen0,
        n_iden: extra.n_iden - n_iden0,
        n_fs: extra.n_fs - n_fs0,
        n_stop: extra.n_stop - n_stop0,
        score: extra.dp_max - score0,
        ..Feature::default()
    };
    if !reg.feat.is_empty() {
        feat.acceptor = acceptor0;
    }
    reg.feat.push(feat);
    if has_stop {
        let ve_mrna = reg.ve + 3;
        reg.feat.push(Feature {
            feature_type: FeatureType::Stop,
            vs: ve_mrna - 3,
            ve: ve_mrna,
            qs: reg.qe + al as i32,
            qe: reg.qe + al as i32,
            blen: 3,
            ..Feature::default()
        });
    }

    if nl as i64 != reg.ve - reg.vs || al as i32 != reg.qe - reg.qs {
        reg.extra = None;
        reg.feat.clear();
        reg.n_feat = 0;
        reg.n_exon = 0;
        return false;
    }
    true
}

pub fn select_multi_exon(regs: &mut [Alignment], single_penalty: i32) {
    if regs.len() < 2 || regs[0].n_exon != 1 {
        return;
    }
    let Some(first_extra) = regs[0].extra.as_ref() else {
        return;
    };
    let mut idx = None;
    for (i, reg) in regs.iter().enumerate().skip(1) {
        if reg.n_exon >= 2 && reg.extra.is_some() {
            idx = Some(i);
            break;
        }
    }
    let Some(i) = idx else {
        return;
    };
    let cand = regs[i].extra.as_ref().expect("extra");
    if first_extra.dp_max < cand.dp_max + single_penalty {
        regs.swap(0, i);
    }
}

pub fn align_reg(
    mi: &Index,
    opt: &MapOptions,
    qlen: i32,
    aa: &[u8],
    reg: &mut Alignment,
    extl0: i32,
    extr0: i32,
) {
    if reg.cnt == 0 || reg.anchors.is_empty() {
        reg.cnt = 0;
        return;
    }

    filter_seed(&mut reg.anchors, 6, 3, opt.kmer2, opt.kmer2 + 1);
    let Some(i0) = reg
        .anchors
        .iter()
        .position(|anchor| anchor.has_query_flag())
    else {
        reg.cnt = 0;
        return;
    };

    let mut extl = opt.max_ext;
    let mut extr = opt.max_ext;
    if reg.qs >= 10 {
        extl = opt.max_intron / 2;
    }
    if qlen - reg.qe >= 10 {
        extr = opt.max_intron / 2;
    }
    if extl0 > 0 {
        extl = extl.min(extl0);
    }
    if extr0 > 0 {
        extr = extr.min(extr0);
    }

    let ctg_len = mi.nt.contigs[reg.vid.contig().index()].len;
    let as_ = if reg.vs > extl as i64 {
        reg.vs - extl as i64
    } else {
        0
    };
    let ae = (reg.ve + extr as i64).min(ctg_len);
    let mut nt = vec![0u8; (ae - as_) as usize];
    let Ok(l_nt) = mi.nt.get_by_v(reg.vid, as_, ae, &mut nt) else {
        reg.cnt = 0;
        return;
    };
    if l_nt != ae - as_ {
        reg.cnt = 0;
        return;
    }
    let ss_buf = if mi.nt.has_spsc() {
        let mut ss = vec![0u8; (ae - as_) as usize];
        let Ok(l_ss) = mi.nt.spsc_get_by_v(reg.vid, as_, ae, &mut ss) else {
            reg.cnt = 0;
            return;
        };
        if l_ss != l_nt {
            reg.cnt = 0;
            return;
        }
        Some(ss)
    } else {
        None
    };
    let ss = ss_buf.as_deref();
    let tables = &mi.tables;
    let vs0 = reg.vs;
    let mut cigar = CigarBuilder::default();
    let mut score = 0i32;
    let mut ne0;
    let mut ae0;

    {
        let vs1 = vs0 + i64::from(reg.anchors[i0].target()) + 1;
        let as1 = low31(reg.anchors[i0]) + 1;
        let mut ns_opt = map_to_ns_opt(opt, tables, NS_F_EXT_LEFT);
        let mut rst = scalar_dp(
            &nt[..(vs1 - as_) as usize],
            &aa[..as1 as usize],
            &ns_opt,
            ss,
        );
        let mut nt_len = rst.nt_len;
        let mut aa_len = rst.aa_len;
        if rst.aa_len != as1 && rst.nt_len < opt.max_ext && opt.io > opt.io_end {
            let as_alt = if vs1 - as_ > opt.max_ext as i64 {
                vs1 - opt.max_ext as i64
            } else {
                as_
            };
            ns_opt.io = opt.io_end;
            rst = scalar_dp(
                &nt[(as_alt - as_) as usize..(vs1 - as_) as usize],
                &aa[..as1 as usize],
                &ns_opt,
                ss,
            );
            if rst.aa_len == as1 {
                nt_len = rst.nt_len;
                aa_len = rst.aa_len;
            }
        }
        reg.vs = vs1 - nt_len as i64;
        reg.qs = as1 - aa_len;
        ne0 = (reg.vs - vs0) as i32;
        ae0 = reg.qs;
    }

    for anchor in reg.anchors.iter().skip(i0) {
        if !anchor.has_query_flag() {
            continue;
        }
        let ne1 = anchor.target() + 1;
        let ae1 = low31(*anchor) + 1;
        let nt_st = (ne0 as i64 + vs0 - as_) as usize;
        let nt_en = (ne1 as i64 + vs0 - as_) as usize;
        score += align_seq(
            opt,
            tables,
            &nt[nt_st..nt_en],
            &aa[ae0 as usize..ae1 as usize],
            ss.map(|ss| &ss[nt_st..nt_en]),
            &mut cigar,
        );
        ne0 = ne1;
        ae0 = ae1;
    }
    reg.ve = ne0 as i64 + vs0;
    reg.qe = ae0;

    if reg.qe < qlen && reg.ve < ae {
        let mut ns_opt = map_to_ns_opt(opt, tables, NS_F_EXT_RIGHT);
        let mut rst = scalar_dp(
            &nt[(reg.ve - as_) as usize..],
            &aa[reg.qe as usize..],
            &ns_opt,
            ss.map(|ss| &ss[(reg.ve - as_) as usize..]),
        );
        let mut nt_len = rst.nt_len;
        let mut aa_len = rst.aa_len;
        if aa_len < qlen - reg.qe && nt_len < opt.max_ext && opt.io > opt.io_end {
            let l_ext = ((ae - reg.ve) as i32).min(opt.max_ext) as usize;
            ns_opt.io = opt.io_end;
            rst = scalar_dp(
                &nt[(reg.ve - as_) as usize..(reg.ve - as_) as usize + l_ext],
                &aa[reg.qe as usize..],
                &ns_opt,
                ss.map(|ss| &ss[(reg.ve - as_) as usize..(reg.ve - as_) as usize + l_ext]),
            );
            if rst.aa_len == qlen - reg.qe {
                nt_len = rst.nt_len;
                aa_len = rst.aa_len;
            }
        }
        score += align_seq(
            opt,
            tables,
            &nt[(reg.ve - as_) as usize..(reg.ve - as_) as usize + nt_len as usize],
            &aa[reg.qe as usize..(reg.qe + aa_len) as usize],
            ss.map(|ss| &ss[(reg.ve - as_) as usize..(reg.ve - as_) as usize + nt_len as usize]),
            &mut cigar,
        );
        reg.ve += nt_len as i64;
        reg.qe += aa_len;
    }

    let cs = if (opt.flag & MP_F_NO_CS) == 0 {
        build_cs(
            tables,
            &nt[(reg.vs - as_) as usize..(reg.ve - as_) as usize],
            &aa[reg.qs as usize..reg.qe as usize],
            &cigar.cigar,
        )
    } else {
        String::new()
    };

    reg.extra = Some(AlignmentExtra {
        dp_score: score,
        cigar: cigar.cigar,
        cs,
        ..AlignmentExtra::default()
    });
    let dist_stop = extra_stop(reg, &nt, as_, ae, tables);
    let dist_start = extra_start(reg, &nt, as_, ae, tables);
    if let Some(extra) = reg.extra.as_mut() {
        extra.dist_stop = dist_stop;
        extra.dist_start = dist_start;
    }
    if !extra_cal(
        reg,
        opt,
        tables,
        &nt[(reg.vs - as_) as usize..],
        &aa[reg.qs as usize..],
        qlen,
    ) {
        reg.cnt = 0;
    }
}
