use crate::align::{AA_STOP, NsOpt, NsResult, acceptor_poly_y_penalty};
use crate::tables::{
    AA_I2C, Kind, NS_F_CIGAR, NS_F_EXT_LEFT, NS_F_EXT_RIGHT, NS_SPSC_OFFSET, Tables, pack_cigar_op,
    unpack_cigar_op,
};

const NEG_INF: i32 = i32::MIN / 4;
const AA_AMBI: u8 = 21;

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

#[derive(Default)]
struct CigarBuilder {
    cigar: Vec<u32>,
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

const fn nt_code(tables: &Tables, byte: u8) -> u8 {
    if byte < 5 {
        byte
    } else {
        tables.nt4[byte as usize]
    }
}

const fn aa_code(tables: &Tables, byte: u8) -> u8 {
    if byte < AA_I2C.len() as u8 {
        byte
    } else {
        tables.aa20[byte as usize]
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
            pen_y = acceptor_poly_y_penalty(&nts, i, opt.sp[5]);
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

const fn score_pair(sc: &[[i8; 22]; 22], nt_aa: u8, aa_aa: u8) -> i32 {
    sc[nt_aa as usize][aa_aa as usize] as i32
}

const fn encode_trace(
    state: u8,
    i_ext: bool,
    d_ext: bool,
    a_ext: bool,
    b_ext: bool,
    c_ext: bool,
) -> u16 {
    (state as u16)
        | ((i_ext as u16) << 4)
        | ((d_ext as u16) << 5)
        | ((a_ext as u16) << 6)
        | ((b_ext as u16) << 7)
        | ((c_ext as u16) << 8)
}

const fn trace_ext(x: u16, state: u8) -> bool {
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

pub(crate) fn global(ns: &[u8], aa: &[u8], opt: &NsOpt<'_>, ss: Option<&[u8]>) -> NsResult {
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

    let sp_default = opt.sp[3];
    let has_splice = prep.donor[..nl].iter().any(|&d| d != sp_default)
        || prep.acceptor[..nl].iter().any(|&a| a != sp_default);

    let mut h_prev3 = vec![NEG_INF; al + 1];
    let mut h_prev2 = vec![NEG_INF; al + 1];
    let mut h_prev1 = vec![NEG_INF; al + 1];
    let mut h_cur = vec![NEG_INF; al + 1];
    let mut d_prev3 = vec![NEG_INF; al + 1];
    let mut d_prev2 = vec![NEG_INF; al + 1];
    let mut d_prev1 = vec![NEG_INF; al + 1];
    let mut d_cur = vec![NEG_INF; al + 1];
    let mut a_prev1 = if has_splice {
        vec![NEG_INF; al + 1]
    } else {
        Vec::new()
    };
    let mut b_prev1 = if has_splice {
        vec![NEG_INF; al + 1]
    } else {
        Vec::new()
    };
    let mut c_prev1 = if has_splice {
        vec![NEG_INF; al + 1]
    } else {
        Vec::new()
    };
    let mut a_cur = if has_splice {
        vec![NEG_INF; al + 1]
    } else {
        Vec::new()
    };
    let mut b_cur = if has_splice {
        vec![NEG_INF; al + 1]
    } else {
        Vec::new()
    };
    let mut c_cur = if has_splice {
        vec![NEG_INF; al + 1]
    } else {
        Vec::new()
    };

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
        if has_splice {
            a_cur.fill(NEG_INF);
            b_cur.fill(NEG_INF);
            c_cur.fill(NEG_INF);
        }

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
            let mut best = h_prev3[j] + score_pair(opt.sc, prep.nas[i], prep.aas[j]);
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

            let mut a_ext = false;
            let mut b_ext = false;
            let mut c_ext = false;

            if has_splice {
                let open_a = h_prev1[col] - opt.io - dim1;
                let ext_a = a_prev1[col];
                t = open_a.max(ext_a);
                a_ext = ext_a > open_a;
                a_cur[col] = t;
                let ta = t - ai;
                if ta > best {
                    best = ta;
                    state = 3;
                }

                let open_b = h_prev1[j] - opt.io - di;
                let ext_b = b_prev1[col];
                t = open_b.max(ext_b);
                b_ext = ext_b > open_b;
                b_cur[col] = t;
                let tb = t - aim2;
                if tb > best {
                    best = tb;
                    state = 4;
                }

                let open_c = h_prev1[j] - opt.io - dip1;
                let ext_c = c_prev1[col];
                t = open_c.max(ext_c);
                c_ext = ext_c > open_c;
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
        if has_splice {
            std::mem::swap(&mut a_prev1, &mut a_cur);
            std::mem::swap(&mut b_prev1, &mut b_cur);
            std::mem::swap(&mut c_prev1, &mut c_cur);
        }
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
