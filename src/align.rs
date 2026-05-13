use crate::format::build_cs;
use crate::index::Index;
use crate::tables::{
    AA_I2C, Kind, NS_F_CIGAR, NS_F_EXT_LEFT, NS_F_EXT_RIGHT, Tables, pack_cigar_op, unpack_cigar_op,
};
use crate::types::{
    Alignment, AlignmentExtra, Anchor, Feature, FeatureType, MP_F_NO_CS, MapOptions,
};

pub(crate) const AA_STOP: u8 = 20;
const AA_AMBI: u8 = 21;

#[derive(Clone, Copy)]
pub(crate) struct NsOpt<'a> {
    pub(crate) flag: i32,
    pub(crate) go: i32,
    pub(crate) ge: i32,
    pub(crate) io: i32,
    pub(crate) fs: i32,
    pub(crate) xdrop: i32,
    pub(crate) end_bonus: i32,
    pub(crate) sp: [i32; 6],
    pub(crate) sp_null_bonus: i32,
    pub(crate) ie_coef: f32,
    pub(crate) sc: &'a [[i8; 22]; 22],
    pub(crate) tables: &'a Tables,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct NsResult {
    pub(crate) cigar: Vec<u32>,
    pub(crate) nt_len: i32,
    pub(crate) aa_len: i32,
    pub(crate) score: i32,
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

const fn aa_code(tables: &Tables, byte: u8) -> u8 {
    if byte < AA_I2C.len() as u8 {
        byte
    } else {
        tables.aa20[byte as usize]
    }
}

#[inline]
const fn codon_aa(tables: &Tables, n1: u8, n2: u8, n3: u8) -> u8 {
    if n1 > 3 || n2 > 3 || n3 > 3 {
        AA_AMBI
    } else {
        tables.codon[((n1 << 4) | (n2 << 2) | n3) as usize]
    }
}

#[inline]
const fn slice_codon_aa(tables: &Tables, nt: &[u8], i: usize) -> u8 {
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

const fn score_pair(sc: &[[i8; 22]; 22], nt_aa: u8, aa_aa: u8) -> i32 {
    sc[nt_aa as usize][aa_aa as usize] as i32
}

#[inline(always)]
pub(crate) fn acceptor_poly_y_penalty(nts: &[u8], i: usize, penalty: i32) -> i32 {
    let mut total = 0;
    for j in (i.saturating_sub(6)..i.saturating_sub(3)).rev() {
        if nts[j] != 1 && nts[j] != 3 {
            total += penalty;
        }
    }
    total
}

fn scalar_dp(ns: &[u8], aa: &[u8], opt: &NsOpt<'_>, ss: Option<&[u8]>) -> NsResult {
    if aa.is_empty() || ns.is_empty() {
        return NsResult::default();
    }
    crate::neon_dp::global_gs16b(ns, aa, opt, ss)
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
            while (i as i32) < k {
                if low31(a[i]) + 1 - t >= trim_back {
                    break;
                }
                i += 1;
            }
            while (i as i32) <= k {
                a[i] = a[i].with_query_flag();
                i += 1;
            }
            i = j - 1;
        }
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::{acceptor_poly_y_penalty, filter_seed};
    use crate::types::Anchor;

    #[test]
    fn acceptor_poly_y_window_is_i4_to_i6_clipped_at_start() {
        let nts = [0, 1, 2, 3, 0, 0, 0, 0, 0];
        assert_eq!(acceptor_poly_y_penalty(&nts, 1, 4), 0);
        assert_eq!(acceptor_poly_y_penalty(&nts, 3, 4), 0);
        assert_eq!(acceptor_poly_y_penalty(&nts, 4, 4), 4);
        assert_eq!(acceptor_poly_y_penalty(&nts, 5, 4), 4);
        assert_eq!(acceptor_poly_y_penalty(&nts, 6, 4), 8);
        assert_eq!(acceptor_poly_y_penalty(&nts, 7, 4), 4);
        assert_eq!(acceptor_poly_y_penalty(&nts, 8, 4), 8);
    }

    #[test]
    fn filter_seed_does_not_mark_when_trim_removes_entire_initial_block() {
        let mut anchors = [
            Anchor::from_parts(0, 0),
            Anchor::from_parts(3, 1),
            Anchor::from_parts(6, 2),
        ];
        filter_seed(&mut anchors, 6, 3, 5, 10);
        assert!(anchors.iter().all(|anchor| !anchor.has_query_flag()));
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
                    let s = score_pair(ns_opt.sc, nt_aa, aa_aa);
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
                    let s = score_pair(ns_opt.sc, nt_aa, aa_aa);
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
        reg.invalidate();
        return;
    }

    filter_seed(&mut reg.anchors, 6, 3, opt.kmer2, opt.kmer2 + 1);
    let Some(i0) = reg
        .anchors
        .iter()
        .position(|anchor| anchor.has_query_flag())
    else {
        reg.invalidate();
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
        reg.invalidate();
        return;
    };
    if l_nt != ae - as_ {
        reg.invalidate();
        return;
    }
    let ss_buf = if mi.nt.has_spsc() {
        let mut ss = vec![0u8; (ae - as_) as usize];
        let Ok(l_ss) = mi.nt.spsc_get_by_v(reg.vid, as_, ae, &mut ss) else {
            reg.invalidate();
            return;
        };
        if l_ss != l_nt {
            reg.invalidate();
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
        reg.invalidate();
    }
}
