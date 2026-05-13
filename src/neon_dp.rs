#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use std::arch::aarch64::*;

    use crate::align::{AA_STOP, NsOpt, NsResult, acceptor_poly_y_penalty};
    use crate::tables::{
        Kind, NS_F_CIGAR, NS_F_EXT_LEFT, NS_F_EXT_RIGHT, pack_cigar_op, unpack_cigar_op,
    };

    const LANES: usize = 8;
    const NEG: i16 = i16::MIN;
    const AA_AMBI: u8 = 21;

    struct PreparedSeq {
        nas: Vec<u8>,
        aas: Vec<u8>,
        donor: Vec<i16>,
        acceptor: Vec<i16>,
        has_splice: bool,
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

    #[inline(always)]
    const fn i8_i16(value: i32) -> i16 {
        value as i8 as i16
    }

    #[inline(always)]
    fn add_i8(slot: &mut i16, delta: i32) {
        *slot = (*slot as i32 + delta) as i8 as i16;
    }

    fn prep_nas(ns: &[u8], opt: &NsOpt<'_>) -> Vec<u8> {
        let mut nas = vec![AA_AMBI; ns.len()];
        let mut codon = 0u8;
        let mut len = 0i32;
        for (i, &byte) in ns.iter().enumerate() {
            let c = opt.tables.nt4[byte as usize];
            if c < 4 {
                codon = ((codon << 2) | c) & 0x3f;
                len += 1;
                if len >= 3 {
                    nas[i] = opt.tables.codon[codon as usize];
                }
            } else {
                codon = 0;
                len = 0;
            }
        }
        nas
    }

    fn spsc(value: u8, max_spsc: i32) -> i32 {
        (((value >> 1) as i8 as i32) - 64).min(max_spsc)
    }

    fn prep_seq(ns: &[u8], aa: &[u8], opt: &NsOpt<'_>, ss: Option<&[u8]>) -> PreparedSeq {
        let aas: Vec<_> = aa
            .iter()
            .map(|&byte| opt.tables.aa20[byte as usize])
            .collect();
        let nts: Vec<_> = ns
            .iter()
            .map(|&byte| opt.tables.nt4[byte as usize])
            .collect();
        let mut donor = vec![i8_i16(opt.sp[3]); nts.len() + 1];
        let mut acceptor = vec![i8_i16(opt.sp[3]); nts.len() + 1];

        for i in 0..nts.len().saturating_sub(3) {
            let mut splice = 3i32;
            if nts[i + 1] == 2 && nts[i + 2] == 3 {
                splice = if i + 3 < nts.len() && (nts[i + 3] == 0 || nts[i + 3] == 2) {
                    if nts[i] == 2 { -1 } else { 4 }
                } else {
                    0
                };
            } else if nts[i + 1] == 2 && nts[i + 2] == 1 && nts[i] == 2 {
                splice = 1;
            } else if nts[i + 1] == 0 && nts[i + 2] == 3 {
                splice = 2;
            }
            donor[i] = if splice < 0 {
                0
            } else {
                i8_i16(opt.sp[splice as usize])
            };
        }
        for i in 1..nts.len() {
            let mut splice = 3i32;
            let mut pen_y = 0i32;
            if nts[i - 1] == 0 && nts[i] == 2 {
                splice = if i >= 2 && (nts[i - 2] == 1 || nts[i - 2] == 3) {
                    -1
                } else {
                    0
                };
                pen_y = acceptor_poly_y_penalty(&nts, i, opt.sp[5]);
            } else if nts[i - 1] == 0 && nts[i] == 1 {
                splice = 2;
            }
            acceptor[i] = if splice < 0 {
                0
            } else {
                i8_i16(opt.sp[splice as usize])
            };
            if splice == -1 || splice == 0 {
                add_i8(&mut acceptor[i], pen_y);
            }
        }

        if let Some(ss) = ss {
            let max_spsc = (opt.io + 1) / 2 - 1;
            for (i, &value) in ss.iter().take(nts.len()).enumerate().skip(1) {
                let spsc = spsc(value, max_spsc);
                if value == 0xff {
                    add_i8(&mut donor[i - 1], -opt.sp_null_bonus);
                    add_i8(&mut acceptor[i - 1], -opt.sp_null_bonus);
                } else if (value & 1) != 0 {
                    add_i8(&mut acceptor[i - 1], -spsc);
                } else {
                    add_i8(&mut donor[i - 1], -spsc);
                }
            }
        }

        let sp_default = i8_i16(opt.sp[3]);
        let n = nts.len();
        let has_splice = donor[..n]
            .iter()
            .chain(acceptor[..n].iter())
            .any(|&v| v != sp_default);
        PreparedSeq {
            nas: prep_nas(ns, opt),
            aas,
            donor,
            acceptor,
            has_splice,
        }
    }

    fn prep_seq_left(ns: &[u8], aa: &[u8], opt: &NsOpt<'_>, ss: Option<&[u8]>) -> PreparedSeq {
        let al = aa.len();
        let nl = ns.len();
        let mut aas = vec![0u8; al];
        for (j, &byte) in aa.iter().enumerate() {
            aas[al - 1 - j] = opt.tables.aa20[byte as usize];
        }

        let mut nts = vec![0u8; nl];
        for (i, &byte) in ns.iter().enumerate() {
            nts[nl - 1 - i] = opt.tables.nt4[byte as usize];
        }
        let mut donor = vec![i8_i16(opt.sp[3]); nl + 1];
        let mut acceptor = vec![i8_i16(opt.sp[3]); nl + 1];

        for i in 0..nl.saturating_sub(3) {
            let mut splice = 3i32;
            let mut pen_y = 0i32;
            if nts[i + 1] == 2 && nts[i + 2] == 0 {
                splice = if i + 3 < nl && (nts[i + 3] == 1 || nts[i + 3] == 3) {
                    -1
                } else {
                    0
                };
                let mut j = i + 5;
                while j < nl && j < i + 8 {
                    if nts[j] != 1 && nts[j] != 3 {
                        pen_y += opt.sp[5];
                    }
                    j += 1;
                }
            } else if nts[i + 1] == 1 && nts[i + 2] == 0 {
                splice = 2;
            }
            donor[i] = if splice < 0 {
                0
            } else {
                i8_i16(opt.sp[splice as usize])
            };
            if splice == -1 || splice == 0 {
                add_i8(&mut donor[i], pen_y);
            }
        }
        for i in 1..nl {
            let mut splice = 3i32;
            if nts[i - 1] == 3 && nts[i] == 2 {
                splice = if i >= 2 && (nts[i - 2] == 0 || nts[i - 2] == 2) {
                    if i + 1 < nl && nts[i + 1] == 2 { -1 } else { 4 }
                } else {
                    0
                };
            } else if nts[i - 1] == 1 && nts[i] == 2 && i + 1 < nl && nts[i + 1] == 1 {
                splice = 1;
            } else if nts[i - 1] == 3 && nts[i] == 0 {
                splice = 2;
            }
            acceptor[i] = if splice < 0 {
                0
            } else {
                i8_i16(opt.sp[splice as usize])
            };
        }

        if let Some(ss) = ss {
            let max_spsc = (opt.io + 1) / 2 - 1;
            for (i, &value) in ss.iter().take(nl).enumerate() {
                let spsc = spsc(value, max_spsc);
                let idx = nl - i - 1;
                if value == 0xff {
                    add_i8(&mut donor[idx], -opt.sp_null_bonus);
                    add_i8(&mut acceptor[idx], -opt.sp_null_bonus);
                } else if (value & 1) != 0 {
                    add_i8(&mut donor[idx], -spsc);
                } else {
                    add_i8(&mut acceptor[idx], -spsc);
                }
            }
        }

        let mut nas = prep_nas(ns, opt);
        nas.reverse();
        if nl >= 2 {
            nas.copy_within(0..nl - 2, 2);
        }
        nas[..nl.min(2)].fill(AA_AMBI);

        let sp_default = i8_i16(opt.sp[3]);
        let has_splice = donor[..nl]
            .iter()
            .chain(acceptor[..nl].iter())
            .any(|&v| v != sp_default);
        PreparedSeq {
            nas,
            aas,
            donor,
            acceptor,
            has_splice,
        }
    }

    #[inline(always)]
    unsafe fn set1(x: i16) -> int16x8_t {
        unsafe { vdupq_n_s16(x) }
    }

    #[inline(always)]
    unsafe fn set_lane0(v: int16x8_t, x: i16) -> int16x8_t {
        unsafe { vsetq_lane_s16::<0>(x, v) }
    }

    #[inline(always)]
    unsafe fn shift_left_2(v: int16x8_t) -> int16x8_t {
        unsafe { vreinterpretq_s16_u8(vextq_u8::<14>(vdupq_n_u8(0), vreinterpretq_u8_s16(v))) }
    }

    #[inline(always)]
    unsafe fn v_or(a: int16x8_t, b: int16x8_t) -> int16x8_t {
        unsafe {
            vreinterpretq_s16_u16(vorrq_u16(
                vreinterpretq_u16_s16(a),
                vreinterpretq_u16_s16(b),
            ))
        }
    }

    #[inline(always)]
    unsafe fn v_and(a: int16x8_t, b: int16x8_t) -> int16x8_t {
        unsafe {
            vreinterpretq_s16_u16(vandq_u16(
                vreinterpretq_u16_s16(a),
                vreinterpretq_u16_s16(b),
            ))
        }
    }

    #[inline(always)]
    unsafe fn cmpgt(a: int16x8_t, b: int16x8_t) -> int16x8_t {
        unsafe { vreinterpretq_s16_u16(vcgtq_s16(a, b)) }
    }

    #[inline(always)]
    unsafe fn select(cond: int16x8_t, a: int16x8_t, b: int16x8_t) -> int16x8_t {
        unsafe { vbslq_s16(vreinterpretq_u16_s16(cond), a, b) }
    }

    #[inline(always)]
    unsafe fn all_le(a: int16x8_t, b: int16x8_t) -> bool {
        unsafe { vmaxvq_u16(vcgtq_s16(a, b)) == 0 }
    }

    #[inline(always)]
    unsafe fn extract(v: int16x8_t, lane: usize) -> i16 {
        let lanes: [i16; LANES] = unsafe { std::mem::transmute(v) };
        lanes[lane]
    }

    #[inline(always)]
    unsafe fn load_lanes(lanes: [i16; LANES]) -> int16x8_t {
        unsafe { vld1q_s16(lanes.as_ptr()) }
    }

    #[inline(always)]
    unsafe fn max_lane(v: int16x8_t) -> i32 {
        unsafe { vmaxvq_s16(v) as i32 }
    }

    #[inline(always)]
    fn approx_log2(x: i32) -> f32 {
        let mut bits = (x as f32).to_bits();
        let mut log_2 = ((bits >> 23) & 255) as f32 - 128.0;
        bits &= !(255 << 23);
        bits += 127 << 23;
        let z = f32::from_bits(bits);
        log_2 += (-0.34484843 * z + 2.02466578) * z - 0.67487759;
        log_2
    }

    unsafe fn gen_profile(
        aas: &[u8],
        al: usize,
        slen: usize,
        sc: &[[i8; 22]; 22],
    ) -> Vec<int16x8_t> {
        let neg = unsafe { set1(NEG) };
        let mut profile = vec![neg; 22 * slen];
        for (a, row) in sc.iter().enumerate() {
            for j in 0..slen {
                let mut lanes = [NEG; LANES];
                for (lane, dst) in lanes.iter_mut().enumerate() {
                    let col = j + lane * slen;
                    if col < al {
                        *dst = row[aas[col] as usize] as i16;
                    }
                }
                profile[a * slen + j] = unsafe { load_lanes(lanes) };
            }
        }
        profile
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

    unsafe fn backtrack(tb: &[int16x8_t], nl: i32, al: i32, slen: usize) -> Vec<u32> {
        let mut i = nl - 1;
        let mut j = al - 1;
        let mut last = 0i32;
        let mut cigar = CigarBuilder::default();
        while i >= 2 && j >= 0 {
            let vec = tb[i as usize * slen + j as usize % slen];
            let mut x = unsafe { extract(vec, j as usize / slen) } as i32;
            if ((x >> 9) & 1) != 0 {
                x = 1 | ((x >> 4) << 4);
            }
            let state = if last == 0 { x & 0x0f } else { last };
            let ext = (1..=5).contains(&state) && ((x >> (state + 3)) & 1) != 0;
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
            let len = ((i + 1) / 3) * 3;
            let tail = (i + 1) % 3;
            if len > 0 {
                cigar.push(Kind::Deletion, len);
            }
            if tail != 0 {
                cigar.push(Kind::FrameshiftGap, tail);
            }
        }
        cigar.cigar.reverse();
        fix_tiny_uv(&mut cigar.cigar);
        cigar.cigar
    }

    #[inline(always)]
    unsafe fn update_boundary(row: &mut [int16x8_t], slen: usize) {
        row[0] = unsafe { set_lane0(shift_left_2(row[slen]), NEG) };
    }

    pub(crate) fn global_gs16b(
        ns: &[u8],
        aa: &[u8],
        opt: &NsOpt<'_>,
        ss: Option<&[u8]>,
    ) -> NsResult {
        if aa.is_empty() || ns.is_empty() {
            return NsResult::default();
        }
        unsafe { global_gs16b_inner(ns, aa, opt, ss) }
    }

    unsafe fn global_gs16b_inner(
        ns: &[u8],
        aa: &[u8],
        opt: &NsOpt<'_>,
        ss: Option<&[u8]>,
    ) -> NsResult {
        let prep = if (opt.flag & NS_F_EXT_LEFT) != 0 {
            prep_seq_left(ns, aa, opt, ss)
        } else {
            prep_seq(ns, aa, opt, ss)
        };
        let nl = prep.nas.len();
        let al = prep.aas.len();
        let is_ext = (opt.flag & (NS_F_EXT_LEFT | NS_F_EXT_RIGHT)) != 0;
        let need_trace = (opt.flag & NS_F_CIGAR) != 0 && !is_ext;
        let slen = al.div_ceil(LANES);
        let neg = unsafe { set1(NEG) };
        let zero = unsafe { set1(0) };
        let go = unsafe { set1(opt.go as i16) };
        let ge = unsafe { set1(opt.ge as i16) };
        let goe = unsafe { set1((opt.go + opt.ge) as i16) };
        let io = unsafe { set1(opt.io as i16) };
        let fs = unsafe { set1(opt.fs as i16) };
        let profile = unsafe { gen_profile(&prep.aas, al, slen, opt.sc) };

        let mut h = vec![neg; slen + 1];
        let mut h1 = vec![neg; slen + 1];
        let mut h2 = vec![neg; slen + 1];
        let mut h3 = vec![neg; slen + 1];
        let mut hmax = vec![neg; slen + 1];
        let mut d = vec![neg; slen];
        let mut d1 = vec![neg; slen];
        let mut d2 = vec![neg; slen];
        let mut d3 = vec![neg; slen];
        let has_splice = prep.has_splice;
        let mut a = if has_splice { vec![neg; slen] } else { Vec::new() };
        let mut b = if has_splice { vec![neg; slen] } else { Vec::new() };
        let mut c = if has_splice { vec![neg; slen] } else { Vec::new() };
        let mut tb = if need_trace {
            vec![zero; nl * slen]
        } else {
            Vec::new()
        };

        h3[0] = unsafe { set_lane0(h3[0], 0) };
        h2[0] = unsafe { set_lane0(h2[0], -(opt.fs as i16)) };
        h1[0] = unsafe { set_lane0(h1[0], -(opt.fs as i16)) };

        let mut max_sc = i32::MIN;
        let mut max_sc_log = i32::MIN;
        let mut max_i = -1i32;
        let pen_len = al as i32 * 3;

        for i in 2..nl {
            let s = &profile[prep.nas[i] as usize * slen..];
            let gei = if prep.nas[i] == AA_STOP { fs } else { ge };
            let dim1 = unsafe { set1(prep.donor[i - 1]) };
            let di = unsafe { set1(prep.donor[i]) };
            let dip1 = unsafe { set1(prep.donor[i + 1]) };
            let ai = unsafe { set1(prep.acceptor[i]) };
            let aim1 = unsafe { set1(prep.acceptor[i - 1]) };
            let aim2 = unsafe { set1(prep.acceptor[i - 2]) };
            let mut i_state = neg;
            let mut last_h = neg;

            if i > 2 {
                unsafe {
                    update_boundary(&mut h3, slen);
                    update_boundary(&mut h2, slen);
                    update_boundary(&mut h1, slen);
                }
            }

            if !need_trace {
                let mut row_max = neg;
                for j in 0..slen {
                    let mut h_vec = unsafe { vqaddq_s16(h3[j], s[j]) };

                    let mut t = unsafe { vqsubq_s16(last_h, go) };
                    t = unsafe { vmaxq_s16(t, i_state) };
                    i_state = unsafe { vqsubq_s16(t, ge) };
                    h_vec = unsafe { vmaxq_s16(h_vec, i_state) };

                    let u = unsafe { vqsubq_s16(h3[j + 1], go) };
                    t = unsafe { vmaxq_s16(u, d3[j]) };
                    t = unsafe { vqsubq_s16(t, gei) };
                    d[j] = t;
                    h_vec = unsafe { vmaxq_s16(h_vec, t) };

                    if has_splice {
                        let mut u = unsafe { vqsubq_s16(h1[j + 1], io) };
                        t = unsafe { vqsubq_s16(u, dim1) };
                        t = unsafe { vmaxq_s16(t, a[j]) };
                        a[j] = t;
                        h_vec = unsafe { vmaxq_s16(h_vec, vqsubq_s16(t, ai)) };

                        u = unsafe { vqsubq_s16(h1[j], io) };
                        t = unsafe { vqsubq_s16(u, di) };
                        t = unsafe { vmaxq_s16(t, b[j]) };
                        b[j] = t;
                        h_vec = unsafe { vmaxq_s16(h_vec, vqsubq_s16(t, aim2)) };

                        t = unsafe { vqsubq_s16(u, dip1) };
                        t = unsafe { vmaxq_s16(t, c[j]) };
                        c[j] = t;
                        h_vec = unsafe { vmaxq_s16(h_vec, vqsubq_s16(t, aim1)) };
                    }

                    t = unsafe { vqsubq_s16(h1[j + 1], fs) };
                    h_vec = unsafe { vmaxq_s16(h_vec, t) };
                    t = unsafe { vqsubq_s16(h2[j + 1], fs) };
                    h_vec = unsafe { vmaxq_s16(h_vec, t) };
                    t = unsafe { vqsubq_s16(h1[j], fs) };
                    h_vec = unsafe { vmaxq_s16(h_vec, t) };
                    t = unsafe { vqsubq_s16(h2[j], fs) };
                    h_vec = unsafe { vmaxq_s16(h_vec, t) };

                    row_max = unsafe { vmaxq_s16(row_max, h_vec) };
                    h[j + 1] = h_vec;
                    last_h = h_vec;
                }

                i_state = unsafe { vmaxq_s16(vqsubq_s16(last_h, goe), vqsubq_s16(i_state, ge)) };
                for _ in 0..LANES {
                    i_state = unsafe { set_lane0(shift_left_2(i_state), NEG) };
                    let mut stopped = false;
                    for j in 0..slen {
                        let mut h_vec = h[j + 1];
                        h_vec = unsafe { vmaxq_s16(h_vec, i_state) };
                        row_max = unsafe { vmaxq_s16(row_max, h_vec) };
                        h[j + 1] = h_vec;
                        let h_gap = unsafe { vqsubq_s16(h_vec, goe) };
                        i_state = unsafe { vqsubq_s16(i_state, ge) };
                        if unsafe { all_le(i_state, h_gap) } {
                            stopped = true;
                            break;
                        }
                    }
                    if stopped {
                        break;
                    }
                }

                let mut tmp_sc = unsafe { max_lane(row_max) };
                let end_sc = unsafe { extract(h[(al - 1) % slen + 1], (al - 1) / slen) } as i32
                    + opt.end_bonus;
                tmp_sc = tmp_sc.max(end_sc);
                let len_pen = if i as i32 - pen_len < 2 {
                    0
                } else {
                    (opt.ie_coef * approx_log2(i as i32 - pen_len) + 0.5) as i32
                };
                let tmp_sc_log = tmp_sc - len_pen;
                if tmp_sc_log > max_sc_log {
                    max_sc = tmp_sc;
                    max_sc_log = tmp_sc_log;
                    max_i = i as i32;
                    hmax.copy_from_slice(&h);
                }

                std::mem::swap(&mut h3, &mut h2);
                std::mem::swap(&mut h2, &mut h1);
                std::mem::swap(&mut h1, &mut h);
                std::mem::swap(&mut d3, &mut d2);
                std::mem::swap(&mut d2, &mut d1);
                std::mem::swap(&mut d1, &mut d);

                if max_sc_log - tmp_sc_log > opt.xdrop {
                    break;
                }
            } else {
                let tb_row = &mut tb[i * slen..(i + 1) * slen];
                for j in 0..slen {
                    let mut y = zero;
                    let mut z = zero;
                    let mut h_vec = unsafe { vqaddq_s16(h3[j], s[j]) };

                    let mut t = unsafe { vqsubq_s16(last_h, go) };
                    z = unsafe { v_or(z, v_and(cmpgt(i_state, t), set1(1 << 4))) };
                    t = unsafe { vmaxq_s16(t, i_state) };
                    i_state = unsafe { vqsubq_s16(t, ge) };
                    y = unsafe { select(cmpgt(i_state, h_vec), set1(1), y) };
                    h_vec = unsafe { vmaxq_s16(h_vec, i_state) };

                    let u0 = unsafe { vqsubq_s16(h3[j + 1], go) };
                    z = unsafe { v_or(z, v_and(cmpgt(d3[j], u0), set1(1 << 5))) };
                    t = unsafe { vmaxq_s16(u0, d3[j]) };
                    t = unsafe { vqsubq_s16(t, gei) };
                    d[j] = t;
                    y = unsafe { select(cmpgt(t, h_vec), set1(2), y) };
                    h_vec = unsafe { vmaxq_s16(h_vec, t) };

                    if has_splice {
                        let mut u = unsafe { vqsubq_s16(h1[j + 1], io) };
                        t = unsafe { vqsubq_s16(u, dim1) };
                        z = unsafe { v_or(z, v_and(cmpgt(a[j], t), set1(1 << 6))) };
                        t = unsafe { vmaxq_s16(t, a[j]) };
                        a[j] = t;
                        t = unsafe { vqsubq_s16(t, ai) };
                        y = unsafe { select(cmpgt(t, h_vec), set1(3), y) };
                        h_vec = unsafe { vmaxq_s16(h_vec, t) };

                        u = unsafe { vqsubq_s16(h1[j], io) };
                        t = unsafe { vqsubq_s16(u, di) };
                        z = unsafe { v_or(z, v_and(cmpgt(b[j], t), set1(1 << 7))) };
                        t = unsafe { vmaxq_s16(t, b[j]) };
                        b[j] = t;
                        t = unsafe { vqsubq_s16(t, aim2) };
                        y = unsafe { select(cmpgt(t, h_vec), set1(4), y) };
                        h_vec = unsafe { vmaxq_s16(h_vec, t) };

                        t = unsafe { vqsubq_s16(u, dip1) };
                        z = unsafe { v_or(z, v_and(cmpgt(c[j], t), set1(1 << 8))) };
                        t = unsafe { vmaxq_s16(t, c[j]) };
                        c[j] = t;
                        t = unsafe { vqsubq_s16(t, aim1) };
                        y = unsafe { select(cmpgt(t, h_vec), set1(5), y) };
                        h_vec = unsafe { vmaxq_s16(h_vec, t) };
                    }

                    t = unsafe { vqsubq_s16(h1[j + 1], fs) };
                    y = unsafe { select(cmpgt(t, h_vec), set1(6), y) };
                    h_vec = unsafe { vmaxq_s16(h_vec, t) };
                    t = unsafe { vqsubq_s16(h2[j + 1], fs) };
                    y = unsafe { select(cmpgt(t, h_vec), set1(7), y) };
                    h_vec = unsafe { vmaxq_s16(h_vec, t) };
                    t = unsafe { vqsubq_s16(h1[j], fs) };
                    y = unsafe { select(cmpgt(t, h_vec), set1(8), y) };
                    h_vec = unsafe { vmaxq_s16(h_vec, t) };
                    t = unsafe { vqsubq_s16(h2[j], fs) };
                    y = unsafe { select(cmpgt(t, h_vec), set1(9), y) };
                    h_vec = unsafe { vmaxq_s16(h_vec, t) };

                    z = unsafe { v_or(z, y) };
                    tb_row[j] = z;
                    h[j + 1] = h_vec;
                    last_h = h_vec;
                }

                i_state = unsafe { vmaxq_s16(vqsubq_s16(last_h, goe), vqsubq_s16(i_state, ge)) };
                for _ in 0..LANES {
                    i_state = unsafe { set_lane0(shift_left_2(i_state), NEG) };
                    let mut stopped = false;
                    for j in 0..slen {
                        let mut z = tb_row[j];
                        let mut h_vec = h[j + 1];
                        z = unsafe { v_or(z, v_and(cmpgt(i_state, h_vec), set1(1 << 9))) };
                        h_vec = unsafe { vmaxq_s16(h_vec, i_state) };
                        tb_row[j] = z;
                        h[j + 1] = h_vec;
                        let h_gap = unsafe { vqsubq_s16(h_vec, goe) };
                        i_state = unsafe { vqsubq_s16(i_state, ge) };
                        if unsafe { all_le(i_state, h_gap) } {
                            stopped = true;
                            break;
                        }
                    }
                    if stopped {
                        break;
                    }
                }

                std::mem::swap(&mut h3, &mut h2);
                std::mem::swap(&mut h2, &mut h1);
                std::mem::swap(&mut h1, &mut h);
                std::mem::swap(&mut d3, &mut d2);
                std::mem::swap(&mut d2, &mut d1);
                std::mem::swap(&mut d1, &mut d);
            }
        }

        if is_ext {
            let mut best_aa = 0i32;
            for j in 0..al {
                let mut sc = unsafe { extract(hmax[j % slen + 1], j / slen) } as i32;
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

        let score = unsafe { extract(h1[(al - 1) % slen + 1], (al - 1) / slen) } as i32;
        let cigar = if need_trace {
            unsafe { backtrack(&tb, nl as i32, al as i32, slen) }
        } else {
            Vec::new()
        };
        NsResult {
            nt_len: nl as i32,
            aa_len: al as i32,
            score,
            cigar,
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub(crate) use aarch64::global_gs16b;

#[cfg(not(target_arch = "aarch64"))]
pub(crate) fn global_gs16b(
    ns: &[u8],
    aa: &[u8],
    opt: &crate::align::NsOpt<'_>,
    ss: Option<&[u8]>,
) -> crate::align::NsResult {
    crate::scalar_dp::global(ns, aa, opt, ss)
}
