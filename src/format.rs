use std::fmt::Write as _;

use noodles::core::Position;
use noodles::gff::{
    self as gff,
    feature::{
        RecordBuf as GffRecordBuf,
        record::{Phase as GffPhase, Strand as GffStrand},
        record_buf::{
            Attributes as GffAttributes,
            attributes::field::{Tag as GffTag, Value as GffValue},
        },
    },
};
use noodles::gtf;

use crate::Index;
use crate::fastx::QueryRecord;
use crate::tables::{AA_I2C, Kind, NT_I2C, Tables, unpack_cigar_op};
use crate::types::{
    Alignment, FeatureType, MP_F_GFF, MP_F_GTF, MP_F_NO_CS, MP_F_NO_PAF, MP_F_SHOW_RESIDUE,
    MP_F_SHOW_TRANS, MP_F_SHOW_UNMAP, MapOptions, VirtualId,
};

fn nt_aa(tables: &Tables, nt: &[u8], i: usize) -> u8 {
    let codon = (nt[i] << 4) | (nt[i + 1] << 2) | nt[i + 2];
    if nt[i] > 3 || nt[i + 1] > 3 || nt[i + 2] > 3 {
        tables.aa20[b'X' as usize]
    } else {
        tables.codon[codon as usize]
    }
}

fn q_aa(tables: &Tables, aa: u8) -> u8 {
    if aa < AA_I2C.len() as u8 {
        aa
    } else {
        tables.aa20[aa as usize]
    }
}

const fn lower_nt(nt: u8) -> char {
    NT_I2C[nt as usize].to_ascii_lowercase() as char
}

const fn upper_aa(aa: u8) -> char {
    (aa as char).to_ascii_uppercase()
}

#[inline]
const fn strand_char(vid: VirtualId) -> char {
    if vid.is_rev() { '-' } else { '+' }
}

#[inline]
const fn project_range(vid: VirtualId, ctg_len: i64, vs: i64, ve: i64) -> (i64, i64) {
    if vid.is_rev() {
        (ctg_len - ve, ctg_len - vs)
    } else {
        (vs, ve)
    }
}

fn push_lower_nts(out: &mut String, nts: &[u8]) {
    out.extend(nts.iter().map(|&nt| lower_nt(nt)));
}

fn push_upper_aas(out: &mut String, aas: &[u8]) {
    out.extend(aas.iter().map(|&aa| upper_aa(aa)));
}

#[inline]
const fn match_marker(mat: &[[i8; 22]; 22], nt_aa: u8, aa_aa: u8) -> char {
    if nt_aa == aa_aa {
        '|'
    } else if mat[nt_aa as usize][aa_aa as usize] > 0 {
        '+'
    } else {
        ' '
    }
}

#[inline]
fn push_optional_attr(attrs: &mut Vec<(&'static str, GffValue)>, tag: &'static str, value: i32) {
    if value > 0 {
        attrs.push((tag, GffValue::from(value.to_string())));
    }
}

#[inline]
fn ratio_str(num: i32, den: i32) -> String {
    format!("{:.4}", num as f64 / den as f64)
}

#[inline]
fn dinuc_str(dinuc: &[u8; 2]) -> String {
    format!("{}{}", dinuc[0] as char, dinuc[1] as char)
}

#[inline]
fn push_padded3(out: &mut String, head: char, pad: char) {
    out.push(head);
    out.push(pad);
    out.push(pad);
}

#[inline]
fn push_nt_triplet(out: &mut String, nt: &[u8], offset: usize) {
    out.push(NT_I2C[nt[offset] as usize] as char);
    out.push(NT_I2C[nt[offset + 1] as usize] as char);
    out.push(NT_I2C[nt[offset + 2] as usize] as char);
}

fn append_cs(out: &mut String, tables: &Tables, nt: &[u8], aa: &[u8], cigar: &[u32]) {
    out.push_str("cs:Z:");
    let mut nt_offset = 0usize;
    let mut aa_offset = 0usize;
    for &cigar in cigar {
        let Some((op, len)) = unpack_cigar_op(cigar) else {
            continue;
        };
        let len3 = len * 3;
        match op {
            Kind::Match => {
                let mut match_run_len = 0usize;
                for (codon_nt, &aa_byte) in nt[nt_offset..nt_offset + len3]
                    .chunks_exact(3)
                    .zip(&aa[aa_offset..aa_offset + len])
                {
                    let nt_aa = nt_aa(tables, codon_nt, 0);
                    let aa_aa = q_aa(tables, aa_byte);
                    if nt_aa != aa_aa {
                        if match_run_len > 0 {
                            out.push(':');
                            let _ = write!(out, "{match_run_len}");
                        }
                        out.push('*');
                        push_lower_nts(out, codon_nt);
                        out.push(upper_aa(aa_byte));
                        match_run_len = 0;
                    } else {
                        match_run_len += 1;
                    }
                }
                if match_run_len > 0 {
                    out.push(':');
                    let _ = write!(out, "{match_run_len}");
                }
                nt_offset += len3;
                aa_offset += len;
            }
            Kind::Insertion => {
                out.push('+');
                push_upper_aas(out, &aa[aa_offset..aa_offset + len]);
                aa_offset += len;
            }
            Kind::Deletion => {
                out.push('-');
                push_lower_nts(out, &nt[nt_offset..nt_offset + len3]);
                nt_offset += len3;
            }
            Kind::FrameshiftGap => {
                out.push('-');
                push_lower_nts(out, &nt[nt_offset..nt_offset + len]);
                nt_offset += len;
            }
            Kind::FrameshiftMatch => {
                out.push('*');
                push_lower_nts(out, &nt[nt_offset..nt_offset + len]);
                out.push(upper_aa(aa[aa_offset]));
                nt_offset += len;
                aa_offset += 1;
            }
            Kind::Skip | Kind::IntronPhase1 | Kind::IntronPhase2 => {
                let intron = &nt[nt_offset..nt_offset + len];
                let (lshift, rshift) = op.splice_shifts().expect("splice op should have shifts");
                if lshift > 0 {
                    out.push('*');
                    push_lower_nts(out, &intron[..lshift]);
                    out.push(upper_aa(aa[aa_offset]));
                }
                out.push('~');
                out.push(lower_nt(nt[nt_offset + lshift]));
                out.push(lower_nt(nt[nt_offset + lshift + 1]));
                let _ = write!(out, "{}", len - (lshift + rshift));
                out.push(lower_nt(nt[nt_offset + len - rshift - 2]));
                out.push(lower_nt(nt[nt_offset + len - rshift - 1]));
                if rshift > 0 {
                    out.push('-');
                    push_lower_nts(out, &intron[len - rshift..]);
                }
                if lshift > 0 {
                    aa_offset += 1;
                }
                nt_offset += len;
            }
            _ => {}
        }
    }
}

pub(crate) fn build_cs(tables: &Tables, nt: &[u8], aa: &[u8], cigar: &[u32]) -> String {
    let mut out = String::new();
    append_cs(&mut out, tables, nt, aa, cigar);
    out
}

fn write_cs(out: &mut String, mi: &Index, aa: &[u8], reg: &Alignment) {
    let Some(extra) = &reg.extra else {
        return;
    };
    let mut nt = vec![0u8; (reg.ve - reg.vs) as usize];
    let Ok(l_nt) = mi.nt.get_by_v(reg.vid, reg.vs, reg.ve, &mut nt) else {
        return;
    };
    if l_nt != reg.ve - reg.vs {
        return;
    }
    append_cs(out, &mi.tables, &nt, aa, &extra.cigar);
}

fn write_residue(out: &mut String, mi: &Index, opt: &MapOptions, aa: &[u8], reg: &Alignment) {
    let Some(extra) = &reg.extra else {
        return;
    };
    let max_flank = opt.max_intron_flank as usize;
    let mut nt = vec![0u8; (reg.ve - reg.vs + 3) as usize];
    let Ok(l_nt) = mi.nt.get_by_v(reg.vid, reg.vs, reg.ve + 3, &mut nt) else {
        return;
    };
    let nt_len = l_nt as usize;

    let mut atn = String::from("##ATN\t");
    let mut ata = String::from("##ATA\t");
    let mut aas = String::from("##AAS\t");
    let mut aqa = String::from("##AQA\t");
    let mut sta = String::from("##STA\t");
    let mut nt_offset = 0usize;
    let mut aa_offset = reg.qs as usize;

    for &cigar in &extra.cigar {
        let Some((op, len)) = unpack_cigar_op(cigar) else {
            continue;
        };
        let len3 = len * 3;
        match op {
            Kind::Match => {
                for l in 0..len {
                    let nt_idx = nt_offset + l * 3;
                    let aa_idx = aa_offset + l;
                    let nt_aa = nt_aa(&mi.tables, &nt, nt_idx);
                    let aa_aa = q_aa(&mi.tables, aa[aa_idx]);
                    push_nt_triplet(&mut atn, &nt, nt_idx);
                    let nt_char = AA_I2C[nt_aa as usize] as char;
                    push_padded3(&mut ata, nt_char, '.');
                    sta.push(nt_char);
                    push_padded3(&mut aas, match_marker(&opt.mat, nt_aa, aa_aa), ' ');
                    push_padded3(&mut aqa, upper_aa(aa[aa_idx]), ' ');
                }
                nt_offset += len3;
                aa_offset += len;
            }
            Kind::Insertion => {
                for j in 0..len {
                    push_padded3(&mut atn, '-', '-');
                    push_padded3(&mut ata, '-', '.');
                    push_padded3(&mut aas, ' ', ' ');
                    push_padded3(&mut aqa, upper_aa(aa[aa_offset + j]), ' ');
                }
                aa_offset += len;
            }
            Kind::Deletion => {
                for l in 0..len {
                    let nt_idx = nt_offset + l * 3;
                    let nt_aa = nt_aa(&mi.tables, &nt, nt_idx);
                    let nt_char = AA_I2C[nt_aa as usize] as char;
                    push_nt_triplet(&mut atn, &nt, nt_idx);
                    push_padded3(&mut ata, nt_char, '.');
                    sta.push(nt_char);
                    push_padded3(&mut aas, ' ', ' ');
                    push_padded3(&mut aqa, '-', ' ');
                }
                nt_offset += len3;
            }
            Kind::FrameshiftGap => {
                for l in 0..len {
                    atn.push(NT_I2C[nt[nt_offset + l] as usize] as char);
                    ata.push('!');
                    aas.push(' ');
                    aqa.push(' ');
                }
                nt_offset += len;
            }
            Kind::FrameshiftMatch => {
                for l in 0..len {
                    atn.push(NT_I2C[nt[nt_offset + l] as usize] as char);
                    ata.push('$');
                    aas.push(' ');
                    aqa.push(if l == 0 { upper_aa(aa[aa_offset]) } else { ' ' });
                }
                nt_offset += len;
                aa_offset += 1;
            }
            Kind::Skip | Kind::IntronPhase1 | Kind::IntronPhase2 => {
                let intron_len = if op == Kind::Skip { len } else { len - 3 };
                if matches!(op, Kind::IntronPhase1 | Kind::IntronPhase2) {
                    let mut codon = [0u8; 3];
                    codon[0] = nt[nt_offset];
                    if op == Kind::IntronPhase1 {
                        codon[1] = nt[nt_offset + len - 2];
                        codon[2] = nt[nt_offset + len - 1];
                    } else {
                        codon[1] = nt[nt_offset + 1];
                        codon[2] = nt[nt_offset + len - 1];
                    }
                    let nt_aa = nt_aa(&mi.tables, &codon, 0);
                    let aa_aa = q_aa(&mi.tables, aa[aa_offset]);
                    let nt_char = AA_I2C[nt_aa as usize] as char;
                    atn.push(NT_I2C[nt[nt_offset] as usize] as char);
                    ata.push(nt_char);
                    aas.push(match_marker(&opt.mat, nt_aa, aa_aa));
                    aqa.push(upper_aa(aa[aa_offset]));
                    sta.push(nt_char);
                    nt_offset += 1;
                    if op == Kind::IntronPhase2 {
                        atn.push(NT_I2C[nt[nt_offset] as usize] as char);
                        ata.push('.');
                        aas.push(' ');
                        aqa.push(' ');
                        nt_offset += 1;
                    }
                    aa_offset += 1;
                }
                if intron_len <= max_flank * 2 {
                    for i in 0..intron_len {
                        atn.push(lower_nt(nt[nt_offset + i]));
                        ata.push(' ');
                        aas.push(' ');
                        aqa.push(' ');
                    }
                } else {
                    for i in 0..max_flank {
                        atn.push(lower_nt(nt[nt_offset + i]));
                        ata.push(' ');
                        aas.push(' ');
                        aqa.push(' ');
                    }
                    atn.push('~');
                    ata.push(' ');
                    aas.push(' ');
                    aqa.push(' ');
                    let s = intron_len.to_string();
                    atn.push_str(&s);
                    let spaces = " ".repeat(s.len());
                    ata.push_str(&spaces);
                    aas.push_str(&spaces);
                    aqa.push_str(&spaces);
                    atn.push('~');
                    ata.push(' ');
                    aas.push(' ');
                    aqa.push(' ');
                    for i in 0..max_flank {
                        atn.push(lower_nt(nt[nt_offset + intron_len - max_flank + i]));
                        ata.push(' ');
                        aas.push(' ');
                        aqa.push(' ');
                    }
                }
                nt_offset += intron_len;
                if matches!(op, Kind::IntronPhase1 | Kind::IntronPhase2) {
                    atn.push(NT_I2C[nt[nt_offset] as usize] as char);
                    ata.push('.');
                    aas.push(' ');
                    aqa.push(' ');
                    nt_offset += 1;
                    if op == Kind::IntronPhase1 {
                        atn.push(NT_I2C[nt[nt_offset] as usize] as char);
                        ata.push('.');
                        aas.push(' ');
                        aqa.push(' ');
                        nt_offset += 1;
                    }
                }
            }
            _ => {}
        }
    }

    if nt_len == (reg.ve - reg.vs + 3) as usize && !sta.ends_with('*') {
        let nt_aa = nt_aa(&mi.tables, &nt, nt_offset);
        let nt_char = AA_I2C[nt_aa as usize] as char;
        push_nt_triplet(&mut atn, &nt, nt_offset);
        push_padded3(&mut ata, nt_char, '.');
        sta.push(nt_char);
        push_padded3(&mut aas, ' ', ' ');
        push_padded3(&mut aqa, ' ', ' ');
    }

    if (opt.flag & MP_F_SHOW_RESIDUE) != 0 {
        out.push_str(&atn);
        out.push('\n');
        out.push_str(&ata);
        out.push('\n');
        out.push_str(&aas);
        out.push('\n');
        out.push_str(&aqa);
        out.push('\n');
    }
    if (opt.flag & MP_F_SHOW_TRANS) != 0 {
        out.push_str(&sta);
        out.push('\n');
    }
}

fn write_paf(
    out: &mut String,
    mi: &Index,
    opt: &MapOptions,
    query: &QueryRecord,
    reg: Option<&Alignment>,
) {
    if (opt.flag & (MP_F_GFF | MP_F_GTF)) != 0 {
        out.push_str("##PAF\t");
    }
    let Some(reg) = reg else {
        let _ = writeln!(
            out,
            "{}\t{}\t0\t0\t*\t*\t0\t0\t0\t0\t0\t0",
            query.name,
            query.seq.len()
        );
        return;
    };

    let ctg = &mi.nt.contigs[reg.vid.contig().index()];
    let (ts, te) = project_range(reg.vid, ctg.len, reg.vs, reg.ve);
    let _ = write!(
        out,
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t",
        query.name,
        query.seq.len(),
        reg.qs,
        reg.qe,
        strand_char(reg.vid),
        ctg.name,
        ctg.len,
        ts,
        te
    );
    if let Some(extra) = &reg.extra {
        let _ = write!(
            out,
            "{}\t{}\t0\tAS:i:{}\tms:i:{}\tnp:i:{}\tfs:i:{}\tst:i:{}\tda:i:{}\tdo:i:{}\t",
            extra.n_iden * 3,
            extra.blen,
            extra.dp_score,
            extra.dp_max,
            extra.n_plus,
            extra.n_fs,
            extra.n_stop,
            extra.dist_start,
            extra.dist_stop
        );
        out.push_str("cg:Z:");
        for &cigar in &extra.cigar {
            if let Some((op, len)) = unpack_cigar_op(cigar) {
                let _ = write!(out, "{}{}", len, op.symbol());
            }
        }
    } else {
        let _ = write!(out, "{}\t{}\t{}", reg.chn_sc, reg.chn_sc_ungap, reg.cnt);
    }
    if (opt.flag & MP_F_NO_CS) == 0 {
        out.push('\t');
        if let Some(extra) = &reg.extra {
            if extra.cs.is_empty() {
                write_cs(out, mi, &query.seq.as_bytes()[reg.qs as usize..], reg);
            } else {
                out.push_str(&extra.cs);
            }
        }
    }
    out.push('\n');
}

#[inline]
fn gff_strand(vid: VirtualId) -> GffStrand {
    if vid.is_rev() {
        GffStrand::Reverse
    } else {
        GffStrand::Forward
    }
}

#[inline]
fn gff_phase(phase: i16) -> Option<GffPhase> {
    match phase {
        0 => Some(GffPhase::Zero),
        1 => Some(GffPhase::One),
        2 => Some(GffPhase::Two),
        _ => None,
    }
}

#[inline]
fn gff_position(n: i64) -> Position {
    usize::try_from(n)
        .ok()
        .and_then(|m| Position::try_from(m).ok())
        .expect("valid 1-based position")
}

#[inline]
fn score(score: i32) -> f32 {
    score as f32
}

fn gff_attributes<I, K, V>(fields: I) -> GffAttributes
where
    I: IntoIterator<Item = (K, V)>,
    K: Into<GffTag>,
    V: Into<GffValue>,
{
    let mut attributes = GffAttributes::default();
    attributes.as_mut().extend(
        fields
            .into_iter()
            .map(|(tag, value)| (tag.into(), value.into())),
    );
    attributes
}

thread_local! {
    static REC_BUF: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn push_gff_record(out: &mut String, record: &GffRecordBuf) {
    REC_BUF.with_borrow_mut(|buf| {
        buf.clear();
        let mut w = gff::io::Writer::new(std::mem::take(buf));
        w.write_record(record).expect("write GFF record");
        *buf = w.into_inner();
        out.push_str(std::str::from_utf8(buf).expect("UTF-8"));
    });
}

fn push_gtf_record(out: &mut String, record: &GffRecordBuf) {
    REC_BUF.with_borrow_mut(|buf| {
        buf.clear();
        let mut w = gtf::io::Writer::new(std::mem::take(buf));
        w.write_record(record).expect("write GTF record");
        *buf = w.into_inner();
        out.push_str(std::str::from_utf8(buf).expect("UTF-8"));
    });
}

fn write_gff(
    out: &mut String,
    mi: &Index,
    query: &QueryRecord,
    reg: &Alignment,
    opt: &MapOptions,
    id: i64,
    hit_idx: i32,
) {
    let Some(extra) = &reg.extra else {
        return;
    };
    let has_stop = reg.qe == query.seq.len() as i32 && extra.dist_stop == 0;
    let ve_mrna = if has_stop { reg.ve + 3 } else { reg.ve };
    let id_str = if (33..=126).contains(&opt.gff_delim) && hit_idx >= 0 {
        format!("{}{}{}", query.name, opt.gff_delim as u8 as char, hit_idx)
    } else {
        format!("{}{:06}", opt.gff_prefix, id)
    };
    let rank_str = hit_idx.to_string();
    let ctg = &mi.nt.contigs[reg.vid.contig().index()];
    let (vs, ve) = project_range(reg.vid, ctg.len, reg.vs, ve_mrna);
    let mut mrna_attrs = vec![
        (
            gff::feature::record_buf::attributes::field::tag::ID,
            GffValue::from(id_str.as_str()),
        ),
        ("Rank", GffValue::from(rank_str.as_str())),
        (
            "Identity",
            GffValue::from(ratio_str(extra.n_iden * 3, extra.blen)),
        ),
        (
            "Positive",
            GffValue::from(ratio_str(extra.n_plus * 3, extra.blen)),
        ),
    ];
    push_optional_attr(&mut mrna_attrs, "Frameshift", extra.n_fs);
    push_optional_attr(&mut mrna_attrs, "StopCodon", extra.n_stop);
    mrna_attrs.push((
        gff::feature::record_buf::attributes::field::tag::TARGET,
        GffValue::from(format!("{} {} {}", query.name, reg.qs + 1, reg.qe)),
    ));
    push_gff_record(
        out,
        &GffRecordBuf::builder()
            .set_reference_sequence_name(ctg.name.as_str())
            .set_source("miniprot")
            .set_type("mRNA")
            .set_start(gff_position(vs + 1))
            .set_end(gff_position(ve))
            .set_score(score(extra.dp_max))
            .set_strand(gff_strand(reg.vid))
            .set_attributes(gff_attributes(mrna_attrs))
            .build(),
    );

    for (j, feat) in reg.feat.iter().enumerate() {
        let mut feat_ve = feat.ve;
        if has_stop
            && feat.feature_type == FeatureType::Cds
            && j + 1 < reg.feat.len()
            && reg.feat[j + 1].feature_type == FeatureType::Stop
        {
            feat_ve += 3;
        }
        let (vs, ve) = project_range(reg.vid, ctg.len, feat.vs, feat_ve);
        let mut attrs = vec![
            (
                gff::feature::record_buf::attributes::field::tag::PARENT,
                GffValue::from(id_str.as_str()),
            ),
            ("Rank", GffValue::from(rank_str.as_str())),
        ];
        if feat.feature_type == FeatureType::Cds {
            attrs.push((
                "Identity",
                GffValue::from(ratio_str(feat.n_iden * 3, feat.blen)),
            ));
            if feat.acceptor[0] != 0 && &feat.acceptor != b"AG" {
                attrs.push(("Acceptor", GffValue::from(dinuc_str(&feat.acceptor))));
            }
            if feat.donor[0] != 0 && &feat.donor != b"GT" {
                attrs.push(("Donor", GffValue::from(dinuc_str(&feat.donor))));
            }
            push_optional_attr(&mut attrs, "Frameshift", feat.n_fs);
            push_optional_attr(&mut attrs, "StopCodon", feat.n_stop);
            attrs.push((
                gff::feature::record_buf::attributes::field::tag::TARGET,
                GffValue::from(format!("{} {} {}", query.name, feat.qs + 1, feat.qe)),
            ));
        }
        let mut builder = GffRecordBuf::builder()
            .set_reference_sequence_name(ctg.name.as_str())
            .set_source("miniprot")
            .set_type(if feat.feature_type == FeatureType::Stop {
                "stop_codon"
            } else {
                "CDS"
            })
            .set_start(gff_position(vs + 1))
            .set_end(gff_position(ve))
            .set_score(score(feat.score))
            .set_strand(gff_strand(reg.vid))
            .set_attributes(gff_attributes(attrs));
        if let Some(phase) = gff_phase(feat.phase) {
            builder = builder.set_phase(phase);
        }
        push_gff_record(out, &builder.build());
    }
}

fn write_gtf(
    out: &mut String,
    mi: &Index,
    query: &QueryRecord,
    reg: &Alignment,
    opt: &MapOptions,
    id: i64,
) {
    let Some(extra) = &reg.extra else {
        return;
    };
    let has_stop = reg.qe == query.seq.len() as i32 && extra.dist_stop == 0;
    let ve_mrna = if has_stop { reg.ve + 3 } else { reg.ve };
    let id_g = format!("{}G{:06}", opt.gff_prefix, id);
    let id_t = format!("{}T{:06}", opt.gff_prefix, id);
    let ctg = &mi.nt.contigs[reg.vid.contig().index()];
    let (vs, ve) = project_range(reg.vid, ctg.len, reg.vs, ve_mrna);
    push_gtf_record(
        out,
        &GffRecordBuf::builder()
            .set_reference_sequence_name(ctg.name.as_str())
            .set_source("miniprot")
            .set_type("gene")
            .set_start(gff_position(vs + 1))
            .set_end(gff_position(ve))
            .set_score(score(extra.dp_max))
            .set_strand(gff_strand(reg.vid))
            .set_attributes(gff_attributes([("gene_id", id_g.as_str())]))
            .build(),
    );
    push_gtf_record(
        out,
        &GffRecordBuf::builder()
            .set_reference_sequence_name(ctg.name.as_str())
            .set_source("miniprot")
            .set_type("transcript")
            .set_start(gff_position(vs + 1))
            .set_end(gff_position(ve))
            .set_score(score(extra.dp_max))
            .set_strand(gff_strand(reg.vid))
            .set_attributes(gff_attributes([
                ("transcript_id", id_t.as_str()),
                ("gene_id", id_g.as_str()),
            ]))
            .build(),
    );
    for feat in &reg.feat {
        if feat.feature_type != FeatureType::Cds {
            continue;
        }
        let (mut vs2, mut ve2) = project_range(reg.vid, ctg.len, feat.vs, feat.ve);
        let (vs, ve) = (vs2, ve2);
        if feat.ve == reg.ve {
            if reg.vid.is_rev() {
                vs2 = ctg.len - ve_mrna;
            } else {
                ve2 = ve_mrna;
            }
        }
        push_gtf_record(
            out,
            &GffRecordBuf::builder()
                .set_reference_sequence_name(ctg.name.as_str())
                .set_source("miniprot")
                .set_type("exon")
                .set_start(gff_position(vs2 + 1))
                .set_end(gff_position(ve2))
                .set_score(score(feat.score))
                .set_strand(gff_strand(reg.vid))
                .set_attributes(gff_attributes([
                    ("transcript_id", id_t.as_str()),
                    ("gene_id", id_g.as_str()),
                ]))
                .build(),
        );
        let mut builder = GffRecordBuf::builder()
            .set_reference_sequence_name(ctg.name.as_str())
            .set_source("miniprot")
            .set_type("CDS")
            .set_start(gff_position(vs + 1))
            .set_end(gff_position(ve))
            .set_score(score(feat.score))
            .set_strand(gff_strand(reg.vid))
            .set_attributes(gff_attributes([
                ("transcript_id", id_t.as_str()),
                ("gene_id", id_g.as_str()),
            ]));
        if let Some(phase) = gff_phase(feat.phase) {
            builder = builder.set_phase(phase);
        }
        push_gtf_record(out, &builder.build());
    }
}

pub fn write_output(
    out: &mut String,
    mi: &Index,
    query: &QueryRecord,
    reg: Option<&Alignment>,
    opt: &MapOptions,
    id: i64,
    hit_idx: i32,
) {
    let Some(reg) = reg else {
        if (opt.flag & MP_F_SHOW_UNMAP) != 0 {
            write_paf(out, mi, opt, query, None);
        }
        return;
    };
    let show_residue = (opt.flag & (MP_F_SHOW_RESIDUE | MP_F_SHOW_TRANS)) != 0;
    if (opt.flag & MP_F_GTF) != 0 {
        if show_residue {
            write_paf(out, mi, opt, query, Some(reg));
            write_residue(out, mi, opt, query.seq.as_bytes(), reg);
        }
        write_gtf(out, mi, query, reg, opt, id);
        return;
    }
    if (opt.flag & MP_F_NO_PAF) == 0 {
        write_paf(out, mi, opt, query, Some(reg));
    }
    if show_residue {
        write_residue(out, mi, opt, query.seq.as_bytes(), reg);
    }
    if (opt.flag & MP_F_GFF) != 0 {
        write_gff(out, mi, query, reg, opt, id, hit_idx);
    }
}
