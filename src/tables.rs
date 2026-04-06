use std::collections::HashMap;
use std::sync::LazyLock;

use noodles::sam::alignment::record::cigar::op::Kind as SamKind;

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kind {
    Match,
    Insertion,
    Deletion,
    Skip,
    SoftClip,
    HardClip,
    Pad,
    SequenceMatch,
    SequenceMismatch,
    Back,
    FrameshiftGap,
    FrameshiftMatch,
    IntronPhase1,
    IntronPhase2,
}

const KIND_BY_CODE: [Option<Kind>; 14] = [
    Some(Kind::Match),
    Some(Kind::Insertion),
    Some(Kind::Deletion),
    Some(Kind::Skip),
    Some(Kind::SoftClip),
    Some(Kind::HardClip),
    Some(Kind::Pad),
    Some(Kind::SequenceMatch),
    Some(Kind::SequenceMismatch),
    None,
    Some(Kind::FrameshiftGap),
    Some(Kind::FrameshiftMatch),
    Some(Kind::IntronPhase1),
    Some(Kind::IntronPhase2),
];

impl TryFrom<SamKind> for Kind {
    type Error = ();

    fn try_from(kind: SamKind) -> Result<Self, Self::Error> {
        #[allow(unreachable_patterns)]
        match kind {
            SamKind::Match => Ok(Self::Match),
            SamKind::Insertion => Ok(Self::Insertion),
            SamKind::Deletion => Ok(Self::Deletion),
            SamKind::Skip => Ok(Self::Skip),
            SamKind::SoftClip => Ok(Self::SoftClip),
            SamKind::HardClip => Ok(Self::HardClip),
            SamKind::Pad => Ok(Self::Pad),
            SamKind::SequenceMatch => Ok(Self::SequenceMatch),
            SamKind::SequenceMismatch => Ok(Self::SequenceMismatch),
            _ => Err(()),
        }
    }
}

impl TryFrom<Kind> for SamKind {
    type Error = ();

    fn try_from(kind: Kind) -> Result<Self, Self::Error> {
        kind.as_sam_kind().ok_or(())
    }
}

impl Kind {
    pub const fn as_sam_kind(self) -> Option<SamKind> {
        match self {
            Self::Match => Some(SamKind::Match),
            Self::Insertion => Some(SamKind::Insertion),
            Self::Deletion => Some(SamKind::Deletion),
            Self::Skip => Some(SamKind::Skip),
            Self::SoftClip => Some(SamKind::SoftClip),
            Self::HardClip => Some(SamKind::HardClip),
            Self::Pad => Some(SamKind::Pad),
            Self::SequenceMatch => Some(SamKind::SequenceMatch),
            Self::SequenceMismatch => Some(SamKind::SequenceMismatch),
            Self::Back
            | Self::FrameshiftGap
            | Self::FrameshiftMatch
            | Self::IntronPhase1
            | Self::IntronPhase2 => None,
        }
    }

    pub const fn code(self) -> u8 {
        self as u8
    }

    pub const fn symbol(self) -> char {
        match self {
            Self::Match => 'M',
            Self::Insertion => 'I',
            Self::Deletion => 'D',
            Self::Skip => 'N',
            Self::SoftClip => 'S',
            Self::HardClip => 'H',
            Self::Pad => 'P',
            Self::SequenceMatch => '=',
            Self::SequenceMismatch => 'X',
            Self::Back => 'B',
            Self::FrameshiftGap => 'F',
            Self::FrameshiftMatch => 'G',
            Self::IntronPhase1 => 'U',
            Self::IntronPhase2 => 'V',
        }
    }

    pub const fn from_code(code: u8) -> Option<Self> {
        let idx = code as usize;
        if idx < KIND_BY_CODE.len() {
            KIND_BY_CODE[idx]
        } else {
            None
        }
    }

    pub const fn splice_shifts(self) -> Option<(usize, usize)> {
        match self {
            Self::Skip => Some((0, 0)),
            Self::IntronPhase1 => Some((1, 2)),
            Self::IntronPhase2 => Some((2, 1)),
            _ => None,
        }
    }
}

pub fn pack_cigar_op(kind: Kind, len: usize) -> u32 {
    ((len as u32) << 4) | kind.code() as u32
}

pub fn unpack_cigar_op(raw: u32) -> Option<(Kind, usize)> {
    Kind::from_code((raw & 0x0f) as u8).map(|kind| (kind, (raw >> 4) as usize))
}

pub const NS_F_CIGAR: i32 = 0x1;
pub const NS_F_EXT_LEFT: i32 = 0x2;
pub const NS_F_EXT_RIGHT: i32 = 0x4;

pub const NS_S_NONE: i32 = 0;
pub const NS_S_GENERIC: i32 = 1;
pub const NS_S_MAMMAL: i32 = 2;

pub const NS_SPSC_OFFSET: u8 = 64;

pub const NT_I2C: &[u8; 5] = b"ACGTN";
pub const AA_I2C: &[u8; 22] = b"ARNDCQEGHILKMFPSTWYV*X";

pub const AA2R: [u8; 22] = [
    0, 2, 4, 4, 6, 5, 5, 8, 3, 10, 11, 2, 11, 12, 7, 1, 1, 13, 12, 10, 14, 15,
];

macro_rules! blosum62 {
    ($($table:tt)*) => {
        parse_blosum62(stringify!($($table)*))
    };
}

const DROP: usize = usize::MAX;
const BLOSUM62_INDEX_MAP: [usize; 25] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, DROP, DROP, DROP, 21, 20,
];

const fn skip_ws(bytes: &[u8], mut i: usize) -> usize {
    while i < bytes.len() && matches!(bytes[i], b' ' | b'\n' | b'\r' | b'\t') {
        i += 1;
    }
    i
}

const fn parse_symbol(bytes: &[u8], i: usize) -> (u8, usize) {
    let i = skip_ws(bytes, i);
    (bytes[i], i + 1)
}

const fn parse_i8(bytes: &[u8], i: usize) -> (i8, usize) {
    let mut i = skip_ws(bytes, i);

    let mut sign = 1i16;
    if bytes[i] == b'-' {
        sign = -1;
        i += 1;
    }

    let mut value = 0i16;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        value = value * 10 + (bytes[i] - b'0') as i16;
        i += 1;
    }

    ((sign * value) as i8, i)
}

const fn parse_blosum62(table: &str) -> [[i8; 22]; 22] {
    let bytes = table.as_bytes();
    let mut pos = 0;
    let mut col = 0;
    while col < BLOSUM62_INDEX_MAP.len() {
        let (_, next) = parse_symbol(bytes, pos);
        pos = next;
        col += 1;
    }

    let mut out = [[0i8; 22]; 22];
    let mut row = 0;
    while row < BLOSUM62_INDEX_MAP.len() {
        let (_, next) = parse_symbol(bytes, pos);
        pos = next;

        let out_row = BLOSUM62_INDEX_MAP[row];
        let mut col = 0;
        while col < BLOSUM62_INDEX_MAP.len() {
            let (score, next) = parse_i8(bytes, pos);
            pos = next;

            let out_col = BLOSUM62_INDEX_MAP[col];
            if out_row != DROP && out_col != DROP {
                out[out_row][out_col] = score;
            }
            col += 1;
        }

        row += 1;
    }

    out
}

/// Entries for the BLOSUM62 matrix at a scale of ln(2)/2.0.
/// The source table is kept verbatim in NCBI's 25-symbol order. This crate only
/// uses the 22 symbols in AA_I2C, so the macro drops B/J/Z and places `*` before `X`.
pub const BLOSUM62: [[i8; 22]; 22] = blosum62! {
       A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V  B  J  Z  X  *
    A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0 -2 -1 -1 -1 -4
    R -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3 -1 -2  0 -1 -4
    N -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3  4 -3  0 -1 -4
    D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3  4 -3  1 -1 -4
    C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1 -3 -1 -3 -1 -4
    Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2  0 -2  4 -1 -4
    E -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2  1 -3  4 -1 -4
    G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3 -1 -4 -2 -1 -4
    H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3  0 -3  0 -1 -4
    I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3 -3  3 -3 -1 -4
    L -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1 -4  3 -3 -1 -4
    K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2  0 -3  1 -1 -4
    M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1 -3  2 -1 -1 -4
    F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1 -3  0 -3 -1 -4
    P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2 -2 -3 -1 -1 -4
    S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2  0 -2  0 -1 -4
    T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0 -1 -1 -1 -1 -4
    W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3 -4 -2 -2 -1 -4
    Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1 -3 -1 -2 -1 -4
    V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4 -3  2 -2 -1 -4
    B -2 -1  4  4 -3  0  1 -1  0 -3 -4  0 -3 -3 -2  0 -1 -4 -3 -3  4 -3  0 -1 -4
    J -1 -2 -3 -3 -1 -2 -3 -4 -3  3  3 -3  2  0 -3 -2 -1 -2 -1  2 -3  3 -3 -1 -4
    Z -1  0  0  1 -3  4  4 -2  0 -3 -3  1 -1 -3 -1  0 -1 -2 -2 -2  0 -3  4 -1 -4
    X -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -4
    * -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4  1
};

#[derive(Clone, Debug)]
pub struct Tables {
    pub nt4: [u8; 256],
    pub aa20: [u8; 256],
    pub aa13: [u8; 256],
    pub codon: [u8; 64],
    pub codon13: [u8; 64],
}

const fn build_tcag_index_by_acgt() -> [usize; 64] {
    let source_digit_by_internal = [2usize, 1, 3, 0];
    let mut map = [0usize; 64];
    let mut i = 0;
    while i < map.len() {
        let b1 = source_digit_by_internal[(i >> 4) & 3];
        let b2 = source_digit_by_internal[(i >> 2) & 3];
        let b3 = source_digit_by_internal[i & 3];
        map[i] = (b1 << 4) | (b2 << 2) | b3;
        i += 1;
    }
    map
}

const TCAG_INDEX_BY_ACGT: [usize; 64] = build_tcag_index_by_acgt();

const fn reorder_translation_table(table: &str) -> [u8; 64] {
    let bytes = table.as_bytes();
    let mut reordered = [0u8; 64];
    let mut i = 0;
    while i < reordered.len() {
        reordered[i] = bytes[TCAG_INDEX_BY_ACGT[i]];
        i += 1;
    }
    reordered
}

macro_rules! translation_tables {
    ($($id:literal $table:literal,)+) => {{
        HashMap::from([
            $(($id, reorder_translation_table($table)),)+
        ])
    }};
}

// Translation tables in NCBI's TCAG codon order:
// TTT TTC TTA TTG ... GGG.
static TRANSLATION_TABLES: LazyLock<HashMap<i32, [u8; 64]>> = LazyLock::new(|| {
    translation_tables! {
    1  "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    2  "FFLLSSSSYY**CCWWLLLLPPPPHHQQRRRRIIMMTTTTNNKKSS**VVVVAAAADDEEGGGG",
    3  "FFLLSSSSYY**CCWWTTTTPPPPHHQQRRRRIIMMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    4  "FFLLSSSSYY**CCWWLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    5  "FFLLSSSSYY**CCWWLLLLPPPPHHQQRRRRIIMMTTTTNNKKSSSSVVVVAAAADDEEGGGG",
    6  "FFLLSSSSYYQQCC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    9  "FFLLSSSSYY**CCWWLLLLPPPPHHQQRRRRIIIMTTTTNNNKSSSSVVVVAAAADDEEGGGG",
    10 "FFLLSSSSYY**CCCWLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    11 "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    12 "FFLLSSSSYY**CC*WLLLSPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    13 "FFLLSSSSYY**CCWWLLLLPPPPHHQQRRRRIIMMTTTTNNKKSSGGVVVVAAAADDEEGGGG",
    14 "FFLLSSSSYYY*CCWWLLLLPPPPHHQQRRRRIIIMTTTTNNNKSSSSVVVVAAAADDEEGGGG",
    15 "FFLLSSSSYY*QCC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    16 "FFLLSSSSYY*LCC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    21 "FFLLSSSSYY**CCWWLLLLPPPPHHQQRRRRIIMMTTTTNNNKSSSSVVVVAAAADDEEGGGG",
    22 "FFLLSS*SYY*LCC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    23 "FF*LSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    24 "FFLLSSSSYY**CCWWLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSSKVVVVAAAADDEEGGGG",
    25 "FFLLSSSSYY**CCGWLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    26 "FFLLSSSSYY**CC*WLLLAPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    27 "FFLLSSSSYYQQCCWWLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    28 "FFLLSSSSYYQQCCWWLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    29 "FFLLSSSSYYYYCC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    30 "FFLLSSSSYYEECC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    31 "FFLLSSSSYYEECCWWLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    32 "FFLLSSSSYY*WCC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    33 "FFLLSSSSYYY*CCWWLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSSKVVVVAAAADDEEGGGG",
    }
});

pub fn make_tables(codon_type: i32) -> crate::Result<Tables> {
    let Some(trans_tab) = TRANSLATION_TABLES.get(&codon_type) else {
        return Err(crate::Error::InvalidTranslationTable(codon_type));
    };

    let mut nt4 = [4u8; 256];
    for (idx, byte) in NT_I2C.iter().copied().enumerate() {
        let idx = idx as u8;
        nt4[idx as usize] = idx;
        nt4[byte as usize] = idx;
        nt4[byte.to_ascii_lowercase() as usize] = idx;
    }

    let mut aa20 = [21u8; 256];
    let mut aa13 = [15u8; 256];
    for (idx, byte) in AA_I2C.iter().copied().enumerate() {
        let idx = idx as u8;
        let aa13_code = AA2R[idx as usize];
        aa20[idx as usize] = idx;
        aa20[byte as usize] = idx;
        aa20[byte.to_ascii_lowercase() as usize] = idx;
        aa13[idx as usize] = aa13_code;
        aa13[byte as usize] = aa13_code;
        aa13[byte.to_ascii_lowercase() as usize] = aa13_code;
    }

    let mut codon = [0u8; 64];
    let mut codon13 = [0u8; 64];
    for (i, &byte) in trans_tab.iter().enumerate() {
        codon[i] = aa20[byte as usize];
        codon13[i] = AA2R[codon[i] as usize];
    }

    Ok(Tables {
        nt4,
        aa20,
        aa13,
        codon,
        codon13,
    })
}

pub fn set_stop_sc(asize: i32, mat: &mut [[i8; 22]; 22], pen: i32) {
    let aa_stop = AA_I2C.iter().position(|&b| b == b'*').unwrap();
    let penalty = -pen as i8;
    let score_ori = mat[aa_stop][aa_stop];
    for i in 0..asize {
        let i = i as usize;
        mat[aa_stop][i] = penalty;
        mat[i][aa_stop] = penalty;
    }
    mat[aa_stop][aa_stop] = score_ori;
}

pub fn opt_set_sp(model: i32) -> [i32; 6] {
    match model {
        NS_S_MAMMAL => [8, 15, 21, 30, 4, 4],
        NS_S_GENERIC => [8, 15, 21, 30, 0, 0],
        _ => [0; 6],
    }
}
