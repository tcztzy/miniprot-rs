use crate::tables;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

pub const MP_VERSION: &str = "0.18-r281";

pub const MP_F_NO_SPLICE: u32 = 0x1;
pub const MP_F_NO_ALIGN: u32 = 0x2;
pub const MP_F_SHOW_UNMAP: u32 = 0x4;
pub const MP_F_GFF: u32 = 0x8;
pub const MP_F_NO_PAF: u32 = 0x10;
pub const MP_F_GTF: u32 = 0x20;
pub const MP_F_NO_PRE_CHAIN: u32 = 0x40;
pub const MP_F_SHOW_RESIDUE: u32 = 0x80;
pub const MP_F_SHOW_TRANS: u32 = 0x100;
pub const MP_F_NO_CS: u32 = 0x200;

pub const MP_BITS_PER_AA: usize = 4;
pub const MP_BLOCK_BONUS: i32 = 2;
pub const MP_CODON_STD: u32 = 1;
pub const MP_IDX_MAGIC: &[u8; 4] = b"MPI\x03";
pub const MP_ANCHOR_QUERY_FLAG: u32 = 1 << 31;

/// Extract the lower 32 bits of a packed u64 as i32.
#[inline(always)]
pub(crate) const fn lo32(v: u64) -> i32 {
    v as u32 as i32
}

/// Pack two 32-bit values into a u64 (high << 32 | low).
#[inline(always)]
pub(crate) const fn pack64(hi: i32, lo: i32) -> u64 {
    ((hi as u32 as u64) << 32) | (lo as u32 as u64)
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Anchor(u64);

impl Anchor {
    #[inline(always)]
    #[must_use]
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    #[inline(always)]
    #[must_use]
    pub const fn from_parts(target: i32, query: i32) -> Self {
        Self(pack64(target, query))
    }

    #[inline(always)]
    pub const fn raw(self) -> u64 {
        self.0
    }

    #[inline(always)]
    pub const fn target(self) -> i32 {
        (self.0 >> 32) as i32
    }

    #[inline(always)]
    pub const fn query(self) -> i32 {
        lo32(self.0)
    }

    #[inline(always)]
    pub const fn query_pos(self) -> i32 {
        self.query() & !(MP_ANCHOR_QUERY_FLAG as i32)
    }

    #[inline(always)]
    pub const fn has_query_flag(self) -> bool {
        (self.query() as u32 & MP_ANCHOR_QUERY_FLAG) != 0
    }

    #[inline(always)]
    pub const fn with_target(self, target: i32) -> Self {
        Self::from_parts(target, self.query())
    }

    #[inline(always)]
    pub const fn with_query_flag(self) -> Self {
        Self::from_parts(self.target(), self.query() | MP_ANCHOR_QUERY_FLAG as i32)
    }
}

impl From<u64> for Anchor {
    fn from(value: u64) -> Self {
        Self::from_raw(value)
    }
}

impl From<Anchor> for u64 {
    #[inline]
    fn from(value: Anchor) -> Self {
        value.raw()
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ChainMeta {
    pub score: i32,
    pub len: usize,
}

impl ChainMeta {
    #[inline(always)]
    #[must_use]
    pub const fn new(score: i32, len: usize) -> Self {
        Self { score, len }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Extents {
    pub left: i32,
    pub right: i32,
}

impl Extents {
    #[inline(always)]
    #[must_use]
    pub const fn new(left: i32, right: i32) -> Self {
        Self { left, right }
    }
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ContigId(pub usize);

impl ContigId {
    #[inline(always)]
    #[must_use]
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    #[inline(always)]
    pub const fn index(self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VirtualId {
    contig: ContigId,
    rev: bool,
}

impl VirtualId {
    #[inline(always)]
    #[must_use]
    pub const fn forward(contig: ContigId) -> Self {
        Self { contig, rev: false }
    }

    #[inline(always)]
    #[must_use]
    pub const fn reverse(contig: ContigId) -> Self {
        Self { contig, rev: true }
    }

    #[inline(always)]
    pub const fn contig(self) -> ContigId {
        self.contig
    }

    #[inline(always)]
    pub const fn is_rev(self) -> bool {
        self.rev
    }

    #[inline(always)]
    pub const fn to_index(self) -> usize {
        self.contig.index() * 2 + if self.rev { 1 } else { 0 }
    }

    #[inline(always)]
    pub fn from_index(index: usize) -> Option<Self> {
        let contig = index >> 1;
        if contig > (u32::MAX >> 1) as usize {
            return None;
        }
        let contig = ContigId::new(contig);
        Some(match index & 1 {
            0 => Self::forward(contig),
            _ => Self::reverse(contig),
        })
    }

    #[inline(always)]
    pub fn encode_u32(self) -> u32 {
        (u32::try_from(self.contig.index()).expect("virtual id exceeds u32") << 1)
            | u32::from(self.rev)
    }

    #[inline(always)]
    pub const fn decode_u32(value: u32) -> Self {
        let contig = ContigId::new((value >> 1) as usize);
        match value & 1 {
            0 => Self::forward(contig),
            _ => Self::reverse(contig),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FeatureType {
    #[default]
    Cds = 0,
    Stop = 1,
}

#[derive(Debug, Clone, Default)]
pub struct AlignmentExtra {
    pub dp_score: i32,
    pub dp_max: i32,
    pub dp_max2: i32,
    pub blen: i32,
    pub n_fs: i32,
    pub n_stop: i32,
    pub dist_stop: i32,
    pub dist_start: i32,
    pub n_iden: i32,
    pub n_plus: i32,
    pub cigar: Vec<u32>,
    pub cs: String,
}

#[derive(Debug, Clone, Default)]
pub struct Feature {
    pub vs: i64,
    pub ve: i64,
    pub qs: i32,
    pub qe: i32,
    pub feature_type: FeatureType,
    pub phase: i16,
    pub n_fs: i32,
    pub n_stop: i32,
    pub score: i32,
    pub n_iden: i32,
    pub blen: i32,
    pub donor: [u8; 2],
    pub acceptor: [u8; 2],
}

#[derive(Debug, Clone, Default)]
pub struct Alignment {
    pub cnt: usize,
    pub id: usize,
    pub parent: Option<usize>,
    pub n_sub: usize,
    pub subsc: i32,
    pub n_feat: usize,
    pub n_exon: usize,
    pub chn_sc: i32,
    pub chn_sc_ungap: i32,
    pub hash: u32,
    pub vid: VirtualId,
    pub qs: i32,
    pub qe: i32,
    pub vs: i64,
    pub ve: i64,
    pub anchors: Vec<Anchor>,
    pub feat: Vec<Feature>,
    pub extra: Option<AlignmentExtra>,
}

impl Alignment {
    /// Effective score: dp_max if alignment was computed, otherwise chain score.
    #[inline]
    pub fn score(&self) -> i32 {
        self.extra
            .as_ref()
            .map_or(self.chn_sc, |extra| extra.dp_max)
    }

    /// Invalidate this alignment so it will be filtered out downstream.
    #[inline]
    pub fn invalidate(&mut self) {
        self.cnt = 0;
        self.anchors.clear();
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct IndexOptions {
    pub bbit: i32,
    pub min_aa_len: i32,
    pub kmer: i32,
    pub mod_bit: i32,
    pub trans_code: u32,
}

impl Default for IndexOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl IndexOptions {
    pub fn new() -> Self {
        Self {
            bbit: 8,
            min_aa_len: 30,
            kmer: 6,
            mod_bit: 1,
            trans_code: MP_CODON_STD,
        }
    }

    #[must_use]
    pub fn n_bucket(&self) -> usize {
        1usize << (self.kmer as usize * MP_BITS_PER_AA - self.mod_bit as usize)
    }

    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        IntoBytes::as_bytes(self)
    }

    pub fn from_reader<R: std::io::Read>(reader: &mut R) -> crate::Result<Self> {
        let mut raw = [0u8; std::mem::size_of::<Self>()];
        reader.read_exact(&mut raw)?;
        Self::read_from_bytes(&raw)
            .map_err(|_| crate::Error::InvalidData("invalid index options header".to_owned()))
    }
}

#[derive(Clone, Debug)]
pub struct MapOptions {
    pub flag: u32,
    pub use_gpu: bool,
    pub mini_batch_size: i64,
    pub max_occ: i32,
    pub max_gap: i32,
    pub max_intron: i32,
    pub min_max_intron: i32,
    pub max_max_intron: i32,
    pub bw: i32,
    pub max_ext: i32,
    pub max_ava: i32,
    pub min_chn_cnt: i32,
    pub max_chn_max_skip: i32,
    pub max_chn_iter: i32,
    pub min_chn_sc: i32,
    pub chn_coef_log: f32,
    pub mask_level: f32,
    pub mask_len: i32,
    pub pri_ratio: f32,
    pub out_sim: f32,
    pub out_cov: f32,
    pub best_n: i32,
    pub out_n: i32,
    pub kmer2: i32,
    pub go: i32,
    pub ge: i32,
    pub io: i32,
    pub fs: i32,
    pub io_end: i32,
    pub ie_coef: f32,
    pub sp_model: i32,
    pub sp_null_bonus: i32,
    pub sp_max_bonus: i32,
    pub sp_scale: f32,
    pub xdrop: i32,
    pub end_bonus: i32,
    pub asize: i32,
    pub gff_delim: i32,
    pub max_intron_flank: i32,
    pub gff_prefix: String,
    pub mat: [[i8; 22]; 22],
}

impl Default for MapOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl MapOptions {
    pub fn new() -> Self {
        let mut opt = Self {
            flag: 0,
            use_gpu: false,
            mini_batch_size: 2_000_000,
            max_occ: 20_000,
            max_gap: 1_000,
            max_intron: 200_000,
            min_max_intron: 10_000,
            max_max_intron: 300_000,
            bw: 200_000,
            max_ext: 10_000,
            max_ava: 1_000,
            min_chn_cnt: 3,
            max_chn_max_skip: 25,
            max_chn_iter: 1_000_000,
            min_chn_sc: 0,
            chn_coef_log: 0.75,
            mask_level: 0.5,
            mask_len: i32::MAX,
            pri_ratio: 0.7,
            out_sim: 0.99,
            out_cov: 0.1,
            best_n: 30,
            out_n: 1_000,
            kmer2: 5,
            go: 11,
            ge: 1,
            io: 29,
            fs: 23,
            io_end: 19,
            ie_coef: 0.5,
            sp_model: tables::NS_S_GENERIC,
            sp_null_bonus: -7,
            sp_max_bonus: 14,
            sp_scale: 1.0,
            xdrop: 100,
            end_bonus: 5,
            asize: 22,
            gff_delim: -1,
            max_intron_flank: 200,
            gff_prefix: "MP".to_owned(),
            mat: tables::BLOSUM62,
        };
        tables::set_stop_sc(opt.asize, &mut opt.mat, opt.fs);
        opt
    }

    pub fn set_fs(&mut self, fs: i32) {
        self.fs = fs;
        tables::set_stop_sc(self.asize, &mut self.mat, fs);
    }

    pub fn set_max_intron(&mut self, gsize: i64) {
        let x = ((((gsize as f64).sqrt() * 3.6) + 1.0) as i64)
            .clamp(self.min_max_intron as i64, self.max_max_intron as i64) as i32;
        self.bw = x;
        self.max_intron = x;
    }

    pub fn check(&self) -> crate::Result<()> {
        if !(tables::NS_S_NONE..=tables::NS_S_MAMMAL).contains(&self.sp_model) {
            return Err(crate::Error::InvalidArgument(
                "option -j should be between 0 and 2".to_owned(),
            ));
        }
        Ok(())
    }
}

const _: () = assert!(std::mem::size_of::<IndexOptions>() == 20);
