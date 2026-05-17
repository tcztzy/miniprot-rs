mod align;
mod chain;
mod cli;
mod cuda_dp;
mod fastx;
mod format;
mod hit;
mod index;
mod io;
mod map;
mod metal_dp;
mod neon_dp;
#[cfg_attr(any(target_arch = "aarch64", target_arch = "x86_64"), allow(dead_code))]
mod scalar_dp;
mod seqdb;
mod sse_dp;
mod wgpu_dp;

mod sketch;
mod sort;
mod tables;
mod types;

pub use cli::run_cli;
pub use fastx::QueryRecord;
pub use index::Index;
pub use map::{map_file, map_file_threads, map_protein};
pub use seqdb::{Contig, NtDb};
pub use tables::{Tables, make_tables};
pub use types::{
    Alignment, AlignmentExtra, ContigId, Feature, FeatureType, IndexOptions, MP_BITS_PER_AA,
    MP_BLOCK_BONUS, MP_CODON_STD, MP_F_GFF, MP_F_GTF, MP_F_NO_ALIGN, MP_F_NO_CS, MP_F_NO_PAF,
    MP_F_NO_PRE_CHAIN, MP_F_NO_SPLICE, MP_F_SHOW_RESIDUE, MP_F_SHOW_TRANS, MP_F_SHOW_UNMAP,
    MP_IDX_MAGIC, MP_VERSION, MapOptions, VirtualId,
};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    InvalidArgument(String),
    InvalidData(String),
    InvalidTranslationTable(i32),
    Unsupported(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(err) => err.fmt(f),
            Self::InvalidArgument(msg) | Self::InvalidData(msg) | Self::Unsupported(msg) => {
                f.write_str(msg)
            }
            Self::InvalidTranslationTable(code) => {
                write!(f, "failed to find translation table {code}")
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

#[cfg(test)]
mod gpu_bench;
#[cfg(test)]
mod tests;
