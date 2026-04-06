use std::ffi::OsString;
use std::process::ExitCode;

use clap::{ArgAction, Parser, error::ErrorKind};

use crate::index::Index;
use crate::map_file;
use crate::types::{
    IndexOptions, MP_F_GFF, MP_F_GTF, MP_F_NO_ALIGN, MP_F_NO_CS, MP_F_NO_PAF, MP_F_NO_PRE_CHAIN,
    MP_F_NO_SPLICE, MP_F_SHOW_RESIDUE, MP_F_SHOW_TRANS, MP_F_SHOW_UNMAP, MP_VERSION, MapOptions,
};

#[derive(Debug, Parser)]
#[command(
    name = "miniprot",
    version = MP_VERSION,
    disable_help_subcommand = true,
    disable_version_flag = false
)]
struct Args {
    #[arg(short = 'k', default_value_t = 6)]
    kmer: i32,
    #[arg(short = 'M', default_value_t = 1)]
    mod_bit: i32,
    #[arg(short = 'L', default_value_t = 30)]
    min_aa_len: i32,
    #[arg(short = 'T', default_value_t = 1)]
    trans_code: u32,
    #[arg(short = 'b', default_value_t = 8)]
    bbit: i32,
    #[arg(short = 'I', action = ArgAction::SetTrue)]
    auto_max_intron: bool,
    #[arg(short = 'd')]
    dump_index: Option<String>,
    #[arg(short = 't', default_value_t = 4)]
    threads: i32,
    #[arg(short = 'A', action = ArgAction::SetTrue)]
    no_align: bool,
    #[arg(short = 'S', action = ArgAction::SetTrue)]
    no_splice: bool,
    #[arg(short = 'u', action = ArgAction::SetTrue)]
    show_unmap: bool,
    #[arg(long = "gff", action = ArgAction::SetTrue)]
    gff: bool,
    #[arg(long = "gff-only", action = ArgAction::SetTrue)]
    gff_only: bool,
    #[arg(long = "gff-delim")]
    gff_delim: Option<String>,
    #[arg(long = "gtf", action = ArgAction::SetTrue)]
    gtf: bool,
    #[arg(long = "aln", action = ArgAction::SetTrue)]
    aln: bool,
    #[arg(long = "trans", action = ArgAction::SetTrue)]
    trans: bool,
    #[arg(long = "no-cs", action = ArgAction::SetTrue)]
    no_cs: bool,
    #[arg(short = 'P', default_value = "MP")]
    gff_prefix: String,
    #[arg(short = 'O')]
    gap_open: Option<i32>,
    #[arg(short = 'E')]
    gap_extend: Option<i32>,
    #[arg(short = 'J')]
    intron_open: Option<i32>,
    #[arg(long = "J2")]
    intron_open_end: Option<i32>,
    #[arg(short = 'F')]
    frameshift: Option<i32>,
    #[arg(short = 'C')]
    sp_scale: Option<f32>,
    #[arg(short = 'B')]
    end_bonus: Option<i32>,
    #[arg(short = 'j')]
    sp_model: Option<i32>,
    #[arg(long = "xdrop")]
    xdrop: Option<i32>,
    #[arg(long = "max-skip")]
    max_skip: Option<i32>,
    #[arg(long = "ie-coef")]
    ie_coef: Option<f32>,
    #[arg(long = "max-intron-out")]
    max_intron_out: Option<i32>,
    #[arg(long = "no-pre-chain", action = ArgAction::SetTrue)]
    no_pre_chain: bool,
    #[arg(long = "spsc")]
    spsc: Option<String>,
    #[arg(long = "spsc0", allow_negative_numbers = true)]
    spsc0: Option<i32>,
    #[arg(long = "spsc-max")]
    spsc_max: Option<i32>,
    #[arg(short = 'c', default_value_t = 20_000)]
    max_occ: i32,
    #[arg(short = 'g', default_value_t = 1_000)]
    max_gap: i32,
    #[arg(short = 'G')]
    max_intron: Option<i32>,
    #[arg(short = 'n', default_value_t = 3)]
    min_chn_cnt: i32,
    #[arg(short = 'm', default_value_t = 0)]
    min_chn_sc: i32,
    #[arg(short = 'l', default_value_t = 5)]
    kmer2: i32,
    #[arg(short = 'e', default_value_t = 10_000)]
    max_ext: i32,
    #[arg(short = 'p', default_value_t = 0.7)]
    pri_ratio: f32,
    #[arg(short = 'N', default_value_t = 30)]
    best_n: i32,
    #[arg(long = "outn", default_value_t = 1000)]
    out_n: i32,
    #[arg(long = "outs", default_value_t = 0.99)]
    out_sim: f32,
    #[arg(long = "outc", default_value_t = 0.1)]
    out_cov: f32,
    #[arg(short = 'w', default_value_t = 0.75)]
    chn_coef_log: f32,
    #[arg(short = 'K', default_value_t = 2_000_000)]
    mini_batch_size: i64,
    #[arg(short = 's', action = ArgAction::SetTrue, hide = true)]
    deprecated_s: bool,
    #[arg(value_name = "REF")]
    reference: Option<String>,
    #[arg(value_name = "QUERY")]
    queries: Vec<String>,
}

pub fn run_cli<I, S>(args: I) -> ExitCode
where
    I: IntoIterator<Item = S>,
    S: Into<OsString>,
{
    match run_cli_inner(args) {
        Ok(code) => code,
        Err(err) => {
            eprintln!("[ERROR]\x1b[1;31m {err}\x1b[0m");
            ExitCode::from(1)
        }
    }
}

#[inline]
fn assign_if_some<T>(slot: &mut T, value: Option<T>) -> bool {
    if let Some(value) = value {
        *slot = value;
        true
    } else {
        false
    }
}

fn run_cli_inner<I, S>(args: I) -> crate::Result<ExitCode>
where
    I: IntoIterator<Item = S>,
    S: Into<OsString>,
{
    let argv: Vec<OsString> = args.into_iter().map(Into::into).collect();
    let args = match Args::try_parse_from(argv) {
        Ok(args) => args,
        Err(err) => match err.kind() {
            ErrorKind::DisplayHelp | ErrorKind::DisplayVersion => {
                err.print().map_err(crate::Error::Io)?;
                return Ok(ExitCode::SUCCESS);
            }
            _ => return Err(crate::Error::InvalidArgument(err.to_string())),
        },
    };

    let io = IndexOptions {
        kmer: args.kmer,
        mod_bit: args.mod_bit,
        min_aa_len: args.min_aa_len,
        trans_code: args.trans_code,
        bbit: args.bbit,
    };

    let mut mo = MapOptions {
        max_occ: args.max_occ,
        max_gap: args.max_gap,
        min_chn_cnt: args.min_chn_cnt,
        min_chn_sc: args.min_chn_sc,
        kmer2: args.kmer2,
        max_ext: args.max_ext,
        pri_ratio: args.pri_ratio,
        best_n: args.best_n,
        out_n: args.out_n,
        out_sim: args.out_sim,
        out_cov: args.out_cov,
        chn_coef_log: args.chn_coef_log,
        mini_batch_size: args.mini_batch_size,
        ..MapOptions::new()
    };
    let mut keep_io = false;
    let explicit_max_intron = args.max_intron.is_some();
    if let Some(max_intron) = args.max_intron {
        mo.max_intron = max_intron;
        mo.bw = max_intron;
    }
    assign_if_some(&mut mo.go, args.gap_open);
    assign_if_some(&mut mo.ge, args.gap_extend);
    keep_io |= assign_if_some(&mut mo.io, args.intron_open);
    keep_io |= assign_if_some(&mut mo.io_end, args.intron_open_end);
    if let Some(fs) = args.frameshift {
        mo.set_fs(fs);
    }
    assign_if_some(&mut mo.sp_scale, args.sp_scale);
    assign_if_some(&mut mo.end_bonus, args.end_bonus);
    assign_if_some(&mut mo.sp_model, args.sp_model);
    assign_if_some(&mut mo.xdrop, args.xdrop);
    assign_if_some(&mut mo.max_chn_max_skip, args.max_skip);
    assign_if_some(&mut mo.ie_coef, args.ie_coef);
    if let Some(max_intron_out) = args.max_intron_out {
        mo.max_intron_flank = (max_intron_out + 1) / 2;
    }
    if let Some(spsc0) = args.spsc0 {
        mo.sp_null_bonus = if spsc0 < 0 { spsc0 } else { -spsc0 };
    }
    assign_if_some(&mut mo.sp_max_bonus, args.spsc_max);
    for (enabled, bit) in [
        (args.no_align, MP_F_NO_ALIGN),
        (args.show_unmap, MP_F_SHOW_UNMAP),
        (args.gff, MP_F_GFF),
        (args.gtf, MP_F_GTF),
        (args.aln, MP_F_SHOW_RESIDUE),
        (args.trans, MP_F_SHOW_TRANS),
        (args.no_cs, MP_F_NO_CS),
        (args.no_pre_chain, MP_F_NO_PRE_CHAIN),
    ] {
        if enabled {
            mo.flag |= bit;
        }
    }
    if args.no_splice {
        mo.flag |= MP_F_NO_SPLICE;
        mo.bw = 1_000;
        mo.max_intron = 1_000;
        mo.max_ext = 1_000;
        mo.io = 10_000;
        mo.io_end = 10_000;
    }
    if args.gff_only {
        mo.flag |= MP_F_GFF | MP_F_NO_PAF;
    }
    if let Some(delim) = args.gff_delim.as_deref() {
        mo.gff_delim = delim.as_bytes().first().copied().unwrap_or_default() as i32;
    }
    mo.gff_prefix = args.gff_prefix;

    if args.deprecated_s {
        eprintln!("Option '-s' is deprecated.");
    }
    mo.check()?;

    let Some(reference) = args.reference.as_deref() else {
        return Err(crate::Error::InvalidArgument(
            "missing reference argument".to_owned(),
        ));
    };
    if args.queries.is_empty() && args.dump_index.is_none() {
        return Err(crate::Error::InvalidArgument(
            "missing query argument".to_owned(),
        ));
    }

    let mut mi = Index::load(reference, &io, args.threads)?;
    if args.auto_max_intron && !explicit_max_intron && !args.no_splice {
        mo.set_max_intron(mi.nt.l_seq);
    }
    if let Some(path) = args.spsc.as_deref() {
        mi.set_spsc_path(path, &mut mo, keep_io)?;
    }
    if let Some(path) = args.dump_index {
        mi.dump(path)?;
    }
    if !args.queries.is_empty() {
        for query in &args.queries {
            print!("{}", map_file(&mi, query, &mo)?);
        }
    }
    Ok(ExitCode::SUCCESS)
}
