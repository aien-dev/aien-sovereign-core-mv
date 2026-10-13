//! `aien-calc`: command line front end. Exit codes: 0 answered, 2 malformed request, 3 refused.

use aien_calc::sweep::sweep;
use aien_calc::{eval, CalcError};
use clap::{Parser, Subcommand};
use std::process::ExitCode;

#[derive(Parser)]
#[command(
    name = "aien-calc",
    version,
    about = "Exact-arithmetic calculator with receipts"
)]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Evaluate one expression exactly.
    Eval {
        /// The expression, e.g. '2^-126' or '0x7f & 7'.
        #[arg(allow_hyphen_values = true)]
        expr: String,
        /// Significant digits for the decimal rendering.
        #[arg(long, default_value_t = 20)]
        precision: u32,
        /// Print the receipt itself (canonical JSON) instead of the human text.
        #[arg(long)]
        json: bool,
    },
    /// Evaluate expressions for every code of a small format (at most 16 bits).
    Sweep {
        /// Format: e4m3, e5m2, f16, bf16, int8, int4, uint8, uint4.
        format: String,
        /// Expected value, in terms of `$b` (the code) and `$x` (its exact value).
        #[arg(long, allow_hyphen_values = true)]
        expect: String,
        /// Value actually produced by the construction under test.
        #[arg(long, allow_hyphen_values = true)]
        got: String,
        /// Optional true/false expression; codes where it is true are listed.
        #[arg(long, allow_hyphen_values = true)]
        flag: Option<String>,
        /// Optional true/false filter; only codes where it is true are compared.
        #[arg(long = "where", allow_hyphen_values = true)]
        where_: Option<String>,
        /// Print the receipt itself (canonical JSON) instead of the human text.
        #[arg(long)]
        json: bool,
    },
}

fn emit(json: bool, text: &str, receipt: &aien_calc::Receipt) {
    if json {
        println!("{}", String::from_utf8_lossy(&receipt.canonical_bytes()));
    } else {
        println!("{text}");
    }
}

fn fail(e: CalcError) -> ExitCode {
    eprintln!("aien-calc: {e}");
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
        Cmd::Eval {
            expr,
            precision,
            json,
        } => match eval(&expr, precision) {
            Ok(out) => {
                emit(json, out.text(), out.receipt());
                if out.refusal().is_some() {
                    ExitCode::from(3)
                } else {
                    ExitCode::SUCCESS
                }
            }
            Err(e) => fail(e),
        },
        Cmd::Sweep {
            format,
            expect,
            got,
            flag,
            where_,
            json,
        } => match sweep(&format, &expect, &got, flag.as_deref(), where_.as_deref()) {
            Ok(out) => {
                emit(json, out.text(), out.receipt());
                ExitCode::SUCCESS
            }
            Err(e) => fail(e),
        },
    }
}
