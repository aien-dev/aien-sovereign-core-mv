//! `aien-calc`: command line front end. Exit codes: 0 answered, 2 malformed request, 3 refused.

use aien_calc::eval;
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
                if json {
                    println!(
                        "{}",
                        String::from_utf8_lossy(&out.receipt().canonical_bytes())
                    );
                } else {
                    println!("{}", out.text());
                }
                if out.refusal().is_some() {
                    ExitCode::from(3)
                } else {
                    ExitCode::SUCCESS
                }
            }
            Err(e) => {
                eprintln!("aien-calc: {e}");
                ExitCode::from(2)
            }
        },
    }
}
