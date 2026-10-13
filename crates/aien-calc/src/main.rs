//! `aien-calc`: command line front end. Exit codes: 0 answered, 2 malformed request, 3 refused.

use aien_calc::store::Store;
use aien_calc::sweep::sweep;
use aien_calc::verify::verify_bytes;
use aien_calc::{eval, CalcError};
use clap::{Parser, Subcommand};
use std::io::Read;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser)]
#[command(
    name = "aien-calc",
    version,
    about = "Exact-arithmetic calculator with receipts"
)]
struct Cli {
    /// Do not save the receipt to the store ($AIEN_CALC_HOME, else ~/.aien/calc).
    #[arg(long, global = true)]
    no_store: bool,
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
    /// Recompute a stored receipt from its own inputs and report MATCH or MISMATCH.
    Verify {
        /// A receipt id (calc:<hex>, 12+ digits) or the path of a receipt file.
        target: String,
    },
    /// Copy a stored receipt to <DIR>/calc-receipts/<hex>.json, to commit next to evidence.
    Export {
        /// Receipt id (calc:<hex>, 12+ digits).
        id: String,
        /// Evidence directory.
        dir: PathBuf,
    },
    /// BLAKE3 of files, in b3sum format (`<hex>  <path>`); `-` reads stdin.
    B3 { files: Vec<String> },
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

/// Save the receipt unless `--no-store`. A store problem is a warning: the answer is still valid.
fn store(no_store: bool, receipt: &aien_calc::Receipt) {
    if no_store {
        return;
    }
    let result = Store::from_env().map_err(|e| e.to_string()).and_then(|s| {
        let cwd = std::env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_default();
        let argv: Vec<String> = std::env::args().collect();
        s.save(receipt, &cwd, &argv)
            .map_err(|e| format!("{}: {e}", s.root().display()))
    });
    if let Err(e) = result {
        eprintln!("aien-calc: warning: receipt not stored: {e}");
    }
}

fn verify_cmd(target: &str) -> ExitCode {
    let file_like =
        target.contains('/') || target.ends_with(".json") || std::path::Path::new(target).is_file();
    let (bytes, named) = if file_like {
        let path = std::path::Path::new(target);
        let named = path
            .file_stem()
            .and_then(|s| s.to_str())
            .filter(|s| s.len() == 64 && s.bytes().all(|c| matches!(c, b'0'..=b'9' | b'a'..=b'f')))
            .map(str::to_string);
        match std::fs::read(path) {
            Ok(b) => (b, named),
            Err(e) => {
                eprintln!("aien-calc: cannot read {target}: {e}");
                return ExitCode::from(2);
            }
        }
    } else {
        let loaded = Store::from_env().and_then(|s| {
            let hex = s.resolve(target)?;
            s.load(&hex).map(|b| (b, Some(hex)))
        });
        match loaded {
            Ok(x) => x,
            Err(e) => {
                eprintln!("aien-calc: {e}");
                return ExitCode::from(2);
            }
        }
    };
    let v = verify_bytes(&bytes, named.as_deref());
    if v.matches {
        println!("MATCH {}", v.id);
        ExitCode::SUCCESS
    } else {
        println!("MISMATCH {}", v.id);
        for d in &v.diffs {
            println!("  {d}");
        }
        ExitCode::from(1)
    }
}

fn b3_cmd(files: &[String]) -> ExitCode {
    let mut code = ExitCode::SUCCESS;
    let targets: Vec<String> = if files.is_empty() {
        vec!["-".into()]
    } else {
        files.to_vec()
    };
    for f in targets {
        let data = if f == "-" {
            let mut buf = Vec::new();
            std::io::stdin().read_to_end(&mut buf).map(|_| buf)
        } else {
            std::fs::read(&f)
        };
        match data {
            Ok(d) => println!("{}  {f}", blake3::hash(&d).to_hex()),
            Err(e) => {
                eprintln!("aien-calc: b3: {f}: {e}");
                code = ExitCode::from(2);
            }
        }
    }
    code
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let no_store = cli.no_store;
    match cli.command {
        Cmd::Eval {
            expr,
            precision,
            json,
        } => match eval(&expr, precision) {
            Ok(out) => {
                store(no_store, out.receipt());
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
                store(no_store, out.receipt());
                emit(json, out.text(), out.receipt());
                ExitCode::SUCCESS
            }
            Err(e) => fail(e),
        },
        Cmd::Verify { target } => verify_cmd(&target),
        Cmd::Export { id, dir } => match Store::from_env().and_then(|s| s.export(&id, &dir)) {
            Ok(p) => {
                println!("exported {}", p.display());
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("aien-calc: {e}");
                ExitCode::from(2)
            }
        },
        Cmd::B3 { files } => b3_cmd(&files),
    }
}
