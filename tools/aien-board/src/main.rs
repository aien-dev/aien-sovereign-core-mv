use aien_board::agent::Agent;
use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = "usage:
  aien-board measure --state-dir DIR --clones DIR --since YYYY-MM-DD --out DIR
  aien-board post   --repo R [--sha SHA] [--base SHA] [--home DIR] [--source DIR]
  aien-board answer --repo R [--post ID] [--dry-run] [--own-git DIR --own-rev REV] [--home DIR] [--source DIR]
  aien-board show   [--home DIR]
  aien-board probe  file|github [--rounds N] [--home DIR] [--gh-repo OWNER/NAME --issue N]
defaults: --home ~/.local/state/aien-board, --source ~/workspace/pipeline/clones (the judge's trusted
clones; only fetched from)";

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn need(args: &[String], name: &str) -> Result<String, String> {
    flag(args, name).ok_or_else(|| format!("{name} is required\n{USAGE}"))
}

fn home_dir() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/tmp".into()))
}

fn agent(args: &[String]) -> Result<Agent, String> {
    Ok(Agent {
        repo: need(args, "--repo")?,
        source: flag(args, "--source").map(PathBuf::from).unwrap_or_else(|| home_dir().join("workspace/pipeline/clones")),
        home: board_home(args),
        own: match (flag(args, "--own-git"), flag(args, "--own-rev")) {
            (Some(d), Some(r)) => Some((PathBuf::from(d), r)),
            (None, None) => None,
            _ => return Err("--own-git and --own-rev go together".into()),
        },
    })
}

fn board_home(args: &[String]) -> PathBuf {
    flag(args, "--home").map(PathBuf::from).unwrap_or_else(|| home_dir().join(".local/state/aien-board"))
}

fn measure(args: &[String]) -> Result<(), String> {
    let state = PathBuf::from(need(args, "--state-dir")?);
    let clones = PathBuf::from(need(args, "--clones")?);
    let since = need(args, "--since")?;
    let out = PathBuf::from(need(args, "--out")?);
    let t = aien_board::measure::epoch(&format!("{since}T00:00:00Z"))
        .ok_or_else(|| format!("--since {since:?} is not YYYY-MM-DD"))?;
    let (receipts, heads) = aien_board::measure::run(&state, &clones, t)?;
    let summary = aien_board::measure::summary(&receipts, &heads, &since);
    std::fs::create_dir_all(&out).map_err(|e| format!("cannot create {}: {e}", out.display()))?;
    let write = |name: &str, body: String| {
        std::fs::write(out.join(name), body).map_err(|e| format!("cannot write {name}: {e}"))
    };
    write("receipts.tsv", aien_board::measure::tsv(&receipts))?;
    let heads_text: String = heads.iter().map(|h| format!("{h}\n")).collect();
    write("heads.jsonl", heads_text)?;
    write("summary.json", serde_json::to_string_pretty(&summary).unwrap_or_default() + "\n")?;
    println!("measured {} heads, {} receipts -> {}", heads.len(), receipts.len(), out.display());
    Ok(())
}

fn probe(args: &[String]) -> Result<(), String> {
    let rounds: usize = flag(args, "--rounds").and_then(|s| s.parse().ok()).unwrap_or(10);
    let v = match args.get(1).map(String::as_str) {
        Some("file") => aien_board::probe::file(&board_home(args).join("probe"), rounds)?,
        Some("github") => aien_board::probe::github(&need(args, "--gh-repo")?, &need(args, "--issue")?, rounds)?,
        _ => return Err(USAGE.to_string()),
    };
    println!("{v}");
    Ok(())
}

fn run(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        Some("measure") => measure(args),
        Some("post") => {
            let r = agent(args)?.post(flag(args, "--sha").as_deref(), flag(args, "--base").as_deref())?;
            println!("{} POST {} {} base {} {}", r.id, r.producer, r.sha, r.base, r.note);
            Ok(())
        }
        Some("answer") => {
            let dry = args.iter().any(|a| a == "--dry-run");
            for line in agent(args)?.answer(flag(args, "--post").as_deref(), dry)? {
                println!("{line}");
            }
            Ok(())
        }
        Some("show") => {
            let b = aien_board::board::Board::open(&board_home(args))?;
            println!("{}", aien_board::board::HEADER);
            for r in b.rows()? {
                println!("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}", r.id, r.utc, r.kind, r.post, r.producer,
                    &r.sha[..r.sha.len().min(8)], &r.base[..r.base.len().min(8)], r.consumer, r.verdict, r.note);
            }
            Ok(())
        }
        Some("probe") => probe(args),
        _ => Err(USAGE.to_string()),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("aien-board: {e}");
            ExitCode::from(2)
        }
    }
}
