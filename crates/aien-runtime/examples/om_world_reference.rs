//! Reference side of the .om world-check parity test (om/effects/run.sh).
//!
//! Usage: om_world_reference TARGET CONTENT_SHA256 PRIOR_SHA256|- INTENT AUTHORIZATION BY
//!
//! Runs the production world check (`effects::world_state`) on TARGET and prints
//! the reconcile record text (`effects::reconcile_text`) on one line, then the
//! sha256 of those text bytes (the `text_sha256` that `append` reports).
use aien_runtime::effects::{reconcile_text, world_state, EffectState, IntentRow};
use sha2::{Digest, Sha256};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 7 {
        eprintln!("usage: om_world_reference TARGET CONTENT PRIOR|- INTENT AUTHORIZATION BY");
        std::process::exit(2);
    }
    let row = IntentRow {
        id: a[4].parse().expect("intent"),
        authorization: a[5].parse().expect("authorization"),
        path: a[1].clone(),
        target: a[1].clone(),
        content_sha256: a[2].clone(),
        prior_sha256: (a[3] != "-").then(|| a[3].clone()),
        executor_pid: 0,
        executor_start: 0,
        state: EffectState::Open,
        state_record: 0,
        unresolved_digest: None,
        objective_id: None,
    };
    let (world, disk) = world_state(&row);
    let text = reconcile_text(&row, world, world, &disk, &a[6], None).to_string();
    let d = Sha256::digest(text.as_bytes());
    let hex: String = d.iter().map(|b| format!("{b:02x}")).collect();
    println!("{text}");
    println!("{hex}");
}
