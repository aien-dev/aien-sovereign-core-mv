// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Drake Stapleton (AIEN / aien-dev)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! `crumbs` command-line tool (sealed side).
//!
//! crumbs families                               list registered families (sealed labels)
//! crumbs vectors DIR                            write Crumb v1 conformance vectors
//! crumbs run --learner PATH [--n N] [--seed S] [--library] [--out DIR]
//! crumbs experiment --learner PATH [--out DIR] [--seed S] [--phase1 N]
//! crumbs gates --learner PATH [--out DIR]       learner-dependent qualification gates

use crumbs::gen::{self, registry};
use crumbs::mixer::{Mixer, MixerConfig};
use crumbs::promotion::PromotionConfig;
use crumbs::protocol::SearchConfig;
use crumbs::session::{Session, SessionConfig};
use std::path::PathBuf;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("help");
    let code = match cmd {
        "families" => {
            for f in &registry().families {
                println!(
                    "{:4} {:32} {:26} {:?} {:?}",
                    f.id,
                    f.key,
                    f.mechanism.key(),
                    f.rung,
                    f.source_family
                );
            }
            0
        }
        "vectors" => crumbs::conformance::write_vectors(&PathBuf::from(args.get(2).expect("DIR"))),
        "run" => run(&args),
        "experiment" => crumbs::experiment::main(&args),
        "gates" => crumbs::gates::main(&args),
        _ => {
            eprintln!("usage: crumbs families | vectors DIR | run --learner PATH | experiment --learner PATH | gates --learner PATH");
            2
        }
    };
    std::process::exit(code);
}

fn run(args: &[String]) -> i32 {
    let learner = arg(args, "--learner").expect("--learner PATH");
    let n: u64 = arg(args, "--n").map_or(40, |v| v.parse().unwrap());
    let seed: u64 = arg(args, "--seed").map_or(1, |v| v.parse().unwrap());
    let cfg = SessionConfig {
        condition: if flag(args, "--library") {
            "learning".into()
        } else {
            "control".into()
        },
        learner,
        run_id: format!("run-{seed}"),
        search: SearchConfig::default(),
        library_enabled: flag(args, "--library"),
        promotion: PromotionConfig::default(),
        record_transcript: false,
        out_dir: arg(args, "--out").map(PathBuf::from),
    };
    let mut s = Session::start(cfg).expect("learner starts");
    let mut mixer = Mixer::new(MixerConfig::default(), seed);
    for _ in 0..n {
        let d = mixer.draw();
        let fam = registry().get(d.family_id).unwrap();
        let o = s
            .present(fam, d.seed, d.population, "run")
            .expect("session");
        mixer.record(fam, o.verified);
        println!(
            "{:4} {:30} pop={:?} supported={} accepted={} verified={} subs={} fh={} gen={} lib_avail={} reuse={} admitted={}",
            o.seq,
            o.family_key,
            o.population,
            o.supported,
            o.accepted,
            o.verified,
            o.evaluations.len(),
            o.falsified_hidden(),
            o.stats.generated,
            o.library_ops_available,
            o.reused_library(),
            o.admissions_after.len()
        );
    }
    let _ = gen::GENERATOR_VERSION;
    let (ledger, _, ladder) = s.finish().expect("clean shutdown");
    println!(
        "ledger head {} records {}",
        ledger.head,
        ledger.records.len()
    );
    println!(
        "ops tracked {} admitted {}",
        ladder.ops.len(),
        ladder.count(crumbs::promotion::Stage::Admitted)
    );
    0
}
