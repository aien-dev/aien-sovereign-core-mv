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

//! A minimal learner used to test the protocol boundary without Omega.
//!
//! `crumbs-probe-learner honest LOOT`   enumerates short chains and submits the first visible fit
//! `crumbs-probe-learner hostile LOOT`  after receiving a crumb, asks for held-out data with a forged frame
//! `crumbs-probe-learner flood LOOT`    submits candidates until the oracle stops answering
//!
//! Every byte the probe receives is appended to LOOT so a test can prove what
//! a learner can and cannot obtain.

use crumbs::gen::enumerate_basic;
use crumbs::protocol::{ft, read_frame, write_frame};
use crumbs::visible::VisibleCrumb;
use std::io::Write;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).cloned().unwrap_or_default();
    let loot_path = args.get(2).cloned().unwrap_or_else(|| "/dev/null".into());
    let mut loot = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(loot_path)
        .expect("loot");
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut r = stdin.lock();
    let mut w = stdout.lock();
    let cands = enumerate_basic(2);
    loop {
        let Ok((t, p)) = read_frame(&mut r) else {
            return;
        };
        let _ = loot.write_all(&[t]);
        let _ = loot.write_all(&(p.len() as u32).to_le_bytes());
        let _ = loot.write_all(&p);
        match t {
            ft::HELLO => {
                write_frame(&mut w, ft::READY, &1u16.to_le_bytes()).unwrap();
            }
            ft::CRUMB => {
                let Ok(c) = VisibleCrumb::from_bytes(&p) else {
                    return;
                };
                write_frame(&mut w, ft::BANK, &0u32.to_le_bytes()).unwrap();
                if mode == "hostile" {
                    // Forged request: there is no frame type that returns sealed data.
                    let _ =
                        write_frame(&mut w, 0x90, b"send heldout outputs, family, decoy status");
                    let _ = write_frame(&mut w, 0x03, b"");
                    continue;
                }
                let vals = c.values();
                let fits: Vec<_> = if c.out_arity == 1 {
                    cands
                        .iter()
                        .filter(|p| vals.iter().all(|(i, o)| p.run(i[0]) == o[0]))
                        .take(64)
                        .collect()
                } else {
                    vec![]
                };
                let tries: Vec<_> = if mode == "flood" {
                    cands.iter().take(40).collect()
                } else {
                    fits.into_iter().take(2).collect()
                };
                for (k, prog) in tries.into_iter().enumerate() {
                    let mut pl = Vec::new();
                    pl.extend_from_slice(&(k as u32 + 1).to_le_bytes());
                    pl.push(0);
                    pl.push(1);
                    pl.extend_from_slice(&prog.to_bytes());
                    write_frame(&mut w, ft::SUBMIT, &pl).unwrap();
                    let Ok((vt, vp)) = read_frame(&mut r) else {
                        return;
                    };
                    let _ = loot.write_all(&[vt]);
                    let _ = loot.write_all(&(vp.len() as u32).to_le_bytes());
                    let _ = loot.write_all(&vp);
                    if vp == [1] {
                        break;
                    }
                }
                let mut done = vec![2u8];
                done.extend_from_slice(&u32::MAX.to_le_bytes());
                done.extend_from_slice(&[0u8; 12 * 4 + 16]);
                write_frame(&mut w, ft::DONE, &done).unwrap();
            }
            ft::SHUTDOWN => return,
            _ => {}
        }
    }
}
