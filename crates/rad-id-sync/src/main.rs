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

use rad_id_sync::identity::IdentityAnchor;
use rad_id_sync::keys::sync_identity_keys;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("status");

    match cmd {
        "status" => {
            let anchor = match IdentityAnchor::load_or_init() {
                Ok(a) => a,
                Err(e) => {
                    eprintln!("Error loading identity anchor: {}", e);
                    std::process::exit(1);
                }
            };
            println!("=== Radicle Identity Anchor ===");
            println!("DID:         {}", anchor.did);
            println!("Name:        {}", anchor.name);
            println!("Type:        {:?}", anchor.entity_type);
            println!(
                "Radicle Key: {}",
                anchor.primary_radicle_key.as_deref().unwrap_or("<none>")
            );
            println!(
                "SSH Key:     {}",
                anchor.primary_ssh_key.as_deref().unwrap_or("<none>")
            );
            println!("Last Synced: {}", anchor.last_synced_at);
            println!("Key History: {} records", anchor.key_history.len());
        }
        "sync" => {
            let anchor = IdentityAnchor::load_or_init().expect("Failed to load anchor");
            let (_, report) = sync_identity_keys(anchor).expect("Sync failed");
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
        }
        "json" => {
            let anchor = IdentityAnchor::load_or_init().expect("Failed to load anchor");
            let (_, report) = sync_identity_keys(anchor).expect("Sync failed");
            println!("{}", serde_json::to_string(&report).unwrap());
        }
        _ => {
            println!("Usage: rad-id-sync [status|sync|json]");
        }
    }
}
