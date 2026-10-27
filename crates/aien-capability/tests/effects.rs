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

use aien_capability::{routing_class, speculation_safe, EffectClass, ToolEffects};

#[test]
fn highest_declared_bit_wins() {
    let read_and_send = ToolEffects::READ_NETWORK | ToolEffects::EXTERNAL_IRREVERSIBLE;
    assert_eq!(
        routing_class(read_and_send),
        EffectClass::ExternalIrreversible
    );

    let write = ToolEffects::READ_FILESYSTEM | ToolEffects::EXTERNAL_WRITE;
    assert_eq!(routing_class(write), EffectClass::ExternalWrite);

    let world = ToolEffects::WORLD_MUTATION | ToolEffects::READ_FILESYSTEM;
    assert_eq!(routing_class(world), EffectClass::WorldMutation);

    let compile = ToolEffects::SPAWN_PROCESS | ToolEffects::LOCAL_EPHEMERAL;
    assert_eq!(routing_class(compile), EffectClass::LocalEphemeral);

    let search = ToolEffects::READ_NETWORK | ToolEffects::SECRET_BEARING;
    assert_eq!(routing_class(search), EffectClass::ReadOnly);

    assert_eq!(routing_class(ToolEffects::PURE), EffectClass::Pure);
}

#[test]
fn speculation_is_safe_only_for_reversible_classes() {
    assert!(speculation_safe(ToolEffects::PURE));
    assert!(speculation_safe(ToolEffects::READ_NETWORK));
    assert!(speculation_safe(
        ToolEffects::READ_FILESYSTEM | ToolEffects::SECRET_BEARING
    ));
    assert!(speculation_safe(ToolEffects::LOCAL_EPHEMERAL));
    assert!(speculation_safe(ToolEffects::SPAWN_PROCESS));

    assert!(!speculation_safe(ToolEffects::WORLD_MUTATION));
    assert!(!speculation_safe(ToolEffects::EXTERNAL_WRITE));
    assert!(!speculation_safe(ToolEffects::EXTERNAL_IRREVERSIBLE));
    assert!(!speculation_safe(
        ToolEffects::READ_NETWORK | ToolEffects::EXTERNAL_IRREVERSIBLE
    ));
}
