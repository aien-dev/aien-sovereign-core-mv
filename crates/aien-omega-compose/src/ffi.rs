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

//! Raw declarations of omega `src/runtime/rxc_host_abi.h` (ABI version 2).
//!
//! Every mirrored constant, callback type, struct and function is declared through the
//! `aien-abi-conformance` macros, which record what they declare. The contract they must
//! match is `abi/rxc_host_abi.manifest`: the crate's `abi_manifest` test checks these Rust
//! declarations against it (every build, stub included) and `build.rs` checks the pinned C
//! header, a compiled layout probe and the archive's symbols against it before linking.
//! At run time `Compose::open` still compares the four struct sizes with
//! `rxc_host_abi_layout` before any other call.

#![allow(non_camel_case_types)]

use aien_abi_conformance::{abi_consts, abi_externs, abi_fnptr, abi_struct};
#[cfg(has_omega_compose)]
use std::os::raw::c_char;
use std::os::raw::{c_int, c_void};

abi_consts! { CONSTS;
    pub const RXC_HOST_ABI_VERSION: u32 = 2;
    pub const RXC_HOST_MAX_HANDLES: u32 = 4;
    pub const RXC_HOST_MAX_SKILLS: u32 = 2;
    pub const RXC_HOST_NAME_MAX: u32 = 64;
    pub const RXC_HOST_NONE: u32 = 0xFFFF_FFFF;
    pub const RXC_HOST_SUBJECT_GOAL: u64 = 1;
    pub const RXC_HOST_SUBJECT_STATE: u64 = 5;
    /// Host records (notes, repair records): no World object.
    pub const RXC_HOST_SUBJECT_HOST: u64 = 0;
    pub const RXC_HOST_TAG_REPAIR_TAIL: u64 = 1;
    pub const RXC_HOST_NOTE_CONSTRAINT: u32 = 2;
    pub const RXC_HOST_NOTE_AUTHORIZATION: u32 = 3;
    pub const RXC_HOST_NOTE_EFFECT: u32 = 4;
    pub const RXC_HOST_NOTE_MAX: usize = 16384;
    pub const RXC_HOST_NP_LEN: usize = 0;
    pub const RXC_HOST_NP_SHA: usize = 1;
    pub const RXC_HOST_NP_BYTES: usize = 5;
    pub const RXC_HOST_RP_CUT_LO: usize = 0;
    pub const RXC_HOST_RP_CUT_HI: usize = 1;
    pub const RXC_HOST_RP_DROPPED: usize = 5;
    pub const RXC_HOST_RP_CAUSE: usize = 6;
    pub const RXC_HOST_OK: c_int = 0;
    pub const RXC_HOST_E_ARG: c_int = -1;
    pub const RXC_HOST_E_IDENTITY: c_int = -2;
    pub const RXC_HOST_E_TORN: c_int = -3;
    pub const RXC_HOST_E_OPEN: c_int = -4;
    pub const RXC_HOST_E_FULL: c_int = -5;
    pub const RXC_HOST_E_STATE: c_int = -6;
    pub const RXC_HOST_E_RUN: c_int = -7;
    pub const RXC_HOST_E_NOMEM: c_int = -8;
    pub const RXC_HOST_E_NOT_FOUND: c_int = -9;
    pub const RXC_HOST_E_DIGEST: c_int = -10;
    pub const RXC_HOST_E_REPLAY: c_int = -11;
}

#[repr(C)]
pub struct RxcHost {
    _private: [u8; 0],
}

abi_fnptr! { FNPTRS;
    pub type RxcHostSkillFn =
        unsafe extern "C" fn(ctx: *mut c_void, task: u64, result: *mut u64) -> c_int;
    pub type RxcHostVerifyFn =
        unsafe extern "C" fn(ctx: *mut c_void, task: u64, result: u64) -> c_int;
}

abi_struct! {
    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default)]
    pub struct RxcHostInfo {
        pub abi_version: u32,
        pub machine_root: u8,
        pub machine_id: [u8; 32],
        pub machine_id_was_stored: u32,
        pub tail_torn: u32,
        pub opened: u32,
        pub open_rc: i32,
        pub records: u64,
        pub recovered_record: u64,
        pub rolled_back: u32,
        pub recovered_completed: u32,
        pub anchor_unknown: u32,
        pub n_skills: u32,
    }
}

abi_struct! {
    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default)]
    pub struct RxcHostResult {
        pub run_rc: i32,
        pub outcome: i32,
        pub committed: u32,
        pub n_branches: u32,
        pub branches_reclaimed: u32,
        pub winner: u32,
        pub winner_skill: u32,
        pub aegis_pass_mask: u32,
        pub task: u64,
        pub result: u64,
        pub cx_goal: u64,
        pub cx_candidate: [u64; 2],
        pub cx_evidence: u64,
        pub cx_promotion: u64,
        pub cx_admission: [u64; 2],
        pub winner_digest: [u8; 32],
        pub record_digest: [u8; 32],
    }
}

abi_struct! {
    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default)]
    pub struct RxcHostRecord {
        pub id: u64,
        pub cls: u32,
        pub kind: u32,
        pub subject: u64,
        pub t: u64,
        pub generation: u64,
        pub tag: u64,
        pub links: [u64; 4],
        pub n_payload: u32,
        pub verified: u32,
        pub digest: [u8; 32],
    }
}

abi_struct! {
    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default)]
    pub struct RxcHostRepair {
        pub repaired: u32,
        pub tail_torn: u32,
        pub anchor_present: u32,
        pub cause: i32,
        pub opens: u32,
        pub open_rc: i32,
        pub rolled_back: u32,
        pub recovered_completed: u32,
        pub cut_lo: u64,
        pub cut_hi: u64,
        pub partial_records: u64,
        pub anchor_records: u64,
        pub records_kept: u64,
        pub dropped_records: u64,
        pub event_id: u64,
        pub cut_bytes_kept: u64,
        pub cut_sha256: [u8; 32],
        pub anchor_head: [u8; 32],
    }
}

abi_externs! { FNS;
    #[cfg(has_omega_compose)]
    extern "C" {
        pub fn rxc_host_abi_layout(out: *mut u32) -> u32;
        pub fn rxc_host_open(
            dir: *const c_char,
            root_kind: u32,
            root: *const u8,
            root_len: usize,
            session: u64,
            flags: u32,
            out: *mut *mut RxcHost,
            info: *mut RxcHostInfo,
        ) -> c_int;
        pub fn rxc_host_register_skill(
            h: *mut RxcHost,
            name: *const c_char,
            digest32: *const u8,
            cost: u64,
            f: RxcHostSkillFn,
            ctx: *mut c_void,
        ) -> c_int;
        pub fn rxc_host_set_verify(
            h: *mut RxcHost,
            f: Option<RxcHostVerifyFn>,
            ctx: *mut c_void,
        ) -> c_int;
        pub fn rxc_host_run(h: *mut RxcHost, task: u64, now_us: u64, out: *mut RxcHostResult) -> c_int;
        pub fn rxc_host_recall(
            h: *mut RxcHost,
            subject: u64,
            out: *mut RxcHostRecord,
            max: u32,
            n_filled: *mut u32,
            n_total: *mut u64,
        ) -> c_int;
        pub fn rxc_host_record(h: *mut RxcHost, id: u64, out: *mut RxcHostRecord) -> c_int;
        pub fn rxc_host_payload(h: *mut RxcHost, id: u64, out: *mut u64, max: u32) -> c_int;
        pub fn rxc_host_info(h: *mut RxcHost, info: *mut RxcHostInfo) -> c_int;
        pub fn rxc_host_note(
            h: *mut RxcHost,
            note: u32,
            links: *const u64,
            bytes: *const u8,
            len: usize,
            out_id: *mut u64,
        ) -> c_int;
        pub fn rxc_host_recover(
            dir: *const c_char,
            root_kind: u32,
            root: *const u8,
            root_len: usize,
            out: *mut RxcHostRepair,
        ) -> c_int;
        pub fn rxc_host_close(h: *mut RxcHost);
    }
}

// Present only when the archive defines it (build.rs sets `has_omega_wait_ms` after finding
// the declaration in rxc_host_abi.h and the symbol in the archive).
abi_externs! { WAIT_FNS;
    #[cfg(has_omega_wait_ms)]
    extern "C" {
        /// Wraps omega `rx_compose_set_wait_ms` on the handle's composition.
        pub fn rxc_host_set_wait_ms(h: *mut RxcHost, wait_ms: u32) -> c_int;
    }
}

/// omega settle wait when `rxc_host_set_wait_ms` is never called.
pub const WAIT_MS_DEFAULT: u32 = 30_000;

/// The Rust side of the `rxc_host` contract, for the manifest check.
pub fn rust_abi() -> aien_abi_conformance::RustAbi {
    aien_abi_conformance::RustAbi::default()
        .with_struct::<RxcHostInfo>()
        .with_struct::<RxcHostResult>()
        .with_struct::<RxcHostRecord>()
        .with_struct::<RxcHostRepair>()
        .with_consts(CONSTS)
        .with_fnptrs(FNPTRS)
        .with_fns(FNS)
        .with_fns(WAIT_FNS)
}
