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

use crate::identity::IdentityAnchor;
use crate::keys::sync_identity_keys;
use std::ffi::CString;
use std::os::raw::c_char;

#[no_mangle]
pub extern "C" fn rad_id_sync_status_json() -> *mut c_char {
    let result = match IdentityAnchor::load_or_init() {
        Ok(anchor) => match sync_identity_keys(anchor) {
            Ok((_, report)) => serde_json::to_string(&report)
                .unwrap_or_else(|e| format!("{{\"error\":\"{}\"}}", e)),
            Err(e) => format!("{{\"error\":\"{}\"}}", e),
        },
        Err(e) => format!("{{\"error\":\"{}\"}}", e),
    };

    CString::new(result).unwrap_or_default().into_raw()
}

#[no_mangle]
/// # Safety
///
/// The `ptr` pointer must be a valid pointer allocated by `rad_id_sync_status_json` or null.
pub unsafe extern "C" fn rad_id_sync_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe {
            let _ = CString::from_raw(ptr);
        }
    }
}

#[no_mangle]
pub extern "C" fn rad_id_sync_is_valid() -> i32 {
    match IdentityAnchor::load_or_init() {
        Ok(anchor) => {
            if anchor.primary_radicle_key.is_some() || anchor.primary_ssh_key.is_some() {
                1
            } else {
                0
            }
        }
        Err(_) => 0,
    }
}
