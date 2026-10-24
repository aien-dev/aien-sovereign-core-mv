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

use crate::PlatformError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ObjectId(pub [u8; 32]);

pub struct ObjectRef<'a> {
    pub id: ObjectId,
    pub data: &'a [u8],
}

pub struct ObjectWrite<'a> {
    pub id: Option<ObjectId>,
    pub data: &'a [u8],
    pub tags: &'a [&'a str],
}

pub trait ObjectStore {
    fn read(&self, id: ObjectId) -> Result<ObjectRef<'_>, PlatformError>;
    fn commit(&self, object: ObjectWrite<'_>) -> Result<ObjectId, PlatformError>;
}
