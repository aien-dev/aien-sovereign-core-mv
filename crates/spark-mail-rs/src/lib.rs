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

pub mod api;
pub mod cortex_sync;
pub mod effect;
pub mod gandi;
pub mod imap;
pub mod models;
pub mod relay;
pub mod smtp_server;
pub mod store;

pub use cortex_sync::CortexSync;
pub use models::{EmailMessage, MailboxStatus, SendEmailRequest};
pub use relay::MailRelay;
pub use store::MailStore;
