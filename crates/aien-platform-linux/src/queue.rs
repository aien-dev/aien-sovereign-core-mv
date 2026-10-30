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

//! Priority-deadline concurrent WorkQueue for the Linux platform.

use aien_platform::{InferenceWork, QueueError, WorkQueue};
use parking_lot::Mutex;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Debug, Clone, PartialEq, Eq)]
struct ScheduledWorkItem(InferenceWork);

impl Ord for ScheduledWorkItem {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.0.priority.cmp(&other.0.priority) {
            Ordering::Equal => match (self.0.deadline, other.0.deadline) {
                (Some(d1), Some(d2)) => d2.cmp(&d1),
                (Some(_), None) => Ordering::Greater,
                (None, Some(_)) => Ordering::Less,
                (None, None) => Ordering::Equal,
            },
            ord => ord,
        }
    }
}

impl PartialOrd for ScheduledWorkItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct LinuxWorkQueue {
    capacity: usize,
    items: Mutex<BinaryHeap<ScheduledWorkItem>>,
}

impl LinuxWorkQueue {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            items: Mutex::new(BinaryHeap::with_capacity(capacity)),
        }
    }

    pub fn with_default_capacity() -> Self {
        Self::new(4096)
    }
}

impl Default for LinuxWorkQueue {
    fn default() -> Self {
        Self::with_default_capacity()
    }
}

impl WorkQueue for LinuxWorkQueue {
    fn push(&self, work: InferenceWork) -> Result<(), QueueError> {
        let mut heap = self.items.lock();
        if heap.len() >= self.capacity {
            return Err(QueueError::Full);
        }
        heap.push(ScheduledWorkItem(work));
        Ok(())
    }

    fn pop(&self) -> Option<InferenceWork> {
        let mut heap = self.items.lock();
        heap.pop().map(|item| item.0)
    }

    fn len(&self) -> usize {
        self.items.lock().len()
    }
}
