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

//! Producer/consumer dependency graph (Headstart M2, issue #535).
//!
//! Nodes are task ids. An edge `producer -> consumer` means the consumer
//! needs the producer's artifact. Every insert is checked: unknown nodes,
//! self edges, duplicate edges and edges that would close a cycle are
//! refused, so the graph is always a DAG. Iteration order is the byte order
//! of task ids (BTreeMap/BTreeSet), which keeps scheduling deterministic.

use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphError {
    DuplicateNode(String),
    MissingNode(String),
    SelfEdge(String),
    DuplicateEdge {
        producer: String,
        consumer: String,
    },
    /// Adding the edge would close this cycle (first and last node equal).
    Cycle(Vec<String>),
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GraphError::DuplicateNode(n) => write!(f, "task '{n}' already exists"),
            GraphError::MissingNode(n) => write!(f, "task '{n}' does not exist"),
            GraphError::SelfEdge(n) => write!(f, "task '{n}' cannot depend on itself"),
            GraphError::DuplicateEdge { producer, consumer } => {
                write!(f, "edge {producer} -> {consumer} already exists")
            }
            GraphError::Cycle(path) => write!(f, "edge would close cycle {}", path.join(" -> ")),
        }
    }
}

impl std::error::Error for GraphError {}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct DependencyGraph {
    /// consumer -> producers
    producers: BTreeMap<String, BTreeSet<String>>,
    /// producer -> consumers
    consumers: BTreeMap<String, BTreeSet<String>>,
}

impl DependencyGraph {
    pub fn contains(&self, id: &str) -> bool {
        self.producers.contains_key(id)
    }

    pub fn add_node(&mut self, id: &str) -> Result<(), GraphError> {
        if self.contains(id) {
            return Err(GraphError::DuplicateNode(id.to_string()));
        }
        self.producers.insert(id.to_string(), BTreeSet::new());
        self.consumers.insert(id.to_string(), BTreeSet::new());
        Ok(())
    }

    /// Check an edge without inserting it.
    pub fn check_edge(&self, producer: &str, consumer: &str) -> Result<(), GraphError> {
        for n in [producer, consumer] {
            if !self.contains(n) {
                return Err(GraphError::MissingNode(n.to_string()));
            }
        }
        if producer == consumer {
            return Err(GraphError::SelfEdge(producer.to_string()));
        }
        if self.consumers[producer].contains(consumer) {
            return Err(GraphError::DuplicateEdge {
                producer: producer.to_string(),
                consumer: consumer.to_string(),
            });
        }
        // The new edge closes a cycle iff the producer is reachable from
        // the consumer along existing edges.
        if let Some(mut path) = self.path(consumer, producer) {
            path.push(consumer.to_string());
            return Err(GraphError::Cycle(path));
        }
        Ok(())
    }

    pub fn add_edge(&mut self, producer: &str, consumer: &str) -> Result<(), GraphError> {
        self.check_edge(producer, consumer)?;
        self.consumers
            .get_mut(producer)
            .map(|s| s.insert(consumer.to_string()));
        self.producers
            .get_mut(consumer)
            .map(|s| s.insert(producer.to_string()));
        Ok(())
    }

    pub fn producers_of(&self, id: &str) -> impl Iterator<Item = &String> {
        self.producers.get(id).into_iter().flatten()
    }

    pub fn consumers_of(&self, id: &str) -> impl Iterator<Item = &String> {
        self.consumers.get(id).into_iter().flatten()
    }

    /// Every task reachable from `id` along producer -> consumer edges, in
    /// breadth-first order, excluding `id`.
    pub fn descendants(&self, id: &str) -> Vec<String> {
        let mut seen = BTreeSet::new();
        let mut out = Vec::new();
        let mut q: VecDeque<&String> = self.consumers_of(id).collect();
        while let Some(n) = q.pop_front() {
            if seen.insert(n.clone()) {
                out.push(n.clone());
                q.extend(self.consumers_of(n));
            }
        }
        out
    }

    /// A path `from -> ... -> to` along existing edges, if one exists.
    fn path(&self, from: &str, to: &str) -> Option<Vec<String>> {
        let mut parent: BTreeMap<&str, &str> = BTreeMap::new();
        let mut q = VecDeque::from([from]);
        let mut seen = BTreeSet::from([from]);
        while let Some(n) = q.pop_front() {
            if n == to {
                let mut path = vec![to.to_string()];
                let mut cur = to;
                while let Some(p) = parent.get(cur) {
                    path.push(p.to_string());
                    cur = p;
                }
                path.reverse();
                return Some(path);
            }
            for c in self.consumers_of(n) {
                if seen.insert(c.as_str()) {
                    parent.insert(c.as_str(), n);
                    q.push_back(c.as_str());
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph(nodes: &[&str]) -> DependencyGraph {
        let mut g = DependencyGraph::default();
        for n in nodes {
            g.add_node(n).unwrap();
        }
        g
    }

    #[test]
    fn refuses_cycles_missing_nodes_and_duplicates() {
        let mut g = graph(&["a", "b", "c"]);
        g.add_edge("a", "b").unwrap();
        g.add_edge("b", "c").unwrap();
        assert_eq!(
            g.add_edge("c", "a"),
            Err(GraphError::Cycle(vec![
                "a".into(),
                "b".into(),
                "c".into(),
                "a".into()
            ]))
        );
        assert_eq!(g.add_edge("a", "a"), Err(GraphError::SelfEdge("a".into())));
        assert_eq!(
            g.add_edge("a", "z"),
            Err(GraphError::MissingNode("z".into()))
        );
        assert!(matches!(
            g.add_edge("a", "b"),
            Err(GraphError::DuplicateEdge { .. })
        ));
        assert_eq!(g.add_node("a"), Err(GraphError::DuplicateNode("a".into())));
        // A diamond is fine.
        g.add_edge("a", "c").unwrap();
        assert_eq!(g.descendants("a"), vec!["b".to_string(), "c".to_string()]);
    }
}
