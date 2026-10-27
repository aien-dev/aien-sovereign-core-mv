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

pub mod adapter_engine;
pub mod emission;
pub mod geometry;
pub mod store;

pub use adapter_engine::{
    AdapterCategory, AdapterPipelineChain, AdapterPipelineReceipt, AdapterSpec, BenchmarkTelemetry,
    ConsumerModel, HardwareProfile, PrSubmissionPlan, SocraticEvaluation, SocraticInquiry,
    UpstreamEngine, build_adapter_spec, emit_adapter_pipeline_combs, evaluate_socratic_reflex,
    find_or_create_adapter, generate_pr_plan, get_catalog_adapters, list_adapter_pipeline_chains,
};
pub use emission::{emit_socratic_comb, emit_subagent_comb};
pub use geometry::{
    DEFAULT_HEX_RADIUS, DetailTier, HEX_DIRECTION_LABELS, HEX_DIRECTIONS, HexCoord,
    PlacementIntent, SQRT_3, axial_to_pixel, hex_distance, neighbors, pixel_to_axial, round_axial,
};
pub use store::{
    BoundingBox, CombStore, CreateForgeTaskInput, ForgeTask, HiveComb, HiveError, NeighborLink,
    PlaceCombInput,
};
