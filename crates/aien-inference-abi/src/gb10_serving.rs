//! STUB (red run): the product path makes no serving reservation call.
use aien_abi_core::ModelConfig;
use aien_omega_gpu::{ServingBounds, ServingBytes};

pub struct ServingLimits {
    pub context_tokens: usize,
    pub max_batch_rows: usize,
    pub prefill_chunk_rows: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServingReservation {
    pub bounds: ServingBounds,
    pub bytes: ServingBytes,
}

pub fn reserve_gb10_serving_with(
    _config: &ModelConfig,
    _limits: &ServingLimits,
    _gpu_native: bool,
    _qwen3_opted_in: bool,
    _reserve: &dyn Fn(&ServingBounds) -> Result<(), String>,
) -> Result<Option<ServingReservation>, String> {
    Ok(None)
}
