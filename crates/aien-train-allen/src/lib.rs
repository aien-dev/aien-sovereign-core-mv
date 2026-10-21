//! Small Allen trainer (L389): teaches the Qwen3.5-0.8B student from the 35B teacher's answers.
//!
//! Host CPU only, our own maths end to end: [`model`] is the forward of the host reference
//! (`aien_inference_abi::qwen38`) with a hand-written backward, [`gemm`] the matrix products,
//! [`train`] the data, loss weights and AdamW, [`io`] the checkpoint reader and the merged
//! (private, never published) model writer.

pub mod decode;
pub mod gemm;
pub mod io;
pub mod model;
pub mod train;
