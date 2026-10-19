//! `RingSource`: a tensor source for the Flash-Next reference whose routed experts come from the
//! Heartbeat ring (published, digest-checked slots) and everything else from the inner source.
//! The bytes a slot holds are the checkpoint's bytes (BLAKE3 equal), so the model's output through
//! this source must be bit for bit the output through the plain source: the test that says so is
//! `wormhole_run ring-check`.

use aien_inference_abi::qwen3_moe_omega::TensorSource;

use crate::cache::ExpertCache;

pub struct RingSource<S: TensorSource> {
    pub inner: S,
    pub cache: ExpertCache,
}

/// `[model.][language_model.]layers.{L}.mlp.experts.{E}.<rest>` (not the MTP head's experts)
pub fn parse_expert(name: &str) -> Option<(u16, u16)> {
    let rest = name.strip_prefix("model.")?;
    let rest = rest.strip_prefix("language_model.").unwrap_or(rest);
    let rest = rest.strip_prefix("layers.")?;
    let (l, rest) = rest.split_once('.')?;
    let rest = rest.strip_prefix("mlp.experts.")?;
    let (e, _) = rest.split_once('.')?;
    Some((l.parse().ok()?, e.parse().ok()?))
}

impl<S: TensorSource> TensorSource for RingSource<S> {
    fn read_into(&mut self, name: &str, dtype: &str, shape: &[usize], dst: &mut [u8]) -> Result<(), String> {
        match parse_expert(name) {
            Some((l, e)) if self.cache.entry(l, e).is_some() => self.cache.read_tensor(l, e, name, dtype, shape, dst),
            _ => self.inner.read_into(name, dtype, shape, dst),
        }
    }
    fn dtype_of(&mut self, name: &str) -> Result<Option<String>, String> {
        if let Some((l, e)) = parse_expert(name) {
            if let Some(ent) = self.cache.entry(l, e) {
                return Ok(ent.tensors.iter().find(|t| t.name == name).map(|t| t.dtype.clone()));
            }
        }
        self.inner.dtype_of(name)
    }
    fn read_range(&mut self, name: &str, dtype: &str, shape: &[usize], offset: u64, dst: &mut [u8]) -> Result<(), String> {
        if parse_expert(name).is_some() {
            return Err(format!("{name}: expert tensors are read whole from the ring"));
        }
        self.inner.read_range(name, dtype, shape, offset, dst)
    }
}

#[cfg(test)]
mod tests {
    use super::parse_expert;
    #[test]
    fn expert_names() {
        assert_eq!(parse_expert("model.language_model.layers.5.mlp.experts.7.up_proj.weight"), Some((5, 7)));
        assert_eq!(parse_expert("model.layers.5.mlp.experts.7.up_proj.weight"), Some((5, 7)));
        assert_eq!(parse_expert("model.language_model.layers.5.mlp.shared_expert.up_proj.weight"), None);
        assert_eq!(parse_expert("mtp.layers.0.mlp.experts.7.up_proj.weight"), None);
        assert_eq!(parse_expert("lm_head.weight"), None);
    }
}
