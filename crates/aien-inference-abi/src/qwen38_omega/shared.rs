//! Shared weights across processes (sc L383; Drake, 2026-10-20: "if they share the memory, then
//! that's one point where we can load something and they can both use it").
//!
//! The owner process (the live 35B server) uploads the checkpoint once and publishes every resident
//! weight as a share token ([`Qwen38Omega::share`]). A test copy imports the tokens
//! ([`Qwen38Omega::open_shared`]): read-only mappings of the same device pages, no copy and no
//! checkpoint read of the big tensors. It keeps its own rows, DeltaNet state, attention caches,
//! router and host-side embedding, so several copies fit beside the owner.
//!
//! The manifest travels as text over a unix socket the owner creates with mode 0600
//! ([`SharedWeights::publish`] / [`SharedWeights::fetch`]): the driver already refuses an import
//! from another user (physics `nvrm_import`: same uid or same pid), the socket keeps the tokens
//! from being read by anyone else in the first place. The pages stay mapped until every holder has
//! dropped them: the owner going away does not pull them from a test copy, a test copy crashing
//! does not touch the owner.
//!
//! Text format, one record per line, whitespace separated, `end` last (a truncated read is refused):
//!
//! ```text
//! aien-shared-weights 1
//! model-dir <canonical checkpoint path>
//! owner-pid <pid>
//! layers <n>
//! fp8 <key> <experts> <n> <k> <w: hclient hmemory flags bytes> <s: hclient hmemory flags bytes>
//! bf16 <key> <n> <k> <hclient hmemory flags bytes>
//! mm <key> <k> <n> <kp> <np> <hclient hmemory flags bytes>
//! head-coef <count> <f32 bits as hex>...
//! end
//! ```
//!
//! Keys: `lm_head`, `lm_head8`, `head_screen`, `layer<l>.mix|ab|o|gate_up|down`, `ab_mm<i>`.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::io::{Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::time::Duration;

use aien_omega_bf16::{Bf16Share, Bf16Tensor};
use aien_omega_fp8::{Fp8Share, Fp8Tensor};
use aien_omega_gpu::{ResidentTensor, ShareToken, TensorShare};

use super::Qwen38Omega;

const MAGIC: &str = "aien-shared-weights";
const VERSION: u32 = 1;
/// How long a fetch waits for the owner to answer.
const FETCH_TIMEOUT: Duration = Duration::from_secs(30);

/// One published resident weight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SharedTensor {
    Fp8(Fp8Share),
    Bf16(Bf16Share),
    Mm(TensorShare),
}

impl SharedTensor {
    fn kind(&self) -> &'static str {
        match self {
            SharedTensor::Fp8(_) => "fp8",
            SharedTensor::Bf16(_) => "bf16",
            SharedTensor::Mm(_) => "mm",
        }
    }
}

/// Everything a second process needs to serve the owner's upload.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SharedWeights {
    /// The checkpoint the owner uploaded (canonical path): an importer must open the same one for
    /// the host-side pieces (config, embedding, norms, router rows).
    pub model_dir: String,
    pub owner_pid: u32,
    pub layers: usize,
    /// The exact head's Row coefficients when the owner built the exact two-pass head (`head_screen`).
    pub head_coef: Option<Vec<f32>>,
    pub tensors: BTreeMap<String, SharedTensor>,
}

fn push_token(out: &mut String, t: &ShareToken) {
    let _ = write!(out, " {} {} {} {}", t.hclient, t.hmemory, t.flags, t.bytes);
}

fn parse_token<'a>(
    it: &mut impl Iterator<Item = &'a str>,
    what: &str,
) -> Result<ShareToken, String> {
    let mut f = |name: &str| -> Result<u64, String> {
        it.next()
            .ok_or_else(|| format!("shared weights: {what}: missing {name}"))?
            .parse::<u64>()
            .map_err(|e| format!("shared weights: {what}: {name}: {e}"))
    };
    let hclient = f("hclient")?;
    let hmemory = f("hmemory")?;
    let flags = f("flags")?;
    let bytes = f("bytes")?;
    let narrow = |v: u64, name: &str| -> Result<u32, String> {
        u32::try_from(v).map_err(|_| format!("shared weights: {what}: {name} out of range"))
    };
    Ok(ShareToken {
        hclient: narrow(hclient, "hclient")?,
        hmemory: narrow(hmemory, "hmemory")?,
        flags: narrow(flags, "flags")?,
        bytes,
    })
}

fn parse_u32<'a>(
    it: &mut impl Iterator<Item = &'a str>,
    what: &str,
    name: &str,
) -> Result<u32, String> {
    it.next()
        .ok_or_else(|| format!("shared weights: {what}: missing {name}"))?
        .parse::<u32>()
        .map_err(|e| format!("shared weights: {what}: {name}: {e}"))
}

impl SharedWeights {
    /// The manifest as text (the wire format above).
    pub fn to_text(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "{MAGIC} {VERSION}");
        let _ = writeln!(out, "model-dir {}", self.model_dir);
        let _ = writeln!(out, "owner-pid {}", self.owner_pid);
        let _ = writeln!(out, "layers {}", self.layers);
        for (key, t) in &self.tensors {
            match t {
                SharedTensor::Fp8(s) => {
                    let _ = write!(out, "fp8 {key} {} {} {}", s.experts, s.n, s.k);
                    push_token(&mut out, &s.w);
                    push_token(&mut out, &s.s);
                }
                SharedTensor::Bf16(s) => {
                    let _ = write!(out, "bf16 {key} {} {}", s.n, s.k);
                    push_token(&mut out, &s.w);
                }
                SharedTensor::Mm(s) => {
                    let _ = write!(out, "mm {key} {} {} {} {}", s.k, s.n, s.kp, s.np);
                    push_token(&mut out, &s.mem);
                }
            }
            out.push('\n');
        }
        if let Some(c) = &self.head_coef {
            let _ = write!(out, "head-coef {}", c.len());
            for v in c {
                let _ = write!(out, " {:08x}", v.to_bits());
            }
            out.push('\n');
        }
        out.push_str("end\n");
        out
    }

    /// Parses the wire format; refuses anything not ending in `end`, an unknown version, a
    /// duplicate key or a malformed record.
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut lines = text.lines();
        let first = lines.next().ok_or("shared weights: empty manifest")?;
        let mut hd = first.split_whitespace();
        if hd.next() != Some(MAGIC) {
            return Err("shared weights: not a shared-weights manifest".into());
        }
        let version = parse_u32(&mut hd, "header", "version")?;
        if version != VERSION {
            return Err(format!(
                "shared weights: version {version} (this build reads {VERSION})"
            ));
        }
        let mut m = SharedWeights::default();
        let mut ended = false;
        for line in lines {
            let mut it = line.split_whitespace();
            let Some(rec) = it.next() else { continue };
            match rec {
                "model-dir" => {
                    m.model_dir = line["model-dir".len()..].trim().to_string();
                    if m.model_dir.is_empty() {
                        return Err("shared weights: empty model-dir".into());
                    }
                }
                "owner-pid" => m.owner_pid = parse_u32(&mut it, "owner-pid", "pid")?,
                "layers" => m.layers = parse_u32(&mut it, "layers", "count")? as usize,
                "fp8" | "bf16" | "mm" => {
                    let key = it
                        .next()
                        .ok_or_else(|| format!("shared weights: {rec}: missing key"))?;
                    let t = match rec {
                        "fp8" => SharedTensor::Fp8(Fp8Share {
                            experts: parse_u32(&mut it, key, "experts")?,
                            n: parse_u32(&mut it, key, "n")?,
                            k: parse_u32(&mut it, key, "k")?,
                            w: parse_token(&mut it, key)?,
                            s: parse_token(&mut it, key)?,
                        }),
                        "bf16" => SharedTensor::Bf16(Bf16Share {
                            n: parse_u32(&mut it, key, "n")?,
                            k: parse_u32(&mut it, key, "k")?,
                            w: parse_token(&mut it, key)?,
                        }),
                        _ => SharedTensor::Mm(TensorShare {
                            k: parse_u32(&mut it, key, "k")?,
                            n: parse_u32(&mut it, key, "n")?,
                            kp: parse_u32(&mut it, key, "kp")?,
                            np: parse_u32(&mut it, key, "np")?,
                            mem: parse_token(&mut it, key)?,
                        }),
                    };
                    if it.next().is_some() {
                        return Err(format!("shared weights: {key}: trailing fields"));
                    }
                    if m.tensors.insert(key.to_string(), t).is_some() {
                        return Err(format!("shared weights: {key}: duplicate key"));
                    }
                }
                "head-coef" => {
                    let n = parse_u32(&mut it, "head-coef", "count")? as usize;
                    let mut c = Vec::with_capacity(n);
                    for v in it.by_ref() {
                        let bits = u32::from_str_radix(v, 16)
                            .map_err(|e| format!("shared weights: head-coef: {e}"))?;
                        c.push(f32::from_bits(bits));
                    }
                    if c.len() != n {
                        return Err(format!(
                            "shared weights: head-coef: {} values, said {n}",
                            c.len()
                        ));
                    }
                    m.head_coef = Some(c);
                }
                "end" => {
                    ended = true;
                    break;
                }
                other => return Err(format!("shared weights: unknown record {other:?}")),
            }
        }
        if !ended {
            return Err("shared weights: manifest cut short (no end line)".into());
        }
        if m.model_dir.is_empty() || m.layers == 0 {
            return Err("shared weights: manifest without model-dir or layers".into());
        }
        Ok(m)
    }

    pub fn has(&self, key: &str) -> bool {
        self.tensors.contains_key(key)
    }

    fn get(&self, key: &str, kind: &str) -> Result<&SharedTensor, String> {
        let t = self
            .tensors
            .get(key)
            .ok_or_else(|| format!("shared weights: {key}: not published by the owner"))?;
        if t.kind() != kind {
            return Err(format!(
                "shared weights: {key}: published as {}, this build wants {kind}",
                t.kind()
            ));
        }
        Ok(t)
    }

    /// Imports the published FP8 tensor `key` into this process (read-only, no copy).
    pub fn fp8(&self, key: &str) -> Result<Fp8Tensor, String> {
        match self.get(key, "fp8")? {
            SharedTensor::Fp8(s) => {
                Fp8Tensor::import(s).map_err(|e| format!("shared weights: {key}: import: {e}"))
            }
            _ => unreachable!("kind checked"),
        }
    }

    /// Imports the published bf16 matrix `key` into this process (read-only, no copy).
    pub fn bf16(&self, key: &str) -> Result<Bf16Tensor, String> {
        match self.get(key, "bf16")? {
            SharedTensor::Bf16(s) => {
                Bf16Tensor::import(s).map_err(|e| format!("shared weights: {key}: import: {e}"))
            }
            _ => unreachable!("kind checked"),
        }
    }

    /// Imports the published matmul weight `key` into this process (read-only, no copy).
    pub fn mm(&self, key: &str) -> Result<ResidentTensor, String> {
        match self.get(key, "mm")? {
            SharedTensor::Mm(s) => {
                ResidentTensor::import(s).map_err(|e| format!("shared weights: {key}: import: {e}"))
            }
            _ => unreachable!("kind checked"),
        }
    }

    /// Serves this manifest to every connection on the unix socket at `path` (created here, mode
    /// 0600, so only this user can read the tokens) from a thread that lives as long as the process.
    /// Refuses a path where another publisher still answers; replaces a dead socket file.
    pub fn publish(&self, path: &Path) -> Result<(), String> {
        use std::os::unix::fs::PermissionsExt;
        let text = self.to_text();
        if path.exists() {
            if UnixStream::connect(path).is_ok() {
                return Err(format!(
                    "{}: another publisher answers there",
                    path.display()
                ));
            }
            std::fs::remove_file(path)
                .map_err(|e| format!("{}: remove stale socket: {e}", path.display()))?;
        }
        let listener =
            UnixListener::bind(path).map_err(|e| format!("{}: bind: {e}", path.display()))?;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
            .map_err(|e| format!("{}: chmod 0600: {e}", path.display()))?;
        std::thread::Builder::new()
            .name("weights-publish".into())
            .spawn(move || {
                for conn in listener.incoming() {
                    let Ok(mut c) = conn else { continue };
                    let _ = c.write_all(text.as_bytes());
                    let _ = c.shutdown(std::net::Shutdown::Both);
                }
            })
            .map_err(|e| format!("weights-publish thread: {e}"))?;
        Ok(())
    }

    /// Reads the manifest a publisher serves at `path`.
    pub fn fetch(path: &Path) -> Result<Self, String> {
        let mut s = UnixStream::connect(path).map_err(|e| {
            format!(
                "{}: connect (is the owner running with --weights-publish?): {e}",
                path.display()
            )
        })?;
        s.set_read_timeout(Some(FETCH_TIMEOUT))
            .map_err(|e| format!("{}: read timeout: {e}", path.display()))?;
        let mut text = String::new();
        s.read_to_string(&mut text)
            .map_err(|e| format!("{}: read: {e}", path.display()))?;
        Self::parse(&text)
    }
}

impl Qwen38Omega {
    /// Publishes every resident weight of this model (the 35B's mixers, experts, heads) for import by
    /// another process of the same user. `dir` is the checkpoint this model was uploaded from. The
    /// returned manifest is what [`Self::open_shared`] takes.
    pub fn share(&self, dir: &Path) -> Result<SharedWeights, String> {
        let gpu = |what: &'static str| {
            move |e: aien_omega_gpu::OmegaGpuError| format!("share {what}: {e}")
        };
        let mut m = SharedWeights {
            model_dir: canonical(dir)?,
            owner_pid: std::process::id(),
            layers: self.config.layers,
            head_coef: self.row_src.as_ref().and_then(|s| s.head_coef.clone()),
            tensors: BTreeMap::new(),
        };
        let mut put = |key: String, t: SharedTensor| m.tensors.insert(key, t);
        put(
            "lm_head".into(),
            SharedTensor::Bf16(self.lm_head.export().map_err(gpu("lm_head"))?),
        );
        if let Some(h8) = &self.lm_head8 {
            put(
                "lm_head8".into(),
                SharedTensor::Fp8(h8.export().map_err(gpu("lm_head8"))?),
            );
        }
        if let Some(s) = &self.head_screen {
            put(
                "head_screen".into(),
                SharedTensor::Fp8(s.export().map_err(gpu("head_screen"))?),
            );
        }
        for (l, t) in self.layers.iter().enumerate() {
            put(
                format!("layer{l}.mix"),
                SharedTensor::Fp8(t.mix.export().map_err(gpu("mix"))?),
            );
            if let Some(ab) = &t.ab {
                put(
                    format!("layer{l}.ab"),
                    SharedTensor::Bf16(ab.export().map_err(gpu("ab"))?),
                );
            }
            put(
                format!("layer{l}.o"),
                SharedTensor::Fp8(t.o.export().map_err(gpu("o"))?),
            );
            put(
                format!("layer{l}.gate_up"),
                SharedTensor::Fp8(t.gate_up.export().map_err(gpu("gate_up"))?),
            );
            put(
                format!("layer{l}.down"),
                SharedTensor::Fp8(t.down.export().map_err(gpu("down"))?),
            );
        }
        for (i, t) in self.ab_mm.iter().enumerate() {
            put(
                format!("ab_mm{i}"),
                SharedTensor::Mm(t.export().map_err(gpu("ab_mm"))?),
            );
        }
        Ok(m)
    }

    /// Opens the 35B at `dir` on another process's upload: every big tensor is imported from
    /// `shared` (read-only mappings, no copy), the host-side pieces (config, embedding, norms,
    /// router rows, DeltaNet and attention parameters) are read from the checkpoint as usual. The
    /// manifest must name this very checkpoint. `heads` conversations resident at once, as
    /// [`Self::open_window_heads`] (1 for the row door, which adds rows after).
    pub fn open_shared(
        dir: &Path,
        shared: &SharedWeights,
        max_pos: usize,
        heads: usize,
    ) -> Result<Self, String> {
        let here = canonical(dir)?;
        if here != shared.model_dir {
            return Err(format!(
                "shared weights: the owner uploaded {}, this process opens {here}",
                shared.model_dir
            ));
        }
        let config = super::Qwen38Config::from_checkpoint(dir)?;
        if config.moe.is_none() {
            return Err(
                "shared weights: the 35B (qwen3_5_moe) only; the dense graph uploads its own"
                    .into(),
            );
        }
        if config.layers != shared.layers {
            return Err(format!(
                "shared weights: the owner published {} layers, the checkpoint has {}",
                shared.layers, config.layers
            ));
        }
        Self::load_moe(
            config,
            0,
            &mut super::ShardSource::open(dir)?,
            max_pos,
            heads,
            Some(shared),
        )
    }
}

fn canonical(dir: &Path) -> Result<String, String> {
    Ok(std::fs::canonicalize(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .to_string_lossy()
        .into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> SharedWeights {
        let t = |a: u32| ShareToken {
            hclient: 0xc1d0_0000 + a,
            hmemory: 0xcaf0_0000 + a,
            flags: a & 1,
            bytes: 4096 * (a as u64 + 1),
        };
        let mut m = SharedWeights {
            model_dir: "/models/with space/q36".into(),
            owner_pid: 4242,
            layers: 2,
            head_coef: Some(vec![1.0, -0.5, f32::MIN_POSITIVE, 3.0e38]),
            tensors: BTreeMap::new(),
        };
        m.tensors.insert(
            "lm_head".into(),
            SharedTensor::Bf16(Bf16Share {
                n: 248320,
                k: 2048,
                w: t(1),
            }),
        );
        m.tensors.insert(
            "layer0.gate_up".into(),
            SharedTensor::Fp8(Fp8Share {
                experts: 257,
                n: 1024,
                k: 2048,
                w: t(2),
                s: t(3),
            }),
        );
        m.tensors.insert(
            "ab_mm0".into(),
            SharedTensor::Mm(TensorShare {
                k: 2048,
                n: 64,
                kp: 2048,
                np: 64,
                mem: t(4),
            }),
        );
        m
    }

    #[test]
    fn text_round_trips_bit_for_bit() {
        let m = sample();
        let text = m.to_text();
        assert!(text.starts_with("aien-shared-weights 1\n"));
        assert!(text.ends_with("end\n"));
        let back = SharedWeights::parse(&text).expect("parses");
        assert_eq!(back, m);
        assert_eq!(back.to_text(), text);
    }

    #[test]
    fn refuses_cut_short_duplicate_and_foreign() {
        let text = sample().to_text();
        let cut = &text[..text.len() - "end\n".len()];
        assert!(SharedWeights::parse(cut).unwrap_err().contains("cut short"));
        let dup = text.replacen("end\n", "bf16 lm_head 1 256 1 2 0 4096\nend\n", 1);
        assert!(SharedWeights::parse(&dup)
            .unwrap_err()
            .contains("duplicate"));
        assert!(SharedWeights::parse("hello 1\nend\n")
            .unwrap_err()
            .contains("not a shared"));
        assert!(SharedWeights::parse("aien-shared-weights 2\nend\n")
            .unwrap_err()
            .contains("version 2"));
        let trailing = text.replacen("\nhead-coef", " 7\nhead-coef", 1);
        assert!(SharedWeights::parse(&trailing)
            .unwrap_err()
            .contains("trailing"));
        let extra_coef = text.replacen("\nend\n", " 3f800000\nend\n", 1);
        assert!(SharedWeights::parse(&extra_coef)
            .unwrap_err()
            .contains("5 values, said 4"));
    }

    #[test]
    fn missing_or_wrong_kind_is_named() {
        let m = sample();
        assert!(m.fp8("layer9.mix").unwrap_err().contains("not published"));
        assert!(m.fp8("lm_head").unwrap_err().contains("published as bf16"));
        assert!(m.has("ab_mm0") && !m.has("ab_mm1"));
    }

    #[test]
    fn publish_and_fetch_over_a_private_socket() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("l383-shared-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let sock = dir.join("weights.sock");
        let m = sample();
        m.publish(&sock).expect("publish");
        assert_eq!(
            std::fs::metadata(&sock).unwrap().permissions().mode() & 0o777,
            0o600
        );
        assert_eq!(SharedWeights::fetch(&sock).expect("fetch"), m);
        assert_eq!(SharedWeights::fetch(&sock).expect("fetch twice"), m);
        // a second publisher must not take over a live socket
        assert!(m.publish(&sock).unwrap_err().contains("another publisher"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
