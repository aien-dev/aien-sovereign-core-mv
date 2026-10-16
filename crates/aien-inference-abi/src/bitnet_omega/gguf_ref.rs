//! Test tool only, never product: reads the ternary tensors of Microsoft's I2_S gguf
//! (`~/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf`) so the T1 test can count how many of our
//! trits agree with theirs. Nothing of theirs is run; the file is parsed here.
//!
//! The I2_S layout, read from their source (spec 7 item 3, resolved by reading):
//! - `3rdparty/llama.cpp/ggml/src/ggml-cpu/quants.c:1333-1354` (`dequantize_row_i2_s`) and
//!   `src/ggml-bitnet-mad.cpp:51-96` (`quantize_i2_s`, x86 branch, `QK_I2_S 128` at `:12`):
//!   the flat element stream is cut into blocks of 128 (64 on an ARM build, `:14`); block `b` is
//!   32 bytes; element `j` of the block sits in byte `b*32 + j%32` at bits `6 - 2*(j/32)`.
//! - Codes: 0 -> -1, 1 -> 0, 2 -> +1 (`quants.c:1334` `map2bit`, `ggml-bitnet-mad.cpp:68-70`).
//! - Tail: one f32 scale at byte `n/4` (`ggml-bitnet-mad.cpp:90-91`), padded to 32 bytes.
//! - Element order: ggml row major, `ne0` (the input dimension) fastest, so row `r` of the
//!   `[out][in]` matrix is elements `r*in .. (r+1)*in`.

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

/// One tensor's entry in the gguf header.
#[derive(Debug, Clone)]
pub struct GgufTensor {
    pub ggml_type: u32,
    pub dims: Vec<u64>,
    /// Absolute byte offset in the file.
    pub offset: u64,
}

/// ggml type id of I2_S (spec 3.3, read from the header's tensor table).
pub const GGML_TYPE_I2_S: u32 = 36;

struct Cursor<R: Read> {
    r: R,
    pos: u64,
}

impl<R: Read> Cursor<R> {
    fn bytes(&mut self, n: usize) -> Result<Vec<u8>, String> {
        let mut b = vec![0u8; n];
        self.r.read_exact(&mut b).map_err(|e| e.to_string())?;
        self.pos += n as u64;
        Ok(b)
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.bytes(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String, String> {
        let n = self.u64()? as usize;
        String::from_utf8(self.bytes(n)?).map_err(|e| e.to_string())
    }
    /// Skips one value of gguf type `t`, returning it as u64 when it is a scalar integer.
    fn value(&mut self, t: u32) -> Result<Option<u64>, String> {
        Ok(match t {
            0 | 1 | 7 => Some(self.bytes(1)?[0] as u64),
            2 | 3 => Some(u16::from_le_bytes(self.bytes(2)?.try_into().unwrap()) as u64),
            4 | 5 => Some(self.u32()? as u64),
            6 => {
                self.bytes(4)?;
                None
            }
            8 => {
                self.string()?;
                None
            }
            9 => {
                let (et, n) = (self.u32()?, self.u64()?);
                for _ in 0..n {
                    self.value(et)?;
                }
                None
            }
            10 | 11 => Some(self.u64()?),
            12 => {
                self.bytes(8)?;
                None
            }
            other => return Err(format!("gguf value type {other} unknown")),
        })
    }
}

/// The tensor table of a gguf v3 file.
pub fn read_header(path: &Path) -> Result<HashMap<String, GgufTensor>, String> {
    let f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut c = Cursor {
        r: std::io::BufReader::new(f),
        pos: 0,
    };
    if c.bytes(4)? != b"GGUF" {
        return Err("not a gguf file".into());
    }
    let version = c.u32()?;
    if version != 3 {
        return Err(format!("gguf version {version}, expected 3"));
    }
    let (n_tensors, n_kv) = (c.u64()?, c.u64()?);
    let mut alignment = 32u64;
    for _ in 0..n_kv {
        let key = c.string()?;
        let t = c.u32()?;
        let v = c.value(t)?;
        if key == "general.alignment" {
            alignment = v.ok_or("general.alignment is not an integer")?;
        }
    }
    let mut infos = Vec::new();
    for _ in 0..n_tensors {
        let name = c.string()?;
        let n_dims = c.u32()?;
        let dims = (0..n_dims)
            .map(|_| c.u64())
            .collect::<Result<Vec<_>, _>>()?;
        let ggml_type = c.u32()?;
        let offset = c.u64()?;
        infos.push((name, dims, ggml_type, offset));
    }
    let data_start = c.pos.div_ceil(alignment) * alignment;
    Ok(infos
        .into_iter()
        .map(|(name, dims, ggml_type, offset)| {
            (
                name,
                GgufTensor {
                    ggml_type,
                    dims,
                    offset: data_start + offset,
                },
            )
        })
        .collect())
}

/// The trits and the f32 scale of one I2_S tensor of `n` elements, decoded with block size
/// `block` (128 on x86, 64 on ARM).
pub fn read_i2s(path: &Path, t: &GgufTensor, block: usize) -> Result<(Vec<i8>, f32), String> {
    use std::os::unix::fs::FileExt;
    if t.ggml_type != GGML_TYPE_I2_S {
        return Err(format!("ggml type {}, not I2_S", t.ggml_type));
    }
    let n = t.dims.iter().product::<u64>() as usize;
    let f = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut raw = vec![0u8; n / 4 + 4];
    f.read_exact_at(&mut raw, t.offset)
        .map_err(|e| e.to_string())?;
    let scale = f32::from_le_bytes(raw[n / 4..n / 4 + 4].try_into().unwrap());
    let group = block / 4;
    let mut trits = vec![0i8; n];
    for (i, out) in trits.iter_mut().enumerate() {
        let (b, j) = (i / block, i % block);
        let byte = raw[b * group + j % group];
        let code = (byte >> (6 - 2 * (j / group))) & 3;
        *out = match code {
            0 => -1,
            1 => 0,
            2 => 1,
            _ => return Err(format!("element {i}: code 3 is not a trit")),
        };
    }
    Ok((trits, scale))
}
