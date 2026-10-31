//! NVFP4 weights as NVIDIA modelopt 0.44.0 writes them (`hf_quant_config.json`: `quant_algo`
//! `W4A16_NVFP4`, `group_size` 16), read from nvidia/Qwen3.6-35B-A3B-NVFP4 rev 1355db6a (DRA-71).
//! The checkpoint layout is written up in `docs/NVFP4_CHECKPOINT_FORMAT.md`.
//!
//! One projection `{p}` of logical shape `[rows][k]` is stored as four tensors:
//! - `{p}.weight`, U8 `[rows][k/2]`: two e2m1 codes per byte. Column `2j` is in the low nibble of
//!   byte `j` and column `2j+1` in the high nibble.
//! - `{p}.weight_scale`, F8_E4M3 `[rows][k/16]`: one e4m3fn scale for each run of 16 consecutive
//!   columns of a row.
//! - `{p}.weight_scale_2`, F32 `[]`: one global scale for the whole projection. Each expert has
//!   its own, and gate and up have different ones.
//! - `{p}.input_scale`, F32 `[]`: an activation scale. W4A16 keeps activations at 16 bits, so it
//!   is not read here.
//!
//! `W[r][c] = e2m1(code(r, c)) * e4m3(weight_scale[r][c / 16]) * weight_scale_2`.
//!
//! This module is the host reference for the CUDApp W4A16 decode GEMV
//! (`omega_blackwell_codegen_w4`). [`Nvfp4Matrix::gemv_grouped`] factors the GEMV the same way the
//! kernel does: an f32 dot of e2m1 values with x for each 16-group, the group scale applied with
//! one fma, and the global scale applied once at the end. The kernel splits groups across lanes,
//! so its summation order differs from this sequential one; the two agree within rounding, not
//! bit for bit. The bit-exact emulation of the kernel's order lives beside the generator in
//! CUDApp.

use crate::qwen3_moe::fp8_e4m3fn_to_f32;
use crate::qwen3_moe_omega::TensorSource;

/// Columns sharing one e4m3 scale.
pub const NVFP4_GROUP: usize = 16;

/// e2m1 code to value. Bit 3 is the sign; the magnitudes of codes 0..=7 are
/// 0, 0.5, 1, 1.5, 2, 3, 4 and 6.
pub const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// The value of an e2m1 code (the low four bits of `code`).
#[inline]
pub fn e2m1(code: u8) -> f32 {
    E2M1[(code & 15) as usize]
}

/// Deliberately wrong dequantisations, used as negative controls. A check that accepts one of
/// these cannot tell the correct layout from a broken one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Nvfp4Mutant {
    /// The correct dequantisation.
    None,
    /// Column `2j` read from the high nibble (the opposite packing order).
    SwapNibbles,
    /// Each group scaled by its neighbour's scale (`group ^ 1`).
    NeighbourScale,
    /// The global scale `weight_scale_2` left out.
    NoGlobal,
}

/// One NVFP4 projection `[rows][k]`, validated when it is built.
#[derive(Clone, Debug)]
pub struct Nvfp4Matrix {
    rows: usize,
    k: usize,
    packed: Vec<u8>,
    scales: Vec<u8>,
    global: f32,
}

impl Nvfp4Matrix {
    /// Builds a matrix from raw tensors. Refuses an empty shape, a `k` that is not a whole
    /// number of 16-groups, wrong tensor lengths, a NaN or negative group scale, and a global
    /// scale that is not finite and positive.
    pub fn new(
        rows: usize,
        k: usize,
        packed: Vec<u8>,
        scales: Vec<u8>,
        global: f32,
    ) -> Result<Self, String> {
        if rows == 0 || k == 0 || !k.is_multiple_of(NVFP4_GROUP) {
            return Err(format!(
                "nvfp4 [{rows}][{k}]: k must be a positive multiple of {NVFP4_GROUP}"
            ));
        }
        if packed.len() != rows * k / 2 {
            return Err(format!(
                "nvfp4 [{rows}][{k}]: {} packed bytes, expected {}",
                packed.len(),
                rows * k / 2
            ));
        }
        let groups = k / NVFP4_GROUP;
        if scales.len() != rows * groups {
            return Err(format!(
                "nvfp4 [{rows}][{k}]: {} group scales, expected {}",
                scales.len(),
                rows * groups
            ));
        }
        if !(global.is_finite() && global > 0.0) {
            return Err(format!(
                "nvfp4 [{rows}][{k}]: global scale {global} is not finite and positive"
            ));
        }
        if let Some(i) = scales
            .iter()
            .position(|&s| s & 0x80 != 0 || s & 0x7f == 0x7f)
        {
            return Err(format!(
                "nvfp4 [{rows}][{k}]: group scale of row {} group {} is e4m3 code {:#04x} (negative or NaN)",
                i / groups,
                i % groups,
                scales[i]
            ));
        }
        Ok(Self {
            rows,
            k,
            packed,
            scales,
            global,
        })
    }

    /// Reads `{prefix}.weight`, `{prefix}.weight_scale` and `{prefix}.weight_scale_2` of a
    /// `[rows][k]` projection. Dtypes and shapes are checked by the source.
    pub fn read(
        source: &mut dyn TensorSource,
        prefix: &str,
        rows: usize,
        k: usize,
    ) -> Result<Self, String> {
        if !k.is_multiple_of(NVFP4_GROUP) {
            return Err(format!(
                "{prefix}: k {k} is not a multiple of {NVFP4_GROUP}"
            ));
        }
        let mut packed = vec![0u8; rows * k / 2];
        source.read_into(
            &format!("{prefix}.weight"),
            "U8",
            &[rows, k / 2],
            &mut packed,
        )?;
        let mut scales = vec![0u8; rows * (k / NVFP4_GROUP)];
        source.read_into(
            &format!("{prefix}.weight_scale"),
            "F8_E4M3",
            &[rows, k / NVFP4_GROUP],
            &mut scales,
        )?;
        let mut global = [0u8; 4];
        source.read_into(&format!("{prefix}.weight_scale_2"), "F32", &[], &mut global)?;
        Self::new(rows, k, packed, scales, f32::from_le_bytes(global))
            .map_err(|e| format!("{prefix}: {e}"))
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn k(&self) -> usize {
        self.k
    }

    /// The packed codes, `[rows][k/2]`, as stored.
    pub fn packed(&self) -> &[u8] {
        &self.packed
    }

    /// The e4m3 group scales, `[rows][k/16]`, as stored.
    pub fn scales(&self) -> &[u8] {
        &self.scales
    }

    /// `weight_scale_2`.
    pub fn global(&self) -> f32 {
        self.global
    }

    /// Bytes of weight data the projection streams: codes, group scales and the global.
    pub fn bytes(&self) -> usize {
        self.packed.len() + self.scales.len() + 4
    }

    /// The e2m1 code of `W[r][c]`.
    #[inline]
    pub fn code(&self, r: usize, c: usize) -> u8 {
        let b = self.packed[r * (self.k / 2) + c / 2];
        if c % 2 == 0 {
            b & 15
        } else {
            b >> 4
        }
    }

    /// The e4m3 scale of row `r`, group `g`.
    #[inline]
    pub fn group_scale(&self, r: usize, g: usize) -> f32 {
        fp8_e4m3fn_to_f32(self.scales[r * (self.k / NVFP4_GROUP) + g])
    }

    /// `W[r][c]` in f32. `e2m1 * e4m3` is exact in f32; the global scale rounds once.
    pub fn weight(&self, r: usize, c: usize) -> f32 {
        e2m1(self.code(r, c)) * self.group_scale(r, c / NVFP4_GROUP) * self.global
    }

    /// Row `r` dequantised into `out` (`k` values).
    pub fn dequant_row(&self, r: usize, out: &mut [f32]) {
        self.dequant_row_mutant(r, Nvfp4Mutant::None, out);
    }

    /// Row `r` dequantised with a deliberate fault (see [`Nvfp4Mutant`]).
    pub fn dequant_row_mutant(&self, r: usize, mutant: Nvfp4Mutant, out: &mut [f32]) {
        assert_eq!(
            out.len(),
            self.k,
            "dequant_row: output of {} values for k {}",
            out.len(),
            self.k
        );
        let groups = self.k / NVFP4_GROUP;
        let global = if mutant == Nvfp4Mutant::NoGlobal {
            1.0
        } else {
            self.global
        };
        for (c, o) in out.iter_mut().enumerate() {
            let code = if mutant == Nvfp4Mutant::SwapNibbles {
                self.code(r, c ^ 1)
            } else {
                self.code(r, c)
            };
            let g = if mutant == Nvfp4Mutant::NeighbourScale {
                ((c / NVFP4_GROUP) ^ 1).min(groups - 1)
            } else {
                c / NVFP4_GROUP
            };
            *o = e2m1(code) * self.group_scale(r, g) * global;
        }
    }

    /// `y[r] = sum_c W[r][c] * x[c]`, factored as the W4A16 kernel factors it: per 16-group
    /// `dot = sum e2m1 * x` (f32 fma chain), `acc = fma(dot, group scale, acc)`, and
    /// `y = acc * weight_scale_2` at the end.
    pub fn gemv_grouped(&self, x: &[f32], y: &mut [f32]) {
        assert_eq!(
            x.len(),
            self.k,
            "gemv: x of {} values for k {}",
            x.len(),
            self.k
        );
        assert_eq!(
            y.len(),
            self.rows,
            "gemv: y of {} values for {} rows",
            y.len(),
            self.rows
        );
        let groups = self.k / NVFP4_GROUP;
        for (r, out) in y.iter_mut().enumerate() {
            let mut acc = 0f32;
            for g in 0..groups {
                let mut dot = 0f32;
                for j in 0..NVFP4_GROUP {
                    let c = g * NVFP4_GROUP + j;
                    dot = e2m1(self.code(r, c)).mul_add(x[c], dot);
                }
                acc = dot.mul_add(self.group_scale(r, g), acc);
            }
            *out = acc * self.global;
        }
    }

    /// The f64 oracle: `y[r] = sum_c W[r][c] * x[c]` with exact weights, and the term scale
    /// `sum_c |W[r][c] * x[c]|` that bounds the rounding of any f32 evaluation.
    pub fn gemv_oracle(&self, x: &[f32]) -> (Vec<f64>, Vec<f64>) {
        assert_eq!(
            x.len(),
            self.k,
            "gemv: x of {} values for k {}",
            x.len(),
            self.k
        );
        let mut y = vec![0f64; self.rows];
        let mut scale = vec![0f64; self.rows];
        for r in 0..self.rows {
            let (mut s, mut a) = (0f64, 0f64);
            for (c, &xc) in x.iter().enumerate() {
                let w = e2m1(self.code(r, c)) as f64
                    * self.group_scale(r, c / NVFP4_GROUP) as f64
                    * self.global as f64;
                s += w * xc as f64;
                a += (w * xc as f64).abs();
            }
            y[r] = s;
            scale[r] = a;
        }
        (y, scale)
    }
}

/// One routed expert's NVFP4 projections: gate and up `[inter][hidden]`, down `[hidden][inter]`.
#[derive(Clone, Debug)]
pub struct Nvfp4Expert {
    pub gate: Nvfp4Matrix,
    pub up: Nvfp4Matrix,
    pub down: Nvfp4Matrix,
}

impl Nvfp4Expert {
    /// The tensor prefix of routed expert `expert` in layer `layer` of the Qwen3.6 text model.
    pub fn prefix(layer: usize, expert: usize) -> String {
        format!("model.language_model.layers.{layer}.mlp.experts.{expert}")
    }

    /// Reads `{prefix}.{gate,up,down}_proj`.
    pub fn read(
        source: &mut dyn TensorSource,
        prefix: &str,
        hidden: usize,
        inter: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            gate: Nvfp4Matrix::read(source, &format!("{prefix}.gate_proj"), inter, hidden)?,
            up: Nvfp4Matrix::read(source, &format!("{prefix}.up_proj"), inter, hidden)?,
            down: Nvfp4Matrix::read(source, &format!("{prefix}.down_proj"), hidden, inter)?,
        })
    }

    /// `down(silu(gate x) * up x)` with [`Nvfp4Matrix::gemv_grouped`] for each projection.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let inter = self.gate.rows();
        let (mut g, mut u) = (vec![0f32; inter], vec![0f32; inter]);
        self.gate.gemv_grouped(x, &mut g);
        self.up.gemv_grouped(x, &mut u);
        let h: Vec<f32> = g
            .iter()
            .zip(&u)
            .map(|(&g, &u)| g / (1.0 + (-g).exp()) * u)
            .collect();
        let mut y = vec![0f32; self.down.rows()];
        self.down.gemv_grouped(&h, &mut y);
        y
    }

    /// Bytes of weight data the expert streams.
    pub fn bytes(&self) -> usize {
        self.gate.bytes() + self.up.bytes() + self.down.bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen3_moe::bf16_to_f32;
    use std::collections::HashMap;

    struct Mem(HashMap<String, (String, Vec<usize>, Vec<u8>)>);

    impl TensorSource for Mem {
        fn read_into(
            &mut self,
            name: &str,
            dtype: &str,
            shape: &[usize],
            dst: &mut [u8],
        ) -> Result<(), String> {
            let (d, s, b) = self.0.get(name).ok_or_else(|| format!("{name}: missing"))?;
            if d != dtype || s != shape || b.len() != dst.len() {
                return Err(format!(
                    "{name}: {d} {s:?} where {dtype} {shape:?} was asked"
                ));
            }
            dst.copy_from_slice(b);
            Ok(())
        }
        fn dtype_of(&mut self, name: &str) -> Result<Option<String>, String> {
            Ok(self.0.get(name).map(|t| t.0.clone()))
        }
    }

    fn lcg(s: &mut u32) -> u32 {
        *s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *s
    }

    /// A random matrix: every code, scales from e4m3 codes 0x00..=0x7e (positive, never NaN).
    fn random(rows: usize, k: usize, seed: u32) -> Nvfp4Matrix {
        let mut s = seed;
        let packed = (0..rows * k / 2)
            .map(|_| (lcg(&mut s) >> 24) as u8)
            .collect();
        let scales = (0..rows * k / 16)
            .map(|_| ((lcg(&mut s) >> 24) % 0x7f) as u8)
            .collect();
        Nvfp4Matrix::new(rows, k, packed, scales, 0.0123).unwrap()
    }

    fn x_bf16(k: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..k)
            .map(|_| {
                let v = (lcg(&mut s) >> 8) as f32 / 16_777_216.0 * 2.0 - 1.0;
                bf16_to_f32((v.to_bits() >> 16) as u16)
            })
            .collect()
    }

    #[test]
    fn e2m1_table_is_sign_magnitude() {
        for c in 0u8..8 {
            assert_eq!(e2m1(c | 8), -e2m1(c));
            assert!(e2m1(c | 8).is_sign_negative());
        }
        assert_eq!(&E2M1[..8], &[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]);
    }

    #[test]
    fn low_nibble_is_the_even_column() {
        // byte 0x21: column 0 = code 1 (0.5), column 1 = code 2 (1.0); byte 0xf9: column 2 = code 9
        // (-0.5), column 3 = code 15 (-6). Scale e4m3 0x38 = 1.0, 0x40 = 2.0; global 2.
        let mut packed = vec![0u8; 16];
        packed[0] = 0x21;
        packed[1] = 0xf9;
        packed[8] = 0x07;
        let m = Nvfp4Matrix::new(1, 32, packed, vec![0x38, 0x40], 2.0).unwrap();
        assert_eq!(m.weight(0, 0), 1.0);
        assert_eq!(m.weight(0, 1), 2.0);
        assert_eq!(m.weight(0, 2), -1.0);
        assert_eq!(m.weight(0, 3), -12.0);
        assert_eq!(m.weight(0, 16), 6.0 * 2.0 * 2.0);
        assert_eq!(m.weight(0, 17), 0.0);
        let mut row = vec![0f32; 32];
        m.dequant_row_mutant(0, Nvfp4Mutant::SwapNibbles, &mut row);
        assert_eq!((row[0], row[1]), (2.0, 1.0));
        m.dequant_row_mutant(0, Nvfp4Mutant::NeighbourScale, &mut row);
        assert_eq!(row[0], 0.5 * 2.0 * 2.0);
        m.dequant_row_mutant(0, Nvfp4Mutant::NoGlobal, &mut row);
        assert_eq!(row[0], 0.5);
    }

    #[test]
    fn construction_refuses_bad_tensors() {
        let ok = |p: usize, s: usize, g: f32, sc: u8| {
            Nvfp4Matrix::new(2, 32, vec![0; p], vec![sc; s], g)
        };
        assert!(ok(32, 4, 1.0, 0x38).is_ok());
        assert!(ok(31, 4, 1.0, 0x38).is_err(), "short codes");
        assert!(ok(32, 3, 1.0, 0x38).is_err(), "short scales");
        assert!(ok(32, 4, 0.0, 0x38).is_err(), "zero global");
        assert!(ok(32, 4, -1.0, 0x38).is_err(), "negative global");
        assert!(ok(32, 4, f32::NAN, 0x38).is_err(), "NaN global");
        assert!(ok(32, 4, f32::INFINITY, 0x38).is_err(), "infinite global");
        assert!(ok(32, 4, 1.0, 0x7f).is_err(), "NaN scale");
        assert!(ok(32, 4, 1.0, 0xff).is_err(), "negative NaN scale");
        assert!(ok(32, 4, 1.0, 0xb8).is_err(), "negative scale");
        assert!(
            ok(32, 4, 1.0, 0x00).is_ok(),
            "a zero scale is a valid all-zero group"
        );
        assert!(
            Nvfp4Matrix::new(1, 24, vec![0; 12], vec![0x38; 1], 1.0).is_err(),
            "k 24"
        );
        assert!(
            Nvfp4Matrix::new(0, 32, vec![], vec![], 1.0).is_err(),
            "no rows"
        );
    }

    #[test]
    fn read_checks_names_dtypes_and_shapes() {
        let mut t = HashMap::new();
        t.insert(
            "p.weight".into(),
            ("U8".into(), vec![2, 16], vec![0x21u8; 32]),
        );
        t.insert(
            "p.weight_scale".into(),
            ("F8_E4M3".into(), vec![2, 2], vec![0x38u8; 4]),
        );
        t.insert(
            "p.weight_scale_2".into(),
            ("F32".into(), vec![], 0.5f32.to_le_bytes().to_vec()),
        );
        let mut src = Mem(t);
        let m = Nvfp4Matrix::read(&mut src, "p", 2, 32).unwrap();
        assert_eq!((m.rows(), m.k(), m.global()), (2, 32, 0.5));
        assert_eq!(m.weight(1, 31), 0.5);
        assert_eq!(m.bytes(), 32 + 4 + 4);
        assert!(
            Nvfp4Matrix::read(&mut src, "p", 4, 16).is_err(),
            "wrong shape"
        );
        assert!(
            Nvfp4Matrix::read(&mut src, "q", 2, 32).is_err(),
            "missing tensor"
        );
        src.0.get_mut("p.weight_scale").unwrap().0 = "BF16".into();
        assert!(
            Nvfp4Matrix::read(&mut src, "p", 2, 32).is_err(),
            "wrong dtype"
        );
    }

    #[test]
    fn grouped_gemv_matches_the_f64_oracle() {
        for &(rows, k) in &[(64usize, 2048usize), (96, 512)] {
            let m = random(rows, k, 7 + k as u32);
            let x = x_bf16(k, 11);
            let mut y = vec![0f32; rows];
            m.gemv_grouped(&x, &mut y);
            let (yo, ts) = m.gemv_oracle(&x);
            for r in 0..rows {
                let rel = (y[r] as f64 - yo[r]).abs() / (ts[r] + 1e-30);
                assert!(
                    rel < 1e-5,
                    "[{rows}][{k}] row {r}: {} vs {} (rel {rel:e})",
                    y[r],
                    yo[r]
                );
            }
        }
    }

    #[test]
    fn mutants_change_the_dequantisation() {
        let m = random(8, 512, 3);
        let mut a = vec![0f32; 512];
        let mut b = vec![0f32; 512];
        for mutant in [
            Nvfp4Mutant::SwapNibbles,
            Nvfp4Mutant::NeighbourScale,
            Nvfp4Mutant::NoGlobal,
        ] {
            m.dequant_row(5, &mut a);
            m.dequant_row_mutant(5, mutant, &mut b);
            assert_ne!(a, b, "{mutant:?} left row 5 unchanged");
        }
    }

    #[test]
    fn expert_forward_is_down_of_silu_gate_times_up() {
        let e = Nvfp4Expert {
            gate: random(32, 64, 1),
            up: random(32, 64, 2),
            down: random(64, 32, 3),
        };
        let x = x_bf16(64, 4);
        let y = e.forward(&x);
        let (g, _) = e.gate.gemv_oracle(&x);
        let (u, _) = e.up.gemv_oracle(&x);
        let h: Vec<f32> = g
            .iter()
            .zip(&u)
            .map(|(&g, &u)| (g / (1.0 + (-g).exp()) * u) as f32)
            .collect();
        let (yo, _) = e.down.gemv_oracle(&h);
        let num: f64 = y
            .iter()
            .zip(&yo)
            .map(|(&a, &b)| (a as f64 - b).powi(2))
            .sum();
        let den: f64 = yo.iter().map(|b| b * b).sum();
        assert!(
            (num / den).sqrt() < 1e-5,
            "expert forward rel-RMS {:e}",
            (num / den).sqrt()
        );
        assert_eq!(e.bytes(), 3 * (32 * 32 + 32 * 4 + 4));
    }

    // --- the real checkpoint -------------------------------------------------------------

    fn rel_rms(a: &[f32], b: &[f32]) -> f64 {
        let num: f64 = a
            .iter()
            .zip(b)
            .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
            .sum();
        let den: f64 = b.iter().map(|&b| (b as f64).powi(2)).sum();
        (num / den).sqrt()
    }

    fn dense(m: &Nvfp4Matrix, mutant: Nvfp4Mutant) -> Vec<f32> {
        let mut d = vec![0f32; m.rows() * m.k()];
        for (r, row) in d.chunks_exact_mut(m.k()).enumerate() {
            m.dequant_row_mutant(r, mutant, row);
        }
        d
    }

    /// The FP8 checkpoint's copy of the same projection, dequantised to f32.
    fn fp8_dense(src: &mut dyn TensorSource, name: &str, n: usize, k: usize) -> Vec<f32> {
        match crate::qwen38::read_matrix(src, name, n, k).unwrap() {
            crate::qwen38::Matrix::Fp8 { w, scales, .. } => {
                let lut = crate::qwen3_moe::fp8_table();
                let kb = k / crate::qwen3_moe::FP8_BLOCK;
                (0..n * k)
                    .map(|i| {
                        let (r, c) = (i / k, i % k);
                        lut[w[i] as usize] * bf16_to_f32(scales[(r / 128) * kb + c / 128])
                    })
                    .collect()
            }
            crate::qwen38::Matrix::Bf16 { w, .. } => w.into_iter().map(bf16_to_f32).collect(),
        }
    }

    fn matmul_f32(d: &[f32], k: usize, x: &[f32]) -> Vec<f32> {
        d.chunks_exact(k)
            .map(|row| row.iter().zip(x).fold(0f32, |s, (&w, &x)| w.mul_add(x, s)))
            .collect()
    }

    /// Writes one projection and its oracle for the CUDApp `gpu_w4gemm_test --real` check:
    /// u32 rows, u32 k, f32 global, codes, scales, x (k f32), y (rows f64), term scale (rows f64).
    fn dump(path: &std::path::Path, m: &Nvfp4Matrix, x: &[f32]) {
        let (yo, ts) = m.gemv_oracle(x);
        let mut b = Vec::new();
        b.extend_from_slice(&(m.rows() as u32).to_le_bytes());
        b.extend_from_slice(&(m.k() as u32).to_le_bytes());
        b.extend_from_slice(&m.global().to_le_bytes());
        b.extend_from_slice(m.packed());
        b.extend_from_slice(m.scales());
        x.iter().for_each(|v| b.extend_from_slice(&v.to_le_bytes()));
        yo.iter()
            .for_each(|v| b.extend_from_slice(&v.to_le_bytes()));
        ts.iter()
            .for_each(|v| b.extend_from_slice(&v.to_le_bytes()));
        std::fs::write(path, b).unwrap();
    }

    /// Predeclared bounds (DRA-71): the grouped GEMV and a plain f32 matmul of the dequantised
    /// matrix within 1e-5 of the term scale of the f64 oracle; the NVFP4 projection within
    /// rel-RMS 0.25 of the FP8 checkpoint's copy; the nibble-swapped mutant at least 0.9 away,
    /// the neighbour-scale mutant at least 0.3 and the mutant without the global at least 10;
    /// the expert's output within rel-RMS 0.35 of the FP8 expert's.
    #[test]
    #[ignore = "needs AIEN_QWEN36_NVFP4_DIR (nvidia/Qwen3.6-35B-A3B-NVFP4) and AIEN_QWEN36_DIR (Qwen/Qwen3.6-35B-A3B-FP8)"]
    fn real_experts_match_host_matmul_and_the_fp8_checkpoint() {
        use crate::qwen3_moe_omega::ShardSource;
        let nv_dir = std::env::var("AIEN_QWEN36_NVFP4_DIR").expect("AIEN_QWEN36_NVFP4_DIR");
        let f8_dir = std::env::var("AIEN_QWEN36_DIR").expect("AIEN_QWEN36_DIR");
        let dump_dir = std::env::var("AIEN_NVFP4_DUMP_DIR").ok();
        let mut nv = ShardSource::open(std::path::Path::new(&nv_dir)).unwrap();
        let mut f8 = ShardSource::open(std::path::Path::new(&f8_dir)).unwrap();
        let (hidden, inter) = (2048usize, 512usize);
        for &(layer, expert) in &[(0usize, 0usize), (20, 137), (39, 255)] {
            let prefix = Nvfp4Expert::prefix(layer, expert);
            let e = Nvfp4Expert::read(&mut nv, &prefix, hidden, inter).unwrap();
            let x = x_bf16(hidden, 17 + layer as u32);
            let mut h = vec![0f32; inter];
            {
                let (g, u) = (e.gate.gemv_oracle(&x).0, e.up.gemv_oracle(&x).0);
                for i in 0..inter {
                    h[i] = (g[i] / (1.0 + (-g[i]).exp()) * u[i]) as f32;
                }
            }
            for (name, m, xin) in [
                ("gate_proj", &e.gate, &x),
                ("up_proj", &e.up, &x),
                ("down_proj", &e.down, &h),
            ] {
                let (yo, ts) = m.gemv_oracle(xin);
                let mut yg = vec![0f32; m.rows()];
                m.gemv_grouped(xin, &mut yg);
                let d = dense(m, Nvfp4Mutant::None);
                let ym = matmul_f32(&d, m.k(), xin);
                let worst = |y: &[f32]| {
                    (0..m.rows())
                        .map(|r| (y[r] as f64 - yo[r]).abs() / (ts[r] + 1e-30))
                        .fold(0f64, f64::max)
                };
                let (wg, wm) = (worst(&yg), worst(&ym));
                let fp = fp8_dense(&mut f8, &format!("{prefix}.{name}.weight"), m.rows(), m.k());
                let rr = rel_rms(&d, &fp);
                let swap = rel_rms(&dense(m, Nvfp4Mutant::SwapNibbles), &fp);
                let nbr = rel_rms(&dense(m, Nvfp4Mutant::NeighbourScale), &fp);
                let nog = rel_rms(&dense(m, Nvfp4Mutant::NoGlobal), &fp);
                println!(
                    "L{layer} E{expert} {name} [{}][{}] global {:e}: grouped GEMV vs f64 oracle {wg:.2e}, f32 matmul vs oracle {wm:.2e} (of the term scale); rel-RMS vs FP8 checkpoint {rr:.4}, swapped nibbles {swap:.4}, neighbour scale {nbr:.4}, no global {nog:.3e}",
                    m.rows(), m.k(), m.global()
                );
                assert!(
                    wg < 1e-5 && wm < 1e-5,
                    "{name}: f32 evaluations off the oracle"
                );
                assert!(
                    rr <= 0.25,
                    "{name}: NVFP4 {rr} away from the FP8 checkpoint"
                );
                assert!(swap >= 0.9, "{name}: nibble-swap control only {swap} away");
                assert!(
                    nbr >= 0.3,
                    "{name}: neighbour-scale control only {nbr} away"
                );
                assert!(nog >= 10.0, "{name}: no-global control only {nog} away");
                if let Some(dir) = &dump_dir {
                    dump(
                        &std::path::Path::new(dir).join(format!("l{layer}e{expert}-{name}.w4ref")),
                        m,
                        xin,
                    );
                }
            }
            let y = e.forward(&x);
            let fg = fp8_dense(
                &mut f8,
                &format!("{prefix}.gate_proj.weight"),
                inter,
                hidden,
            );
            let fu = fp8_dense(&mut f8, &format!("{prefix}.up_proj.weight"), inter, hidden);
            let fd = fp8_dense(
                &mut f8,
                &format!("{prefix}.down_proj.weight"),
                hidden,
                inter,
            );
            let (g, u) = (matmul_f32(&fg, hidden, &x), matmul_f32(&fu, hidden, &x));
            let hf: Vec<f32> = g
                .iter()
                .zip(&u)
                .map(|(&g, &u)| g / (1.0 + (-g).exp()) * u)
                .collect();
            let yf = matmul_f32(&fd, inter, &hf);
            let rr = rel_rms(&y, &yf);
            println!("L{layer} E{expert} expert output: rel-RMS vs the FP8 expert {rr:.4} ({} B NVFP4 vs {} B FP8)", e.bytes(), 3 * inter * hidden + 3 * 4 * 16 * 2);
            assert!(
                rr <= 0.35,
                "L{layer} E{expert}: expert output {rr} away from FP8"
            );
        }
    }

    /// The NVFP4 head against the FP8 checkpoint's bf16 head on a sample of rows: rel-RMS of the
    /// weights at most 0.25, nibble-swap control at least 0.9 (predeclared).
    #[test]
    #[ignore = "needs AIEN_QWEN36_NVFP4_DIR and AIEN_QWEN36_DIR; reads the 0.29 GB NVFP4 head and the 1 GB bf16 head"]
    fn real_head_matches_the_bf16_head() {
        use crate::qwen3_moe_omega::ShardSource;
        let nv_dir = std::env::var("AIEN_QWEN36_NVFP4_DIR").expect("AIEN_QWEN36_NVFP4_DIR");
        let f8_dir = std::env::var("AIEN_QWEN36_DIR").expect("AIEN_QWEN36_DIR");
        let mut nv = ShardSource::open(std::path::Path::new(&nv_dir)).unwrap();
        let mut f8 = ShardSource::open(std::path::Path::new(&f8_dir)).unwrap();
        let (vocab, hidden) = (248_320usize, 2048usize);
        let m = Nvfp4Matrix::read(&mut nv, "lm_head", vocab, hidden).unwrap();
        let bf = fp8_dense(&mut f8, "lm_head.weight", vocab, hidden);
        let (mut a, mut b, mut s) = (Vec::new(), Vec::new(), Vec::new());
        let mut row = vec![0f32; hidden];
        for r in (0..vocab).step_by(997) {
            m.dequant_row(r, &mut row);
            a.extend_from_slice(&row);
            m.dequant_row_mutant(r, Nvfp4Mutant::SwapNibbles, &mut row);
            s.extend_from_slice(&row);
            b.extend_from_slice(&bf[r * hidden..(r + 1) * hidden]);
        }
        let (rr, swap) = (rel_rms(&a, &b), rel_rms(&s, &b));
        println!(
            "lm_head [{vocab}][{hidden}] global {:e}, {} B: {} sampled rows, rel-RMS vs bf16 head {rr:.4}, swapped nibbles {swap:.4}",
            m.global(),
            m.bytes(),
            a.len() / hidden
        );
        assert!(rr <= 0.25, "head {rr} away from bf16");
        assert!(swap >= 0.9, "nibble-swap control only {swap} away");
    }
}
