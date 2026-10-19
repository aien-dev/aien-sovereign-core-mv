//! The storage map of [`super::Qwen38Omega`] (sc #678): every block the upload, the decode graph
//! and the prefill make, from the config, with the omega source that fixes each size. One entry
//! per block; `call` names the allocating call the load probes, so plan and measurement meet
//! call by call. `need_bytes` of the binaries is [`plan`]'s peak.
//!
//! Every omega driver allocation is rounded up to a 4 KiB page (`page_up`, omega
//! `src/omega_gpu_session.c` and `omega_gpu_pf_alloc_dev`); an FP8 tensor is two allocations
//! (weights, f32 scales; `omega_gpu_fp8_tensor_upload`).
use aien_omega_q38 as q;

use crate::qwen38::Qwen38Config;
use crate::storage_map::{Lifetime, Space, StorageMap};

const PAGE: u64 = 4096;
const BLOCK: u64 = 128;

/// omega `page_up`: whole 4 KiB pages, at least one.
pub const fn page(b: u64) -> u64 {
    let r = b.div_ceil(PAGE) * PAGE;
    if r < PAGE {
        PAGE
    } else {
        r
    }
}

/// omega `al` (`src/omega_gpu_q38_graph.c`, `pf_layout`): 256-byte sections.
const fn al(b: u64) -> u64 {
    b.div_ceil(256) * 256
}

/// An FP8 tensor `[n][k]` as resident: weights plus one f32 scale per 128 x 128 block.
pub const fn fp8(n: u64, k: u64) -> u64 {
    page(n * k) + page((n / BLOCK) * (k / BLOCK) * 4)
}

// omega constants (file: line of the define in omega qwen38/prefill-graph 6e24f7fd).
/// `OMEGA_GDN_PAR_BYTES` (omega_blackwell_codegen_gdn.h:45).
const GDN_PAR: u64 = 164_736;
/// `OMEGA_GDN_TAIL_BYTES` = 48 heads x 384 slots x 16 (omega_blackwell_codegen_gdn.h:49-50).
const GDN_TAIL: u64 = 48 * 384 * 16;
/// `OMEGA_GDN_STATE_BYTES` = 48 heads x 128 x 128 f32 (omega_blackwell_codegen_gdn.h:52-53).
const GDN_STATE: u64 = 48 * 128 * 128 * 4;
/// The DeltaNet launch buffer `LB_BYTES` = qkv 10240 | z 6144 | ab 48 x 8 | out 6144 (omega_gpu_gdn_api.c:18-22).
const GDN_LB: u64 = 10240 * 4 + 6144 * 4 + 48 * 8 + 6144 * 4;
/// `OMEGA_DATTN256_ROW_BYTES` and the plane slack `MAX_DEPTH 4 x 4 x row + 64` (omega_blackwell_codegen_attn256.h:55-60).
const ATT_ROW: u64 = 512;
const ATT_SLACK: u64 = 4 * 4 * ATT_ROW + 64;
/// `OMEGA_DATTN256_ROPE_ROW_BYTES`, `OMEGA_DATTN256_PARTIAL_WORDS`, `OMEGA_DATTN256_CHUNK`.
const ATT_ROPE_ROW: u64 = 1024;
const ATT_PARTIAL_WORDS: u64 = 1548;
const ATT_CHUNK: u64 = 256;
/// `OMEGA_GPU_CMDLIST_MAX_LAUNCHES` and its per-launch bank and QMD (omega_gpu_cmdlist.h:44-46).
const CL_CAP: u64 = 2048;
const CL_BANK: u64 = 0x400;
const CL_QMD: u64 = 0x400;
/// `OMEGA_DEC_ARGMAX_SPAN` (omega_gpu_decode_graph.h:46).
const ARGMAX_SPAN: u64 = 2048;
/// `OMEGA_PATTN256_QF_PAIR_BYTES`, `OMEGA_PATTN256_OUT_WORDS` (omega_blackwell_codegen_pattn256.h:38-44).
const PATTN_QF_PAIR: u64 = 24576;
const PATTN_OUT_WORDS: u64 = 6144;
/// `OMEGA_GDN_REC_BYTES`, `OMEGA_GDN_SCAN_CHUNK` (omega_blackwell_codegen_gdn.h:105,122).
const GDN_REC: u64 = 1552;
const GDN_SCAN_CHUNK: u64 = 8;
/// `OMEGA_GPU_MATMUL_TILE_M/K/N` (omega_gpu_matmul_api.h:46-48).
const MM_TILE_M: u64 = 16;
const MM_TILE_K: u64 = 16;
const MM_TILE_N: u64 = 8;
/// `TILE_BYTES` of one FP8 GEMM tile record (aien-omega-pf).
const TILE: u64 = aien_omega_pf::TILE_BYTES as u64;

/// Measured allowances (no formula on this side: kernel code sizes come out of omega's code
/// generator, the session and driver maps out of the platform). Each is the figure measured on
/// the chip, exact (first run: evidence/f6-storage-map-20261019T0156Z/storage-l4.storage-map-end.txt,
/// omega archive 373d122c, qwen38/prefill-graph 6e24f7fd). A different omega build that
/// changes one of them fails the storage check by name: re-measure, do not widen.
pub const SESSION_ALLOWANCE: u64 = 1_323_008;
pub const GRAPH_CODE_ALLOWANCE: u64 = 69_632;
pub const PREFILL_CODE_ALLOWANCE: u64 = 126_976;
pub const HEAD_CODE_ALLOWANCE: u64 = 8_192;
pub const DECODE_FIRST_ALLOWANCE: u64 = 61_440;
/// Process heap beyond the named vectors: the safetensors index, read buffers, rayon arenas
/// keeping freed chunk vectors (glibc keeps blocks under its 32 MiB mmap threshold). Measured:
/// 54,784,000 bytes unaccounted after a 256-token chunk run and 1.2 MB over the plan at the
/// load peak (evidence/f6-storage-map-20261019T0156Z, storage-l4-c256); 64 MiB.
pub const HEAP_ALLOWANCE: u64 = 64 << 20;
/// Drift a call with a measured allowance may show before the check fails (kernel code is
/// made per chunk shape: a 300-token prompt in chunks of 128 made one more 4 KiB code page
/// than 512 in chunks of 256). Calls with only derived entries must match exactly.
pub const ALLOWANCE_SLACK: u64 = 64 << 10;

/// Call names (the load and the prefill probe these).
pub const CALL_LM_HEAD: &str = "lm_head upload (first omega call: opens the session)";
pub const CALL_GDN: &str = "gdn alloc";
pub const CALL_ATTN: &str = "attn256 cache alloc";
pub const CALL_GRAPH: &str = "q38 graph create";
pub const CALL_PF_BUFS: &str = "prefill buffers";
pub const CALL_PF_FIRST: &str = "first prefill (omega's lazy scratch and kernels)";
pub const CALL_DECODE_FIRST: &str = "first decode step";
pub const CALL_HEAD_FIRST: &str = "first host-path lm_head GEMV";

/// The per-layer entry names (call = name).
pub fn layer_names(l: usize, attention: bool) -> Vec<String> {
    let mut v = if attention {
        vec![
            format!("L{l:02} attn q|gate|k|v fp8"),
            format!("L{l:02} attn o_proj fp8"),
        ]
    } else {
        vec![
            format!("L{l:02} gdn qkv|z fp8"),
            format!("L{l:02} gdn a|b bf16 (graph)"),
            format!("L{l:02} gdn a|b bf16 (prefill matmul)"),
            format!("L{l:02} gdn out_proj fp8"),
        ]
    };
    v.push(format!("L{l:02} mlp gate|up fp8"));
    v.push(format!("L{l:02} mlp down fp8"));
    v
}

/// The whole model resident, its graph and (with `chunk`) the prefill for chunks of that many
/// tokens, `bm` the FP8 GEMM tile rows (`omega_gpu_pf_qgemm_bm`).
pub fn plan(c: &Qwen38Config, max_pos: usize, chunk: Option<(usize, usize)>) -> StorageMap {
    let mut s = StorageMap::new();
    let (h, v, i) = (c.hidden as u64, c.vocab as u64, c.intermediate as u64);
    let n_attn = c.full_attention.iter().filter(|&&a| a).count() as u64;
    let n_gdn = c.layers as u64 - n_attn;
    let mp = max_pos as u64;
    let w = Lifetime::Weights;
    let d = Space::Driver;

    // In load order: the peak is a running maximum over the entries as registered.
    let hp = Space::Heap;
    s.add(
        "embedding bf16 [vocab][hidden]",
        hp,
        v * h * 2,
        "Qwen38Omega.embed",
        &["load"],
        w,
    );
    let e = s.add(
        "process heap: checkpoint index, read buffers, allocator slack",
        hp,
        HEAP_ALLOWANCE,
        "process",
        &["load", "prefill"],
        Lifetime::Engine,
    );
    e.derived = false;
    s.add(
        "lm_head bf16 [vocab][hidden]",
        d,
        page(v * h * 2),
        "Qwen38Omega.lm_head",
        &["load"],
        w,
    )
    .call = CALL_LM_HEAD.into();
    let e = s.add(
        "omega session (pushbuffer, banks, marker, QMD, driver maps)",
        d,
        SESSION_ALLOWANCE,
        "omega session",
        &["omega session open"],
        Lifetime::Engine,
    );
    e.call = CALL_LM_HEAD.into();
    e.derived = false;
    // The host copy is still held while its upload runs: counted with the uploaded twin.
    s.add(
        "lm_head host copy (read before the upload)",
        hp,
        v * h * 2,
        "load_at",
        &["load"],
        w,
    )
    .transient = true;

    if n_gdn > 0 {
        let g = "Qwen38Omega.gdn";
        let parts: [(&str, u64, &[&str], Lifetime); 4] = [
            (
                "gdn params (conv, A_log, dt_bias, norm)",
                page(n_gdn * GDN_PAR),
                &["load (set_layer)"],
                w,
            ),
            (
                "gdn conv tails",
                page(n_gdn * GDN_TAIL),
                &["gdn decode step", "gdn prefill post"],
                Lifetime::Conversation,
            ),
            (
                "gdn states (48 x 128x128 f32 per layer)",
                page(n_gdn * GDN_STATE),
                &["gdn decode step", "gdn prefill scan"],
                Lifetime::Conversation,
            ),
            (
                "gdn launch buffer",
                page(GDN_LB),
                &["gdn decode step"],
                Lifetime::Token,
            ),
        ];
        for (name, b, wr, lt) in parts {
            s.add(name, d, b, g, wr, lt).call = CALL_GDN.into();
        }
    }
    if n_attn > 0 {
        let a = "Qwen38Omega.attn";
        let plane = 4 * mp * ATT_ROW + ATT_SLACK;
        let splits = mp / ATT_CHUNK;
        let parts: [(&str, u64, &[&str], Lifetime); 4] = [
            (
                "attn256 KV planes (K|V per layer)",
                n_attn * page(2 * plane),
                &["attn256 decode append", "attn256 prefill append"],
                Lifetime::Conversation,
            ),
            (
                "attn256 RoPE rows",
                page(mp * ATT_ROPE_ROW),
                &["attn256 cache alloc"],
                Lifetime::Cache,
            ),
            (
                "attn256 q/k norms",
                page(n_attn * 2 * 256 * 4),
                &["load (set_norms)"],
                w,
            ),
            (
                "attn256 decode scratch (partials, counters, qkv, out)",
                page(4 * splits * ATT_PARTIAL_WORDS * 4)
                    + page(64)
                    + page(q::ATT_QKV as u64 * 4)
                    + page(q::MIXER_OUT as u64 * 4),
                &["attn256 decode"],
                Lifetime::Token,
            ),
        ];
        for (name, b, wr, lt) in parts {
            s.add(name, d, b, a, wr, lt).call = CALL_ATTN.into();
        }
    }

    for l in 0..c.layers {
        let attention = c.full_attention[l];
        let names = layer_names(l, attention);
        let o = "Qwen38Omega.layers";
        let mo = (c.heads * c.head_dim) as u64;
        let bytes: Vec<u64> = if attention {
            vec![fp8(2 * mo + 2 * c.kv_dim() as u64, h), fp8(h, mo)]
        } else {
            let ab_n = 2 * c.lin_v_heads as u64;
            let mm = h.div_ceil(MM_TILE_K) * MM_TILE_K * ab_n.div_ceil(MM_TILE_N) * MM_TILE_N * 2;
            vec![
                fp8((c.conv_dim() + c.lin_value_dim()) as u64, h),
                page(ab_n * h * 2),
                page(mm),
                fp8(h, c.lin_value_dim() as u64),
            ]
        };
        for (name, b) in names.iter().zip(bytes) {
            s.add(name.clone(), d, b, o, &["load"], w);
        }
        let n = names.len();
        s.add(names[n - 2].clone(), d, fp8(2 * i, h), o, &["load"], w);
        s.add(names[n - 1].clone(), d, fp8(h, i), o, &["load"], w);
    }
    // One layer's host copies while it uploads, the largest being gate|up: both parts as read
    // and the fused copy (weights and bf16 scales). Freed before the next layer reads.
    let gu = 2 * i * h + (2 * i / BLOCK) * (h / BLOCK) * 2;
    s.add(
        "layer upload host copies (gate|up parts and fused)",
        hp,
        2 * gu,
        "load_at",
        &["load"],
        w,
    )
    .transient = true;

    // The decode graph: one buffer at fixed 256-byte sections (omega_gpu_q38_graph.c:125-130).
    let am_ctas = v.div_ceil(ARGMAX_SPAN);
    let layers = c.layers as u64;
    let gbuf = [
        h * 4,
        h * 4,
        q::GDN_MIX as u64 * 4,
        q::AB as u64 * 4,
        q::ATT_MIX as u64 * 4,
        q::MIXER_OUT as u64 * 4,
        h * 4,
        2 * i * 4,
        i * 4,
        h * 4,
        am_ctas * ARGMAX_SPAN * 4,
        ARGMAX_SPAN * 4,
        ARGMAX_SPAN * 4,
        16,
        layers * 2 * h * 4,
        h * 4,
        layers * 4 * 16,
        q::ATT_MIX as u64 * 4,
    ]
    .iter()
    .map(|&b| al(b))
    .sum::<u64>();
    let g = "Qwen38Omega.graph";
    s.add(
        "q38 graph buffer (activations, norms, tables)",
        d,
        page(gbuf),
        g,
        &["decode graph"],
        Lifetime::Token,
    )
    .call = CALL_GRAPH.into();
    s.add(
        "q38 graph command list (banks, QMDs, semaphores, stamps, scratch)",
        d,
        page(CL_CAP * CL_BANK)
            + page(CL_CAP * CL_QMD)
            + page(CL_CAP * 16)
            + page((CL_CAP + 1) * 16)
            + page(0x4000),
        g,
        &["decode graph record"],
        Lifetime::Engine,
    )
    .call = CALL_GRAPH.into();
    let e = s.add(
        "q38 graph kernel code",
        d,
        GRAPH_CODE_ALLOWANCE,
        g,
        &["omega code generator"],
        Lifetime::Engine,
    );
    e.call = CALL_GRAPH.into();
    e.derived = false;
    let e = s.add(
        "first decode step (lazy omega allocations)",
        d,
        DECODE_FIRST_ALLOWANCE,
        g,
        &["omega"],
        Lifetime::Engine,
    );
    e.call = CALL_DECODE_FIRST.into();
    e.derived = false;

    s.add(
        "bf16 GEMV staging (lm_head x, y)",
        d,
        page(h * 4) + page(v * 4),
        "omega bf16 gemv",
        &["head_logits"],
        Lifetime::Token,
    )
    .call = CALL_HEAD_FIRST.into();
    let e = s.add(
        "lm_head GEMV kernel code",
        d,
        HEAD_CODE_ALLOWANCE,
        "omega bf16 gemv",
        &["omega code generator"],
        Lifetime::Engine,
    );
    e.call = CALL_HEAD_FIRST.into();
    e.derived = false;

    // Host: the norms the prefill applies.
    s.add(
        "norms f32 (input, post-attention per layer; final)",
        hp,
        layers * 2 * h * 4 + h * 4,
        "Qwen38Omega.norms",
        &["load"],
        w,
    );

    if let Some((m, bm)) = chunk {
        prefill_entries(&mut s, c, m as u64, bm as u64);
    }
    s
}

/// Puts the prefill's blocks for chunks of `m` tokens into `s`, replacing an earlier sizing
/// (its buffers were freed before the new ones were made, so their measurement starts over;
/// omega's lazy scratch grows in place and keeps its measurement).
pub fn set_prefill(s: &mut StorageMap, c: &Qwen38Config, m: usize, bm: usize) {
    let mut fresh = StorageMap::new();
    prefill_entries(&mut fresh, c, m as u64, bm as u64);
    s.entries.retain(|e| fresh.get(&e.name).is_none());
    s.measured.retain(|(call, _)| call != CALL_PF_BUFS);
    s.entries.extend(fresh.entries);
}

/// The prefill's blocks for chunks of `m` tokens.
fn prefill_entries(s: &mut StorageMap, c: &Qwen38Config, m: u64, bm: u64) {
    let h = c.hidden as u64;
    let (d, t) = (Space::Driver, Lifetime::Token);
    let kq = q::INTER as u64;
    let max_tiles = m.div_ceil(bm) + 1;
    let p = "Qwen38Omega.pf";
    let bufs: [(&str, u64, &[&str]); 7] = [
        (
            "prefill xin HOST f32 [m][6144]",
            page(m * q::MIXER_OUT as u64 * 4),
            &["host norm", "host gate"],
        ),
        (
            "prefill xq DEV e4m3 [m+bm][17408]",
            page((m + bm) * kq),
            &["pf quant"],
        ),
        (
            "prefill xs DEV f32 scales [m+bm][136]",
            page((m + bm) * (kq / BLOCK) * 4),
            &["pf quant"],
        ),
        (
            "prefill mix HOST f32 [m][16384]",
            page(m * q::GDN_MIX as u64 * 4),
            &["pf qgemm (mixer)"],
        ),
        (
            "prefill act DEV bf16 [m][17408]",
            page(m * kq * 2),
            &["pf qgemm (gate|up SwiGLU)"],
        ),
        (
            "prefill out HOST f32 [m][5120]",
            page(m * h * 4),
            &["pf qgemm (o_proj, down)"],
        ),
        (
            "prefill GEMM tile table",
            page(max_tiles * TILE),
            &["host tiles"],
        ),
    ];
    for (name, b, wr) in bufs {
        s.add(name, d, b, p, wr, t).call = CALL_PF_BUFS.into();
    }
    s.add(
        "prefill gather table",
        Space::Heap,
        q::gather_table(c.rotary_dim).len() as u64 * 4,
        p,
        &["pf open"],
        Lifetime::Cache,
    );

    // omega's scratch made on first use, sized by the largest chunk (omega_gpu_session_scratch
    // grows to the high water and keeps it).
    let t8 = m.div_ceil(GDN_SCAN_CHUNK) * GDN_SCAN_CHUNK;
    let gdn_pf = al(m * q::GDN_QKV as u64 * 4)
        + al(m * q::MIXER_OUT as u64 * 4)
        + al(m * 48 * 8)
        + 2 * al(m * q::MIXER_OUT as u64 * 4)
        + al(t8 * 48 * GDN_REC);
    let mmp = m.div_ceil(MM_TILE_M) * MM_TILE_M;
    // (omega's 64 MiB host-entry stage, PF_STAGE_BYTES, is made only by omega_gpu_pf_upload /
    // download, which this path does not call: not planned, and the chip run shows it absent.)
    let lazy: [(&str, u64, &str, &[&str]); 4] = [
        (
            "gdn prefill scratch (qkv, z, ab, out, o, records)",
            page(gdn_pf),
            "omega gdn",
            &["gdn prefill"],
        ),
        (
            "attn256 prefill staging (qf, qkv, out)",
            page(m.div_ceil(2) * PATTN_QF_PAIR)
                + page(m * q::ATT_QKV as u64 * 4)
                + page(m * PATTN_OUT_WORDS * 4),
            "omega prefill attn256",
            &["attn256 prefill"],
        ),
        (
            "resident matmul staging (a|b: A bf16, C f32)",
            page(mmp * h * 2)
                + page(mmp * (2 * c.lin_v_heads as u64).div_ceil(MM_TILE_N) * MM_TILE_N * 4),
            "omega matmul",
            &["a|b matmul"],
        ),
        (
            "prefill kernel code",
            PREFILL_CODE_ALLOWANCE,
            "omega",
            &["omega code generator"],
        ),
    ];
    for (name, b, owner, wr) in lazy {
        let e = s.add(
            name,
            d,
            b,
            owner,
            wr,
            if name.ends_with("code") {
                Lifetime::Engine
            } else {
                t
            },
        );
        e.call = CALL_PF_FIRST.into();
        e.derived = !name.ends_with("code");
    }

    // Host vectors of one chunk (prefill.rs prefill_chunk), the DeltaNet layer's being the
    // largest: x, xn, att, mix, qkv, z, ab live together.
    let per_token = (h
        + h
        + q::MIXER_OUT as u64
        + q::GDN_MIX as u64
        + q::GDN_QKV as u64
        + q::MIXER_OUT as u64
        + q::AB as u64)
        * 4;
    let e = s.add(
        "prefill chunk host vectors (x, xn, att, mix, qkv, z, ab)",
        Space::Heap,
        m * per_token,
        "prefill_chunk",
        &["host"],
        t,
    );
    e.transient = true;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Qwen3.8-27B text config (HF config @ 017b9c7a, every 4th layer full attention).
    fn config() -> Qwen38Config {
        Qwen38Config {
            hidden: 5120,
            layers: 64,
            vocab: 248320,
            intermediate: 17408,
            full_attention: (0..64).map(|l| l % 4 == 3).collect(),
            heads: 24,
            kv_heads: 4,
            head_dim: 256,
            rotary_dim: 64,
            rope_theta: 1e7,
            lin_k_heads: 16,
            lin_v_heads: 48,
            lin_k_dim: 128,
            lin_v_dim: 128,
            conv_kernel: 4,
            eps: 1e-6,
            eos_token_ids: vec![1],
        }
    }

    #[test]
    fn page_and_fp8() {
        assert_eq!(page(0), 4096);
        assert_eq!(page(1), 4096);
        assert_eq!(page(4096), 4096);
        assert_eq!(page(4097), 8192);
        // gate|up [34816][5120]: weights + 272 x 40 scales.
        assert_eq!(fp8(34816, 5120), 34816 * 5120 + page(272 * 40 * 4));
    }

    #[test]
    fn plan_names_every_layer_and_is_stable() {
        let c = config();
        let a = plan(&c, 4096, Some((256, 128)));
        let b = plan(&c, 4096, Some((256, 128)));
        assert_eq!(a, b);
        for l in 0..c.layers {
            for n in layer_names(l, c.full_attention[l]) {
                assert!(a.get(&n).is_some(), "{n}");
            }
        }
        let names: std::collections::HashSet<_> = a.entries.iter().map(|e| &e.name).collect();
        assert_eq!(names.len(), a.entries.len(), "names are unique");
    }

    #[test]
    fn plan_counts_what_the_old_estimate_missed() {
        let c = config();
        let s = plan(&c, 131072, Some((4096, 128)));
        // The host embedding: 248320 x 5120 bf16.
        assert_eq!(
            s.get("embedding bf16 [vocab][hidden]").unwrap().bytes,
            2_542_796_800
        );
        // The KV cache at 128k: 16 layers x 2 x 4 heads x 131072 x 512 bytes (plus slack).
        let kv = s.get("attn256 KV planes (K|V per layer)").unwrap().bytes;
        assert!(kv >= 16 * 2 * 4 * 131072 * 512);
        assert!(
            s.peak() > s.total(None),
            "the chunk vectors count in the peak only"
        );
    }
}
