//! Host tests of the batch scheduler over a fake row model: every row's words equal the same
//! request alone, a row-state swap is caught, next turns reuse their row, cancel and stop work.
use super::*;
use aien_omega_q38::GraphStats;
use std::sync::atomic::AtomicUsize;

const VOCAB: usize = 97;

fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// A row model whose next token depends on every token fed so far and its position.
struct Fake {
    rows: Vec<(u64, usize)>,
    /// Swap the first two fed rows' states once, at the first step from this one on that feeds two.
    swap_at: Option<u64>,
    steps: u64,
    fed: Arc<AtomicUsize>,
}

impl Fake {
    fn new(n: usize, swap_at: Option<u64>) -> (Self, Arc<AtomicUsize>) {
        let fed = Arc::new(AtomicUsize::new(0));
        (
            Self {
                rows: vec![(0, 0); n],
                swap_at,
                steps: 0,
                fed: Arc::clone(&fed),
            },
            fed,
        )
    }
}

impl RowStep for Fake {
    type Snap = (u64, usize);
    fn row_snapshot(&mut self, r: usize, snap: &mut (u64, usize)) -> Result<(), String> {
        *snap = self.rows[r];
        Ok(())
    }
    fn row_restore(&mut self, r: usize, snap: &(u64, usize)) -> Result<(), String> {
        self.rows[r] = *snap;
        Ok(())
    }
    fn row_count(&self) -> usize {
        self.rows.len()
    }
    fn row_reset(&mut self, r: usize) -> Result<(), String> {
        self.rows[r] = (0, 0);
        Ok(())
    }
    fn step_rows(&mut self, feed: &[RowFeed]) -> Result<Vec<RowOut>, String> {
        self.steps += 1;
        // A step takes a moment, as on the chip, so requests started together overlap.
        std::thread::sleep(Duration::from_micros(300));
        // As the chip's control: once, at the first step from k on that feeds two rows.
        if feed.len() >= 2 && self.swap_at.is_some_and(|k| self.steps >= k) {
            self.swap_at = None;
            self.rows.swap(feed[0].row, feed[1].row);
        }
        let mut outs = Vec::new();
        for f in feed {
            let (st, pos) = &mut self.rows[f.row];
            *st = mix(*st ^ (u64::from(f.token) << 20) ^ *pos as u64);
            *pos += 1;
            self.fed.fetch_add(1, Ordering::SeqCst);
            let logits: Vec<f32> = (0..VOCAB)
                .map(|j| (mix(*st ^ j as u64) % 1_000_000) as f32)
                .collect();
            // The chip's argmax follows the host greedy rule.
            let token = crate::sampling::sample_next(&logits, &greedy(1), 0);
            outs.push(RowOut {
                token,
                logits: f.logits.then_some(logits),
                stats: GraphStats::default(),
            });
        }
        Ok(outs)
    }
}

fn lock_file() -> PathBuf {
    static N: AtomicUsize = AtomicUsize::new(0);
    std::env::temp_dir().join(format!(
        "l354-test-lock-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::SeqCst)
    ))
}

fn door(rows: usize, swap_at: Option<u64>, per_beat: usize) -> (Qwen36BatchDoor, Arc<AtomicUsize>) {
    door_with(rows, swap_at, per_beat, true)
}

fn door_with(
    rows: usize,
    swap_at: Option<u64>,
    per_beat: usize,
    chip_greedy: bool,
) -> (Qwen36BatchDoor, Arc<AtomicUsize>) {
    let (f, fed) = Fake::new(rows, swap_at);
    let d = Qwen36BatchDoor::with_stepper(
        f,
        4096,
        lock_file(),
        Duration::from_secs(5),
        per_beat,
        chip_greedy,
    )
    .unwrap();
    (d, fed)
}

fn greedy(n: usize) -> GenerateParams {
    GenerateParams {
        max_new_tokens: n,
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        seed: 0,
        stop_token_ids: Vec::new(),
    }
}

fn run(d: &Qwen36BatchDoor, prompt: &[u32], p: &GenerateParams) -> (Vec<u32>, GenerateOutcome) {
    let mut words = Vec::new();
    let o = d
        .generate(prompt, p, &mut |t| {
            words.push(t);
            true
        })
        .unwrap();
    (words, o)
}

fn prompts() -> Vec<(Vec<u32>, usize)> {
    (0..12u32)
        .map(|i| {
            let len = 1 + (i as usize * 7) % 40;
            let p = (0..len as u32)
                .map(|k| (k * 31 + i * 17 + 3) % VOCAB as u32)
                .collect();
            (p, 5 + (i as usize * 11) % 26)
        })
        .collect()
}

fn alone(prompt: &[u32], n: usize) -> Vec<u32> {
    let (d, _) = door(1, None, 1);
    run(&d, prompt, &greedy(n)).0
}

/// Every prompt at once on `rows` rows (more prompts than rows queue for a free one), each
/// thread starting a little later than the one before: rows join and leave mid-spell.
fn concurrent(rows: usize, swap_at: Option<u64>, per_beat: usize) -> Vec<Vec<u32>> {
    let (d, _) = door(rows, swap_at, per_beat);
    let d = Arc::new(d);
    let hs: Vec<_> = prompts()
        .into_iter()
        .enumerate()
        .map(|(i, (p, n))| {
            let d = Arc::clone(&d);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(i as u64 * 2));
                run(&d, &p, &greedy(n)).0
            })
        })
        .collect();
    hs.into_iter().map(|h| h.join().unwrap()).collect()
}

#[test]
fn every_row_equals_the_request_alone() {
    let want: Vec<Vec<u32>> = prompts().iter().map(|(p, n)| alone(p, *n)).collect();
    for (rows, per_beat) in [(1, 1), (4, 1), (4, 16), (8, 3), (12, 16)] {
        let got = concurrent(rows, None, per_beat);
        assert_eq!(got, want, "rows {rows}, prompt tokens per beat {per_beat}");
    }
    for (w, (_, n)) in want.iter().zip(prompts()) {
        assert_eq!(w.len(), n);
    }
}

#[test]
fn row_swap_mutant_is_caught() {
    let want: Vec<Vec<u32>> = prompts().iter().map(|(p, n)| alone(p, *n)).collect();
    let got = concurrent(12, Some(3), 1);
    let differ = got.iter().zip(&want).filter(|(g, w)| g != w).count();
    assert!(differ >= 2, "the swap changed {differ} conversations");
}

#[test]
fn a_next_turn_continues_its_row() {
    let (d, fed) = door(2, None, 4);
    let p1: Vec<u32> = (0..20).map(|k| (k * 5 + 1) % VOCAB as u32).collect();
    let (r1, o1) = run(&d, &p1, &greedy(8));
    assert_eq!(o1.finish, FinishReason::Length);
    // Turn 2 = turn 1's prompt, its reply, and a new question.
    let mut p2 = p1.clone();
    p2.extend(&r1);
    p2.extend([7, 8, 9]);
    let before = fed.load(Ordering::SeqCst);
    let (r2, _) = run(&d, &p2, &greedy(10));
    let tail = fed.load(Ordering::SeqCst) - before;
    assert_eq!(r2, alone(&p2, 10), "reused row == cold row");
    // The row held p1 and r1 but its last token (never fed): 1 + 3 prompt tokens, 9 decoded.
    assert_eq!(tail, 4 + 9);
    // A prompt that does not continue any row's tokens goes in cold.
    let p3: Vec<u32> = (0..6).map(|k| (k * 3 + 2) % VOCAB as u32).collect();
    let before = fed.load(Ordering::SeqCst);
    let (r3, _) = run(&d, &p3, &greedy(3));
    assert_eq!(r3, alone(&p3, 3));
    assert_eq!(fed.load(Ordering::SeqCst) - before, 6 + 2);
}

#[test]
fn cancel_frees_the_row_and_stop_ends() {
    let (d, _) = door(1, None, 1);
    let p: Vec<u32> = vec![1, 2, 3];
    let mut got = Vec::new();
    let o = d
        .generate(&p, &greedy(20), &mut |t| {
            got.push(t);
            got.len() < 3
        })
        .unwrap();
    assert_eq!(o.finish, FinishReason::Cancelled);
    assert_eq!(o.emitted_tokens, 3);
    let full = alone(&p, 20);
    assert_eq!(got, full[..3]);
    // The one row is free again; a stop token ends the reply before it.
    let mut stop = greedy(20);
    stop.stop_token_ids = vec![full[5]];
    let first = full.iter().position(|&t| t == full[5]).unwrap();
    let (w, o) = run(&d, &p, &stop);
    assert_eq!(o.finish, FinishReason::Stop);
    assert_eq!(w, full[..first]);
}

#[test]
fn a_reworded_next_turn_restores_the_turn_header_copy() {
    let (d, fed) = door(1, None, 4);
    let im = IM_START;
    let p1 = vec![1, 2, 3, im, 5, 6];
    let (r1, _) = run(&d, &p1, &greedy(4));
    // Turn 2 renders turn 1's reply differently (so the live tokens do not match) but keeps
    // everything before turn 1's header: the copy taken there is restored.
    let other = (r1[0] + 1) % VOCAB as u32;
    let p2 = vec![1, 2, 3, im, 5, 6, other, 78, im, 4];
    let before = fed.load(Ordering::SeqCst);
    let (r2, _) = run(&d, &p2, &greedy(5));
    assert_eq!(r2, alone(&p2, 5), "restored row == cold row");
    assert_eq!(fed.load(Ordering::SeqCst) - before, 7 + 4);
    // Turn 3 continues turn 2 live: its copy (taken at p2's last header) is shorter than the live
    // tokens, so the live state wins.
    let mut p3 = p2.clone();
    p3.extend(&r2);
    p3.extend([im, 9]);
    let before = fed.load(Ordering::SeqCst);
    let (r3, _) = run(&d, &p3, &greedy(2));
    assert_eq!(r3, alone(&p3, 2));
    assert_eq!(fed.load(Ordering::SeqCst) - before, 3 + 1);
}

#[test]
fn the_graphs_greedy_token_equals_the_host_argmax_path() {
    let want: Vec<Vec<u32>> = prompts()
        .iter()
        .map(|(p, n)| {
            let (d, _) = door_with(1, None, 1, false);
            run(&d, p, &greedy(*n)).0
        })
        .collect();
    assert_eq!(concurrent(8, None, 16), want);
}

// ---- the merge train's per-request switches (L384) ----

/// The fake honors the switches as the chip does: a mutant changes the row's own words and no
/// other row's; strokes and the exact head leave the words alone (the chip's contract for them).
impl Fake {
    fn step_rows_switched(&mut self, feed: &[RowFeed], tests: &[RowTest]) -> Vec<RowOut> {
        let mut outs = Vec::new();
        for (i, f) in feed.iter().enumerate() {
            let t = tests.get(i).copied().unwrap_or_default();
            let (st, pos) = &mut self.rows[f.row];
            *st = mix(*st ^ (u64::from(f.token) << 20) ^ *pos as u64);
            if t.mutant != 0 {
                *st = mix(*st ^ 0xdead_0000 ^ t.mutant as u64);
            }
            *pos += 1;
            self.fed.fetch_add(1, Ordering::SeqCst);
            let logits: Vec<f32> = (0..VOCAB)
                .map(|j| (mix(*st ^ j as u64) % 1_000_000) as f32)
                .collect();
            let token = crate::sampling::sample_next(&logits, &greedy(1), 0);
            outs.push(RowOut {
                token,
                logits: f.logits.then_some(logits),
                stats: GraphStats::default(),
            });
        }
        outs
    }
}

/// A fake whose `step_rows_test` applies the switches (the chip's `Qwen38Omega` does).
struct SwitchedFake(Fake);

impl RowStep for SwitchedFake {
    type Snap = (u64, usize);
    fn row_snapshot(&mut self, r: usize, snap: &mut (u64, usize)) -> Result<(), String> {
        self.0.row_snapshot(r, snap)
    }
    fn row_restore(&mut self, r: usize, snap: &(u64, usize)) -> Result<(), String> {
        self.0.row_restore(r, snap)
    }
    fn row_count(&self) -> usize {
        self.0.row_count()
    }
    fn row_reset(&mut self, r: usize) -> Result<(), String> {
        self.0.row_reset(r)
    }
    fn step_rows(&mut self, feed: &[RowFeed]) -> Result<Vec<RowOut>, String> {
        self.0.step_rows(feed)
    }
    fn step_rows_test(
        &mut self,
        feed: &[RowFeed],
        tests: &[RowTest],
    ) -> Result<Vec<RowOut>, String> {
        self.0.steps += 1;
        std::thread::sleep(Duration::from_micros(300));
        Ok(self.0.step_rows_switched(feed, tests))
    }
}

fn switched_door(rows: usize) -> Qwen36BatchDoor {
    let (f, _) = Fake::new(rows, None);
    Qwen36BatchDoor::with_stepper(
        SwitchedFake(f),
        4096,
        lock_file(),
        Duration::from_secs(5),
        4,
        true,
    )
    .unwrap()
}

fn run_test(d: &Qwen36BatchDoor, prompt: &[u32], n: usize, t: TestSwitches) -> Vec<u32> {
    let mut words = Vec::new();
    d.generate_test(prompt, &greedy(n), t, &mut |tok| {
        words.push(tok);
        true
    })
    .unwrap();
    words
}

/// Every prompt at once on one door; prompt `i` carries `switch(i)`.
fn concurrent_switched(
    rows: usize,
    switch: impl Fn(usize) -> TestSwitches + Send + Sync + 'static,
) -> Vec<Vec<u32>> {
    let d = Arc::new(switched_door(rows));
    let switch = Arc::new(switch);
    let hs: Vec<_> = prompts()
        .into_iter()
        .enumerate()
        .map(|(i, (p, n))| {
            let (d, switch) = (Arc::clone(&d), Arc::clone(&switch));
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(i as u64 * 2));
                run_test(&d, &p, n, switch(i))
            })
        })
        .collect();
    hs.into_iter().map(|h| h.join().unwrap()).collect()
}

#[test]
fn a_mutant_switch_changes_only_its_own_row() {
    let want: Vec<Vec<u32>> = prompts().iter().map(|(p, n)| alone(p, *n)).collect();
    let got = concurrent_switched(4, |i| TestSwitches {
        mutant: (i == 5).then_some(4),
        ..TestSwitches::default()
    });
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        if i == 5 {
            assert_ne!(g, w, "the mutant row must go red");
        } else {
            assert_eq!(
                g, w,
                "prompt {i} rode beside the mutant and must not change"
            );
        }
    }
}

#[test]
fn a_negative_control_leaves_no_state_for_a_later_request() {
    // L387 (train run 20261022T0119Z, row-swap-red): the mutant row's copy at its last turn
    // header was kept, and the next request with the same prompt restored it and went red.
    let mut p: Vec<u32> = (0..12u32).map(|k| 1 + (k * 13) % 90).collect();
    p.push(IM_START);
    p.extend((0..6u32).map(|k| 2 + (k * 7) % 90));
    let want = alone(&p, 6);
    let d = switched_door(1);
    let red = run_test(
        &d,
        &p,
        6,
        TestSwitches {
            mutant: Some(4),
            ..TestSwitches::default()
        },
    );
    assert_ne!(red, want, "the mutant must go red");
    assert_eq!(
        run_test(&d, &p, 6, TestSwitches::default()),
        want,
        "a plain request after a mutant must not reuse the mutant's state"
    );
}

#[test]
fn strokes_off_and_exact_head_off_equal_the_served_path() {
    let want: Vec<Vec<u32>> = prompts().iter().map(|(p, n)| alone(p, *n)).collect();
    let got = concurrent_switched(8, |i| TestSwitches {
        stroke: Some(i % 2 == 0),
        exact_head: Some(i % 3 != 0),
        ..TestSwitches::default()
    });
    assert_eq!(got, want);
}

#[test]
fn a_row_swap_switch_trades_two_conversations_and_both_go_red() {
    let want: Vec<Vec<u32>> = prompts().iter().map(|(p, n)| alone(p, *n)).collect();
    let got = concurrent_switched(12, |i| TestSwitches {
        row_swap: (i == 3).then_some(2),
        ..TestSwitches::default()
    });
    let differ: Vec<usize> = got
        .iter()
        .zip(&want)
        .enumerate()
        .filter(|(_, (g, w))| g != w)
        .map(|(i, _)| i)
        .collect();
    assert!(
        differ.contains(&3),
        "the swapping conversation must go red: {differ:?}"
    );
    assert!(differ.len() >= 2, "its partner must go red too: {differ:?}");
}

#[test]
fn no_switches_is_the_served_path_on_a_stepper_without_them() {
    // The plain `Fake` refuses `step_rows_test` with a switch: a request without switches never
    // reaches it, and a request with one fails loudly instead of being served as plain.
    let (d, _) = door(2, None, 4);
    let p: Vec<u32> = vec![1, 2, 3, 4];
    assert_eq!(run_test(&d, &p, 5, TestSwitches::default()), alone(&p, 5));
    let r = d.generate_test(
        &p,
        &greedy(5),
        TestSwitches {
            mutant: Some(4),
            ..TestSwitches::default()
        },
        &mut |_| true,
    );
    assert!(matches!(r, Err(EngineError::Failed(_))), "{r:?}");
}

#[test]
fn the_switch_names_round_trip_from_json() {
    let t = TestSwitches::from_json(&serde_json::json!({
        "stroke": false, "exact_head": true, "mutant": 4, "row_swap": 7,
        "prefill_chunk": 128, "prefix_pool": false, "state_bf16": null
    }))
    .unwrap();
    assert_eq!(
        t.names(),
        [
            "stroke",
            "exact_head",
            "mutant",
            "row_swap",
            "prefill_chunk",
            "prefix_pool"
        ]
    );
    assert!(TestSwitches::from_json(&serde_json::json!({"strokes": true})).is_err());
    assert!(TestSwitches::from_json(&serde_json::json!({"mutant": "4"})).is_err());
    assert!(TestSwitches::from_json(&serde_json::json!({"mutant": 99})).is_err());
    assert!(TestSwitches::from_json(&serde_json::json!([])).is_err());
    assert!(TestSwitches::from_json(&serde_json::json!({}))
        .unwrap()
        .is_empty());
}

#[test]
fn long_prompts_go_in_as_chunks_and_keep_the_turn_header_copy() {
    // L369: a row with more than Q36_PREFILL_MIN prompt tokens left reads a chunk per beat
    // (prefill_row); the replies are the request's alone, and a chunk stops at the last turn
    // header, so the copy there is taken and a reworded next turn restores it.
    let im = IM_START;
    let body = |seed: u32, n: u32| -> Vec<u32> {
        (0..n)
            .map(|i| 1 + (seed.wrapping_mul(7919) ^ i.wrapping_mul(31)) % 90)
            .collect()
    };
    let mut p1 = body(1, 300);
    p1.push(im);
    p1.extend(body(2, 40));
    let (d, fed) = door(2, None, 4);
    let (r1, _) = run(&d, &p1, &greedy(4));
    assert_eq!(r1, alone(&p1, 4));
    let other = (r1[0] + 1) % VOCAB as u32;
    let mut p2 = p1.clone();
    p2.extend([other, 78, im]);
    p2.extend(body(3, 150));
    let before = fed.load(Ordering::SeqCst);
    let (r2, _) = run(&d, &p2, &greedy(5));
    assert_eq!(r2, alone(&p2, 5), "restored row == cold row");
    assert_eq!(fed.load(Ordering::SeqCst) - before, (p2.len() - 300) + 4);
}
