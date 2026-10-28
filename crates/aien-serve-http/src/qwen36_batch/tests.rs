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
    /// Swap two fed rows' states once, at the first step from this one on that feeds two rows
    /// whose states differ (two rows seated this beat both hold the empty state, and swapping
    /// equal states changes nothing).
    swap_at: Option<u64>,
    steps: u64,
    fed: Arc<AtomicUsize>,
    /// The rows each step fed, in order.
    log: StepLog,
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
                log: Arc::new(Mutex::new(Vec::new())),
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
        self.log
            .lock()
            .unwrap()
            .push(feed.iter().map(|f| f.row).collect());
        // A step takes a moment, as on the chip, so requests started together overlap.
        std::thread::sleep(Duration::from_micros(300));
        // As the chip's control: once, at the first step from k on that feeds two rows holding
        // different states.
        if self.swap_at.is_some_and(|k| self.steps >= k) {
            let pair = (0..feed.len())
                .flat_map(|a| (a + 1..feed.len()).map(move |b| (a, b)))
                .find(|&(a, b)| self.rows[feed[a].row] != self.rows[feed[b].row]);
            if let Some((a, b)) = pair {
                self.swap_at = None;
                self.rows.swap(feed[a].row, feed[b].row);
            }
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

type StepLog = Arc<Mutex<Vec<Vec<usize>>>>;

/// A door whose fake logs the rows of every step.
fn logged_door(rows: usize) -> (Arc<Qwen36BatchDoor>, StepLog) {
    let (f, _) = Fake::new(rows, None);
    let log = Arc::clone(&f.log);
    let d = Qwen36BatchDoor::with_stepper(f, 4096, lock_file(), Duration::from_secs(5), 1, true)
        .unwrap();
    (Arc::new(d), log)
}

/// Eight short prompts (under `Q36_PREFILL_MIN`, so every token goes in through `step_rows` and a
/// beat is one logged step).
fn group_jobs() -> Vec<GroupJob> {
    prompts()
        .into_iter()
        .take(8)
        .map(|(mut p, n)| GroupJob {
            prompt: {
                p.truncate(Q36_PREFILL_MIN - 1);
                p
            },
            params: greedy(n),
            cancel: None,
            test: TestSwitches::default(),
        })
        .collect()
}

fn run_group(d: &Qwen36BatchDoor, jobs: &[GroupJob]) -> Vec<Vec<u32>> {
    let mut words = vec![Vec::new(); jobs.len()];
    // Through the trait object, as the server calls it (an inherent method of the same name would
    // shadow a missing trait override).
    let outs = (d as &dyn Engine).generate_group(jobs, &mut |i, t| {
        words[i].push(t);
        true
    });
    for o in &outs {
        assert!(o.is_ok(), "{o:?}");
    }
    words
}

/// The readers' group (sc L371): eight jobs seated in one beat, beside a conversation already
/// running, each job's words those of the same request alone, in job order.
#[test]
fn a_group_of_eight_joins_in_one_beat_and_keeps_its_order() {
    let (d, log) = logged_door(10);
    let solo: Vec<u32> = (0..6).collect();
    let h = {
        let d = Arc::clone(&d);
        let solo = solo.clone();
        std::thread::spawn(move || run(&d, &solo, &greedy(300)).0)
    };
    while log.lock().unwrap().len() < 5 {
        std::thread::sleep(Duration::from_millis(1));
    }
    let jobs = group_jobs();
    let got = run_group(&d, &jobs);
    let want: Vec<Vec<u32>> = jobs
        .iter()
        .map(|j| alone(&j.prompt, j.params.max_new_tokens))
        .collect();
    assert_eq!(got, want);
    assert_eq!(h.join().unwrap(), alone(&solo, 300));
    // Before the group one row steps; the first step with more has all nine.
    let log = log.lock().unwrap();
    let first = log
        .iter()
        .position(|s| s.len() > 1)
        .expect("the group stepped");
    assert!(first >= 5);
    assert_eq!(log[first].len(), 9, "step {first}: {:?}", log[first]);
}

/// A group waits for enough free rows and nothing seats a part of it early.
#[test]
fn a_group_waits_for_its_rows_then_joins_whole() {
    let (d, log) = logged_door(8);
    let solo: Vec<u32> = (0..4).collect();
    let h = {
        let d = Arc::clone(&d);
        std::thread::spawn(move || run(&d, &solo, &greedy(40)).0)
    };
    while log.lock().unwrap().is_empty() {
        std::thread::sleep(Duration::from_millis(1));
    }
    let got = run_group(&d, &group_jobs());
    h.join().unwrap();
    assert_eq!(got.len(), 8);
    let log = log.lock().unwrap();
    // One row (the single conversation) until it leaves, then all eight at once.
    let first = log
        .iter()
        .position(|s| s.len() > 1)
        .expect("the group stepped");
    assert!(log[..first].iter().all(|s| s.len() == 1));
    assert_eq!(log[first].len(), 8);
    assert!(
        first >= 40,
        "the group started before the single left (step {first})"
    );
}

#[test]
fn a_group_larger_than_the_door_is_refused_whole() {
    let (d, _) = door(4, None, 1);
    let outs = d.generate_group(&group_jobs(), &mut |_, _| true);
    assert_eq!(outs.len(), 8);
    assert!(outs
        .iter()
        .all(|o| matches!(o, Err(EngineError::Failed(m)) if m.contains("needs 8 rows"))));
}

/// The caller's flag stops a job still taking in its prompt (the readers' client left).
#[test]
fn a_group_job_stops_on_the_callers_flag_while_prefilling() {
    let (d, _) = door(8, None, 1);
    let flag = Arc::new(AtomicBool::new(true));
    let jobs: Vec<GroupJob> = group_jobs()
        .into_iter()
        .map(|mut j| {
            j.cancel = Some(Arc::clone(&flag));
            j
        })
        .collect();
    let outs = d.generate_group(&jobs, &mut |_, _| true);
    for o in outs {
        let o = o.unwrap();
        assert_eq!(o.finish, FinishReason::Cancelled);
        assert_eq!(o.emitted_tokens, 0);
    }
}

/// Records the lock's release, for the load admission order.
struct Release<'a>(&'a std::cell::RefCell<Vec<&'static str>>);

impl Drop for Release<'_> {
    fn drop(&mut self) {
        self.0.borrow_mut().push("release");
    }
}

const LOAD_NEED: u64 = 30 << 30;
const LOAD_KEEP: u64 = 8 << 30;

/// The load reads free memory only once the chip lock is ours (the wait can last up to
/// `LOAD_LOCK_WAIT`, and free memory on a shared box swings by tens of GiB meanwhile), and
/// uploads only after that reading admitted it.
#[test]
fn load_reads_memory_after_the_lock_and_before_the_upload() {
    let log = std::cell::RefCell::new(Vec::new());
    let (lock, avail) = admit_then_open(
        || {
            log.borrow_mut().push("lock");
            Ok(Release(&log))
        },
        || {
            log.borrow_mut().push("memory");
            LOAD_NEED + LOAD_KEEP
        },
        LOAD_NEED,
        LOAD_KEEP,
        |avail| {
            log.borrow_mut().push("open");
            Ok(avail)
        },
    )
    .expect("exactly need plus keep-free is enough");
    assert_eq!(avail, LOAD_NEED + LOAD_KEEP);
    drop(lock);
    assert_eq!(*log.borrow(), ["lock", "memory", "open", "release"]);
}

/// 59 GiB free before the wait, 6 GiB after it: the reading after the lock refuses, the lock is
/// let go and nothing is uploaded.
#[test]
fn load_refused_after_the_wait_releases_the_lock_and_uploads_nothing() {
    let log = std::cell::RefCell::new(Vec::new());
    let e = admit_then_open(
        || {
            log.borrow_mut().push("lock");
            Ok(Release(&log))
        },
        || {
            log.borrow_mut().push("memory");
            6 << 30
        },
        LOAD_NEED,
        LOAD_KEEP,
        |_| -> Result<(), String> {
            log.borrow_mut().push("open");
            Ok(())
        },
    )
    .err()
    .expect("6 GiB is refused");
    assert!(
        e.contains("MemAvailable 6144 MiB after taking the GPU lock"),
        "{e}"
    );
    assert_eq!(*log.borrow(), ["lock", "memory", "release"]);
}

#[test]
fn load_without_the_lock_reads_no_memory_and_uploads_nothing() {
    let r: Result<((), ()), String> = admit_then_open(
        || Err("lock wait timed out".into()),
        || panic!("memory must not be read without the lock"),
        LOAD_NEED,
        LOAD_KEEP,
        |_| panic!("nothing is uploaded without the lock"),
    );
    assert_eq!(r.unwrap_err(), "lock wait timed out");
}

/// A reader that stops taking its events (its request thread blocked, as a streaming reply's
/// `blocking_send` blocks when the client stops reading) must not stall the scheduler, which
/// holds the chip lock: once the reader's event channel is full its row is dropped, the other
/// conversation finishes exactly as alone, the lock is let go when the spell ends, and the
/// stalled request ends with an error that says why.
#[test]
fn a_reader_that_stops_reading_is_dropped_and_the_others_finish() {
    use std::os::fd::AsRawFd;
    let (f, _) = Fake::new(2, None);
    let lock = lock_file();
    let d = Arc::new(
        Qwen36BatchDoor::with_stepper(f, 4096, lock.clone(), Duration::from_secs(5), 1, true)
            .unwrap(),
    );
    let (release, hold) = mpsc::channel::<()>();
    let stalled = {
        let d = Arc::clone(&d);
        std::thread::spawn(move || {
            let mut first = true;
            d.generate(&[1, 2, 3], &greedy(2000), &mut |_| {
                if std::mem::take(&mut first) {
                    // Stops reading after its first token, until the test lets go.
                    let _ = hold.recv();
                }
                true
            })
        })
    };
    // The other conversation joins before the stalled reader's channel fills (256 events, some
    // 80 ms of fake steps) and is long enough to be mid-reply when it does.
    std::thread::sleep(Duration::from_millis(10));
    let other: Vec<u32> = (0..6).collect();
    let (done_tx, done_rx) = mpsc::channel();
    {
        let d = Arc::clone(&d);
        let other = other.clone();
        std::thread::spawn(move || {
            let _ = done_tx.send(run(&d, &other, &greedy(1000)).0);
        });
    }
    let got = done_rx
        .recv_timeout(Duration::from_secs(30))
        .expect("the other conversation must finish while a reader is stalled");
    assert_eq!(got, alone(&other, 1000));
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&lock)
        .unwrap();
    let t = Instant::now();
    // SAFETY: flock on a valid descriptor of the test's private lock file.
    while unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        assert!(
            t.elapsed() < Duration::from_secs(30),
            "the scheduler still holds the chip lock"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    drop(file);
    release.send(()).unwrap();
    match stalled.join().unwrap() {
        Err(EngineError::Failed(m)) => assert!(m.contains("fell behind"), "{m}"),
        r => panic!("the stalled reader's request: {r:?}"),
    }
}

/// The interleaving the stall test cannot force: the reader falls behind (an event is dropped and
/// the scheduler cancels the job), then resumes and drains before the scheduler sends the job's
/// last event, so `Done` fits. The request must still fail as fallen behind, not report a normal
/// completion of a truncated reply, and only the delivered tokens reach the caller.
#[test]
fn a_reader_that_fell_behind_fails_even_when_its_done_fits() {
    let (otx, orx) = mpsc::sync_channel::<(usize, Ev)>(2);
    let cancel = Arc::new(AtomicBool::new(false));
    let fell = Arc::new(AtomicBool::new(false));
    let job = Job {
        prompt: vec![1, 2, 3],
        params: greedy(10),
        out: otx,
        tag: 0,
        cancel: Arc::clone(&cancel),
        fell_behind: Arc::clone(&fell),
        t_queued: Instant::now(),
        test: TestSwitches::default(),
    };
    assert!(job.send(Ev::Token(11)).is_ok());
    assert!(job.send(Ev::Token(12)).is_ok());
    assert!(job.send(Ev::Token(13)).is_err(), "the channel is full");
    assert!(
        cancel.load(Ordering::SeqCst),
        "the scheduler cancels the job"
    );
    let (drained_tx, drained_rx) = mpsc::channel();
    let consumer = std::thread::spawn(move || {
        let mut got = Vec::new();
        let r = relay(&orx, &[cancel], &[fell], &mut |_, t| {
            got.push(t);
            if got.len() == 2 {
                let _ = drained_tx.send(());
            }
            true
        });
        (r, got)
    });
    // The reader has taken both queued tokens: the job's last event fits now.
    drained_rx.recv().unwrap();
    job.send(Ev::Done(GenerateOutcome {
        emitted_tokens: 3,
        finish: FinishReason::Cancelled,
        prefill_ms: 0,
        decode_ms: 0,
        sampling: None,
    }))
    .expect("Done fits after the drain");
    drop(job);
    let (mut r, got) = consumer.join().unwrap();
    assert_eq!(got, [11, 12]);
    match r.pop().expect("one outcome") {
        Err(EngineError::Failed(m)) => assert!(m.contains("fell behind"), "{m}"),
        other => panic!("a truncated reply reported as {other:?}"),
    }
}

/// The decode timeline over the fake: [`Fake`] plus the row timeline a chip graph gives, a fixed
/// launch sequence (one layer and the exact head) with 100 ns chip stamps.
struct StampFake {
    inner: Fake,
    on: bool,
    asked: Arc<AtomicUsize>,
    last: Vec<RowTimeline>,
}

const STAMP_KINDS: [&str; 21] = [
    "norm", "mix", "ab", "gdn", "o_proj", "res", "router", "gate_up", "swiglu", "down", "gather",
    "res", "norm", "lm_head", "lm_head", "lm_head", "argmax", "argmax", "lm_head", "argmax",
    "argmax",
];

impl RowStep for StampFake {
    type Snap = (u64, usize);
    fn row_snapshot(&mut self, r: usize, snap: &mut (u64, usize)) -> Result<(), String> {
        self.inner.row_snapshot(r, snap)
    }
    fn row_restore(&mut self, r: usize, snap: &(u64, usize)) -> Result<(), String> {
        self.inner.row_restore(r, snap)
    }
    fn row_count(&self) -> usize {
        self.inner.row_count()
    }
    fn row_reset(&mut self, r: usize) -> Result<(), String> {
        self.inner.row_reset(r)
    }
    fn step_rows(&mut self, feed: &[RowFeed]) -> Result<Vec<RowOut>, String> {
        if !self.on || feed.len() != 1 {
            return self.inner.step_rows(feed);
        }
        let h0 = timeline_now_ns();
        let h1 = timeline_now_ns();
        std::thread::sleep(Duration::from_micros(50)); // the list build
        let rec = timeline_now_ns() - h1;
        let mut outs = self.inner.step_rows(feed)?; // submit and wait
        let total = timeline_now_ns() - h1;
        let h2 = timeline_now_ns();
        let h3 = timeline_now_ns();
        let n = STAMP_KINDS.len();
        let base = 7_000_000_000 + h1;
        self.last[feed[0].row] = RowTimeline {
            host: [h0, h1, h2, h3],
            stamps: (0..=n as u64).map(|i| base + i * 100).collect(),
            kinds: STAMP_KINDS.iter().map(|k| k.to_string()).collect(),
        };
        outs[0].stats = GraphStats {
            launches: n as u32,
            chip_ns: n as u64 * 100,
            record_ns: rec,
            total_ns: total,
            ..Default::default()
        };
        Ok(outs)
    }
    fn set_row_stamps(&mut self, on: bool) -> Result<(), String> {
        self.asked.fetch_add(1, Ordering::SeqCst);
        self.on = on;
        Ok(())
    }
    fn row_timeline(&self, r: usize) -> Result<RowTimeline, String> {
        self.asked.fetch_add(1, Ordering::SeqCst);
        Ok(self.last[r].clone())
    }
}

fn stamp_fake(rows: usize) -> (StampFake, Arc<AtomicUsize>) {
    let (inner, _) = Fake::new(rows, None);
    let asked = Arc::new(AtomicUsize::new(0));
    (
        StampFake {
            inner,
            on: false,
            asked: Arc::clone(&asked),
            last: vec![RowTimeline::default(); rows],
        },
        asked,
    )
}

/// Waits for the scheduler's idle flush: `n` lines in `path`.
fn timeline_lines(path: &Path, n: usize) -> String {
    for _ in 0..200 {
        let t = std::fs::read_to_string(path).unwrap_or_default();
        if t.lines().count() >= n {
            return t;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    std::fs::read_to_string(path).unwrap_or_default()
}

/// Every decode beat of one conversation is one complete, gap-free record; the words are the
/// request alone's; dropping one record, one host phase or one launch turns the checker red.
#[test]
fn decode_timeline_records_every_single_row_beat() {
    let path = lock_file().with_extension("timeline.jsonl");
    let (f, asked) = stamp_fake(1);
    let d = Qwen36BatchDoor::with_stepper_timeline(
        f,
        4096,
        lock_file(),
        Duration::from_secs(5),
        1,
        true,
        Some(path.clone()),
    )
    .unwrap();
    let prompt: Vec<u32> = (0..5).map(|k| (k * 13 + 1) % VOCAB as u32).collect();
    let (words, _) = run(&d, &prompt, &greedy(20));
    assert_eq!(words, alone(&prompt, 20), "the timeline changed the words");
    // 20 tokens: the first from the prompt's last step, then 19 decode steps.
    let text = timeline_lines(&path, 19);
    let recs = crate::decode_timeline::read_jsonl(&text).unwrap();
    assert_eq!(recs.len(), 19, "{text}");
    assert_eq!(crate::decode_timeline::check(&recs), Vec::<String>::new());
    for w in recs.windows(2) {
        assert_eq!(w[1].beat, w[0].beat + 1);
        assert_eq!(w[1].step, w[0].step + 1);
    }
    assert!(asked.load(Ordering::SeqCst) >= 20);
    let mut no_rec = recs.clone();
    no_rec.remove(7);
    assert!(!crate::decode_timeline::check(&no_rec).is_empty());
    let mut no_host = recs.clone();
    no_host[3].host.remove(4);
    assert!(!crate::decode_timeline::check(&no_host).is_empty());
    let mut no_dev = recs.clone();
    no_dev[11].dev.remove(9);
    assert!(!crate::decode_timeline::check(&no_dev).is_empty());
    let s = crate::decode_timeline::summary(&recs);
    assert!(s.contains("\"dev:head_recheck\" -> {100, "), "{s}");
    // `AIEN_Q36_DECODE_TIMELINE_KEEP=<file>`: keep this host run for the checker CLI.
    if let Some(keep) = std::env::var_os("AIEN_Q36_DECODE_TIMELINE_KEEP") {
        std::fs::copy(&path, keep).unwrap();
    }
    let _ = std::fs::remove_file(&path);
}

/// Switched off (the default), the door never asks the stepper for stamps or a timeline.
#[test]
fn decode_timeline_off_never_asks() {
    let (f, asked) = stamp_fake(2);
    let d = Qwen36BatchDoor::with_stepper(f, 4096, lock_file(), Duration::from_secs(5), 1, true)
        .unwrap();
    let prompt = [3u32, 1, 4, 1, 5];
    let (words, _) = run(&d, &prompt, &greedy(10));
    assert_eq!(words, alone(&prompt, 10));
    assert_eq!(asked.load(Ordering::SeqCst), 0);
}

/// A stepper without a timeline refuses a switched-on door instead of recording nothing.
#[test]
fn decode_timeline_refused_without_stamps() {
    let (f, _) = Fake::new(1, None);
    let path = lock_file().with_extension("refused.jsonl");
    let r = Qwen36BatchDoor::with_stepper_timeline(
        f,
        4096,
        lock_file(),
        Duration::from_secs(5),
        1,
        true,
        Some(path.clone()),
    );
    assert!(r.is_err());
    assert!(!path.exists());
}

/// A stepper that never returns logits, even when a feed asks for them.
struct NoLogits(Vec<u64>);

impl RowStep for NoLogits {
    type Snap = u64;
    fn row_snapshot(&mut self, r: usize, snap: &mut u64) -> Result<(), String> {
        *snap = self.0[r];
        Ok(())
    }
    fn row_restore(&mut self, r: usize, snap: &u64) -> Result<(), String> {
        self.0[r] = *snap;
        Ok(())
    }
    fn row_count(&self) -> usize {
        self.0.len()
    }
    fn row_reset(&mut self, r: usize) -> Result<(), String> {
        self.0[r] = 0;
        Ok(())
    }
    fn step_rows(&mut self, feed: &[RowFeed]) -> Result<Vec<RowOut>, String> {
        Ok(feed
            .iter()
            .map(|f| {
                self.0[f.row] = mix(self.0[f.row] ^ u64::from(f.token));
                RowOut {
                    token: (self.0[f.row] % 89) as u32 + 5,
                    logits: None,
                    stats: GraphStats::default(),
                }
            })
            .collect())
    }
}

/// The prompt's last feed asks for logits; a stepper that returns none fails the request with a
/// clear error instead of leaving it waiting forever (the door used to hang here).
#[test]
fn missing_logits_fail_the_request() {
    for chip_greedy in [true, false] {
        let d = Qwen36BatchDoor::with_stepper(
            NoLogits(vec![0; 2]),
            4096,
            lock_file(),
            Duration::from_secs(5),
            4,
            chip_greedy,
        )
        .unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let r = d.generate(&[3, 1, 4, 1, 5], &greedy(8), &mut |_| true);
            let _ = tx.send(r.map(|_| ()).map_err(|e| format!("{e:?}")));
        });
        let r = rx
            .recv_timeout(Duration::from_secs(20))
            .expect("the door hung on a step without the logits it asked for");
        let e = r.expect_err("a step without asked-for logits must fail the request");
        assert!(e.contains("no logits"), "error names the cause: {e}");
    }
}
