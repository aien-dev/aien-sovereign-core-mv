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

type StepLog = Arc<Mutex<Vec<Vec<usize>>>>;

/// A door whose fake logs the rows of every step.
fn logged_door(rows: usize) -> (Arc<Qwen36BatchDoor>, StepLog) {
    let (f, _) = Fake::new(rows, None);
    let log = Arc::clone(&f.log);
    let d = Qwen36BatchDoor::with_stepper(f, 4096, lock_file(), Duration::from_secs(5), 1, true)
        .unwrap();
    (Arc::new(d), log)
}

fn group_jobs() -> Vec<GroupJob> {
    prompts()
        .into_iter()
        .take(8)
        .map(|(p, n)| GroupJob {
            prompt: p,
            params: greedy(n),
            cancel: None,
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
