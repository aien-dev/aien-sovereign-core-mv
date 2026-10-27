// THROWAWAY A/B driver (evidence only, not product code): appended to
// crates/aien-serve-http/src/qwen36_batch/tests.rs in BOTH arms, unchanged.
// One request of BENCH_TOKENS greedy tokens through the batch door over a
// fake model that does no work, so the measured time is the door's own beat
// loop: the code the switched-off decode timeline touches.

struct BenchFake(Vec<(u64, usize)>);

impl RowStep for BenchFake {
    type Snap = (u64, usize);
    fn row_snapshot(&mut self, r: usize, snap: &mut (u64, usize)) -> Result<(), String> {
        *snap = self.0[r];
        Ok(())
    }
    fn row_restore(&mut self, r: usize, snap: &(u64, usize)) -> Result<(), String> {
        self.0[r] = *snap;
        Ok(())
    }
    fn row_count(&self) -> usize {
        self.0.len()
    }
    fn row_reset(&mut self, r: usize) -> Result<(), String> {
        self.0[r] = (0, 0);
        Ok(())
    }
    fn step_rows(&mut self, feed: &[RowFeed]) -> Result<Vec<RowOut>, String> {
        Ok(feed
            .iter()
            .map(|f| {
                let (st, pos) = &mut self.0[f.row];
                *st = mix(*st ^ (u64::from(f.token) << 20) ^ *pos as u64);
                *pos += 1;
                let token = (*st % 89) as u32 + 5;
                // The prompt's last feed asks for logits: the door samples the first reply
                // token from them. A one-hot row keeps that pick equal to `token`.
                let logits = f.logits.then(|| {
                    let mut l = vec![0.0f32; VOCAB];
                    l[token as usize] = 1.0;
                    l
                });
                RowOut {
                    token,
                    logits,
                    stats: GraphStats::default(),
                }
            })
            .collect())
    }
}

#[test]
#[ignore]
fn bench_door_beat() {
    const BENCH_TOKENS: usize = 20_000;
    let d = Qwen36BatchDoor::with_stepper(
        BenchFake(vec![(0, 0); 8]),
        1 << 16,
        lock_file(),
        Duration::from_secs(5),
        16,
        true,
    )
    .unwrap();
    let prompt = [3u32, 1, 4, 1, 5, 9, 2, 6];
    let t = Instant::now();
    let (words, _) = run(&d, &prompt, &greedy(BENCH_TOKENS));
    let ns = t.elapsed().as_nanos();
    assert_eq!(words.len(), BENCH_TOKENS);
    let digest = words.iter().fold(0u64, |h, &w| mix(h ^ u64::from(w)));
    println!("BENCH tokens {} ns {} digest {:016x}", words.len(), ns, digest);
}
