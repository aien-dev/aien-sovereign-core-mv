#[test]
fn entropy_failure_never_publishes_plaintext_or_partial_file() {
    let root = temp_root("entropy-error");
    let f = test_fields();
    let ns = f.namespace_id().unwrap();
    let key = crate::seal::SealKey::from_bytes([7; 32])
        .test_source(Arc::new(|_| Err(crate::seal::SealError::Random)));
    let mut cfg = HostFsConfig::from_fields(root.clone(), &f).unwrap();
    cfg.seal_key = Some(Arc::new(key));
    cfg.worker = WorkerMode::Manual;
    let s = HostFsStore::open(cfg).unwrap();
    while !s.recover_step(50).done {}
    let writes_before = s.stats().writes; // Recovery may write its bookkeeping before this job.
    let toks: Vec<u32> = (0..4).collect();
    let k = chunk_chain(&ns, 8, &toks).unwrap()[0];
    let h = f.chunk_header(&ns, k, None, 0, 4, 1).unwrap();
    let (src, _buf) = make_src(1, 4);
    s.submit_store(h, src, Incarnation::seq(1)).unwrap();
    s.run_until_idle();
    let completions = s.poll(16);
    assert!(
        !completions
            .iter()
            .any(|c| c.terminal == Some(Terminal::Done(JobStage::Durable))),
        "entropy failure cannot complete durable publication"
    );
    assert_eq!(
        s.lookup(&[k]),
        vec![None],
        "entropy failure cannot publish an object"
    );
    assert_eq!(
        s.stats().writes,
        writes_before,
        "entropy failure must precede all filesystem writes"
    );
    assert!(
        !object_path_for(&root, &k.0 .0).exists(),
        "unsealed object file must not exist"
    );
    fn no_temp(p: &Path) {
        for e in std::fs::read_dir(p).unwrap() {
            let e = e.unwrap();
            if e.file_type().unwrap().is_dir() {
                no_temp(&e.path())
            } else {
                assert!(
                    !e.file_name().to_string_lossy().ends_with(".tmp"),
                    "no partial temporary file"
                );
            }
        }
    }
    no_temp(&root);
    s.audit().unwrap();
    drop(s);
    std::fs::remove_dir_all(root).unwrap();
}
