#[test]
fn restart_child_fixture() {
    let Ok(path)=std::env::var("AIEN_TEST_NONCE_OUTPUT") else {return};
    let value:u8=std::env::var("AIEN_TEST_NONCE_BYTE").unwrap().parse().unwrap();
    let key=SealKey::from_bytes([7;32]).test_source(Arc::new(move|out|{out.fill(value);Ok(())}));
    let mut file=plain(3);key.seal_file(&mut file).unwrap();std::fs::write(path,file).unwrap();
}
#[test]
fn actual_process_restart_preserves_authenticated_disk_object_not_rng_state() {
    let base=std::env::temp_dir().join(format!("kv-nonce-process-{}",std::process::id()));
    std::fs::create_dir(&base).unwrap();
    for value in [1u8,2] {
        let output=base.join(format!("{value}.seal"));
        let status=std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact","seal::nonce_tests::restart_child_fixture","--nocapture"])
            .env("AIEN_TEST_NONCE_OUTPUT",&output).env("AIEN_TEST_NONCE_BYTE",value.to_string())
            .status().unwrap();assert!(status.success(),"real child process fixture must complete");
        let mut file=std::fs::read(output).unwrap();assert_eq!(&file[O_NONCE..O_TAG],&[value;12]);
        let key=SealKey::from_bytes([7;32]);let(r,p)=file.split_at_mut(HEADER_REGION_LEN);key.open_payload(r,p).unwrap();assert_eq!(p,&[3;32]);
    }
    // Parent holds the first durable bytes while child 2 starts independently.
    // The injected source's lifetime is controlled externally, not restored from disk.
    let first=std::fs::read(base.join("1.seal")).unwrap();let second=std::fs::read(base.join("2.seal")).unwrap();
    assert_ne!(&first[O_NONCE..O_TAG],&second[O_NONCE..O_TAG]);
    std::fs::remove_dir_all(base).unwrap();
}
