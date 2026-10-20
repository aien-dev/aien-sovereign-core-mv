//! Operator input for the ALLEN end-to-end demo (scripts/allen_e2e_demo.sh).
//! NOT a test: it is `#[ignore]`d and does nothing unless the demo driver sets
//! its environment. It writes ONE exported kind-24 subject head object, bound
//! to the lineage of the demo's Cortex journal, using the TEST-ONLY fixture
//! encoder. Real subjects are created only by Aienux `cs_provision`, so the
//! demo labels this subject a host-built fixture.
//!   DEMO_SUBJECT_OUT      path of the head object file to write
//!   DEMO_LINEAGE_HEX      64 hex: digest of Cortex record 1 of the demo home
//!   DEMO_SUBJECT_NAME     fixture name (derives root/agent ids)
//! Prints `AGENT=<64 hex>` and `ROOT=<64 hex>` on stdout.
mod support;

#[test]
#[ignore = "operator input for scripts/allen_e2e_demo.sh, not a test"]
fn write_demo_subject() {
    let (Ok(out), Ok(lineage), Ok(name)) = (
        std::env::var("DEMO_SUBJECT_OUT"),
        std::env::var("DEMO_LINEAGE_HEX"),
        std::env::var("DEMO_SUBJECT_NAME"),
    ) else {
        panic!("set DEMO_SUBJECT_OUT, DEMO_LINEAGE_HEX and DEMO_SUBJECT_NAME");
    };
    let mut f = support::fx(&name);
    f.cortex = aien_allen::unhex32(&lineage).expect("DEMO_LINEAGE_HEX is 64 hex");
    let chain = support::chain(&f, 2);
    support::write_head(std::path::Path::new(&out), chain.last().unwrap());
    println!("AGENT={}", aien_allen::hex(&f.agent));
    println!("ROOT={}", aien_allen::hex(&f.root));
}
