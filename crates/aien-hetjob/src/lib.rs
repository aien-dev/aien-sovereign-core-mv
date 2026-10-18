//! Heterogeneous CPU jobs for AIEN (aien-sovereign-core#637, parent aienix#6).
//!
//! A coordinator (the Spark) sends one allowlisted, integer-exact job to a
//! hosted worker machine (an M2 MacBook, or any Unix host) over PST1 frames on
//! a byte stream, normally SSH stdio to a forced-command key. The worker
//! admits it against fixed ceilings, runs it, signs a receipt with its own
//! pinned key and stores it durably; the coordinator checks the receipt and
//! compares the output digest with its own recompute.
//!
//! What this proves and what it does not: a match proves the worker returned
//! the right answer for that job. The receipt's platform, program and timing
//! fields are the worker's own claims (self-attested). There is no remote
//! shell: the worker runs only the kinds in `task`.

pub mod coordinator;
pub mod host;
pub mod receipt;
pub mod store;
pub mod task;
pub mod wire;
pub mod worker;
