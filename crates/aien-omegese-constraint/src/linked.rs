//! [`LinkedOracle`]: `omegese next --machine` as one function call into omega's reader, linked as a
//! library (`tools/omegese/omegese_lib.c`, pinned by `omegese.lock`; see `build.rs`). The answer
//! is the command's own bytes and exit code (omega `make test-omegese-lib` checks that on every
//! prefix of the dev and worked-example programs), so it reads with the same [`parse_machine`] as
//! [`SubprocessOracle`](crate::SubprocessOracle), without a process per query.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::oracle::{parse_machine, Answer, ItemOracle};

/// Where the linked library came from: `omega <sha>`, `prebuilt <path>`, or `none`.
pub const LIB_SOURCE: &str = env!("AIEN_OMEGESE_LIB_SOURCE");

/// The omega commit `omegese.lock` pins.
pub const PINNED_SHA: &str = env!("AIEN_OMEGESE_PINNED_SHA");

#[cfg(has_omegese_lib)]
mod ffi {
    use std::os::raw::{c_char, c_int};
    pub const NEXT_MACHINE: c_int = 0;
    pub const COMPLETE: c_int = 1;
    extern "C" {
        pub fn omegese_lib_query(
            mode: c_int,
            src: *const c_char,
            len: usize,
            out: *mut *mut c_char,
            out_len: *mut usize,
        ) -> c_int;
        pub fn omegese_lib_free(p: *mut c_char);
    }
}

/// The reader's state is global: one query at a time in the whole process.
static READER: Mutex<()> = Mutex::new(());

/// One query: the command's stdout bytes and exit code.
#[cfg(has_omegese_lib)]
fn query(mode: std::os::raw::c_int, src: &[u8]) -> Result<(Vec<u8>, i32), String> {
    let _one = READER.lock().unwrap_or_else(|p| p.into_inner());
    let mut out: *mut std::os::raw::c_char = std::ptr::null_mut();
    let mut n = 0usize;
    // SAFETY: src is valid for src.len() bytes; the library writes a malloc'd buffer of n bytes
    // (plus a NUL) into out, freed below with its own free.
    let rc = unsafe {
        ffi::omegese_lib_query(mode, src.as_ptr().cast(), src.len(), &mut out, &mut n)
    };
    let bytes = if out.is_null() {
        Vec::new()
    } else {
        // SAFETY: out holds n bytes (see above).
        let v = unsafe { std::slice::from_raw_parts(out.cast::<u8>(), n) }.to_vec();
        unsafe { ffi::omegese_lib_free(out) };
        v
    };
    Ok((bytes, rc))
}

#[cfg(not(has_omegese_lib))]
fn query(_mode: i32, _src: &[u8]) -> Result<(Vec<u8>, i32), String> {
    let _one = READER.lock();
    Err(not_linked())
}

fn not_linked() -> String {
    "this build has no linked Omegese reader: build with AIEN_OMEGESE_DIR=<omega checkout at \
     omegese.lock> (or AIEN_OMEGESE_LIB=<libomegese.a>)"
        .into()
}

/// True when this build linked the reader.
pub fn is_linked() -> bool {
    cfg!(has_omegese_lib)
}

/// The real oracle, linked: one call per query.
#[derive(Debug, Default)]
pub struct LinkedOracle {
    calls: u64,
    elapsed: Duration,
}

impl LinkedOracle {
    /// `Err` when this build did not link the library.
    pub fn new() -> Result<Self, String> {
        if !is_linked() {
            return Err(not_linked());
        }
        Ok(Self::default())
    }

    /// Queries made so far.
    pub fn calls(&self) -> u64 {
        self.calls
    }

    /// Wall time spent in queries so far.
    pub fn elapsed(&self) -> Duration {
        self.elapsed
    }
}

impl ItemOracle for LinkedOracle {
    fn next(&mut self, prefix: &[u8]) -> Result<Answer, String> {
        let t = Instant::now();
        #[cfg(has_omegese_lib)]
        let r = query(ffi::NEXT_MACHINE, prefix);
        #[cfg(not(has_omegese_lib))]
        let r = query(0, prefix);
        self.calls += 1;
        self.elapsed += t.elapsed();
        let (out, rc) = r?;
        if !matches!(rc, 0 | 1) {
            return Err(format!(
                "omegese_lib exit {rc}: {}",
                String::from_utf8_lossy(&out).trim()
            ));
        }
        let out = String::from_utf8(out).map_err(|_| "omegese_lib answer is not UTF-8")?;
        parse_machine(&out, rc == 0)
    }
}

/// `omegese complete` on `prefix`: `Ok(Some(text))` that, written after the prefix, makes a
/// program `omegese emit` accepts; `Ok(None)` when the reader refuses the prefix.
pub fn complete(prefix: &[u8]) -> Result<Option<Vec<u8>>, String> {
    #[cfg(has_omegese_lib)]
    let (out, rc) = query(ffi::COMPLETE, prefix)?;
    #[cfg(not(has_omegese_lib))]
    let (out, rc) = query(1, prefix)?;
    match rc {
        0 => Ok(Some(out)),
        1 => Ok(None),
        _ => Err(format!(
            "omegese_lib exit {rc}: {}",
            String::from_utf8_lossy(&out).trim()
        )),
    }
}
