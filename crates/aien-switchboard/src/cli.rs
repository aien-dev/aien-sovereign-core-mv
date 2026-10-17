//! STUB (red step).
use crate::doctor::VersionProbe;
use crate::launch::{Env, Spawner};
use std::io::Write;
use std::path::Path;
pub fn run(_a: &[String], _e: &Env, _cwd: &Path, _s: &mut dyn Spawner, _p: &mut dyn VersionProbe, _o: &mut dyn Write, _r: &mut dyn Write) -> i32 { todo!() }
