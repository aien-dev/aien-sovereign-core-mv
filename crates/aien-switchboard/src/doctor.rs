//! STUB (red step).
use crate::launch::Env;
use crate::rules::RuleTable;
use std::fmt;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level { Ok, Warn, Fail }
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding { pub level: Level, pub subject: String, pub message: String }
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Report { pub findings: Vec<Finding> }
impl Report { pub fn failed(&self) -> bool { todo!() } }
impl fmt::Display for Report { fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result { todo!() } }
pub trait VersionProbe { fn version(&mut self, program: &Path, env: &Env) -> Result<String, String>; }
#[derive(Debug)]
pub struct FakeProbe { pub answer: Result<String, String>, pub calls: Vec<PathBuf> }
impl VersionProbe for FakeProbe { fn version(&mut self, _p: &Path, _e: &Env) -> Result<String, String> { todo!() } }
pub fn doctor(_r: &RuleTable, _e: &Env, _cwd: &Path, _p: &mut dyn VersionProbe) -> Report { todo!() }
