//! STUB (red step).
use crate::account::Account;
use crate::rules::RuleTable;
use std::collections::BTreeMap;
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::path::{Path, PathBuf};

pub const STRIPPED_ENV: &[&str] = &[];
pub const PROXY_ENV: &[&str] = &[];
pub const CREDENTIALS_FILE: &str = ".credentials.json";
pub type Env = BTreeMap<OsString, OsString>;

pub fn is_credential_selecting(_name: &str) -> bool { todo!() }
pub fn launch_env(_i: &Env, _d: &Path, _p: &[String]) -> Env { todo!() }
pub fn find_program(_p: &str, _path: Option<&OsStr>) -> Option<PathBuf> { todo!() }
pub fn claude_profiles(_r: &RuleTable) -> Vec<(&Account, &Path)> { todo!() }
pub fn profile<'a>(_r: &'a RuleTable, _a: &str) -> Result<(&'a Account, &'a Path), LaunchError> { todo!() }

#[derive(Clone, PartialEq, Eq)]
pub struct LaunchPlan { pub alias: String, pub program: PathBuf, pub config_dir: PathBuf, pub env: Env }
impl fmt::Debug for LaunchPlan { fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result { todo!() } }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirState { Missing, Private }
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DirProblem { NotADirectory, Symlink, TooOpen { mode: u32 }, Io(String) }
pub fn check_profile_dir(_p: &Path) -> Result<DirState, DirProblem> { todo!() }
pub fn prepare_profile_dir(_p: &Path) -> Result<(), LaunchError> { todo!() }
pub fn plan_launch(_r: &RuleTable, _a: &str, _e: &Env) -> Result<LaunchPlan, LaunchError> { todo!() }

pub trait Spawner { fn spawn(&mut self, plan: &LaunchPlan) -> Result<i32, LaunchError>; }
#[derive(Debug, Default)]
pub struct FakeSpawner { pub plans: Vec<LaunchPlan>, pub exit_code: i32 }
impl Spawner for FakeSpawner { fn spawn(&mut self, _p: &LaunchPlan) -> Result<i32, LaunchError> { todo!() } }
pub fn launch(_r: &RuleTable, _a: &str, _e: &Env, _s: &mut dyn Spawner) -> Result<i32, LaunchError> { todo!() }

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LaunchError {
    UnknownAlias { alias: String, known: Vec<String> },
    NotAClaudeProfile { alias: String },
    ExecutableNotFound { program: String },
    ProfileDir { path: PathBuf, problem: DirProblem },
    Spawn { program: PathBuf, message: String },
}
impl fmt::Display for LaunchError { fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result { todo!() } }
