// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Drake Stapleton (AIEN / aien-dev)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::ffi::{CString, OsStr, OsString};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd};
use std::os::unix::ffi::OsStrExt;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;

use crate::safety::validate_workspace_path;

#[repr(C)]
struct OpenHow {
    flags: u64,
    mode: u64,
    resolve: u64,
}

#[derive(Debug)]
pub enum BindError {
    /// The path is not the authorized workspace object, or it cannot be bound.
    Denied(String),
    NotFound(String),
    Exists(String),
    Failed(String),
}

impl BindError {
    pub fn message(&self) -> &str {
        match self {
            Self::Denied(m) | Self::NotFound(m) | Self::Exists(m) | Self::Failed(m) => m,
        }
    }
}

struct Authorized {
    /// Workspace that contains the path the caller named.
    workspace: PathBuf,
    /// That path, relative to `workspace`, exactly as named. `..` is kept:
    /// the kernel resolves it under `RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS`,
    /// so a `..` reaches the same parent the allow-list resolved, and a
    /// symlink before it is refused instead of being dropped as text.
    rel: PathBuf,
    /// Workspace that contains the allow-listed canonical object.
    canon_workspace: PathBuf,
    /// Canonical object relative to `canon_workspace`. Plain components.
    /// Used only when the caller's final component is a symlink to it.
    canon_rel: PathBuf,
    canonical: PathBuf,
    display: String,
}

/// Workspace roots the CLI file tools may authorize. Same set the tool
/// allow-list already used: platform defaults plus `/tmp`.
pub fn permitted_workspaces() -> Vec<PathBuf> {
    #[cfg(test)]
    if let Some(roots) = test_hooks::workspaces() {
        return roots;
    }
    let mut allowed = crate::platform::PlatformContext::detect().default_workspaces;
    allowed.push(PathBuf::from("/tmp"));
    allowed
}

fn authorize(user_path: &str) -> Result<Authorized, BindError> {
    let workspaces = permitted_workspaces();
    let canonical = validate_workspace_path(user_path, &workspaces).map_err(|_| {
        BindError::Denied(format!(
            "refusing {}: it is outside the workspace allow-list",
            safe_label(Path::new(user_path))
        ))
    })?;
    let named = absolute_named(user_path)?;
    // Open the path the caller named. Opening `canonical` instead would
    // follow a directory symlink into another allowed directory.
    let Some((workspace, rel)) = longest_workspace(&named, &workspaces) else {
        return Err(BindError::Denied(format!(
            "refusing {}: it resolves outside permitted workspace roots",
            safe_label(Path::new(user_path))
        )));
    };
    let Some((canon_workspace, canon_rel)) = longest_workspace(&canonical, &workspaces) else {
        return Err(BindError::Denied(format!(
            "refusing {}: it resolves outside permitted workspace roots",
            safe_label(Path::new(user_path))
        )));
    };
    plain_rel(&canon_rel, user_path)?;
    Ok(Authorized {
        workspace,
        rel,
        canon_workspace,
        canon_rel,
        canonical,
        display: user_path.to_string(),
    })
}

fn plain_rel(rel: &Path, user_path: &str) -> Result<(), BindError> {
    if rel.components().any(|c| !matches!(c, Component::Normal(_))) {
        return Err(BindError::Denied(format!(
            "refusing {}: the authorized path is not a plain path under the workspace",
            safe_label(Path::new(user_path))
        )));
    }
    Ok(())
}

/// True when `rel` holds only names and `..`, and no `..` climbs above the
/// directory it starts from. `RESOLVE_BENEATH` enforces the same bound.
fn stays_beneath(rel: &Path) -> bool {
    let mut depth = 0usize;
    for c in rel.components() {
        match c {
            Component::Normal(_) => depth += 1,
            Component::ParentDir if depth > 0 => depth -= 1,
            _ => return false,
        }
    }
    true
}

/// Deepest workspace that `path` names lexically and never climbs out of,
/// returned as (canonical root, path relative to it). A root configured
/// through a symlink also matches by its configured name; it is opened at
/// its canonical place, since `open_workspace` trusts the root and its
/// ancestors.
fn longest_workspace(path: &Path, workspaces: &[PathBuf]) -> Option<(PathBuf, PathBuf)> {
    let mut best: Option<(usize, PathBuf, PathBuf)> = None;
    for ws in workspaces {
        let Ok(ws_canon) = ws.canonicalize() else {
            continue;
        };
        let Some((prefix, rel)) = [ws_canon.as_path(), ws.as_path()]
            .into_iter()
            .find_map(|prefix| path.strip_prefix(prefix).ok().map(|rel| (prefix, rel)))
        else {
            continue;
        };
        if !stays_beneath(rel) {
            continue;
        }
        let depth = prefix.components().count();
        if best.as_ref().is_none_or(|(cur, _, _)| depth > *cur) {
            best = Some((depth, ws_canon.clone(), rel.to_path_buf()));
        }
    }
    best.map(|(_, ws, rel)| (ws, rel))
}

/// `user_path` made absolute against the current directory. `.` and repeated
/// separators go; `..` stays. Collapsing `link/..` as text names a different
/// object from the one the OS (and `validate_workspace_path`) resolves when
/// `link` is a symlink.
fn absolute_named(user_path: &str) -> Result<PathBuf, BindError> {
    let raw = Path::new(user_path);
    let abs = if raw.is_absolute() {
        raw.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|e| BindError::Failed(e.to_string()))?
            .join(raw)
    };
    Ok(abs.components().collect())
}

fn symlink_component(err: &std::io::Error) -> bool {
    matches!(err.raw_os_error(), Some(libc::ELOOP) | Some(libc::ENOTDIR))
}

fn name_is_symlink(dirfd: libc::c_int, name: &OsStr) -> bool {
    let Ok(c) = c_string(name.as_bytes()) else {
        return false;
    };
    let mut st = std::mem::MaybeUninit::<libc::stat>::uninit();
    // SAFETY: `dirfd` is a directory we hold and `c` is one component.
    let rc = unsafe {
        libc::fstatat(
            dirfd,
            c.as_ptr(),
            st.as_mut_ptr(),
            libc::AT_SYMLINK_NOFOLLOW,
        )
    };
    if rc != 0 {
        return false;
    }
    let st = unsafe { st.assume_init() };
    st.st_mode as u32 & libc::S_IFMT == libc::S_IFLNK
}

/// True only when every component except the last opened, and the last is a
/// symlink. An intermediate symlink makes the parent open fail, so a swapped
/// directory is not treated as a final-component link.
fn final_component_is_symlink(ws: &OwnedFd, rel: &Path) -> bool {
    let Some(name) = rel.file_name() else {
        return false;
    };
    let parent = rel.parent().unwrap_or(Path::new(""));
    let parent_fd = if parent.as_os_str().is_empty() {
        dup_fd(ws.as_raw_fd()).ok()
    } else {
        open_beneath(
            ws.as_raw_fd(),
            parent,
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
            0,
        )
        .ok()
    };
    let Some(parent_fd) = parent_fd else {
        return false;
    };
    name_is_symlink(parent_fd.as_raw_fd(), name)
}

/// Workspace-relative text, or a generic label when `path` is absolute.
/// Error strings must not carry a host absolute path.
fn safe_label(path: &Path) -> String {
    if path.as_os_str().is_empty() {
        "the workspace root".to_string()
    } else if path.is_absolute() {
        "the requested path".to_string()
    } else {
        path.display().to_string()
    }
}

fn map_io(path: &Path, err: std::io::Error) -> BindError {
    let label = safe_label(path);
    match err.raw_os_error() {
        Some(libc::ELOOP) => BindError::Denied(format!(
            "refusing '{label}': a symlink is on the authorized path"
        )),
        Some(libc::EXDEV) => BindError::Denied(format!(
            "refusing '{label}': the path resolves outside the workspace root"
        )),
        Some(libc::ENOSYS) => BindError::Denied(format!(
            "refusing '{label}': openat2 is unavailable, so the open cannot be bound to the authorized target"
        )),
        // EPERM and EINVAL are confinement failures, not a generic I/O miss.
        // A kernel that rejects the resolve flags must not fall through to a
        // pathname open.
        Some(libc::EPERM) | Some(libc::EINVAL) => BindError::Denied(format!(
            "refusing '{label}': the open cannot be bound to the authorized target"
        )),
        // A symlink opened with O_DIRECTORY and RESOLVE_NO_SYMLINKS is reported
        // as ENOTDIR (the link itself is not a directory). Do not follow it.
        Some(libc::ENOTDIR) => BindError::Denied(format!(
            "refusing '{label}': a path component is not a directory"
        )),
        Some(libc::ENOENT) => BindError::NotFound(err.to_string()),
        Some(libc::EEXIST) => BindError::Exists(err.to_string()),
        _ => BindError::Failed(err.to_string()),
    }
}

fn c_string(bytes: &[u8]) -> std::io::Result<CString> {
    CString::new(bytes).map_err(|_| std::io::Error::from_raw_os_error(libc::EINVAL))
}

/// Open `rel` beneath `dirfd` in one `openat2` step. `rel` empty duplicates
/// `dirfd` (the workspace root itself).
fn open_beneath(
    dirfd: libc::c_int,
    rel: &Path,
    flags: libc::c_int,
    mode: u64,
) -> std::io::Result<OwnedFd> {
    if rel.as_os_str().is_empty() {
        return dup_fd(dirfd);
    }
    #[cfg(test)]
    if let Some(code) = test_hooks::take_openat2_error() {
        return Err(std::io::Error::from_raw_os_error(code));
    }
    let c = c_string(rel.as_os_str().as_bytes())?;
    let how = OpenHow {
        flags: (flags | libc::O_CLOEXEC) as u64,
        mode,
        resolve: (libc::RESOLVE_BENEATH | libc::RESOLVE_NO_SYMLINKS | libc::RESOLVE_NO_MAGICLINKS)
            as u64,
    };
    // SAFETY: `c` is a valid C string, `how` matches the kernel `open_how`
    // layout, and a non-negative fd is owned by this function alone.
    let fd = unsafe {
        libc::syscall(
            libc::SYS_openat2,
            dirfd,
            c.as_ptr(),
            &how as *const OpenHow,
            std::mem::size_of::<OpenHow>(),
        )
    };
    if fd >= 0 {
        return Ok(unsafe { OwnedFd::from_raw_fd(fd as libc::c_int) });
    }
    Err(std::io::Error::last_os_error())
}

fn dup_fd(fd: libc::c_int) -> std::io::Result<OwnedFd> {
    // SAFETY: `fd` is an open descriptor; a non-negative result is a new fd.
    let n = unsafe { libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, 0) };
    if n < 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(unsafe { OwnedFd::from_raw_fd(n) })
}

fn open_workspace(path: &Path) -> Result<OwnedFd, BindError> {
    let c = c_string(path.as_os_str().as_bytes()).map_err(|e| map_io(path, e))?;
    // SAFETY: `c` is a valid C string. `O_NOFOLLOW` applies to the final
    // component. Ancestors of the workspace root are the trust boundary, the
    // same rule as runtime `open_confined`. A non-negative fd is ours.
    let fd = unsafe {
        libc::open(
            c.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
        )
    };
    if fd < 0 {
        return Err(map_io(path, std::io::Error::last_os_error()));
    }
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

fn mkdir_at(dirfd: libc::c_int, name: &OsStr) -> std::io::Result<()> {
    let c = c_string(name.as_bytes())?;
    // SAFETY: `dirfd` is a directory and `c` is one component, not a path.
    let rc = unsafe { libc::mkdirat(dirfd, c.as_ptr(), 0o755) };
    if rc == 0 {
        return Ok(());
    }
    let err = std::io::Error::last_os_error();
    if err.raw_os_error() == Some(libc::EEXIST) {
        Ok(())
    } else {
        Err(err)
    }
}

/// Stop a single file from being pulled into the grep pipe without a bound.
const READ_LIMIT: u64 = 8 * 1024 * 1024;

fn read_all(file: &mut File) -> Result<Vec<u8>, BindError> {
    file.seek(SeekFrom::Start(0))
        .map_err(|e| BindError::Failed(e.to_string()))?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)
        .map_err(|e| BindError::Failed(e.to_string()))?;
    if buf.len() as u64 > READ_LIMIT {
        return Err(BindError::Denied(
            "refusing the file: it is larger than the read limit".into(),
        ));
    }
    Ok(buf)
}

fn write_all_trunc(file: &mut File, content: &[u8]) -> Result<(), BindError> {
    file.set_len(0)
        .map_err(|e| BindError::Failed(e.to_string()))?;
    file.seek(SeekFrom::Start(0))
        .map_err(|e| BindError::Failed(e.to_string()))?;
    file.write_all(content)
        .map_err(|e| BindError::Failed(e.to_string()))?;
    Ok(())
}

fn reject_special(path: &Path, file: &File) -> Result<(), BindError> {
    let meta = file
        .metadata()
        .map_err(|e| BindError::Failed(e.to_string()))?;
    if meta.file_type().is_file() {
        return Ok(());
    }
    if meta.is_dir() {
        return Err(BindError::Failed("Is a directory (os error 21)".into()));
    }
    Err(BindError::Denied(format!(
        "refusing '{}': authorized target is not a regular file",
        safe_label(path)
    )))
}

/// A file authorized under a workspace, pinned by the descriptor in `cached`
/// once it has been opened. Later reads and writes use that descriptor.
#[derive(Debug)]
pub struct HeldFile {
    workspace: OwnedFd,
    rel: PathBuf,
    authorized: PathBuf,
    display: String,
    cached: Mutex<Option<File>>,
    writable: bool,
}

impl HeldFile {
    pub fn authorized(&self) -> &Path {
        &self.authorized
    }

    pub fn display(&self) -> &str {
        &self.display
    }

    fn cache_lock(&self) -> std::sync::MutexGuard<'_, Option<File>> {
        self.cached.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn open_rel(&self, flags: libc::c_int, mode: u64) -> Result<OwnedFd, BindError> {
        open_beneath(self.workspace.as_raw_fd(), &self.rel, flags, mode)
            .map_err(|e| map_io(&self.authorized, e))
    }

    /// Bytes of the pinned file, or `None` when the authorized name is absent.
    /// A symlink occupying that name is a denial, not a read of its target.
    pub fn read_checkpoint(&self) -> Result<Option<Vec<u8>>, String> {
        self.read_bytes().map_err(|e| e.message().to_string())
    }

    fn read_bytes(&self) -> Result<Option<Vec<u8>>, BindError> {
        let mut guard = self.cache_lock();
        if let Some(file) = guard.as_mut() {
            return read_all(file).map(Some);
        }
        match self.open_rel(
            libc::O_RDONLY | libc::O_NONBLOCK | libc::O_NOFOLLOW | libc::O_CLOEXEC,
            0,
        ) {
            Ok(fd) => {
                let mut file = File::from(fd);
                reject_special(&self.authorized, &file)?;
                let bytes = read_all(&mut file)?;
                *guard = Some(file);
                Ok(Some(bytes))
            }
            Err(BindError::NotFound(_)) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Create missing parent directories beneath the workspace, then write
    /// `content`. Returns whether any parent directory was created.
    /// `overwrite` false refuses an existing file without truncating it.
    pub fn write_bytes(&self, content: &[u8], overwrite: bool) -> Result<bool, BindError> {
        {
            let mut guard = self.cache_lock();
            if let Some(file) = guard.as_mut() {
                if !self.writable {
                    return Err(BindError::Failed(format!(
                        "Permission denied writing {}",
                        safe_label(&self.rel)
                    )));
                }
                if !overwrite {
                    return Err(BindError::Failed(format!(
                        "File already exists: {}. Pass overwrite: true to replace.",
                        safe_label(&self.rel)
                    )));
                }
                write_all_trunc(file, content)?;
                return Ok(false);
            }
        }
        let parent_rel = self.rel.parent().unwrap_or(Path::new(""));
        let created = self.ensure_dirs(parent_rel)?;
        let create_flags =
            libc::O_RDWR | libc::O_CREAT | libc::O_EXCL | libc::O_CLOEXEC | libc::O_NOFOLLOW;
        match self.open_rel(create_flags, 0o666) {
            Ok(fd) => {
                let mut file = File::from(fd);
                write_all_trunc(&mut file, content)?;
                *self.cache_lock() = Some(file);
                Ok(created)
            }
            Err(BindError::Exists(_)) if overwrite => {
                let fd = self.open_rel(
                    libc::O_RDWR | libc::O_NONBLOCK | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                    0,
                )?;
                let mut file = File::from(fd);
                reject_special(&self.authorized, &file)?;
                write_all_trunc(&mut file, content)?;
                *self.cache_lock() = Some(file);
                Ok(created)
            }
            Err(BindError::Exists(_)) => Err(BindError::Failed(format!(
                "File already exists: {}. Pass overwrite: true to replace.",
                safe_label(&self.rel)
            ))),
            Err(e) => Err(e),
        }
    }

    /// Replace the bytes of a file that was already pinned. Does not create.
    pub fn overwrite_existing(&self, content: &[u8]) -> Result<(), BindError> {
        let mut guard = self.cache_lock();
        let Some(file) = guard.as_mut() else {
            return Err(BindError::NotFound(
                "No such file or directory (os error 2)".into(),
            ));
        };
        if !self.writable {
            return Err(BindError::Failed(format!(
                "Permission denied writing {}",
                safe_label(&self.rel)
            )));
        }
        write_all_trunc(file, content)
    }

    /// Create every missing directory in `dir_rel` beneath the workspace fd.
    /// Each new directory is re-opened from the workspace root, so a rename
    /// out of the workspace fails closed instead of continuing the walk.
    /// Names up to the last `..` were resolved by the allow-list; they are
    /// opened as they stand and never created.
    fn ensure_dirs(&self, dir_rel: &Path) -> Result<bool, BindError> {
        if dir_rel.as_os_str().is_empty() {
            return Ok(false);
        }
        let mut created = false;
        let names: Vec<OsString> = dir_rel
            .components()
            .map(|c| c.as_os_str().to_os_string())
            .collect();
        let fixed = names.iter().rposition(|n| n == "..").map_or(0, |i| i + 1);
        let mut acc: PathBuf = names[..fixed].iter().collect();
        for name in &names[fixed..] {
            let parent_fd = if acc.as_os_str().is_empty() {
                dup_fd(self.workspace.as_raw_fd()).map_err(|e| map_io(&self.authorized, e))?
            } else {
                open_beneath(
                    self.workspace.as_raw_fd(),
                    &acc,
                    libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                    0,
                )
                .map_err(|e| map_io(&self.authorized, e))?
            };
            acc.push(name);
            match open_beneath(
                self.workspace.as_raw_fd(),
                &acc,
                libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                0,
            ) {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    mkdir_at(parent_fd.as_raw_fd(), name)
                        .map_err(|err| map_io(&self.authorized, err))?;
                    created = true;
                    open_beneath(
                        self.workspace.as_raw_fd(),
                        &acc,
                        libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                        0,
                    )
                    .map_err(|err| map_io(&self.authorized, err))?;
                }
                Err(e) => return Err(map_io(&self.authorized, e)),
            }
        }
        Ok(created)
    }
}

struct PinnedOpen {
    workspace: OwnedFd,
    rel: PathBuf,
    cached: Option<File>,
}

fn open_pinned(
    auth: &Authorized,
    flags: libc::c_int,
    regular_only: bool,
) -> Result<PinnedOpen, BindError> {
    let workspace = open_workspace(&auth.workspace)?;
    run_before();
    match open_beneath(workspace.as_raw_fd(), &auth.rel, flags, 0) {
        Ok(fd) => {
            let file = File::from(fd);
            if regular_only {
                reject_special(&auth.canonical, &file)?;
            }
            Ok(PinnedOpen {
                workspace,
                rel: auth.rel.clone(),
                cached: Some(file),
            })
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound && !regular_only => {
            Err(map_io(&auth.canonical, e))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(PinnedOpen {
            workspace,
            rel: auth.rel.clone(),
            cached: None,
        }),
        Err(e) if symlink_component(&e) && final_component_is_symlink(&workspace, &auth.rel) => {
            let canon_ws = open_workspace(&auth.canon_workspace)?;
            let fd = open_beneath(canon_ws.as_raw_fd(), &auth.canon_rel, flags, 0)
                .map_err(|err| map_io(&auth.canonical, err))?;
            let file = File::from(fd);
            if regular_only {
                reject_special(&auth.canonical, &file)?;
            }
            Ok(PinnedOpen {
                workspace: canon_ws,
                rel: auth.canon_rel.clone(),
                cached: Some(file),
            })
        }
        Err(e) => Err(map_io(&auth.canonical, e)),
    }
}

fn hold(path: &str, write: bool) -> Result<HeldFile, BindError> {
    let auth = authorize(path)?;
    // `Path` drops a trailing `/` or `/.`; POSIX does not. Such a name is a
    // directory, so a write must not create a regular file under it.
    if auth.rel.as_os_str().is_empty() || (write && (path.ends_with('/') || path.ends_with("/.")))
    {
        return Err(BindError::Failed("Is a directory (os error 21)".into()));
    }
    let flags = if write {
        libc::O_RDWR | libc::O_NONBLOCK | libc::O_NOFOLLOW | libc::O_CLOEXEC
    } else {
        libc::O_RDONLY | libc::O_NONBLOCK | libc::O_NOFOLLOW | libc::O_CLOEXEC
    };
    let pinned = open_pinned(&auth, flags, true)?;
    run_after();
    Ok(HeldFile {
        workspace: pinned.workspace,
        rel: pinned.rel,
        authorized: auth.canonical,
        display: auth.display,
        cached: Mutex::new(pinned.cached),
        writable: write,
    })
}

pub fn hold_for_write(path: &str) -> Result<HeldFile, BindError> {
    hold(path, true)
}

pub fn read_text(path: &str) -> Result<(PathBuf, String), BindError> {
    let held = hold(path, false)?;
    let Some(bytes) = held.read_bytes()? else {
        return Err(BindError::NotFound(
            "No such file or directory (os error 2)".into(),
        ));
    };
    let text = String::from_utf8(bytes)
        .map_err(|_| BindError::Failed("stream did not contain valid UTF-8".into()))?;
    Ok((held.authorized, text))
}

#[derive(Debug)]
enum NodeKind {
    File,
    Dir,
    Link,
    Other,
}

fn kind_of(mode: u32) -> NodeKind {
    match mode & libc::S_IFMT {
        libc::S_IFREG => NodeKind::File,
        libc::S_IFDIR => NodeKind::Dir,
        libc::S_IFLNK => NodeKind::Link,
        _ => NodeKind::Other,
    }
}

struct DirEnt {
    name: OsString,
    kind: NodeKind,
    len: u64,
}

fn list_dirfd(dirfd: libc::c_int) -> std::io::Result<Vec<DirEnt>> {
    let raw = dup_fd(dirfd)?.into_raw_fd();
    // SAFETY: `raw` is a live fd. On success fdopendir owns it; on failure
    // we close it below.
    let dir = unsafe { libc::fdopendir(raw) };
    if dir.is_null() {
        let err = std::io::Error::last_os_error();
        unsafe { libc::close(raw) };
        return Err(err);
    }
    let mut out = Vec::new();
    let walked = loop {
        // SAFETY: `dir` is a live DIR. A null readdir is EOF when errno is 0.
        unsafe { *libc::__errno_location() = 0 };
        let ent = unsafe { libc::readdir(dir) };
        if ent.is_null() {
            let errno = unsafe { *libc::__errno_location() };
            break if errno == 0 {
                Ok(())
            } else {
                Err(std::io::Error::from_raw_os_error(errno))
            };
        }
        let name_c = unsafe { std::ffi::CStr::from_ptr((*ent).d_name.as_ptr()) };
        let name = OsStr::from_bytes(name_c.to_bytes()).to_os_string();
        if name == "." || name == ".." {
            continue;
        }
        let Ok(c_name) = c_string(name.as_bytes()) else {
            continue;
        };
        let mut st = std::mem::MaybeUninit::<libc::stat>::uninit();
        // SAFETY: `dirfd` is still the caller's directory (fdopendir consumed
        // only the duplicate). `c_name` is one entry name. On success `st`
        // is initialized.
        let rc = unsafe {
            libc::fstatat(
                dirfd,
                c_name.as_ptr(),
                st.as_mut_ptr(),
                libc::AT_SYMLINK_NOFOLLOW,
            )
        };
        if rc != 0 {
            continue;
        }
        let st = unsafe { st.assume_init() };
        let len = if st.st_size < 0 { 0 } else { st.st_size as u64 };
        out.push(DirEnt {
            name,
            kind: kind_of(st.st_mode as u32),
            len,
        });
    };
    // SAFETY: `dir` came from fdopendir and has not been closed.
    unsafe { libc::closedir(dir) };
    walked?;
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

fn open_node(path: &str, directory: bool) -> Result<(Authorized, OwnedFd), BindError> {
    let auth = authorize(path)?;
    let flags = if directory {
        libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC
    } else {
        libc::O_RDONLY | libc::O_NONBLOCK | libc::O_NOFOLLOW | libc::O_CLOEXEC
    };
    let pinned = open_pinned(&auth, flags, false)?;
    run_after();
    let Some(file) = pinned.cached else {
        return Err(BindError::NotFound(
            "No such file or directory (os error 2)".into(),
        ));
    };
    Ok((auth, OwnedFd::from(file)))
}

#[derive(Debug, Clone)]
pub struct ListedEntry {
    pub name: String,
    pub is_dir: bool,
    pub size_bytes: u64,
}

pub fn list_dir(path: &str) -> Result<(PathBuf, Vec<ListedEntry>), BindError> {
    let (auth, fd) = open_node(path, true)?;
    let ents = list_dirfd(fd.as_raw_fd()).map_err(|e| map_io(&auth.canonical, e))?;
    let rows = ents
        .into_iter()
        .map(|e| ListedEntry {
            name: e.name.to_string_lossy().into_owned(),
            is_dir: matches!(e.kind, NodeKind::Dir),
            size_bytes: e.len,
        })
        .collect();
    Ok((auth.canonical, rows))
}

pub fn make_dir(path: &str) -> Result<PathBuf, BindError> {
    let auth = authorize(path)?;
    // An existing directory, or a final-component symlink to one, is opened
    // the way `list_dir` opens it. Only a missing name goes on to be created.
    match open_pinned(
        &auth,
        libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        false,
    ) {
        Ok(_) => {
            run_after();
            return Ok(auth.canonical);
        }
        Err(BindError::NotFound(_)) => {}
        Err(e) => return Err(e),
    }
    let held = HeldFile {
        workspace: open_workspace(&auth.workspace)?,
        rel: auth.rel.clone(),
        authorized: auth.canonical.clone(),
        display: auth.display.clone(),
        cached: Mutex::new(None),
        writable: true,
    };
    held.ensure_dirs(&auth.rel)?;
    // Confirm the final directory is still the one beneath the workspace.
    open_beneath(
        held.workspace.as_raw_fd(),
        &auth.rel,
        libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        0,
    )
    .map_err(|e| map_io(&auth.canonical, e))?;
    run_after();
    Ok(auth.canonical)
}

fn grep_bytes(query: &str, bytes: &[u8]) -> Result<Vec<String>, BindError> {
    let mut child = Command::new("grep")
        .args(["-nI", "-e", query])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| BindError::Failed(format!("failed to run grep: {e}")))?;
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| BindError::Failed("grep stdin is unavailable".into()))?;
    let mut stdout = child
        .stdout
        .take()
        .ok_or_else(|| BindError::Failed("grep stdout is unavailable".into()))?;
    let mut stderr = child
        .stderr
        .take()
        .ok_or_else(|| BindError::Failed("grep stderr is unavailable".into()))?;
    // Write stdin on another thread. Reading stdout only after the write
    // fills both pipes once the match output exceeds the kernel buffer.
    let payload = bytes.to_vec();
    let writer = std::thread::spawn(move || {
        let _ = stdin.write_all(&payload);
    });
    let err_reader = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = stderr.read_to_end(&mut buf);
        buf
    });
    let mut out = Vec::new();
    let read_result = stdout.read_to_end(&mut out);
    let status = child.wait();
    let _ = writer.join();
    let err_buf = err_reader.join().unwrap_or_default();
    read_result.map_err(|e| BindError::Failed(format!("grep failed: {e}")))?;
    let status = status.map_err(|e| BindError::Failed(format!("grep failed: {e}")))?;
    if status.code() == Some(2) {
        let err = String::from_utf8_lossy(&err_buf);
        return Err(BindError::Failed(format!(
            "grep refused the pattern: {}",
            err.trim()
        )));
    }
    let text = String::from_utf8_lossy(&out);
    Ok(text.lines().map(|s| s.to_string()).collect())
}

fn display_of(display_root: &str, root: &Path, child: &Path) -> String {
    match child.strip_prefix(root) {
        Ok(rel) if !rel.as_os_str().is_empty() => {
            Path::new(display_root).join(rel).display().to_string()
        }
        _ => display_root.to_string(),
    }
}

fn name_matches(pattern: &str, name: &str) -> bool {
    let Ok(pat) = CString::new(pattern) else {
        return false;
    };
    let Ok(nm) = CString::new(name) else {
        return false;
    };
    // SAFETY: both pointers are valid C strings. fnmatch does not retain them.
    unsafe { libc::fnmatch(pat.as_ptr(), nm.as_ptr(), 0) == 0 }
}

fn open_child(dirfd: libc::c_int, name: &OsStr, flags: libc::c_int) -> std::io::Result<OwnedFd> {
    let c = c_string(name.as_bytes())?;
    // SAFETY: `dirfd` is a directory we hold and `c` is a single component.
    let fd = unsafe { libc::openat(dirfd, c.as_ptr(), flags) };
    if fd < 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

/// Directory descriptors held by one walk: the root plus the ancestors of
/// the directory currently being listed. Sibling directories stay as names
/// until their turn, so a wide tree does not open one fd per child.
const MAX_WALK_DEPTH: usize = 64;

#[cfg(test)]
thread_local! {
    static WALK_FD_MAX: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn note_walk_fds(n: usize) {
    WALK_FD_MAX.with(|c| {
        if n > c.get() {
            c.set(n);
        }
    });
}

#[cfg(test)]
fn take_walk_fd_max() -> usize {
    WALK_FD_MAX.with(|c| c.replace(0))
}

#[cfg(not(test))]
fn note_walk_fds(_n: usize) {}

struct WalkFrame {
    dir: OwnedFd,
    path: PathBuf,
    listed: bool,
    subdirs: Vec<OsString>,
}

/// Walk `root`, never following symlinks. `visit` returns false to stop.
fn walk_nofollow(
    root: &Path,
    dirfd: libc::c_int,
    limit_hit: &mut bool,
    mut visit: impl FnMut(&Path, &DirEnt, Option<File>) -> Result<bool, BindError>,
) -> Result<(), BindError> {
    let mut stack = vec![WalkFrame {
        dir: dup_fd(dirfd).map_err(|e| map_io(root, e))?,
        path: root.to_path_buf(),
        listed: false,
        subdirs: Vec::new(),
    }];
    note_walk_fds(stack.len());
    while !stack.is_empty() {
        if *limit_hit {
            break;
        }
        if !stack.last().is_some_and(|f| f.listed) {
            let fd = stack.last().unwrap().dir.as_raw_fd();
            let dir_path = stack.last().unwrap().path.clone();
            let ents = list_dirfd(fd).map_err(|e| map_io(&dir_path, e))?;
            let mut subdirs = Vec::new();
            for ent in ents {
                let child_path = dir_path.join(&ent.name);
                match ent.kind {
                    NodeKind::Dir => {
                        let cont = visit(&child_path, &ent, None)?;
                        if !cont {
                            *limit_hit = true;
                            break;
                        }
                        subdirs.push(ent.name);
                    }
                    NodeKind::File => {
                        match open_child(
                            fd,
                            &ent.name,
                            libc::O_RDONLY | libc::O_NONBLOCK | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                        ) {
                            Ok(raw) => {
                                let file = File::from(raw);
                                let cont = visit(&child_path, &ent, Some(file))?;
                                if !cont {
                                    *limit_hit = true;
                                    break;
                                }
                            }
                            Err(e) if e.raw_os_error() == Some(libc::ELOOP) => {}
                            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                            Err(_) => {}
                        }
                    }
                    NodeKind::Link | NodeKind::Other => {
                        let cont = visit(&child_path, &ent, None)?;
                        if !cont {
                            *limit_hit = true;
                            break;
                        }
                    }
                }
            }
            let frame = stack.last_mut().unwrap();
            frame.subdirs = subdirs;
            frame.listed = true;
            continue;
        }
        let name = stack.last_mut().unwrap().subdirs.pop();
        let Some(name) = name else {
            stack.pop();
            continue;
        };
        if stack.len() >= MAX_WALK_DEPTH {
            return Err(BindError::Denied(
                "refusing the directory walk: the tree is too deep".into(),
            ));
        }
        let parent = stack.last().unwrap().dir.as_raw_fd();
        let child_path = stack.last().unwrap().path.join(&name);
        match open_child(
            parent,
            &name,
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        ) {
            Ok(child) => {
                stack.push(WalkFrame {
                    dir: child,
                    path: child_path,
                    listed: false,
                    subdirs: Vec::new(),
                });
                note_walk_fds(stack.len());
            }
            Err(e) if e.raw_os_error() == Some(libc::ELOOP) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(map_io(&child_path, e)),
        }
    }
    Ok(())
}

pub fn grep_search(path: &str, query: &str, limit: usize) -> Result<Vec<String>, BindError> {
    // One open. A second lookup after this returns would authorize whatever
    // the name points at now, which is a different request.
    let (auth, fd) = open_node(path, false)?;
    let mut file = File::from(fd);
    let meta = file
        .metadata()
        .map_err(|e| BindError::Failed(e.to_string()))?;
    let mut matches = Vec::new();
    if meta.is_file() {
        let bytes = read_all(&mut file)?;
        for line in grep_bytes(query, &bytes)? {
            matches.push(format!("{}:{line}", auth.display));
            if matches.len() >= limit {
                break;
            }
        }
        return Ok(matches);
    }
    if !meta.is_dir() {
        return Err(BindError::Denied(format!(
            "refusing '{}': authorized target is not a file or directory",
            safe_label(&auth.rel)
        )));
    }
    let root = auth.canonical.clone();
    let shown = auth.display.clone();
    let mut stop = false;
    walk_nofollow(&root, file.as_raw_fd(), &mut stop, |child, ent, file| {
        if !matches!(ent.kind, NodeKind::File) {
            return Ok(true);
        }
        let Some(mut file) = file else {
            return Ok(true);
        };
        let bytes = read_all(&mut file)?;
        for line in grep_bytes(query, &bytes)? {
            matches.push(format!("{}:{line}", display_of(&shown, &root, child)));
            if matches.len() >= limit {
                return Ok(false);
            }
        }
        Ok(true)
    })?;
    Ok(matches)
}

pub fn find_names(path: &str, pattern: &str, limit: usize) -> Result<Vec<String>, BindError> {
    let (auth, fd) = open_node(path, false)?;
    let file = File::from(fd);
    let meta = file
        .metadata()
        .map_err(|e| BindError::Failed(e.to_string()))?;
    let mut matches = Vec::new();
    // Match the root by the name the caller gave, as find(1) does: a
    // symlinked root keeps its own name. `..` or `.` has no name to match.
    let root_name = Path::new(&auth.display)
        .file_name()
        .or_else(|| auth.canonical.file_name())
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    if meta.is_file() {
        if name_matches(pattern, &root_name) {
            matches.push(auth.display.clone());
        }
        return Ok(matches);
    }
    if !meta.is_dir() {
        return Err(BindError::Denied(format!(
            "refusing '{}': authorized target is not a file or directory",
            safe_label(&auth.rel)
        )));
    }
    if name_matches(pattern, &root_name) {
        matches.push(auth.display.clone());
    }
    let root = auth.canonical.clone();
    let shown = auth.display.clone();
    let mut stop = matches.len() >= limit;
    walk_nofollow(&root, file.as_raw_fd(), &mut stop, |child, ent, _| {
        let name = ent.name.to_string_lossy();
        if name_matches(pattern, &name) {
            matches.push(display_of(&shown, &root, child));
            if matches.len() >= limit {
                return Ok(false);
            }
        }
        Ok(true)
    })?;
    Ok(matches)
}

#[cfg(test)]
pub mod test_hooks {
    use std::cell::RefCell;

    thread_local! {
        static BEFORE: RefCell<Option<Box<dyn FnOnce()>>> = const { RefCell::new(None) };
        static AFTER: RefCell<Option<Box<dyn FnOnce()>>> = const { RefCell::new(None) };
        static OPENAT2_ERR: std::cell::Cell<Option<i32>> = const { std::cell::Cell::new(None) };
        static WORKSPACES: RefCell<Option<Vec<std::path::PathBuf>>> = const { RefCell::new(None) };
    }

    /// `permitted_workspaces` returns exactly `roots` on this thread, so a
    /// test can hold two permitted roots without touching real user dirs.
    pub fn set_workspaces(roots: Vec<std::path::PathBuf>) {
        WORKSPACES.with(|c| *c.borrow_mut() = Some(roots));
    }

    pub(super) fn workspaces() -> Option<Vec<std::path::PathBuf>> {
        WORKSPACES.with(|c| c.borrow().clone())
    }

    /// Next `openat2` returns this errno instead of calling the kernel.
    /// One shot, so a later open in the same call is real.
    pub fn set_openat2_error(code: i32) {
        OPENAT2_ERR.with(|c| c.set(Some(code)));
    }

    pub(super) fn take_openat2_error() -> Option<i32> {
        OPENAT2_ERR.with(|c| c.take())
    }

    pub fn set_before_open(f: impl FnOnce() + 'static) {
        BEFORE.with(|c| *c.borrow_mut() = Some(Box::new(f)));
    }

    pub fn set_after_open(f: impl FnOnce() + 'static) {
        AFTER.with(|c| *c.borrow_mut() = Some(Box::new(f)));
    }

    pub fn clear() {
        BEFORE.with(|c| *c.borrow_mut() = None);
        AFTER.with(|c| *c.borrow_mut() = None);
        OPENAT2_ERR.with(|c| c.set(None));
        WORKSPACES.with(|c| *c.borrow_mut() = None);
    }

    pub(super) fn run_before() {
        if let Some(f) = BEFORE.with(|c| c.borrow_mut().take()) {
            f();
        }
    }

    pub(super) fn run_after() {
        if let Some(f) = AFTER.with(|c| c.borrow_mut().take()) {
            f();
        }
    }
}

#[cfg(not(test))]
fn run_before() {}

#[cfg(not(test))]
fn run_after() {}

#[cfg(test)]
fn run_before() {
    test_hooks::run_before();
}

#[cfg(test)]
fn run_after() {
    test_hooks::run_after();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};

    const INSIDE: &str = "INSIDE_MARKER";
    const OTHER: &str = "OTHER_MARKER";
    const OUTSIDE: &str = "OUTSIDE_MARKER";

    struct ClearHooks;
    impl Drop for ClearHooks {
        fn drop(&mut self) {
            test_hooks::clear();
        }
    }

    struct Scratch {
        _root: tempfile::TempDir,
        ws: PathBuf,
        other: PathBuf,
    }

    fn scratch() -> Scratch {
        let root = tempfile::tempdir_in("/tmp").unwrap();
        let root_path = root.path().canonicalize().unwrap();
        let ws = root_path.join("ws");
        let other = root_path.join("other");
        std::fs::create_dir_all(ws.join("d")).unwrap();
        std::fs::create_dir_all(&other).unwrap();
        std::fs::write(ws.join("d/f"), INSIDE).unwrap();
        std::fs::write(other.join("f"), OTHER).unwrap();
        Scratch {
            _root: root,
            ws,
            other,
        }
    }

    fn outside_scratch() -> tempfile::TempDir {
        let base = PathBuf::from("/dev/shm");
        assert!(
            base.is_dir(),
            "/dev/shm is required for an unpermitted scratch dir"
        );
        let dir = tempfile::tempdir_in(&base).unwrap();
        let marker = dir.path().join("secret");
        std::fs::write(&marker, OUTSIDE).unwrap();
        assert!(
            validate_workspace_path(&marker, &permitted_workspaces()).is_err(),
            "unpermitted scratch resolved inside a workspace; the check would not prove confinement"
        );
        dir
    }

    /// A fresh open during a directory swap may see the original file or a
    /// denial. The other workspace file and the unpermitted file are rejected.
    /// Accepting either of those is the broken implementation.
    fn fresh_swap_read_is_inside(text: &str) -> bool {
        text == INSIDE
    }

    #[test]
    fn fresh_swap_read_rejects_the_other_and_outside_markers() {
        assert!(fresh_swap_read_is_inside(INSIDE));
        assert!(!fresh_swap_read_is_inside(OTHER));
        assert!(!fresh_swap_read_is_inside(OUTSIDE));
    }

    fn swap_dir(dir: &Path, elsewhere: &Path) {
        let parked = dir.with_file_name("d.held");
        std::fs::rename(dir, &parked).unwrap();
        symlink(elsewhere, dir).unwrap();
    }

    #[test]
    fn ordinary_read_write_list_create_search_round_trip() {
        let root = tempfile::tempdir_in("/tmp").unwrap();
        let root = root.path().canonicalize().unwrap();
        let file = root.join("note.txt");
        let held = hold_for_write(file.to_str().unwrap()).unwrap();
        held.write_bytes(b"alpha beta", true).unwrap();
        let (path, text) = read_text(file.to_str().unwrap()).unwrap();
        assert_eq!(path, file);
        assert_eq!(text, "alpha beta");
        held.overwrite_existing(b"alpha gamma").unwrap();
        let (_, text) = read_text(file.to_str().unwrap()).unwrap();
        assert_eq!(text, "alpha gamma");

        let nested = root.join("sub/leaf");
        let created = make_dir(nested.to_str().unwrap()).unwrap();
        assert!(created.is_dir());
        let (listed_at, rows) = list_dir(root.to_str().unwrap()).unwrap();
        assert_eq!(listed_at, root);
        assert!(rows.iter().any(|r| r.name == "note.txt" && !r.is_dir));
        assert!(rows.iter().any(|r| r.name == "sub" && r.is_dir));

        let hits = grep_search(root.to_str().unwrap(), "gamma", 50).unwrap();
        assert!(hits
            .iter()
            .any(|h| h.contains("note.txt") && h.contains("gamma")));
        let found = find_names(root.to_str().unwrap(), "note.txt", 50).unwrap();
        assert_eq!(found, vec![file.display().to_string()]);
    }

    #[test]
    fn in_workspace_symlink_reads_the_canonical_file() {
        let s = scratch();
        let link = s.ws.join("alias");
        symlink(s.ws.join("d/f"), &link).unwrap();
        let (_, text) = read_text(link.to_str().unwrap()).unwrap();
        assert_eq!(text, INSIDE);
    }

    #[test]
    fn symlink_to_unpermitted_target_is_denied() {
        let s = scratch();
        let outside = outside_scratch();
        let link = s.ws.join("leak");
        symlink(outside.path().join("secret"), &link).unwrap();
        let err = read_text(link.to_str().unwrap()).unwrap_err();
        assert!(matches!(err, BindError::Denied(_)), "{err:?}");
        assert!(!err.message().contains(OUTSIDE));
        assert_eq!(
            std::fs::read_to_string(outside.path().join("secret")).unwrap(),
            OUTSIDE
        );
    }

    #[test]
    fn parent_dotdot_through_symlink_is_denied() {
        let s = scratch();
        let outside = outside_scratch();
        std::fs::create_dir(outside.path().join("child")).unwrap();
        symlink(outside.path().join("child"), s.ws.join("link")).unwrap();
        let probe = s.ws.join("link/../secret");
        let err = read_text(probe.to_str().unwrap()).unwrap_err();
        assert!(matches!(err, BindError::Denied(_)), "{err:?}");
        assert!(!err.message().contains(OUTSIDE));
    }

    const DECOY: &str = "DECOY_MARKER";
    const REAL: &str = "REAL_MARKER";

    /// `ws/link -> ws/a/b`. `ws/a/x` holds REAL when `real` is set and `ws/x`
    /// holds DECOY when `decoy` is set. The OS and the allow-list resolve
    /// `ws/link/../x` to `ws/a/x`; dropping `link/..` as text names `ws/x`.
    fn dotdot_layout(real: bool, decoy: bool) -> Scratch {
        let s = scratch();
        std::fs::create_dir_all(s.ws.join("a/b")).unwrap();
        symlink(s.ws.join("a/b"), s.ws.join("link")).unwrap();
        if real {
            std::fs::write(s.ws.join("a/x"), REAL).unwrap();
        }
        if decoy {
            std::fs::write(s.ws.join("x"), DECOY).unwrap();
        }
        s
    }

    /// The call either acted on the authorized object (`ok`) or was denied.
    fn authorized_or_denied<T: std::fmt::Debug>(r: Result<T, BindError>, ok: impl FnOnce(T)) {
        match r {
            Ok(v) => ok(v),
            Err(BindError::Denied(m)) => {
                assert!(!m.contains(DECOY) && !m.contains(REAL), "{m}")
            }
            Err(e) => panic!("expected the authorized object or a denial, got {e:?}"),
        }
    }

    fn write_through(path: &Path, content: &[u8]) -> Result<bool, BindError> {
        hold_for_write(path.to_str().unwrap())?.write_bytes(content, true)
    }

    #[test]
    fn dotdot_after_a_symlink_never_reads_the_lexical_decoy() {
        let s = dotdot_layout(true, true);
        let probe = s.ws.join("link/../x");
        assert_eq!(
            validate_workspace_path(&probe, &permitted_workspaces()).unwrap(),
            s.ws.join("a/x")
        );
        assert_eq!(std::fs::read_to_string(&probe).unwrap(), REAL);
        authorized_or_denied(read_text(probe.to_str().unwrap()), |(path, text)| {
            assert_eq!(text, REAL, "opened a different file from the authorized one");
            assert_eq!(path, s.ws.join("a/x"));
        });
        authorized_or_denied(
            grep_search(probe.to_str().unwrap(), "MARKER", 50),
            |hits| assert!(hits.iter().all(|h| !h.contains(DECOY)), "{hits:?}"),
        );
    }

    #[test]
    fn dotdot_after_a_symlink_never_overwrites_the_lexical_decoy() {
        let s = dotdot_layout(true, true);
        let probe = s.ws.join("link/../x");
        authorized_or_denied(write_through(&probe, b"NEW"), |_| {
            assert_eq!(std::fs::read_to_string(s.ws.join("a/x")).unwrap(), "NEW")
        });
        assert_eq!(std::fs::read_to_string(s.ws.join("x")).unwrap(), DECOY);
    }

    #[test]
    fn dotdot_after_a_symlink_never_creates_the_lexical_decoy() {
        for real in [true, false] {
            let s = dotdot_layout(real, false);
            let probe = s.ws.join("link/../x");
            authorized_or_denied(write_through(&probe, b"NEW"), |_| {
                assert!(real, "created a file the allow-list never authorized");
                assert_eq!(std::fs::read_to_string(s.ws.join("a/x")).unwrap(), "NEW")
            });
            assert!(
                s.ws.join("x").symlink_metadata().is_err(),
                "write through link/.. created ws/x (real={real})"
            );
        }
    }

    #[test]
    fn dotdot_after_a_symlink_never_lists_or_makes_the_lexical_parent() {
        let s = dotdot_layout(true, true);
        authorized_or_denied(
            list_dir(s.ws.join("link/..").to_str().unwrap()),
            |(path, rows)| {
                assert_eq!(path, s.ws.join("a"));
                let names: Vec<_> = rows.iter().map(|r| r.name.as_str()).collect();
                assert_eq!(names, ["b", "x"], "listed a different directory");
            },
        );
        authorized_or_denied(
            make_dir(s.ws.join("link/../made").to_str().unwrap()),
            |path| {
                assert_eq!(path, s.ws.join("a/made"));
                assert!(path.is_dir());
            },
        );
        assert!(s.ws.join("made").symlink_metadata().is_err());
    }

    #[test]
    fn dotdot_after_a_symlink_never_crosses_into_the_lexical_workspace() {
        let _clear = ClearHooks;
        let root = tempfile::tempdir_in("/tmp").unwrap();
        let base = root.path().canonicalize().unwrap();
        let (ws_a, ws_b) = (base.join("A"), base.join("B"));
        std::fs::create_dir_all(ws_b.join("sub")).unwrap();
        std::fs::create_dir_all(&ws_a).unwrap();
        std::fs::write(ws_a.join("x"), DECOY).unwrap();
        std::fs::write(ws_b.join("x"), REAL).unwrap();
        symlink(ws_b.join("sub"), ws_a.join("link")).unwrap();
        test_hooks::set_workspaces(vec![ws_a.clone(), ws_b.clone()]);
        let probe = ws_a.join("link/../x");
        // Both files are permitted; the allow-list picked the one in B.
        assert_eq!(
            validate_workspace_path(&probe, &permitted_workspaces()).unwrap(),
            ws_b.join("x")
        );
        assert!(validate_workspace_path(ws_a.join("x"), &permitted_workspaces()).is_ok());
        authorized_or_denied(read_text(probe.to_str().unwrap()), |(_, text)| {
            assert_eq!(text, REAL, "read workspace A for a path authorized in B")
        });
        authorized_or_denied(write_through(&probe, b"NEW"), |_| {
            assert_eq!(std::fs::read_to_string(ws_b.join("x")).unwrap(), "NEW")
        });
        assert_eq!(std::fs::read_to_string(ws_a.join("x")).unwrap(), DECOY);
    }

    #[test]
    fn dotdot_through_real_directories_still_opens_the_resolved_path() {
        let s = scratch();
        let (_, text) = read_text(s.ws.join("d/../d/f").to_str().unwrap()).unwrap();
        assert_eq!(text, INSIDE);
        symlink(s.ws.join("d/f"), s.ws.join("alias")).unwrap();
        let (_, text) = read_text(s.ws.join("d/../alias").to_str().unwrap()).unwrap();
        assert_eq!(text, INSIDE);
        write_through(&s.ws.join("d/../d/g"), b"G").unwrap();
        assert_eq!(std::fs::read_to_string(s.ws.join("d/g")).unwrap(), "G");
        let made = make_dir(s.ws.join("d/../d/sub/leaf").to_str().unwrap()).unwrap();
        assert_eq!(made, s.ws.join("d/sub/leaf"));
        assert!(made.is_dir());
    }

    #[test]
    fn dotdot_above_the_deepest_workspace_uses_the_enclosing_one() {
        let _clear = ClearHooks;
        let s = scratch();
        test_hooks::set_workspaces(vec![s.ws.parent().unwrap().to_path_buf(), s.ws.clone()]);
        let (_, text) = read_text(s.ws.join("../other/f").to_str().unwrap()).unwrap();
        assert_eq!(text, OTHER);
    }

    #[test]
    fn write_to_a_trailing_slash_name_never_creates_a_file() {
        let s = scratch();
        let named = |rest: &str| format!("{}/{rest}", s.ws.display());
        for name in ["d/new/", "d/new/."] {
            let err = hold_for_write(&named(name))
                .and_then(|h| h.write_bytes(b"x", false))
                .unwrap_err();
            assert!(err.message().contains("Is a directory"), "{name}: {err:?}");
            assert!(
                s.ws.join("d/new").symlink_metadata().is_err(),
                "{name} created d/new"
            );
        }
        // Directory-shaped names keep working where a directory is meant.
        assert!(matches!(
            read_text(&named("d/new/")),
            Err(BindError::NotFound(_))
        ));
        make_dir(&named("d/new/")).unwrap();
        assert!(s.ws.join("d/new").is_dir());
        let (_, rows) = list_dir(&named("d/")).unwrap();
        assert!(rows.iter().any(|r| r.name == "new" && r.is_dir));
    }

    /// `real` is configured as `alias -> real`, alone or under an enclosing
    /// root that also matches the named path.
    fn symlinked_root_is_usable(nested: bool) {
        let _clear = ClearHooks;
        let s = scratch();
        let base = s.ws.parent().unwrap().to_path_buf();
        let real = s.ws.clone();
        let alias = base.join("alias");
        symlink(&real, &alias).unwrap();
        let at = |rest: &str| alias.join(rest).to_str().unwrap().to_string();
        let roots = if nested {
            vec![base.clone(), alias.clone()]
        } else {
            vec![alias.clone()]
        };
        test_hooks::set_workspaces(roots);
        assert_eq!(
            validate_workspace_path(alias.join("d/f"), &permitted_workspaces()).unwrap(),
            real.join("d/f")
        );
        let (path, text) = read_text(&at("d/f")).unwrap();
        assert_eq!((path, text.as_str()), (real.join("d/f"), INSIDE));
        hold_for_write(&at("g"))
            .and_then(|h| h.write_bytes(b"G", false))
            .unwrap();
        assert_eq!(std::fs::read_to_string(real.join("g")).unwrap(), "G");
        assert_eq!(make_dir(&at("n")).unwrap(), real.join("n"));
        assert!(real.join("n").is_dir());
        let (listed, _) = list_dir(&at("d")).unwrap();
        assert_eq!(listed, real.join("d"));
    }

    #[test]
    fn a_workspace_root_configured_through_a_symlink_is_usable() {
        symlinked_root_is_usable(false);
    }

    #[test]
    fn a_symlinked_root_nested_under_another_root_is_usable() {
        symlinked_root_is_usable(true);
    }

    #[test]
    fn make_dir_accepts_an_existing_directory_symlink_like_list_dir() {
        let s = scratch();
        let dl = s.ws.join("dl");
        symlink(s.ws.join("d"), &dl).unwrap();
        assert_eq!(list_dir(dl.to_str().unwrap()).unwrap().0, s.ws.join("d"));
        assert_eq!(make_dir(dl.to_str().unwrap()).unwrap(), s.ws.join("d"));
        assert!(dl.symlink_metadata().unwrap().file_type().is_symlink());
        // A symlink to a file is not a directory.
        let fl = s.ws.join("fl");
        symlink(s.ws.join("d/f"), &fl).unwrap();
        assert!(make_dir(fl.to_str().unwrap()).is_err());
        assert_eq!(std::fs::read_to_string(s.ws.join("d/f")).unwrap(), INSIDE);
        // A symlink before the last component stays refused.
        assert!(matches!(
            make_dir(dl.join("new").to_str().unwrap()),
            Err(BindError::Denied(_))
        ));
        assert!(!s.ws.join("d/new").exists());
    }

    #[test]
    fn find_names_matches_a_symlinked_root_by_the_name_given() {
        let s = scratch();
        std::fs::write(s.ws.join("d/hit.txt"), "HIT\n").unwrap();
        let dl = s.ws.join("dl");
        let fl = s.ws.join("fl");
        symlink(s.ws.join("d"), &dl).unwrap();
        symlink(s.ws.join("d/hit.txt"), &fl).unwrap();
        let find = |p: &Path, pat: &str| find_names(p.to_str().unwrap(), pat, 50).unwrap();
        let (dl_s, fl_s) = (dl.display().to_string(), fl.display().to_string());
        // find(1) semantics: the root is named by the caller's last component.
        assert_eq!(find(&dl, "dl"), vec![dl_s.clone()]);
        assert!(!find(&dl, "d").contains(&dl_s));
        assert_eq!(find(&fl, "fl"), vec![fl_s.clone()]);
        assert!(find(&fl, "hit.txt").is_empty());
        // No name to match (`..`): the resolved directory's name is used.
        let up = s.ws.join("d/..");
        assert_eq!(find(&up, "ws"), vec![up.display().to_string()]);
    }

    fn assert_no_host_path(msg: &str, hidden: &[&Path]) {
        for prefix in ["/tmp", "/dev", "/etc", "/home", "/workspace", "/usr"] {
            assert!(!msg.contains(prefix), "{msg}");
        }
        for path in hidden {
            let text = path.display().to_string();
            if !text.is_empty() {
                assert!(!msg.contains(&text), "{msg} contains {text}");
            }
        }
    }

    #[test]
    fn error_text_omits_host_absolute_paths() {
        let s = scratch();
        let outside = outside_scratch();
        let link = s.ws.join("leak");
        symlink(outside.path().join("secret"), &link).unwrap();
        let err = read_text(link.to_str().unwrap()).unwrap_err();
        assert_no_host_path(err.message(), &[outside.path(), &s.ws, &link]);

        swap_dir(&s.ws.join("d"), &s.other);
        let err = read_text(s.ws.join("d/f").to_str().unwrap()).unwrap_err();
        assert_no_host_path(err.message(), &[&s.ws, &s.other, &s.ws.join("d/f")]);

        let err = read_text("/etc/passwd").unwrap_err();
        assert_no_host_path(
            err.message(),
            &[Path::new("/etc/passwd"), Path::new("/etc")],
        );
    }

    #[test]
    fn directory_symlink_does_not_read_the_other_file() {
        let s = scratch();
        let target = s.ws.join("d/f");
        swap_dir(&s.ws.join("d"), &s.other);
        let err = read_text(target.to_str().unwrap()).unwrap_err();
        assert!(
            matches!(err, BindError::Denied(_)),
            "a directory symlink must not open the other file, got {err:?}"
        );
        assert!(!err.message().contains(OTHER));
        assert_eq!(std::fs::read_to_string(s.other.join("f")).unwrap(), OTHER);
        assert_eq!(
            std::fs::read_to_string(s.ws.join("d.held/f")).unwrap(),
            INSIDE
        );
    }

    #[test]
    fn allowlist_would_permit_the_other_file_but_a_swap_before_open_does_not_read_it() {
        let _clear = ClearHooks;
        let s = scratch();
        let target = s.ws.join("d/f");
        let other_file = s.other.join("f");
        // Both paths sit under /tmp, so a pathname allow-list admits either.
        assert!(validate_workspace_path(&target, &permitted_workspaces()).is_ok());
        assert!(validate_workspace_path(&other_file, &permitted_workspaces()).is_ok());
        let other = s.other.clone();
        let dir = s.ws.join("d");
        test_hooks::set_before_open(move || swap_dir(&dir, &other));
        let err = read_text(target.to_str().unwrap()).unwrap_err();
        assert!(matches!(err, BindError::Denied(_)), "{err:?}");
        assert!(!err.message().contains(OTHER));
        assert_eq!(std::fs::read_to_string(&other_file).unwrap(), OTHER);
        assert_eq!(
            std::fs::read_to_string(s.ws.join("d.held/f")).unwrap(),
            INSIDE
        );
    }

    #[test]
    fn held_descriptor_ignores_a_parent_swap() {
        let _clear = ClearHooks;
        let s = scratch();
        let target = s.ws.join("d/f");
        let other = s.other.clone();
        let dir = s.ws.join("d");
        test_hooks::set_after_open(move || swap_dir(&dir, &other));
        let (_, text) = read_text(target.to_str().unwrap()).unwrap();
        assert_eq!(text, INSIDE);
        // By path the name now resolves at the other file.
        assert_eq!(std::fs::read_to_string(&target).unwrap(), OTHER);
        assert_eq!(
            std::fs::read_to_string(s.ws.join("d.held/f")).unwrap(),
            INSIDE
        );
    }

    #[test]
    fn swap_before_open_does_not_write_the_other_file() {
        let _clear = ClearHooks;
        let s = scratch();
        let target = s.ws.join("d/f");
        let other_file = s.other.join("f");
        let other = s.other.clone();
        let dir = s.ws.join("d");
        test_hooks::set_before_open(move || swap_dir(&dir, &other));
        let err = hold_for_write(target.to_str().unwrap()).unwrap_err();
        assert!(matches!(err, BindError::Denied(_)), "{err:?}");
        assert_eq!(std::fs::read_to_string(&other_file).unwrap(), OTHER);
        assert_eq!(
            std::fs::read_to_string(s.ws.join("d.held/f")).unwrap(),
            INSIDE
        );
    }

    #[test]
    fn held_descriptor_write_stays_on_the_authorized_inode() {
        let _clear = ClearHooks;
        let s = scratch();
        let target = s.ws.join("d/f");
        let other = s.other.clone();
        let dir = s.ws.join("d");
        test_hooks::set_after_open(move || swap_dir(&dir, &other));
        let held = hold_for_write(target.to_str().unwrap()).unwrap();
        held.write_bytes(b"UPDATED", true).unwrap();
        assert_eq!(std::fs::read_to_string(s.other.join("f")).unwrap(), OTHER);
        assert_eq!(
            std::fs::read_to_string(s.ws.join("d.held/f")).unwrap(),
            "UPDATED"
        );
        assert_eq!(held.read_checkpoint().unwrap().unwrap(), b"UPDATED");
    }

    #[test]
    fn create_does_not_land_in_a_swapped_parent() {
        let _clear = ClearHooks;
        let s = scratch();
        let target = s.ws.join("d/newfile");
        let other = s.other.clone();
        let dir = s.ws.join("d");
        test_hooks::set_after_open(move || swap_dir(&dir, &other));
        let held = hold_for_write(target.to_str().unwrap()).unwrap();
        let err = held.write_bytes(b"nope", true).unwrap_err();
        assert!(matches!(err, BindError::Denied(_)), "{err:?}");
        assert!(!s.other.join("newfile").exists());
        assert!(!s.ws.join("d.held/newfile").exists());
    }

    #[test]
    fn make_dir_refuses_a_swapped_parent() {
        let _clear = ClearHooks;
        let s = scratch();
        let target = s.ws.join("d/created");
        let other = s.other.clone();
        let dir = s.ws.join("d");
        test_hooks::set_before_open(move || swap_dir(&dir, &other));
        let err = make_dir(target.to_str().unwrap()).unwrap_err();
        assert!(matches!(err, BindError::Denied(_)), "{err:?}");
        assert!(!s.other.join("created").exists());
    }

    #[test]
    fn search_does_not_descend_a_symlink() {
        let s = scratch();
        let outside = outside_scratch();
        std::fs::write(s.ws.join("d/hit.txt"), "HIT_MARKER").unwrap();
        symlink(outside.path(), s.ws.join("d/leak")).unwrap();
        let hits = grep_search(s.ws.join("d").to_str().unwrap(), OUTSIDE, 50).unwrap();
        assert!(hits.is_empty(), "{hits:?}");
        let found = find_names(s.ws.join("d").to_str().unwrap(), "secret", 50).unwrap();
        assert!(found.is_empty(), "{found:?}");
        let leak = find_names(s.ws.join("d").to_str().unwrap(), "leak", 50).unwrap();
        assert_eq!(leak.len(), 1);
        assert!(leak[0].ends_with("/leak"));
    }

    #[test]
    fn search_keeps_the_opened_directory_after_a_swap() {
        let _clear = ClearHooks;
        let s = scratch();
        std::fs::write(s.ws.join("d/hit.txt"), "HIT_MARKER").unwrap();
        std::fs::write(s.other.join("secret"), OUTSIDE).unwrap();
        let other = s.other.clone();
        let dir = s.ws.join("d");
        let search = dir.clone();
        test_hooks::set_after_open(move || swap_dir(&dir, &other));
        let hits = grep_search(search.to_str().unwrap(), "HIT_MARKER", 50).unwrap();
        assert!(hits.iter().any(|h| h.contains("HIT_MARKER")), "{hits:?}");
        assert!(hits.iter().all(|h| !h.contains(OUTSIDE)));
    }

    #[test]
    fn concurrent_swap_keeps_a_pinned_file_and_refuses_an_unpermitted_target() {
        let s = scratch();
        let outside = outside_scratch();
        std::fs::write(outside.path().join("f"), OUTSIDE).unwrap();
        let target = s.ws.join("d/f");
        let held = hold_for_write(target.to_str().unwrap()).unwrap();
        let dir = s.ws.join("d");
        let other = s.other.clone();
        let outside_dir = outside.path().to_path_buf();
        let stop = std::sync::Arc::new(AtomicBool::new(false));
        let swapper = {
            let (stop, dir, other, outside_dir) = (
                std::sync::Arc::clone(&stop),
                dir.clone(),
                other.clone(),
                outside_dir.clone(),
            );
            std::thread::spawn(move || {
                let parked = dir.with_file_name("d.held");
                let mut flip = false;
                while !stop.load(Ordering::Relaxed) {
                    let dest = if flip { &other } else { &outside_dir };
                    flip = !flip;
                    if dir
                        .symlink_metadata()
                        .map(|m| m.file_type().is_symlink())
                        .unwrap_or(false)
                    {
                        let _ = std::fs::remove_file(&dir);
                        let _ = std::fs::rename(&parked, &dir);
                    } else {
                        let _ = std::fs::rename(&dir, &parked);
                        let _ = symlink(dest, &dir);
                    }
                }
                if dir
                    .symlink_metadata()
                    .map(|m| m.file_type().is_symlink())
                    .unwrap_or(false)
                {
                    let _ = std::fs::remove_file(&dir);
                    let _ = std::fs::rename(&parked, &dir);
                }
            })
        };
        let end = Instant::now() + Duration::from_millis(400);
        let mut reads = 0u32;
        while Instant::now() < end {
            let pinned = held
                .read_checkpoint()
                .unwrap()
                .expect("pinned file missing");
            assert_eq!(pinned, INSIDE.as_bytes());
            match read_text(target.to_str().unwrap()) {
                Ok((_, text)) => {
                    assert!(
                        fresh_swap_read_is_inside(&text),
                        "fresh open followed a swapped directory: {text}"
                    );
                }
                Err(BindError::Denied(_)) | Err(BindError::NotFound(_)) => {}
                Err(BindError::Failed(msg))
                    if msg.contains("No such file") || msg.contains("Not a directory") => {}
                Err(e) => panic!("unexpected bind failure: {e:?}"),
            }
            reads += 1;
        }
        stop.store(true, Ordering::Relaxed);
        swapper.join().unwrap();
        assert!(reads > 0);
        assert_eq!(
            std::fs::read_to_string(outside.path().join("f")).unwrap(),
            OUTSIDE
        );
        let (_, text) = read_text(target.to_str().unwrap()).unwrap();
        assert_eq!(text, INSIDE);
    }

    fn openat2_errno_is_denied(code: i32) {
        let _clear = ClearHooks;
        let root = tempfile::tempdir_in("/tmp").unwrap();
        let file = root.path().join("f");
        std::fs::write(&file, b"kept").unwrap();
        test_hooks::set_openat2_error(code);
        let err = read_text(file.to_str().unwrap()).unwrap_err();
        assert!(
            matches!(err, BindError::Denied(_)),
            "errno {code} must be a confinement denial, got {err:?}"
        );
        assert_eq!(std::fs::read(&file).unwrap(), b"kept");
    }

    #[test]
    fn wide_tree_walk_holds_one_fd_per_depth_not_per_sibling() {
        let root = tempfile::tempdir_in("/tmp").unwrap();
        let root = root.path().canonicalize().unwrap();
        let n = 80;
        for i in 0..n {
            let dir = root.join(format!("dir{i:03}"));
            std::fs::create_dir(&dir).unwrap();
            std::fs::write(dir.join("f.txt"), b"x").unwrap();
        }
        let _ = take_walk_fd_max();
        let found = find_names(root.to_str().unwrap(), "f.txt", 200).unwrap();
        assert_eq!(found.len(), n);
        let max_fds = take_walk_fd_max();
        assert!(
            max_fds <= 3,
            "wide tree held {max_fds} directory fds; one per sibling would be about {n}"
        );
    }

    #[test]
    fn openat2_enosys_fails_closed() {
        openat2_errno_is_denied(libc::ENOSYS);
    }

    #[test]
    fn openat2_eperm_is_denied() {
        openat2_errno_is_denied(libc::EPERM);
    }

    #[test]
    fn openat2_einval_is_denied() {
        openat2_errno_is_denied(libc::EINVAL);
    }

    #[test]
    fn grep_bytes_makes_progress_on_large_input_and_output() {
        let line = "match-token\n";
        let copies = (1024 * 1024) / line.len();
        let bytes = line.repeat(copies).into_bytes();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(grep_bytes("match-token", &bytes));
        });
        let result = match rx.recv_timeout(Duration::from_secs(8)) {
            Ok(result) => result,
            Err(_) => {
                let _ = Command::new("pkill")
                    .args(["-P", &std::process::id().to_string(), "grep"])
                    .status();
                panic!("grep_bytes stalled; stdout was not read while stdin was written");
            }
        };
        let lines = result.expect("grep");
        assert!(
            lines.len() > 1000,
            "expected a large match set, got {}",
            lines.len()
        );
    }
}
