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

//! Rust side: declaration macros that record what they declare, and the manifest renderer.

use crate::manifest::{canon_sig, make_sig};

/// Kind of a field's element type, as the C probe names it.
pub trait CKind {
    const KIND: &'static str;
    const COUNT: usize = 1;
}

macro_rules! ckind {
    ($($t:ty => $k:literal),*) => { $(impl CKind for $t { const KIND: &'static str = $k; })* };
}
ckind!(bool => "bool", i8 => "i8", u8 => "u8", i16 => "i16", u16 => "u16", i32 => "i32",
       u32 => "u32", i64 => "i64", u64 => "u64", f32 => "f32", f64 => "f64");
#[cfg(target_pointer_width = "64")]
ckind!(usize => "u64", isize => "i64");

impl<T: CKind, const N: usize> CKind for [T; N] {
    const KIND: &'static str = T::KIND;
    const COUNT: usize = N;
}

/// One field as recorded by [`abi_struct!`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Field {
    pub name: &'static str,
    pub offset: usize,
    pub size: usize,
    pub kind: &'static str,
    pub count: usize,
}

impl Field {
    /// `ty` is the declared type's tokens: a `c_char` element is reported as `c_char`
    /// (the C probe's name for plain `char`), whatever its signedness on this target.
    pub fn of<T: CKind>(name: &'static str, offset: usize, ty: &'static str) -> Field {
        let kind = if ty.contains("c_char") {
            "c_char"
        } else {
            T::KIND
        };
        Field {
            name,
            offset,
            size: std::mem::size_of::<T>(),
            kind,
            count: T::COUNT,
        }
    }
}

/// A `#[repr(C)]` struct declared through [`abi_struct!`].
pub trait AbiStruct: Sized {
    const NAME: &'static str;
    fn fields() -> Vec<Field>;
    fn manifest_lines() -> Vec<String> {
        let mut v = vec![format!(
            "struct {} {} {}",
            Self::NAME,
            std::mem::size_of::<Self>(),
            std::mem::align_of::<Self>()
        )];
        for f in Self::fields() {
            v.push(format!(
                "field {} {} {} {} {} {}",
                Self::NAME,
                f.name,
                f.offset,
                f.size,
                f.kind,
                f.count
            ));
        }
        v
    }
}

/// A recorded extern or callback signature: (name, return type tokens, parameter type tokens).
pub type FnSig = (&'static str, &'static str, &'static [&'static str]);

/// Everything one boundary's Rust mirror declares.
#[derive(Default)]
pub struct RustAbi {
    pub structs: Vec<Vec<String>>,
    pub consts: Vec<(&'static str, i128)>,
    pub fnptrs: Vec<FnSig>,
    pub fns: Vec<FnSig>,
}

impl RustAbi {
    pub fn with_struct<T: AbiStruct>(mut self) -> Self {
        self.structs.push(T::manifest_lines());
        self
    }
    pub fn with_consts(mut self, c: &[(&'static str, i128)]) -> Self {
        self.consts.extend_from_slice(c);
        self
    }
    pub fn with_fnptrs(mut self, f: &[FnSig]) -> Self {
        self.fnptrs.extend_from_slice(f);
        self
    }
    pub fn with_fns(mut self, f: &[FnSig]) -> Self {
        self.fns.extend_from_slice(f);
        self
    }
}

fn sig_text(ret: &str, params: &[&str]) -> String {
    let s = make_sig(params, ret);
    canon_sig(&s).unwrap_or(s)
}

/// The Rust mirror rendered as manifest fact lines, in the same order as
/// [`crate::Manifest::fact_lines`] renders a file (structs with their fields, constants,
/// callbacks, functions).
pub fn rust_manifest(abi: &RustAbi) -> Vec<String> {
    let mut v: Vec<String> = abi.structs.iter().flatten().cloned().collect();
    for (n, val) in &abi.consts {
        v.push(format!("const {n} {val}"));
    }
    for (n, r, p) in &abi.fnptrs {
        v.push(format!("fnptr {n} {}", sig_text(r, p)));
    }
    for (n, r, p) in &abi.fns {
        v.push(format!("fn {n} {}", sig_text(r, p)));
    }
    v
}

/// Declare a `#[repr(C)]` struct and record its fields in declaration order
/// (implements [`AbiStruct`]). Attributes and doc comments pass through unchanged.
#[macro_export]
macro_rules! abi_struct {
    ($(#[$m:meta])* $vis:vis struct $name:ident {
        $($(#[$fm:meta])* $fvis:vis $f:ident : $t:ty),* $(,)?
    }) => {
        $(#[$m])* $vis struct $name { $($(#[$fm])* $fvis $f : $t),* }
        impl $crate::AbiStruct for $name {
            const NAME: &'static str = stringify!($name);
            fn fields() -> ::std::vec::Vec<$crate::Field> {
                ::std::vec![$($crate::Field::of::<$t>(
                    stringify!($f), ::core::mem::offset_of!($name, $f), stringify!($t))),*]
            }
        }
    };
}

/// Declare an `extern "C"` block (attributes on the block, such as a `cfg`, apply to it as
/// usual) and a `pub const $table: &[FnSig]` naming every function's types. The table exists
/// in stub builds too, so the signatures are checked even where nothing links.
#[macro_export]
macro_rules! abi_externs {
    ($table:ident; $(#[$bm:meta])* extern "C" {
        $( $(#[$m:meta])* pub fn $name:ident ( $($p:ident : $t:ty),* $(,)? ) $(-> $r:ty)? ; )*
    }) => {
        $(#[$bm])* extern "C" { $( $(#[$m])* pub fn $name($($p: $t),*) $(-> $r)?; )* }
        pub const $table: &[$crate::FnSig] =
            &[$( (stringify!($name), stringify!($($r)?), &[$(stringify!($t)),*]) ),*];
    };
}

/// Declare `unsafe extern "C" fn` callback types and a table of their signatures.
#[macro_export]
macro_rules! abi_fnptr {
    ($table:ident; $( $(#[$m:meta])* pub type $name:ident =
        unsafe extern "C" fn( $($p:ident : $t:ty),* $(,)? ) $(-> $r:ty)? ; )*) => {
        $( $(#[$m])* pub type $name = unsafe extern "C" fn($($p: $t),*) $(-> $r)?; )*
        pub const $table: &[$crate::FnSig] =
            &[$( (stringify!($name), stringify!($($r)?), &[$(stringify!($t)),*]) ),*];
    };
}

/// Declare integer constants mirrored from C and a table of their values.
#[macro_export]
macro_rules! abi_consts {
    ($table:ident; $( $(#[$m:meta])* pub const $name:ident : $t:ty = $v:expr ; )*) => {
        $( $(#[$m])* pub const $name: $t = $v; )*
        pub const $table: &[(&str, i128)] = &[$( (stringify!($name), $name as i128) ),*];
    };
}

/// Compare a Rust mirror with its checked-in manifest file. With `AIEN_ABI_BLESS=1` the file's
/// fact lines are rewritten from the Rust side instead (its `abi`/`header` lines and comments
/// are kept); the C side then has to agree in `build.rs` before anything links.
pub fn assert_manifest(abi: &RustAbi, path: &std::path::Path) {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let rust = rust_manifest(abi);
    if std::env::var("AIEN_ABI_BLESS").as_deref() == Ok("1") {
        let mut out: Vec<String> = text
            .lines()
            .take_while(|l| {
                let t = l.trim_start();
                t.is_empty()
                    || t.starts_with('#')
                    || t.starts_with("abi ")
                    || t.starts_with("header ")
            })
            .map(str::to_string)
            .collect();
        out.extend(rust.iter().cloned());
        std::fs::write(path, out.join("\n") + "\n").expect("write manifest");
        return;
    }
    let m = crate::Manifest::parse(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let file = m.fact_lines();
    if file != rust {
        let only_rust: Vec<&String> = rust.iter().filter(|l| !file.contains(l)).collect();
        let only_file: Vec<&String> = file.iter().filter(|l| !rust.contains(l)).collect();
        panic!(
            "Rust FFI declarations differ from {}\n  Rust only: {only_rust:#?}\n  manifest only: {only_file:#?}\n  (order differs if both are empty). \
             An intended ABI change updates the manifest (AIEN_ABI_BLESS=1) and the omega pin together.",
            path.display()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::raw::{c_char, c_int, c_void};

    crate::abi_struct! {
        #[repr(C)]
        #[derive(Clone, Copy)]
        pub struct Demo {
            /// doc on a field
            pub a: u8,
            pub name: [c_char; 4],
            pub b: u64,
            pub flag: bool,
            pub n: usize,
        }
    }
    crate::abi_fnptr! { CB; #[allow(dead_code)] pub type Cb = unsafe extern "C" fn(ctx: *mut c_void, x: u64) -> c_int; }
    crate::abi_externs! { FNS; #[cfg(any())] extern "C" {
        /// doc
        pub fn demo_open(p: *const c_char, cb: Option<Cb>, out: *mut *mut Demo) -> c_int;
        pub fn demo_close();
    } }
    crate::abi_consts! { K; pub const DEMO_E: c_int = -3; pub const DEMO_V: u32 = 2; }

    #[test]
    fn macros_record_layout_and_signatures() {
        let abi = RustAbi::default()
            .with_struct::<Demo>()
            .with_consts(K)
            .with_fnptrs(CB)
            .with_fns(FNS);
        let got = rust_manifest(&abi);
        let want = [
            "struct Demo 32 8",
            "field Demo a 0 1 u8 1",
            "field Demo name 1 4 c_char 4",
            "field Demo b 8 8 u64 1",
            "field Demo flag 16 1 bool 1",
            "field Demo n 24 8 u64 1",
            "const DEMO_E -3",
            "const DEMO_V 2",
            "fnptr Cb fn(*mut c_void, u64) -> i32",
            "fn demo_open fn(*const c_char, Cb, *mut *mut Demo) -> i32",
            "fn demo_close fn() -> ()",
        ];
        assert_eq!(got, want);
    }
}
