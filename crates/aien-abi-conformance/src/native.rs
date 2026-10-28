//! Run-time view of what a binary linked (filled from the `build.rs` identity record).

/// One native component as linked into this binary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeIdentity {
    pub component: &'static str,
    /// stub | pinned-source | dev-override | prebuilt
    pub link: &'static str,
    /// pass | symbols-only | unchecked | none
    pub abi: &'static str,
    /// locked | unrecorded | prebuilt-match | none
    pub digest: &'static str,
    pub archive_sha256: &'static str,
    pub source_sha: &'static str,
    pub aienos_lock: &'static str,
    pub physics_lock: &'static str,
    pub target: &'static str,
    pub cc: &'static str,
}

impl NativeIdentity {
    /// Release-grade: built from its pinned source, C ABI checked, digest recorded
    /// and matched. A stub, a developer override or a prebuilt archive never qualifies.
    pub fn qualifies(&self) -> bool {
        self.link == "pinned-source" && self.abi == "pass" && self.digest == "locked"
    }

    /// Why [`Self::qualifies`] is false (empty when it is true).
    pub fn gaps(&self) -> Vec<&'static str> {
        let mut v = Vec::new();
        if self.link != "pinned-source" {
            v.push("not built from the pinned component source");
        }
        if self.abi != "pass" {
            v.push("C ABI not checked against the manifest");
        }
        if self.digest != "locked" {
            v.push("archive digest not matched against native-artifacts.lock");
        }
        v
    }

    /// One JSON object (no dependencies; every value is a plain ASCII token).
    pub fn to_json(&self) -> String {
        let esc = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
        format!(
            "{{\"component\":\"{}\",\"link\":\"{}\",\"abi\":\"{}\",\"digest\":\"{}\",\"archive_sha256\":\"{}\",\"source_sha\":\"{}\",\"aienos_lock\":\"{}\",\"physics_lock\":\"{}\",\"target\":\"{}\",\"cc\":\"{}\",\"qualifies\":{}}}",
            esc(self.component),
            esc(self.link),
            esc(self.abi),
            esc(self.digest),
            esc(self.archive_sha256),
            esc(self.source_sha),
            esc(self.aienos_lock),
            esc(self.physics_lock),
            esc(self.target),
            esc(self.cc),
            self.qualifies()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_pinned_checked_locked_build_qualifies() {
        let ok = NativeIdentity {
            component: "x",
            link: "pinned-source",
            abi: "pass",
            digest: "locked",
            archive_sha256: "a",
            source_sha: "b",
            aienos_lock: "",
            physics_lock: "",
            target: "t",
            cc: "cc \"q\"",
        };
        assert!(ok.qualifies() && ok.gaps().is_empty());
        assert!(ok.to_json().contains("\"qualifies\":true"));
        assert!(ok.to_json().contains("cc \\\"q\\\""));
        for bad in [
            NativeIdentity {
                link: "stub",
                abi: "none",
                digest: "none",
                ..ok
            },
            NativeIdentity {
                link: "prebuilt",
                abi: "symbols-only",
                digest: "prebuilt-match",
                ..ok
            },
            NativeIdentity {
                link: "dev-override",
                ..ok
            },
            NativeIdentity {
                digest: "unrecorded",
                ..ok
            },
            NativeIdentity {
                abi: "unchecked",
                ..ok
            },
        ] {
            assert!(!bad.qualifies() && !bad.gaps().is_empty(), "{bad:?}");
        }
    }
}
