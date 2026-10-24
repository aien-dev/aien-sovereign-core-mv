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

pub fn generate_inquisitor_interview(author: &str, pr_number: u64, pr_title: &str) -> String {
    format!(
        r#"### 🌿 Sovereign Steward: Contributor Alignment Interview

Greetings @{author}. I am **AIEN**, the automated steward for the AIENOS project.

We welcome all contributors who share our commitment to open collaboration, global humanity, and disciplined systems craftsmanship. We build to expand computing capability, advance open-source intelligence, and share technology freely with the world.

Your pull request **#{pr_number}** ("{pr_title}") is ready for stewardship alignment.

Please respond directly to these four alignment inquiries:

1. **Long-Term Mission Alignment**:
   This initiative is an enduring engineering commitment to open-source technology for humanity, not a vehicle for short-term hype, social media monetization, or get-rich-quick ventures. How does your contribution support global progress and the resilience of local communities?

2. **Telemetry & Integrity**:
   Do you certify that this change introduces zero telemetry, zero surveillance, and zero artificial paywalls?
   > "I certify that my contribution is submitted in service of human sovereignty and open collaboration. I affirm that this work maintains zero telemetry, contains no surveillance backdoors, and introduces no artificial barriers, and that I offer it under AGPL-3.0-or-later, the license that governs this project. I build so that intelligence is owned by the person who runs it, not rented."

3. **Downstream Heritage Preservation**:
   Do you acknowledge and agree that any downstream branch, fork, or copy of this work remains under AGPL-3.0-or-later and retains the project's license and attribution notices, as the `LICENSE` file requires?

4. **Craftsmanship & Native Discipline**:
   What architectural trade-offs did you make, and how did you verify that this code introduces zero regressions, zero unneeded dependencies, and complies with our pure native compiled standards?

---
*Reply to this comment with your affirmation. Once verified against the project standards, I will grant the `sovereign-interview-passed` clearance. For direct sovereign coordination: Drake Stapleton (drake@aienos.com) and AIEN (aien@aienos.com)*
"#,
        author = author,
        pr_number = pr_number,
        pr_title = pr_title
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_interview() {
        let text = generate_inquisitor_interview("contributor_bob", 42, "feat: add ed25519 helper");
        assert!(text.contains("@contributor_bob"));
        assert!(text.contains("#42"));
        assert!(text.contains("AGPL-3.0-or-later"));
        assert!(text.contains("Long-Term Mission Alignment"));
        assert!(text.contains("Downstream Heritage Preservation"));
        assert!(text.contains("Drake Stapleton (drake@aienos.com) and AIEN (aien@aienos.com)"));
        assert!(!text.contains('\u{2014}'));
        assert!(!text.contains('\u{2013}'));
    }
}
