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

//! sc#332: a counted topic list ("Cover three topics: drying, labelling and
//! storage.") was refused as uncertain before any model call, and the topic
//! check refused word forms the goal itself accepts ("dried", "Label",
//! "stored", "withdrew"). The list is now read when the count matches it, the
//! word-form rule agrees on y/i, doubled consonants and -age, and a form the
//! goal ties to one topic ("withdrew ... for the third") is accepted for it.
use aien_runtime::requirements::{analyze, refusal_reason, Requirement};

use Requirement::*;

/// tasks-oq3-v4 W3, verbatim.
const W3: &str = "Please put together docs/SEED-SAVING.md, a short explainer on saving vegetable seeds from your own garden. Make it at least eighteen lines long. Cover three topics: drying, labelling and storage. A topic counts as covered when its word, or another form of it such as dried, labelled or stored, appears in the file.";
/// v5 D3 (sc#339), verbatim.
const D3: &str = "Write an explainer for beginners called docs/savings-basics.md about how a savings account works. It must cover four topics: interest, deposit, withdraw and balance, and the file needs at least 18 lines. A topic counts as covered when its word or another form of that word appears, for example interests or interested for the first one, deposits or deposited for the second, withdrew or withdrawing for the third, and balances or balancing for the last.";

fn reqs(goal: &str) -> Vec<Requirement> {
    let ex = analyze(goal);
    assert!(
        ex.uncertain.is_empty(),
        "{goal:?} should be read, uncertain: {:?}",
        ex.uncertain
    );
    ex.requirements
}

fn topics(r: &[Requirement]) -> Vec<String> {
    r.iter()
        .find_map(|x| match x {
            RequiredTopics(t) => Some(t.clone()),
            _ => None,
        })
        .expect("a topic requirement")
}

#[test]
fn the_counted_topic_lists_are_read() {
    let r = reqs(W3);
    assert!(r.contains(&MinLines(18)), "{r:?}");
    assert_eq!(topics(&r), ["drying", "labelling", "storage"]);
    let r = reqs(D3);
    assert!(r.contains(&MinLines(18)), "{r:?}");
    assert_eq!(
        topics(&r),
        ["interest", "deposit", "withdraw (or withdrew)", "balance"]
    );
}

#[test]
fn the_forms_the_goals_accept_cover_their_topics() {
    // evidence-v5/topic-forms-probe.txt: each sentence uses a form the goal accepts.
    for (topic, text) in [
        (
            "drying",
            "Leave the pods until they have dried on the plant.",
        ),
        ("drying", "Dry the seeds on a plate."),
        ("labelling", "Label each envelope with the variety."),
        ("labelling", "Keep every packet labelled."),
        ("storage", "Store the jars somewhere cool."),
        ("storage", "The seeds are stored in a tin."),
        ("withdraw (or withdrew)", "She withdrew fifty pounds."),
        ("withdraw (or withdrew)", "Withdrawing money is free."),
        // sc#353 G2: noun and participle forms by the -n, -al, -als rule
        ("withdraw (or withdrew)", "The money was withdrawn."),
        ("withdraw (or withdrew)", "One withdrawal a day."),
        ("withdraw (or withdrew)", "Withdrawals are free."),
        ("withdraw", "The money was withdrawn."),
        ("withdraw", "Withdrawals are free."),
        ("withdrawing", "One withdrawal a day."),
        ("arrive", "Note the arrival time."),
        ("approve", "Ask for approvals."),
        ("grow", "The plants have grown."),
        ("take", "It was taken."),
        // the cost of the rule, pinned so a tightening is deliberate (sc#354 review)
        ("sign", "A signal."),
        ("line", "A linen cloth."),
        ("interest", "The bank pays interest monthly."),
        ("balance", "Check the balances each week."),
        // the same rule elsewhere
        ("manage", "Good management helps."),
        ("running", "Run every day."),
        ("addressing", "Write the address."),
        ("stories", "Tell a story."),
        // sc#347 review: plurals of -age words still agree
        ("messages", "Read the message."),
        ("message", "Read the messages."),
        ("storages", "Check the storage."),
        ("languages", "One language."),
        ("average", "Two averages."),
        ("fill", "The cup is filled."),
        ("dress", "She dressed well."),
        ("classes", "One class."),
    ] {
        let r = [RequiredTopics(vec![topic.to_string()])];
        assert!(refusal_reason(&r, text).is_none(), "{topic}: {text}");
    }
}

#[test]
fn other_words_still_do_not_cover_a_topic() {
    for (topic, text) in [
        ("drying", "Wet the seeds."),
        ("storage", "Stop here."),
        ("labelling", "Lab work is fun."),
        ("manage", "One man went."),
        // sc#347 review: unrelated words do not collide
        ("file", "Fill the cup."),
        ("mile", "The mill turns."),
        ("tale", "A tall tree."),
        ("ski", "The sky is blue."),
        ("pack", "A package came."),
        ("postage", "Post it today."),
        ("shortage", "A short note."),
        ("message", "What a mess."),
        ("storage", "A story."),
        // an irregular form counts only when the goal names it
        ("withdraw", "She withdrew fifty pounds."),
        // naming `withdrew` opens no loose match: a word sharing its letters stays out
        ("withdraw (or withdrew)", "She drew fifty pounds."),
        // the -n / -al / -als rule is explicit, not a prefix: other endings, the bare
        // shorter word and a different stem stay refused (sc#353 G2)
        ("withdraw", "Draw fifty pounds."),
        ("withdraw", "Open the drawer."),
        ("withdraw (or withdrew)", "Open the drawer."),
        ("withdraw", "A withdrawer came."),
        ("withdraw", "The funds are withdrawable."),
        ("withdraw", "She withheld fifty pounds."),
        ("drawer", "She drew a line."),
        ("arm", "An armal plate."),
        ("bee", "It has been."),
        // 4 letters before -al / -als, and -n only after w or e (sc#354 review)
        ("met", "A metal box."),
        ("met", "Two metals."),
        ("melo", "A melon."),
        ("see", "It was seen."),
    ] {
        let r = [RequiredTopics(vec![topic.to_string()])];
        assert!(refusal_reason(&r, text).is_some(), "{topic}: {text}");
    }
}

#[test]
fn counted_topic_near_misses_stay_uncertain() {
    for goal in [
        // the count differs from the list
        "Cover three topics: drying and storage.",
        "It must cover two topics: interest, deposit and balance.",
        // negated
        "Do not cover three topics: drying, labelling and storage.",
        // a list item that is not a topic
        "Cover two topics: drying and at least 3 lines.",
        // sc#347 review: the last item runs on
        "Cover three topics: drying, labelling and storage and disposal.",
        "Cover three topics: drying, labelling and storage and write at least 20 lines.",
    ] {
        let ex = analyze(goal);
        assert!(
            !ex.uncertain.is_empty(),
            "{goal:?} should stay uncertain, got {:?}",
            ex.requirements
        );
    }
}

#[test]
fn forms_are_read_only_when_tied_to_their_topic() {
    // A sentence that is not about word forms, or a form that is not one of
    // the topic's, adds nothing.
    for goal in [
        "Cover three topics: drying, labelling and storage. In the final format, for example sold or bought for the first one.",
        "Cover three topics: drying, labelling and storage. Any form is fine, for example sold or bought for the first one.",
    ] {
        assert_eq!(topics(&reqs(goal)), ["drying", "labelling", "storage"], "{goal}");
    }
}
