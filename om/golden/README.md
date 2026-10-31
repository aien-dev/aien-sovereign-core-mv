# Golden-token harness (Core #853, DRA-52)

Records the greedy token ids an aien-serve-http engine produces for a fixed prompt set, and compares
two such records token by token. Every judgment is made in `golden.om`; `record.sh` sequences and
curl moves bytes. Formats: [SCHEMA.md](SCHEMA.md).

| File | Role |
|---|---|
| `golden.om` | `req` writes request bodies, `rec` turns engine replies into a record, `cmp` compares two records |
| `fake.om` | TEST ONLY: HTTP responder on a random 127.0.0.1 port serving canned replies; never an engine |
| `build.sh` | Builds a `.om` with the pinned omc (omega-mv `bf54b19f`, sha256 `62775d2a...3712b`); exit 3 NOT_RUN otherwise |
| `record.sh` | Sequencer: build, `golden req`, one curl POST per prompt, `golden rec`, no-clobber link to `--out` |
| `check.sh` | Host arms with synthetic fixtures, the fake responder and guard mutants |
| `fixtures/` | Hand-written records labelled `SYNTHETIC` (`synthetic/not-a-model`, revision of zeros) and a 2-prompt set |

Reference records are not committed. A reference is only meaningful when recorded on an independent
machine with a known engine sha256 and model revision.

## Record

```sh
OMC=/path/to/omega-mv/seed/out/omc bash om/golden/record.sh \
  --url http://127.0.0.1:PORT --prompts prompts.json \
  --model-repo ORG/NAME --model-rev <40 hex> --max-tokens 256 \
  --engine-sha256 "$(sha256sum /path/to/aien-serve-http | cut -c1-64)" \
  --host-id "$(hostname -s)" --label REAL --out ref.json [--header-file auth.hdr]
```

`--label REAL` is for engine replies only; a run against a test responder must say `FAKE`.
Exit: 0 written, 2 usage or existing `--out`, 3 compiler not pinned, 4 prompt/file refusal,
5 engine reply refused (error, no `usage.aien_token_ids`, not greedy), 6 transport.

## Compare

```sh
golden cmp ref.json cand.json
```

Exit 0 every prompt matches, 1 a token differs (`DIVERGE <id> index=<K> ref=<X> cand=<Y>`),
3 not comparable (label, engine sha256, model repo or revision, sampling, max_tokens, prompts sha256
or prompt ids differ), 4 file or schema refusal, 2 usage.

## Check

```sh
OMC=/path/to/omega-mv/seed/out/omc bash om/golden/check.sh
```

27 arms: 9 comparator arms on the synthetic fixtures (match, 1-token divergence with its index,
engine and model-revision header mismatch, early end, malformed JSON, non-greedy record, unknown
key, usage), 15 recorder arms against `fake.om` (ids from `usage.aien_token_ids`, FAKE label,
prompts sha256, UTC window, request bodies byte-equal to `golden req`, header file sent, re-record
divergence, sampled reply, missing ids, HTTP 500, existing output) and 3 guard mutants of
`golden.om` (header check disabled, divergence index off by one, sampled replies accepted), each of
which must fail its arm.

NOT_RUN: a real reference recorded from the served model on an independent machine.
