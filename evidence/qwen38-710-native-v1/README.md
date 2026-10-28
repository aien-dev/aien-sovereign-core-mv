# CPU receipt-checker evidence, not model measurements

Source under `native/qwen38/qualification/`. `synthetic.input.txt` contains
invented test data produced by `fixture.om`. Never cite its durations or memory
values as Spark inference results. `historical-model.sha256` is an older manifest
of expected model bytes, not evidence that those files are currently present.

Executed on Spark Linux AArch64, 2026-10-23:

- 23 structured cases: two valid summaries and 21 refusals, expected exits matched.
- Five malformed integer/input cases refused with exit 2 and no summary.
- Exact independent expected summary matched, including warmup exclusion,
  median 1803, nearest-rank p95 1805 and integer rate floor 5546.
- Three compiled mutants caught: removing boundary, packed-read or contamination
  checks incorrectly accepts its negative fixture. Raw verdicts in `mutants.txt`.
- Recompiling the unchanged source produced byte-identical executable output.
- Rebuilt the pinned compiler using its existing `om4.bin` parent, then rebuilt
  it with itself. Both outputs match the consumed compiler exactly. Identities
  are in `seed-rebuild.sha256`. This is reconstruction, not independent semantic proof.
- The receipt executable has no dynamic section; it does not link a C/Rust library.

Repository preflight: **FAIL**, preserved without waivers. The sandbox run
reported four failed crates and could not fetch advisories. The unrestricted
retry reported 57 crates, four executed and 53 reused stamps, with one failure:
`aien-cli` browser screenshot test (`websocket read: Resource temporarily
unavailable`). Its scope already has Core #688. The separately retried audit
fetched the database and reported eight vulnerabilities in unchanged locked
dependencies. Raw preflight logs are retained byte-for-byte as `.log.txt.gz`
(read with `gzip -cd`); `raw.sha256` records compressed and original hashes.
None is relabelled as a native-checker
failure or a passing repository gate. No unrelated dependency update is included.

Initial source development hit compiler error 12 (register-expression capacity)
and then error 19 (passing a global array as an argument). Functions were split
and summary sorting now uses a local array. Those refusals were not test passes.
No compiler changes or foreign-language runtime were introduced.

## Replay

From repository root, on AArch64, with the pinned Omega checkout in CONTRACT.md:

```sh
set -eu
OMC=/home/drakestapleton/workspace/omega-osc-manual-migration/seed/out/omc
mkdir -p /tmp/sc679-tests
sha256sum "$OMC"
# Must equal 01b1545b23e8ff40cb5691e20b5df910ea4acaefde3495f0d62d270465b98cd6.
"$OMC" < native/qwen38/qualification/receipt.om > /tmp/sc679-tests/receipt
"$OMC" < native/qwen38/qualification/fixture.om > /tmp/sc679-tests/fixture
chmod +x /tmp/sc679-tests/receipt /tmp/sc679-tests/fixture
for pair in A:0 B:4 C:5 D:3 E:3 F:4 G:5 H:5 I:3 J:3 K:3 L:3 M:3 N:3 O:3 P:3 Q:3 R:2 S:2 T:3 U:3 V:3 W:0; do
  case_id=${pair%:*}
  expected=${pair#*:}
  printf '%s' "$case_id" | /tmp/sc679-tests/fixture > "/tmp/sc679-tests/$case_id.input"
  actual=0
  /tmp/sc679-tests/receipt < "/tmp/sc679-tests/$case_id.input" > "/tmp/sc679-tests/$case_id.output" || actual=$?
  printf '%s expected=%s actual=%s\n' "$case_id" "$expected" "$actual"
  test "$actual" = "$expected"
  if test "$actual" != 0; then test ! -s "/tmp/sc679-tests/$case_id.output"; fi
done
for arm in 0 1 2 3; do
  printf '%s\n1803\n1805\n103\n5546\n40000000000\n' "$arm"
done > /tmp/sc679-tests/expected
cmp /tmp/sc679-tests/expected /tmp/sc679-tests/A.output
cmp /tmp/sc679-tests/expected /tmp/sc679-tests/W.output
for value in '-1' '1000000000001' '18446744073709551616' '1x' ''; do
  actual=0
  printf '%s' "$value" | /tmp/sc679-tests/receipt > /tmp/sc679-tests/invalid.output || actual=$?
  test "$actual" = 2
  test ! -s /tmp/sc679-tests/invalid.output
done
sed 's/if or ne get row 21 boundary ne get row 23 1/if ne get row 23 1/' native/qwen38/qualification/receipt.om > /tmp/sc679-tests/wrong-boundary.om
sed 's/if ne get row 18 packed/if false/' native/qwen38/qualification/receipt.om > /tmp/sc679-tests/no-packed.om
sed 's/if or gt get row 14 0 gt get row 15 0/if false/' native/qwen38/qualification/receipt.om > /tmp/sc679-tests/no-contamination.om
for item in wrong-boundary:D:3 no-packed:C:5 no-contamination:B:4; do
  name=${item%%:*}
  rest=${item#*:}
  case_id=${rest%%:*}
  required=${rest#*:}
  "$OMC" < "/tmp/sc679-tests/$name.om" > "/tmp/sc679-tests/$name"
  chmod +x "/tmp/sc679-tests/$name"
  actual=0
  "/tmp/sc679-tests/$name" < "/tmp/sc679-tests/$case_id.input" > "/tmp/sc679-tests/$name.output" || actual=$?
  test "$actual" != "$required"
done
"$OMC" < native/qwen38/qualification/receipt.om > /tmp/sc679-tests/rebuild
cmp /tmp/sc679-tests/rebuild /tmp/sc679-tests/receipt
```

These commands are temporary external test orchestration. A complete native
test/build runner remains an unmet migration dependency. No hardware run was
requested without the full native serving artifact and verified model inputs.
No FP8/W4/MTP throughput improvement or achieved model quality is available.
Independent review of the final source remains NOT_RUN. This source-only
packet must not be promoted or used to close any of the five issues.
