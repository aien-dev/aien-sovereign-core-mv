# Native receipt arithmetic v2 and recovered checkpoint

These are CPU-only synthetic checks and file-integrity receipts. No inference
speed, model quality, packed GPU execution or serving qualification is claimed.

25 structured cases passed (3 valid, 22 refused), plus 5 malformed inputs.
Exact summaries cover shuffled phase costs, differing ordinary/MTP phases,
per-request rates, queue-free service and warmup exclusion. Two compiled
mutants remove sorting or select the wrong p95; both are caught. A regression
control compiles the previous checker: it accepts the zero-service X receipt,
while v2 refuses it. Repeated compilation produced identical bytes.

The 81 model files passed download-time SHA256 and an independent second read
with sha256sum against ../qwen38-710-native-v1/historical-model.sha256.
They are 30,890,049,597 bytes total. Download timing is not inference timing.
Restore used two concurrent curl readers of immutable public Hugging Face
resolve URLs, staged each file, verified its hash before renaming, and refused
existing mismatches. No model bytes are in Git. Reproduce possession checks:

```sh
cd /home/drakestapleton/models/qwen3.8-27b-fp8-017b9c7a
sha256sum -c /home/drakestapleton/workspace/sc-679-native/evidence/qwen38-710-native-v1/historical-model.sha256
```

Prior repository preflight failures remain open. Hosted verification of the
previous PR head also refused one stale generated crumb; raw log retained.
The existing advisory-crumb migration owns that policy discrepancy. No gate
was waived, crumb regenerated, live service restarted or GPU trial attempted.
Final independent source review and qualified native model execution remain
NOT_RUN. Preserve v1 evidence as historical: its replay must use source commit
1b37dcf70a83a112f555ee043ecc407b8f00eeef, not the changed v2 output schema.

## Reproduce synthetic checks

Run from this repository root on AArch64 with the unchanged pinned compiler
from CONTRACT.md. These shell invocations are external test orchestration,
not a retained first-party replacement implementation.

```sh
mkdir -p /tmp/sc679-v2
set -eu
OMC=/home/drakestapleton/workspace/omega-osc-manual-migration/seed/out/omc
"$OMC" < native/qwen38/qualification/receipt.om > /tmp/sc679-v2/receipt
"$OMC" < native/qwen38/qualification/fixture.om > /tmp/sc679-v2/fixture
chmod +x /tmp/sc679-v2/receipt /tmp/sc679-v2/fixture
for pair in A:0 B:4 C:5 D:3 E:3 F:4 G:5 H:5 I:3 J:3 K:3 L:3 M:3 N:3 O:3 P:3 Q:3 R:2 S:2 T:3 U:3 V:3 W:0 X:3 Y:0; do
  case_id=${pair%:*}
  expected=${pair#*:}
  printf '%s' "$case_id" | /tmp/sc679-v2/fixture > "/tmp/sc679-v2/$case_id.input"
  actual=0
  /tmp/sc679-v2/receipt < "/tmp/sc679-v2/$case_id.input" > "/tmp/sc679-v2/$case_id.output" || actual=$?
  printf '%s expected=%s actual=%s\n' "$case_id" "$expected" "$actual"
  test "$actual" = "$expected"
  if test "$actual" != 0; then test ! -s "/tmp/sc679-v2/$case_id.output"; fi
done
for arm in 0 1 2 3; do
  decode=1000
  draft=0
  verify=0
  if test $((arm % 2)) = 1; then decode=700; draft=100; verify=200; fi
  printf '%s\n' "$arm" 10 10 103 105 200 200 300 300 703 705 "$decode" "$decode" "$draft" "$draft" "$verify" "$verify" 100 100 100 100 1803 1805 40000000000 40000000000 5546 5552 1700 1700 5882 5882 40000000000
done > /tmp/sc679-v2/expected-A
cmp /tmp/sc679-v2/expected-A /tmp/sc679-v2/A.output
cmp /tmp/sc679-v2/expected-A /tmp/sc679-v2/W.output
for arm in 0 1 2 3; do
  decode=1000
  dm=0; dp=0; vm=0; vp=0; factor=16
  if test $((arm % 2)) = 1; then decode=700; dm=103; dp=105; vm=206; vp=210; factor=19; fi
  tm=$((1800 + 3 * factor)); tp=$((1800 + 5 * factor))
  sm=$((tm - 103)); sp=$((tp - 105))
  printf '%s\n' "$arm" 13 15 103 105 203 205 306 310 712 720 $((decode + 9)) $((decode + 15)) "$dm" "$dp" "$vm" "$vp" 112 120 115 125 "$tm" "$tp" 40000000000 40000000000 $((13000000 / tm)) $((15000000 / tp)) "$sm" "$sp" $((13000000 / sm)) $((15000000 / sp)) 40000000000
done > /tmp/sc679-v2/expected-Y
cmp /tmp/sc679-v2/expected-Y /tmp/sc679-v2/Y.output
printf 'Exact A/W/Y summaries PASS\n'
for value in '-1' '1000000000001' '18446744073709551616' '1x' ''; do
  actual=0
  printf '%s' "$value" | /tmp/sc679-v2/receipt > /tmp/sc679-v2/invalid.output || actual=$?
  test "$actual" = 2
  test ! -s /tmp/sc679-v2/invalid.output
done
printf 'Five malformed inputs PASS\n'
sed 's/sort a 0/note sorting removed/' native/qwen38/qualification/receipt.om > /tmp/sc679-v2/no-sort.om
sed 's/if not emit get a 4/if not emit get a 3/' native/qwen38/qualification/receipt.om > /tmp/sc679-v2/wrong-p95.om
for name in no-sort wrong-p95; do
  "$OMC" < "/tmp/sc679-v2/$name.om" > "/tmp/sc679-v2/$name"
  chmod +x "/tmp/sc679-v2/$name"
  "/tmp/sc679-v2/$name" < /tmp/sc679-v2/Y.input > "/tmp/sc679-v2/$name.output"
  if cmp -s /tmp/sc679-v2/expected-Y "/tmp/sc679-v2/$name.output"; then exit 1; fi
  printf '%s mutant caught\n' "$name"
done
git show 1b37dcf70a83a112f555ee043ecc407b8f00eeef:native/qwen38/qualification/receipt.om > /tmp/sc679-v2/old.om
"$OMC" < /tmp/sc679-v2/old.om > /tmp/sc679-v2/old
chmod +x /tmp/sc679-v2/old
/tmp/sc679-v2/old < /tmp/sc679-v2/X.input > /tmp/sc679-v2/old-X.output
printf 'Old checker accepts zero-service X; new checker refuses X PASS\n'
"$OMC" < native/qwen38/qualification/receipt.om > /tmp/sc679-v2/rebuild
cmp /tmp/sc679-v2/rebuild /tmp/sc679-v2/receipt
printf 'Deterministic rebuild PASS\n'
```
