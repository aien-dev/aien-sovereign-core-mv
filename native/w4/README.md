# Native packed W4 group converter

Partial implementation for core issue #711. This handwritten seeded Omega
module migrates the existing `speed3/w4-gemv` converter's OCP, FIT and BEST
rules. It does not implement a GPU kernel, checkpoint reader or model runner.
No inference speed or full-model quality claim is made.

## Contract

`qpack(scale, rule)` consumes 32 E4M3FN bytes in `qinput`, a BF16 scale
represented by its unsigned 16-bit encoding, and rule 0 (OCP), 1 (FIT) or
2 (BEST). It produces 16 bytes in `qoutput`: even inputs occupy the low
nibble, odd inputs the high nibble. E2M1 magnitudes are
0, 0.5, 1, 1.5, 2, 3, 4, 6; bit 3 carries the sign. Zero has canonical
positive sign. Ties select the even magnitude code.

OCP chooses X = 2^(floor(log2(maximum absolute scaled input)) - 2).
FIT doubles X if the OCP group clips any magnitude above 6X. BEST tries
that same doubled scale only when clipping occurs and selects it only
when exact squared error is strictly lower. Equal errors keep OCP.
All-zero groups produce zero bytes and E8M0 scale 127. The supported
E8M0 output range is 0..192, matching the existing device implementation.

The five `qmeta` words are: valid, E8M0 scale, all-zero flag, clipped
count, and squared error in the internal integer units. Public outputs
and metadata are cleared before every call, including refusals. Status
0 means success; 65 means scale input exceeds 16 bits; 69 means unsupported
rule; 75 means unrepresentable output scale; 76 means nonfinite input.
E4M3FN encodings 0x7f and 0xff and all BF16 exponent-255 scales are refused.
Finite negative, zero and subnormal BF16 scales are supported.

Arithmetic is exact unsigned 64-bit integer arithmetic. The raw magnitude
is the FP8 magnitude in units of 1/512 multiplied by the BF16 significand.
Its maximum is 58,490,880. The actual value is raw times 2^(bexp-143)
for a normal BF16 scale, or raw times 2^-142 for a subnormal scale.
Squared error sums differences in eight-times-raw units; multiply by
2^(2*(bexp-146)), or 2^-290 for subnormals, to obtain physical squared
error. This diagnostic is not comparable across differently scaled groups
without that conversion. Reviewed bounds keep each 32-element sum below
2^59. Global scratch arrays make the module nonreentrant: callers must
serialize calls or use separate instances.

## Diagnostic transport

Concatenate `pack.om` with `stream.om` to build a bounded binary adapter.
Each request is 35 bytes: rule u8, BF16 scale u16 little-endian, then 32
FP8 bytes. Each response is 20 bytes: status u8, E8M0 u8, 16 packed bytes,
clipped count u8, all-zero flag u8. A rejected group emits a status with
cleared remaining fields and does not prevent later groups succeeding.

At most 512 records are buffered. Short reads and returned short writes
are handled; EOF after complete records returns 0, incomplete final
records return 65 after previously completed responses. Returned I/O
errors or zero writes return 74. Interrupted calls are treated as errors;
broken-pipe signals can terminate the process. Short-write retries were
source-reviewed, not fault-injected. The transport supplies no model
identity, checkpoint validation or concurrency protection.

## Build and check

Use the reconstructed seeded `omc` executable with SHA256
`191b776ef98821597c3e30655da6d680a9b66edbc9ceaca3c39c61ea6df098e9`
(omega-bridge source `0fc7d8662cb799e7c206b17c18ea4c3ec2b8fea9`, seed
Omega `29b12760`), on the qualified AArch64 host. Set `OMC` to that executable.
Development shell commands concatenate native source and invoke the compiler:

```sh
cat native/w4/pack.om native/w4/tests.om > /tmp/w4-tests.om
"$OMC" < /tmp/w4-tests.om > /tmp/w4-tests
chmod 700 /tmp/w4-tests
/tmp/w4-tests
cat native/w4/pack.om native/w4/review/probes.om > /tmp/w4-independent.om
"$OMC" < /tmp/w4-independent.om > /tmp/w4-independent
chmod 700 /tmp/w4-independent
/tmp/w4-independent
```

Require compilation success before executing an output. Author checks
report 1,219,868 and zero failures. The separate nearest-distance oracle
reports 4,381,347 and zero failures. Compiled intentional defects verify
that the tests detect tie, nibble, exponent, BEST and nonfinite mistakes.
The independent oracle covers every BF16 encoding and exact rounding
boundaries. `review/fixture.om` and `review/reply.om` independently generate
and verify 1,536 transport records. Their entry points must be compiled
separately, not appended to the other test entry points.

Top-level `fixture.om` and `reply.om` are different, test-only tools for the
real-weight differential: the former repeats one three-byte rule/scale
header across raw 32-byte input groups; the latter accepts exactly 512
successful replies and writes 8,192 packed bytes followed by 512 scales.
Neither is a production checkpoint reader.

## Existing implementation and evidence

The reference is `aien-dev/omega-mv` branch `speed3/w4-gemv`, commit
`8655007f0f09d454b47b75fb3d83bc80e8f6d4da`, specifically
`src/omega_gpu_w4_api.c` and `src/omega_blackwell_codegen_w4.h`.
Its physics dependency was built at the locked commit
`6d7cf0d4d8eb2cda7b512100ff6058e25dbb3ddf`. That existing C implementation
is a development reference only, not a dependency of these native binaries.
Historical GPU results on that branch are not new qualification evidence.

See `evidence/w4-711/` for recorded source identities, raw check results,
review reports, controls and the exact real-model sample differential.
The real checkpoint sample agrees byte-for-byte for all three rules.
Model data and generated binaries are not included in this repository.
Repository preflight remains failed on the same base due to the existing
browser screenshot test and dependency advisories recorded by L790.

Remaining issue acceptance includes a native device path and model loading,
whole-checkpoint layout and quality validation, safe serving attachment,
and uncontended full-model latency/throughput/memory measurements. The
issue must remain open until those requirements are met.

## Whole raw-tensor conversion

`pack.om` plus `tensor.om` forms a native file converter. This entry point uses
the separately qualified native-host compiler at Omega
`f0499f415571bdd7496b6de391096915e78f7556`, executable SHA256
`e5d60db892c8f21cc8c9128af26d8bded0e96d7f92946f5a549397c43a872649`.
The author and independent arithmetic suites were rebuilt and passed under
this compiler too. It is a different profile from the earlier pure compiler;
no unsupported combination of compiler features is assumed.

```text
w4-tensor raw-fp8 raw-bf16 packed-output e8-output rows columns rule
```

Arguments are unsigned decimal integers without signs, whitespace or prefixes.
Rows must be a positive multiple of 128, at most 524,288. Columns must be a
positive multiple of 128, at most 32,768. Rule is 0, 1 or 2. A width accepted
by this converter is not necessarily accepted by a particular device kernel.

Input weights are exactly rows*columns contiguous E4M3FN bytes, row major.
The scale file contains exactly (rows/128)*(columns/128) little-endian BF16
values, row major. The source scale for element (r,c) is
scale[r/128][c/128]. Outputs are rows*columns/2 packed bytes and
rows*columns/32 E8M0 scale bytes, each row major, identical to the historical
converter's separate output arrays. This is MXFP4, with one E8M0 scale per
32 values. It is not NVFP4 and does not accept another format's scale/layout
under the same interface. The two raw outputs have no self-describing header.

Inputs must be immutable, seekable regular model extracts supplied by a caller
that has already authenticated their identities and dimensions. This command
does not parse safetensors, select model tensors or authenticate checkpoints.
It verifies exact input sizes before creating outputs, then checks for extra
bytes at EOF. That does not detect arbitrary concurrent same-size input edits.

The converter reads one weight row and one 128-row scale block at a time;
its buffers have fixed capacities independent of tensor size. It converts
every group through qpack and retains packed output, with no whole-tensor
expansion. Output files are created exclusively with owner-only permissions.
Existing outputs are refused without overwriting them. Handled conversion,
read, write, synchronization or close failures remove only outputs created
by this invocation. A private output directory under the caller's control is
required; concurrent path replacement is outside this contract.

A successful exit is required before any consumer opens either result. Both
files are synchronized before success, but publication of the pair is not
atomic, directory entries are not synchronized, and a crash or signal may
leave incomplete files. There is no restart/resume or in-place update mode.
Exit 64 means malformed command arguments, 65 invalid shape/length/truncation,
69 unsupported rule, 74 an I/O failure, 75 unrepresentable scale and 76
nonfinite data. Clean up failed-run files before retrying with fresh output
names. This diagnostic converter is not a serving loader or model registry.

A complete 17,408x5,120 gate projection from the pinned real model was
converted under all three rules. Each result matched the existing converter
exactly: 44,564,480 weight bytes and 2,785,280 scale bytes. Those file bytes
are about 53% of the input weight+scale bytes, not measured physical GPU
allocation or traffic. See `evidence/w4-711/tensor/` for exact identities,
commands and independent file-lifecycle/layout checks. Whole-model quality,
GPU kernels, loading and serving integration remain unqualified.
