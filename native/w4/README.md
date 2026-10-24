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

## Packed host memory image

`resident.om` loads the two raw packed files into one managed anonymous host
mapping. It consumes the existing L788 compiler extension from Omega functional
source `d80b428c6ed9e3864fd7fc02cd7392bd4ad67f56`, source/test head
`7bc13ac26a99d47c7afec9d9f1e84a05cd904d45` (draft Omega PR #676), compiler
SHA256 `1d1aad082539399920bc6c90ff41f0a0c3a1ca3225e518fd0d259e0aeb71732d`.
The extension is proposed branch work, not merged default-branch capability.
No compiler, mapping implementation or GPU launcher is duplicated here.

Set `wlimit[0]` to the remaining host allocation budget, after accounting for
other live allocations and reservations, then call `wload(wfd,sfd,rows,cols)`.
The descriptors are borrowed, must be distinct, readable and seekable, and
remain the caller's responsibility to close. This operation changes their file
positions. The caller must authenticate and freeze the raw MXFP4 file identities,
layout and dimensions first. There is no checkpoint parser or format detection.

Rows are 1..262144; columns are a positive multiple of 512, at most32768,
matching the existing packed upload's alignment class. The loader verifies
exact file lengths, copies packed bytes unchanged, checks each E8M0 scale is
0..192, and rechecks EOF. Weights start at zero; scales start at the next
4KiB boundary, matching the historical upload layout. Padding comes from the
zero-filled anonymous allocation. No weights expand into FP8 or floating point.
At the maximum supported shape each copy loop stays within the compiler's
one-million-iteration limit.

Only after both files copy and validate does `wstate` expose its five words:
managed handle, rows, columns, scale offset, total requested mapping bytes.
Failed loads publish no handle and release their private mapping. A cleanup
failure retains internal ownership for a subsequent `wdrop` retry; another
load is refused until it is cleared. Loading over an existing image refuses
without changing it. The caller must serialize the module and must not modify
its internal globals or unmap its handle externally.

`wread(offset,count)` copies at most8192 bytes to `wbuffer` after validating
both ends against the published allocation. It permits a zero-length read at
the exact end. This is a host inspection/copy method, not device execution.
`wdrop` releases the image and clears its descriptor, and is idempotent when
empty. Status0 means success,65 invalid descriptor/shape/length/range,
73 empty/busy lifecycle,74 I/O/unmap failure,75 budget/allocation refusal,
and76 invalid E8M0 scale. Returned failure means the buffer is not a result.
The underlying compiler traps on corrupted or externally retired handles;
this module does not turn arbitrary process memory corruption into recoverable
errors.

The configured limit bounds requested mapping bytes. Linux may lazily commit
pages, so successful mapping is not proof of guaranteed physical residency or
recovery from an OS out-of-memory kill. The image is not pinned or registered
with the GPU, and no device coherency, transfer, kernel or full-model speed
claim follows from it. Input files must remain immutable during loading; after
successful copying they can be closed without invalidating the owned image.

Concatenate `resident.om` with `resident-test.om` and compile with the mapping
profile to run the8715-check native author suite in a fresh private directory.
`review/resident-projection.om` is a separate test entrypoint for the exact
17408x5120 real projection: two input paths, then raw weights+scales on stdout.
It closes both source files before reading the image. All three rule outputs
round-tripped through the native mapping byte-for-byte (47349760bytes each).
Independent coverage, failure controls and exact hashes are recorded under
`evidence/w4-711/resident/`. Actual GPU upload and serving remain open gates.

## Corrected compiler compatibility

The compiler safety correction in Omega PR676 also reproduces this component
without changing its executable bytes. See [evidence/w4-711/compiler-correction](../../evidence/w4-711/compiler-correction/README.md) for the exact compiler/source pins, binary comparisons and fresh host replays. The original qualification remains preserved.
