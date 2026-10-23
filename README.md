# Recovery archive, 2026-10-23

This branch is a preservation index, not a product implementation or merge proposal.
Original committed work and unreviewed working/staged/conflict snapshots are reachable
through this commit's parent history. The index tree deliberately contains only recovery
records. No active branch, original staging area, service or working file was replaced.

## Contents

- 74 local branch references absent from the published heads/PR histories inspected.
- 85 working-folder snapshots, 2 separate staged snapshots, 9 unresolved conflict alternatives.
- 7 excluded untracked generated/model/nested-repository entries; see working-snapshots.json.
- 367 previously unreferenced commits, including the new recovery snapshots.
- Published main at capture: 754c07b118f41ad899607b60329bd851b1ccb13e.

The capture covered discovered workspace clones and their registered worktrees for
Sovereign Core, Omega and CUDApp, plus their existing omega-bridge migration repository.
Ignored files, arbitrary reflog-only objects and unrelated repositories were not swept.
Six missing registered worktree directories were recorded in the cross-repository summary.
Known compiler/device owners retain implementation ownership. No test, quality, hardware,
license-transition or release approval is implied by publication. Historical Rust/C/.osc
is retained here as recovery history, not adopted as the final first-party implementation.

## Recover saved work

Use the commit links below, or fetch this branch and create a new local branch at the
desired commit. Do not reset an active working folder. Working snapshots capture eligible
on-disk files; staged and conflict snapshots preserve additional index alternatives.
Generated/model artifacts and nested repositories are excluded explicitly.

### Previously unpublished branch tips

| Original local branch | Saved commit |
|---|---|
| adr0004-step3 | [857d43cf5fe4](https://github.com/aien-dev/aien-sovereign-core-mv/commit/857d43cf5fe48d75bddc0918415f64d811ed906f) |
| backup/249-pre-rebase-2058 | [ca17ce03407c](https://github.com/aien-dev/aien-sovereign-core-mv/commit/ca17ce03407c4a65cc89346c1bbb9aac01b4a7f1) |
| c1/serving-reserve | [5b1d246e56df](https://github.com/aien-dev/aien-sovereign-core-mv/commit/5b1d246e56df8d7a85e241147f597d3154cb2d8b) |
| calc/aien-calc | [96dbb47f22a4](https://github.com/aien-dev/aien-sovereign-core-mv/commit/96dbb47f22a4e2a68de3180c4dc44cee3ce5563b) |
| crumb/advisory | [f7646e38c39f](https://github.com/aien-dev/aien-sovereign-core-mv/commit/f7646e38c39f30d3d0029257279d577a2bbc2cde) |
| docs/front-sheet | [4a4b95a7ba3f](https://github.com/aien-dev/aien-sovereign-core-mv/commit/4a4b95a7ba3f24e635807f46c8ebff6d8e29dee3) |
| e2e/shakedown-local | [e906c629a08f](https://github.com/aien-dev/aien-sovereign-core-mv/commit/e906c629a08f1c46505845dff52dad70d592b1fc) |
| fastload/parallel-load | [48613a4df1cb](https://github.com/aien-dev/aien-sovereign-core-mv/commit/48613a4df1cb98e5e10142b485b848061a8ece0f) |
| feat/gandi-mail-connector | [4829c8d2bc3f](https://github.com/aien-dev/aien-sovereign-core-mv/commit/4829c8d2bc3f438429cc8480f7f6f7d6ed1e0b82) |
| feat/native-probe-endpoint | [95a4bdc081d3](https://github.com/aien-dev/aien-sovereign-core-mv/commit/95a4bdc081d3cf04d22da1b6093439d01e55c8cb) |
| feat/token-native-platform-linux-runtime/05-end-to-end | [9c5e06954a5e](https://github.com/aien-dev/aien-sovereign-core-mv/commit/9c5e06954a5e78c0e0e35d709f0e390d4bfd2b84) |
| fix/adapters-submilli-flake | [df6a0c2fc5dc](https://github.com/aien-dev/aien-sovereign-core-mv/commit/df6a0c2fc5dcfee177993e5f4d59d3c4476b3f9f) |
| fix/cortex-vault-auth | [eb325cf42858](https://github.com/aien-dev/aien-sovereign-core-mv/commit/eb325cf4285836800c4a98f8f4d818eead58474c) |
| fix/gb10-gate-prepare | [85e0cb953e10](https://github.com/aien-dev/aien-sovereign-core-mv/commit/85e0cb953e107a30f04b332f9e8f1744ede90d76) |
| fix/gb10-qwen3-default-reservation | [20ef0ecfde8b](https://github.com/aien-dev/aien-sovereign-core-mv/commit/20ef0ecfde8b327272c0c0a49df628f68f44c7fb) |
| fix/kv-concurrency-progress-under-load | [5b96443a7a09](https://github.com/aien-dev/aien-sovereign-core-mv/commit/5b96443a7a0952df7b3210834db6535378da7605) |
| fix/kv-flake-575 | [dcc23651f1ff](https://github.com/aien-dev/aien-sovereign-core-mv/commit/dcc23651f1ffdcf84c38a097629d619398637bb6) |
| fix/license-and-readme-standardization | [bd4349af910c](https://github.com/aien-dev/aien-sovereign-core-mv/commit/bd4349af910cb18a2944cda214a3a4a2c377af63) |
| fix/own30b-server-522 | [aa41e32068eb](https://github.com/aien-dev/aien-sovereign-core-mv/commit/aa41e32068eb44d9f0ae2afaf57cfeab684e47db) |
| fix/pkv-bench-backpressure | [6d52e08ed27f](https://github.com/aien-dev/aien-sovereign-core-mv/commit/6d52e08ed27fe3bb42cf9cad1f943a52267264e2) |
| fix/siblings-synapse | [f4d98295b4b6](https://github.com/aien-dev/aien-sovereign-core-mv/commit/f4d98295b4b65cc97143df179380d2274bda5267) |
| ops/renda-8b-L720 | [2242fc1d9b50](https://github.com/aien-dev/aien-sovereign-core-mv/commit/2242fc1d9b50a43221773df37caad9c6a89dd18f) |
| own30b/a16 | [b1d6742ae316](https://github.com/aien-dev/aien-sovereign-core-mv/commit/b1d6742ae316c8929383e2b50789de31d0fda336) |
| own30b/a16b | [dc56b5371974](https://github.com/aien-dev/aien-sovereign-core-mv/commit/dc56b53719741c27600f3596ff4cd37773335df9) |
| own30b/decode-mega | [8b9be2b8cb1b](https://github.com/aien-dev/aien-sovereign-core-mv/commit/8b9be2b8cb1bf66c118658fc12f2b04be0dea9dc) |
| own30b/rc2 | [d26b3c6ae1f1](https://github.com/aien-dev/aien-sovereign-core-mv/commit/d26b3c6ae1f1b7c7d132431e21af2317823e2b34) |
| own30b/rc3 | [0826a255ae52](https://github.com/aien-dev/aien-sovereign-core-mv/commit/0826a255ae52c1bca738f7feafc3722f4984d6dd) |
| own30b/resident | [70a567210cfe](https://github.com/aien-dev/aien-sovereign-core-mv/commit/70a567210cfe792841c8686645f8b487b63c249a) |
| own30b/spec-batch | [d2e88e97a3ae](https://github.com/aien-dev/aien-sovereign-core-mv/commit/d2e88e97a3aee079ee2aa17fab11c6da2a29b758) |
| own30b/spec-decode | [a453bc2bcb61](https://github.com/aien-dev/aien-sovereign-core-mv/commit/a453bc2bcb61420873423319c22da397e60b07f0) |
| own30b/spec-wp | [c2608aa2627f](https://github.com/aien-dev/aien-sovereign-core-mv/commit/c2608aa2627f3b27faffcf0f37525d05c21236fa) |
| own30b/stream-prefill | [15c3a33a75b1](https://github.com/aien-dev/aien-sovereign-core-mv/commit/15c3a33a75b1ff6bc83cd0507c833bb6dc67eafe) |
| perf/pipeline-power-warm | [3f58d0a398ea](https://github.com/aien-dev/aien-sovereign-core-mv/commit/3f58d0a398eade2966ce9dc1c8d0b05f4fb5eff4) |
| pipeline/slug-aien-spark-first | [bb5c1c8d6049](https://github.com/aien-dev/aien-sovereign-core-mv/commit/bb5c1c8d6049422f4adfa7f241dec5eb1d8f50a5) |
| pr-68 | [2caecc68374e](https://github.com/aien-dev/aien-sovereign-core-mv/commit/2caecc68374e2ae0dab156597378194b8df8a8e8) |
| pr-84 | [9a10442364b1](https://github.com/aien-dev/aien-sovereign-core-mv/commit/9a10442364b181c5d3a04e2146642f8c08cea884) |
| ship/motor-l387 | [ecf64eedb9ab](https://github.com/aien-dev/aien-sovereign-core-mv/commit/ecf64eedb9ab83e9dc54127e83806b54c97ee0b2) |
| sub4bit/drift-study | [4451fec3aa53](https://github.com/aien-dev/aien-sovereign-core-mv/commit/4451fec3aa53863d2cd7e5197157097afafea095) |
| test/crypto-reuse-708 | [106339ad2836](https://github.com/aien-dev/aien-sovereign-core-mv/commit/106339ad2836f900d1acfe770720dc2b15599e85) |
| train/cycle3 | [352e4208dee2](https://github.com/aien-dev/aien-sovereign-core-mv/commit/352e4208dee25515193458cb3c4f0f184a5954f8) |
| fix/452-prefill-mutants | [1415d9450988](https://github.com/aien-dev/aien-sovereign-core-mv/commit/1415d94509882c85e66a4397a25e25637fbc2331) |
| fix/408-receipt-continuity | [382dfc0190d5](https://github.com/aien-dev/aien-sovereign-core-mv/commit/382dfc0190d5660153e6b0747cebc356bf4f2f4c) |
| c619 | [068c0140e869](https://github.com/aien-dev/aien-sovereign-core-mv/commit/068c0140e869c118fe1484c4231fb406d41ea450) |
| c620 | [e01e16d1c0ca](https://github.com/aien-dev/aien-sovereign-core-mv/commit/e01e16d1c0caa9211f3809fe744787ec6f1e7253) |
| c621 | [2ebe70876ae4](https://github.com/aien-dev/aien-sovereign-core-mv/commit/2ebe70876ae4c7316d5aadd4fa47a3f5d2bb632f) |
| carry604 | [81b9c34d34f6](https://github.com/aien-dev/aien-sovereign-core-mv/commit/81b9c34d34f65fdfcf773c8a5794bb1cc183651b) |
| carry606 | [33466599e4ef](https://github.com/aien-dev/aien-sovereign-core-mv/commit/33466599e4ef3453fe4b316dffe3948aa447069f) |
| fix/native-imap-reader | [b3db92ef223a](https://github.com/aien-dev/aien-sovereign-core-mv/commit/b3db92ef223a66f8b842c8f858cfb2b633943680) |
| md604 | [aee99c56d5c7](https://github.com/aien-dev/aien-sovereign-core-mv/commit/aee99c56d5c7ce28e212af3bd6bab2e5971d55d8) |
| pr578 | [3ff38b810f48](https://github.com/aien-dev/aien-sovereign-core-mv/commit/3ff38b810f48aa7adf900d7cd878ea1634d01a90) |
| pr591 | [794cd8e7acec](https://github.com/aien-dev/aien-sovereign-core-mv/commit/794cd8e7acec27e6146c98930d2ce4b19b59f102) |
| pr604 | [d2cba088ba1a](https://github.com/aien-dev/aien-sovereign-core-mv/commit/d2cba088ba1a0eef080370e7b8c921321c1a1443) |
| pr606 | [5e062333fd75](https://github.com/aien-dev/aien-sovereign-core-mv/commit/5e062333fd750e64ee91e271cf7b5307bf88a13a) |
| sr604 | [310c6ff27394](https://github.com/aien-dev/aien-sovereign-core-mv/commit/310c6ff27394abf61ab1a9e1093939fe6c139b35) |
| sr619 | [12d1c0743d70](https://github.com/aien-dev/aien-sovereign-core-mv/commit/12d1c0743d7079cfed47ef991fc80c8da1df09e2) |
| sr620 | [efcbc782d2cd](https://github.com/aien-dev/aien-sovereign-core-mv/commit/efcbc782d2cd9e076716dc92d9e7aa2642ed7396) |
| sr621 | [aa4163c91000](https://github.com/aien-dev/aien-sovereign-core-mv/commit/aa4163c91000c7968194b21e41a7433f661678d2) |
| feat/swarm-bridge-587 | [804b808160c6](https://github.com/aien-dev/aien-sovereign-core-mv/commit/804b808160c66dd2eae89f682948b03c57c87957) |
| feat/592-switchboard-launcher | [314331ddd0ae](https://github.com/aien-dev/aien-sovereign-core-mv/commit/314331ddd0aef6e3d0145b8b39ec9289972ecf43) |
| fix-654-rebased | [c5d89ab93cad](https://github.com/aien-dev/aien-sovereign-core-mv/commit/c5d89ab93cad7ace3509263b4edc907cc28462c3) |
| fix/gb10-lock-held-signal | [0c3a1d4a20e9](https://github.com/aien-dev/aien-sovereign-core-mv/commit/0c3a1d4a20e96d53ef04fcfc06eeb34cfb3780d1) |
| fix/gpu-lock-bounded-wait | [1905a614360f](https://github.com/aien-dev/aien-sovereign-core-mv/commit/1905a614360f21eab1f99d98d37257aab4ea59c6) |
| fix/kv-corrupt-byte-flip | [c61bb3354d63](https://github.com/aien-dev/aien-sovereign-core-mv/commit/c61bb3354d63889924b6e24af4451d1ac7096f86) |
| fix/readiness-crumbless-repo | [c289ca3c9843](https://github.com/aien-dev/aien-sovereign-core-mv/commit/c289ca3c98438d0682c3c3a7400cab3a674e1734) |
| fix/wt-mirror-lock | [a65f6a29fbf5](https://github.com/aien-dev/aien-sovereign-core-mv/commit/a65f6a29fbf526ff4474ad220082764af62c5f14) |
| pipeline/aien-verification-phase3 | [3c561a35a561](https://github.com/aien-dev/aien-sovereign-core-mv/commit/3c561a35a561d08e04023ca4cf3f080109f9ba89) |
| pipeline/slug-aien-spark-first | [5e05902c1143](https://github.com/aien-dev/aien-sovereign-core-mv/commit/5e05902c1143f6a2babb565c520bb1926d6222c0) |
| pr-580 | [a10c00b821c3](https://github.com/aien-dev/aien-sovereign-core-mv/commit/a10c00b821c35473f52c8a23efd9a49ac635ed35) |
| pr-582 | [812d23b3bdb7](https://github.com/aien-dev/aien-sovereign-core-mv/commit/812d23b3bdb702caa0900df631bec82110899dc4) |
| pr-584 | [91a31b570633](https://github.com/aien-dev/aien-sovereign-core-mv/commit/91a31b570633f4a6e819172bb657de0063bfade0) |
| pr654 | [0be74e595b05](https://github.com/aien-dev/aien-sovereign-core-mv/commit/0be74e595b051f841a9e874e0041b61a4ade48f4) |
| sc-667 | [8d0fb7596ab0](https://github.com/aien-dev/aien-sovereign-core-mv/commit/8d0fb7596ab0c224433a4cb0cd2be6f89dee89da) |
| trial-693-on-695 | [3d88fe85f2f2](https://github.com/aien-dev/aien-sovereign-core-mv/commit/3d88fe85f2f24760eab74a3f853bfc9245c5929e) |
| adr0004-step3 | [64118f72e050](https://github.com/aien-dev/aien-sovereign-core-mv/commit/64118f72e0509717b09ee26af76c507a4506bad8) |

### Unfinished working-folder snapshots

| Source folder | Saved commit |
|---|---|
| sc-main-run7 | [4d7acb73792a](https://github.com/aien-dev/aien-sovereign-core-mv/commit/4d7acb73792a58817c9943f050544059e0ceb01e) |
| sc-main-smoke | [464d1a8a1864](https://github.com/aien-dev/aien-sovereign-core-mv/commit/464d1a8a1864790c901b0bd62f31b35fb082aa9d) |
| aien-mcp-broker | [6fbc881305e8](https://github.com/aien-dev/aien-sovereign-core-mv/commit/6fbc881305e82668e3712343a933bdc77c5e82b1) |
| aien-sovereign-core | [e072503165b6](https://github.com/aien-dev/aien-sovereign-core-mv/commit/e072503165b6d18cab65a4a8875b85d483a570b9) |
| aien-mcp-transport | [f3f801fe7e05](https://github.com/aien-dev/aien-sovereign-core-mv/commit/f3f801fe7e0504ea923043e8faf4f508df874bcb) |
| complete-system-architecture | [cf73a7d90b43](https://github.com/aien-dev/aien-sovereign-core-mv/commit/cf73a7d90b430902b250adfb96364c2e033e88e1) |
| gb10-release | [2304d5079004](https://github.com/aien-dev/aien-sovereign-core-mv/commit/2304d5079004b9efb10e13dfe3b749d5522557d1) |
| cta1024-base | [df1b0a6e172e](https://github.com/aien-dev/aien-sovereign-core-mv/commit/df1b0a6e172ecab29dc168001f869d8d5e802b3a) |
| mail-effect-gate | [724be23a8a6a](https://github.com/aien-dev/aien-sovereign-core-mv/commit/724be23a8a6aa6ebde551af42b4c7ac6707c4ddf) |
| golden-path-rebase | [3237510530d6](https://github.com/aien-dev/aien-sovereign-core-mv/commit/3237510530d6467ce3474cfcc83ce105df1d7f2e) |
| model-capsule-integration | [ae05cb8be71a](https://github.com/aien-dev/aien-sovereign-core-mv/commit/ae05cb8be71af1a7f8390393e59df183f45a080e) |
| pearl-cortex-m1 | [d435c3b41e5f](https://github.com/aien-dev/aien-sovereign-core-mv/commit/d435c3b41e5f83eb63af1c866373882ff180e6cf) |
| native-ends | [579469d4ee36](https://github.com/aien-dev/aien-sovereign-core-mv/commit/579469d4ee36633bf97f99c6f04dbfdc0bdf4539) |
| qwen3-coder-a3b-moe | [29406bfa8081](https://github.com/aien-dev/aien-sovereign-core-mv/commit/29406bfa8081e13b37f90803fb3ab6454a57fbb3) |
| aien-sandbox | [72d0f66efb4a](https://github.com/aien-dev/aien-sovereign-core-mv/commit/72d0f66efb4a704e10ac90bb4d609326d37a0f7e) |
| bug-sweep | [fc40e526f16f](https://github.com/aien-dev/aien-sovereign-core-mv/commit/fc40e526f16fc16767ec11d8757a6f1e3f1059ad) |
| c1-campaign-spec | [ea888c95dad4](https://github.com/aien-dev/aien-sovereign-core-mv/commit/ea888c95dad4560ec3a9f81c805c4a7f9faf7f5b) |
| license-canonical | [834197b79532](https://github.com/aien-dev/aien-sovereign-core-mv/commit/834197b79532e5ea5952391328e48c43581e6e2a) |
| linux-baseline | [11b4aad8c564](https://github.com/aien-dev/aien-sovereign-core-mv/commit/11b4aad8c564c022cdb88a325e5f99b597fcdce3) |
| canonical-branch-native-runtime | [f31885b31db3](https://github.com/aien-dev/aien-sovereign-core-mv/commit/f31885b31db3e25c30da6c6ef7ab26a7d0d17fb4) |
| core | [de5197154008](https://github.com/aien-dev/aien-sovereign-core-mv/commit/de5197154008a83cc918328eefe27a372e3b10b9) |
| fix-eff-serve-door | [bee5176fce16](https://github.com/aien-dev/aien-sovereign-core-mv/commit/bee5176fce16ae523418c6898f4372fe40dbc2cb) |
| L676-browser | [6a1c9c5ff0a5](https://github.com/aien-dev/aien-sovereign-core-mv/commit/6a1c9c5ff0a5fc037610881b181560e486d92a07) |
| renda-delivery-L679 | [81de996e6d77](https://github.com/aien-dev/aien-sovereign-core-mv/commit/81de996e6d7785909acf7b459ff997ad71b08041) |
| 03-sequence-runtime | [f7711efdfd04](https://github.com/aien-dev/aien-sovereign-core-mv/commit/f7711efdfd04c431820dcc17c78ce8e5c65d63f2) |
| GOSUB1-base | [ba0ca73cb316](https://github.com/aien-dev/aien-sovereign-core-mv/commit/ba0ca73cb316defebddf31004391e180eb59e508) |
| GOSUB1-sc | [460ef3c9dd54](https://github.com/aien-dev/aien-sovereign-core-mv/commit/460ef3c9dd5446d2aa7dc1ed3a490f18f8a6d50c) |
| GOSUB1-sc2 | [e6d9db1483c2](https://github.com/aien-dev/aien-sovereign-core-mv/commit/e6d9db1483c2ead7109082f29bacee8cae1163cd) |
| aien-m6 | [d44143bcb02d](https://github.com/aien-dev/aien-sovereign-core-mv/commit/d44143bcb02d0dd661dd300cad39ef9526b9b859) |
| L088-repo-agents | [7fbf511190e8](https://github.com/aien-dev/aien-sovereign-core-mv/commit/7fbf511190e80eee8133e30fd0fc8c7d240a4720) |
| aien-proof | [b1cd6943230d](https://github.com/aien-dev/aien-sovereign-core-mv/commit/b1cd6943230dbb13e43070241a3b6772e6894fee) |
| c4-sc | [2055672dd398](https://github.com/aien-dev/aien-sovereign-core-mv/commit/2055672dd398b328da61c1cb43d51f7ec4b0b360) |
| cortex-vault-auth | [c65afa39e2b2](https://github.com/aien-dev/aien-sovereign-core-mv/commit/c65afa39e2b225054e752049780fb52f2bd38359) |
| crumbs-v1 | [9dac0f9d4d83](https://github.com/aien-dev/aien-sovereign-core-mv/commit/9dac0f9d4d835abe31690df47fe1e188c57198f8) |
| formal-5 | [8bce6bfcd069](https://github.com/aien-dev/aien-sovereign-core-mv/commit/8bce6bfcd069397781ae15f3ddd00483b64325ff) |
| gandi-mail-connector | [f26641a32f23](https://github.com/aien-dev/aien-sovereign-core-mv/commit/f26641a32f232b30afe94f4236c02a6a8503233a) |
| l40-core | [794fcee1d6fa](https://github.com/aien-dev/aien-sovereign-core-mv/commit/794fcee1d6fa475166a96578c152b68975daafbc) |
| native-fp8-capsule | [765bcd246fe6](https://github.com/aien-dev/aien-sovereign-core-mv/commit/765bcd246fe6f3442259e71b4470a54bac3a4d24) |
| sc-138 | [4bca55c52e7c](https://github.com/aien-dev/aien-sovereign-core-mv/commit/4bca55c52e7c97186c58fb2afc197d39ab4a9662) |
| sc-ci-fix | [fc159b74d47b](https://github.com/aien-dev/aien-sovereign-core-mv/commit/fc159b74d47bcbcb05628df18d440b2e57e59df5) |
| sc-cockpit-access | [2d5d841d7f0b](https://github.com/aien-dev/aien-sovereign-core-mv/commit/2d5d841d7f0b7b228c6bd6ba55326a48e85da729) |
| sc-deploy | [b6e2afab678e](https://github.com/aien-dev/aien-sovereign-core-mv/commit/b6e2afab678e6f8fd23859115e34f126a7ba81be) |
| sc-fb1-cut3b | [48ffa29b4abc](https://github.com/aien-dev/aien-sovereign-core-mv/commit/48ffa29b4abc4d3c3af066ff9d0f773aa781c3da) |
| sc-fb1-cut2 | [094042bc04fc](https://github.com/aien-dev/aien-sovereign-core-mv/commit/094042bc04fc043258ddee182fceb687b557af83) |
| sc-hermetic | [bf17014da79d](https://github.com/aien-dev/aien-sovereign-core-mv/commit/bf17014da79da759187d2e694c4bbb41afd6ebba) |
| sc-kv-admission | [d773c0e48993](https://github.com/aien-dev/aien-sovereign-core-mv/commit/d773c0e4899332e3a4920d8d5b2704a14e3d7b85) |
| sc-kvfix | [f9d7f5e8ed76](https://github.com/aien-dev/aien-sovereign-core-mv/commit/f9d7f5e8ed76acbfe41c2e600e42c1d59649d490) |
| sc-lt-c6 | [2c7370dd1798](https://github.com/aien-dev/aien-sovereign-core-mv/commit/2c7370dd1798dfd89908819aeb42b064a1af96c5) |
| sc-mail-acceptance | [35afbe9d852a](https://github.com/aien-dev/aien-sovereign-core-mv/commit/35afbe9d852a5321ef88ee9770d3180de3a04897) |
| sc-path-m1 | [f253580b80c3](https://github.com/aien-dev/aien-sovereign-core-mv/commit/f253580b80c32602ea30d17e2210b5f3a85a1164) |
| sc-repro-install | [452667fa04ba](https://github.com/aien-dev/aien-sovereign-core-mv/commit/452667fa04ba02d204f39e1dcae4f61c30725b75) |
| sc-strict | [60723e8cbac8](https://github.com/aien-dev/aien-sovereign-core-mv/commit/60723e8cbac8234badf0fe943167df46160b31be) |
| sovereign-core-142 | [ea6d2ee806f6](https://github.com/aien-dev/aien-sovereign-core-mv/commit/ea6d2ee806f6388066f540d90d1e23537a97d6a1) |
| sovereign-core-dm0 | [3dc7fa645f5f](https://github.com/aien-dev/aien-sovereign-core-mv/commit/3dc7fa645f5f975fb6032c1a7ffe31f9f8be68f8) |
| sovereign-fb1-cut3a | [f7eba4b4b434](https://github.com/aien-dev/aien-sovereign-core-mv/commit/f7eba4b4b43429c4ddec73595f24d626b74bbd07) |
| sovereign-core-prefill-gate | [dea344c4143b](https://github.com/aien-dev/aien-sovereign-core-mv/commit/dea344c4143bbc60fec2d395fa065f46a9553f0f) |
| wt-power-warm | [403273a2af99](https://github.com/aien-dev/aien-sovereign-core-mv/commit/403273a2af993a6e1cfbec5fbdfc154ffcb89afc) |
| sc-gates-qwen38 | [7405b6ff501b](https://github.com/aien-dev/aien-sovereign-core-mv/commit/7405b6ff501b7c26aaf8bb7fdfd9d818d66032db) |
| sc-hum-exact-batch | [e9a5998dae76](https://github.com/aien-dev/aien-sovereign-core-mv/commit/e9a5998dae765879547e5fe98375120f5cd31ea9) |
| sc-l355 | [768a8f2c0fad](https://github.com/aien-dev/aien-sovereign-core-mv/commit/768a8f2c0fad4a9540a19b88290755ed98e4ac01) |
| sc-l313-door | [65b9063e1e28](https://github.com/aien-dev/aien-sovereign-core-mv/commit/65b9063e1e28d2a5df63f638e2265c7308f9305a) |
| sc-l383 | [9bff5d56d251](https://github.com/aien-dev/aien-sovereign-core-mv/commit/9bff5d56d25156e65e3f66ff4f28721e553a2e34) |
| sc-l384-train | [472209bb46a1](https://github.com/aien-dev/aien-sovereign-core-mv/commit/472209bb46a1546fe8203928663d4f114e239ba4) |
| sc-levers-stroke | [8b4a57438dbf](https://github.com/aien-dev/aien-sovereign-core-mv/commit/8b4a57438dbf4c955f161d10f053580c1204325e) |
| sc-motor | [cbeb612b9232](https://github.com/aien-dev/aien-sovereign-core-mv/commit/cbeb612b9232caa95ee6f9db6c917977f4bd07a8) |
| sc-motor-ship | [3a437a98530c](https://github.com/aien-dev/aien-sovereign-core-mv/commit/3a437a98530ccf21715b326b067c054fabfb3c13) |
| sc-own30b-decode | [d66015b7b83a](https://github.com/aien-dev/aien-sovereign-core-mv/commit/d66015b7b83a0b35e73a8c9703c04e5bc4301260) |
| sc-own30b-rc2 | [d0a228a8c6ba](https://github.com/aien-dev/aien-sovereign-core-mv/commit/d0a228a8c6ba119dfaeab5336d7c1bc6557fc1dd) |
| sc-own30b-rc3 | [7e7b4ca039a8](https://github.com/aien-dev/aien-sovereign-core-mv/commit/7e7b4ca039a8f0fb69c7731c693e6231d3e54885) |
| sc-own30b-resident | [decd3be64219](https://github.com/aien-dev/aien-sovereign-core-mv/commit/decd3be642192597a765aee48f8ad6cd2bbe5fde) |
| sc-q36 | [f647af630435](https://github.com/aien-dev/aien-sovereign-core-mv/commit/f647af63043568bc3ab6ed4a8a64ab62f45625e3) |
| sc-q36-batch | [d79076caa346](https://github.com/aien-dev/aien-sovereign-core-mv/commit/d79076caa346d827faa257542624644f538a2213) |
| sc-q36-prefix | [8ba978d1fe51](https://github.com/aien-dev/aien-sovereign-core-mv/commit/8ba978d1fe51259e541b710f1436b2f2c0873f60) |
| sc-q36-readers | [78b07b23c59c](https://github.com/aien-dev/aien-sovereign-core-mv/commit/78b07b23c59c1c13a86d0c95071b3115ea32f1e9) |
| sc-renda-8b-L720 | [89033ddcd1d2](https://github.com/aien-dev/aien-sovereign-core-mv/commit/89033ddcd1d280d9b4d8a97d0478953adccc5c99) |
| sc-train3 | [d57d4ab776f5](https://github.com/aien-dev/aien-sovereign-core-mv/commit/d57d4ab776f5a94548337ae4f916d42cdcc097cc) |
| sc-typewriter | [54ce612481e5](https://github.com/aien-dev/aien-sovereign-core-mv/commit/54ce612481e574cd5fea0366182530e9cf4ad25e) |
| sc-w1 | [d629ed9c263d](https://github.com/aien-dev/aien-sovereign-core-mv/commit/d629ed9c263d2addc38301866ad5cc796a8f1dd1) |
| sc-w3 | [9c1d24c7dcc2](https://github.com/aien-dev/aien-sovereign-core-mv/commit/9c1d24c7dcc24a0347e2795b767819b796b75229) |
| sc-w4 | [69ee13f00999](https://github.com/aien-dev/aien-sovereign-core-mv/commit/69ee13f00999e28d8e680e735538541bdb50a089) |
| sc-w5 | [01d890071da1](https://github.com/aien-dev/aien-sovereign-core-mv/commit/01d890071da1cae0b0048039e9bbda6df507c19c) |
| sc-w7 | [586244103c1b](https://github.com/aien-dev/aien-sovereign-core-mv/commit/586244103c1b7edbfbf891b6655863f915c52798) |
| github-polling-L714 | [41160adf31d3](https://github.com/aien-dev/aien-sovereign-core-mv/commit/41160adf31d39f1f898387a016aab667cf963e97) |
| sc-adr0004 | [3d0e99f36c75](https://github.com/aien-dev/aien-sovereign-core-mv/commit/3d0e99f36c75748069b74633f52c54b58bb27387) |
| sc-switchboard | [449d230c9332](https://github.com/aien-dev/aien-sovereign-core-mv/commit/449d230c9332f915b94851ef6cfe28cd9d488fff) |

## Verification and limits

Unpublished blobs were read and checked for credential filenames, common token/private-key
patterns and oversized objects. Two CUDApp audit-selftest blobs contain intentional fake
secret negative controls; those exact blobs were reviewed. Pattern screening is not a
formal security audit. Original source tests were not rerun for recovery publication.
Working files selected for capture were reread and compared before creating snapshots.
Root publication and reachability are verified separately after upload.
