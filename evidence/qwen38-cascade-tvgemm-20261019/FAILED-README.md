# Failed first attempt (binary built from sc e2080d1e, omega lib without 37034f03)
Both runs (tree and exact) stopped at layer 0: "down gemm x50: CODEGEN_FAIL: fp8 gemm codegen refused the shape". Cause: the row-pair GEMM refuses k > 16384 and down_proj has k = 17408;
the lift to k 32768 is omega 37034f03, missing from the omega branch lib that was linked. These are NOT measurements. Fixed by taking 37034f03 into omega cascade/tree-verify-gemm (22446aef); rerun follows with a rebuilt binary.
