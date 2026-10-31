(* fit.wl: DRA-75 (#869) HISTORICAL per-class decomposition and ladder fits. Data: lever-2 shared-chip runs
   (in_window false) from evidence/motor-chip-20261021 and DRA-64 benchmarks-mv 35528552 inputs.wl.
   Per-class times are ONE last-stroke timeline at N = 8 (per-class/stroke8l2-stamps.tsv, parser output of
   om/timeline/timeline.om). Ladder medians are the gate upper medians. Nothing here is a chip measurement of main.
   Evaluated 2026-10-31 on the Wolfram MCP kernel (15.0.1); output in fit-output.txt. *)
l2 = <|"ab"->{30,0.420},"argmax"->{16,0.138},"attn"->{80,5.301},"down"->{40,15.858},"gate"->{80,0.221},"gate_up"->{40,22.840},"gather"->{10,0.059},"gdn"->{240,7.237},"group"->{40,0.509},"lm_head"->{2,9.619},"mix"->{40,6.502},"norm"->{1,0.012},"o_proj"->{40,2.433},"perm"->{370,1.500},"res"->{360,2.833},"router"->{40,2.131},"swiglu"->{40,0.126}|>;
perConv = {7.237/8, 5.301/8, 1.500/8, 1.500*240/370/8, 0.221/8};
four = 7.237/8 + 5.301/8 + 0.123 + 0.221/8;
gdnBytes = 2*32*128*128*4*30*8;
gbs = gdnBytes/(7.237*10^-3)/10^9;
attnUs = 5.301/80*1000;
kvBytesConv = 10*2*256*2*2*71;
kvGBs = kvBytesConv/(5.301/8*10^-3)/10^9;
lMain[n_] := 531 + 82 n + Ceiling[n/4]; lL2[n_] := 491 + 122 n + Ceiling[n/4];
sumL = Total[l2[[All,1]]]; sumMs = Total[l2[[All,2]]];
ns = {1,2,4,8}; ms = {25.320206, 32.464756, 44.350601, 85.427837}; u = {9, 16.3, 30.7, 55.6};
f1 = LinearModelFit[Transpose[{ns, ms}], x, x];
f2 = LinearModelFit[Transpose[{ns, Ceiling[ns/4], ms}], {x, y}, {x, y}];
dra64 = 9.13 + 3.29 ns + 0.696 u + 4.81 Ceiling[ns/4];
cExp = (22.840 + 15.858)/55.6;
perHead = Total[{7.237, 5.301, 0.221, 0.138}]/8 + 1.500*240/370/8;
(* MODELED main-graph forecast (CN rows): per-head classes scale with N, perm splits 240 per-head / 130 fixed of 370,
   experts c U, lm_head h Ceiling[N/4], rows classes held at their N = 8 time (upper bound), res at 80/360 launches. *)
mainPred[n_, un_] := n*(7.237 + 5.301 + 0.221 + 0.138 + 1.500*240/370)/8 + (1.500*130/370) + cExp*un + (9.619/2)*Ceiling[n/4] + (0.420 + 0.059 + 0.509 + 6.502 + 0.012 + 2.433 + 2.131 + 0.126) + 2.833*80/360;
<|"perConvGdnAttnPermPermOGate"->perConv, "fourClassSum"->four, "gdnBytes"->gdnBytes, "gdnGBs"->gbs, "attnUsPerLaunch"->attnUs, "kvBytesPerConvPos70"->kvBytesConv, "attnEffGBs"->kvGBs, "launchMain"->lMain/@{1,2,4,8}, "launchL2"->lL2/@{1,2,4,8}, "sumLaunches"->sumL, "sumMs"->sumMs, "fitLin"->{f1["BestFitParameters"], f1["FitResiduals"], Sqrt[Mean[f1["FitResiduals"]^2]]}, "fitLinCeil"->{f2["BestFitParameters"], f2["FitResiduals"], Sqrt[Mean[f2["FitResiduals"]^2]]}, "dra64Pred"->dra64, "dra64Resid"->ms - dra64, "cExp"->cExp, "perHeadMs"->perHead, "mainPredStroke"->Table[mainPred[ns[[i]], u[[i]]], {i,4}], "mainPredDup9"->Table[mainPred[n, 9], {n, {2,4,8}}], "dupMinusDistinct8"->cExp*(55.6-9)|>
