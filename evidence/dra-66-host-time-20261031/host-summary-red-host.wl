(* records 0..19 (seq) of evidence/dra-66-host-time-20261031/host-token-receipt.jsonl; the whole file checked green *)
(* aien.q36-token-receipt/1: 19 decoding beats; integer ns, host clock *)
receipt = <|
  "rows" -> {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
  "prompt_rows" -> {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
  "steps" -> {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
  "submits" -> {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
  "total_ns" -> {1055251, 1255684, 1210228, 1091411, 1071987, 1072675, 1075875, 1069572, 1071571, 1066499, 1069827, 1085523, 1073763, 1071523, 1183108, 1105299, 1067795, 1051843, 1053811},
  "step_ns" -> {447795, 587124, 558516, 454899, 451651, 455171, 451938, 451459, 450659, 451283, 452003, 465027, 454867, 454803, 568116, 448931, 444659, 444995, 446163},
  "graph_ns" -> {447731, 586980, 558308, 454739, 451587, 455075, 451826, 451363, 450563, 451203, 451891, 464947, 454787, 454723, 568036, 448547, 444595, 444947, 446099},
  "record_ns" -> {93233, 96881, 197378, 97473, 95809, 95025, 95120, 95024, 94624, 95648, 94768, 97280, 97649, 94401, 95297, 94673, 91777, 91857, 92337},
  "step_chip_ns" -> {354402, 490019, 360818, 357202, 355714, 359938, 356642, 356243, 355875, 355491, 356979, 367571, 357010, 360226, 472675, 353810, 352754, 353042, 353714},
  "chip_ns" -> {354402, 490019, 360818, 357202, 355714, 359938, 356642, 356243, 355875, 355491, 356979, 367571, 357010, 360226, 472675, 353810, 352754, 353042, 353714},
  "host_overhead_ns" -> {700849, 765665, 849410, 734209, 716273, 712737, 719233, 713329, 715696, 711008, 712848, 717952, 716753, 711297, 710433, 751489, 715041, 698801, 700097}
|>;
clean[k_] := Pick[Range[Length[receipt["rows"]]], MapThread[#1 == k && #2 == 0 &, {receipt["rows"], receipt["prompt_rows"]}]];
(* DRA-66 host lower bound per token at 1 conversation (wall - chip_ns; chip_ns holds the submit and the marker poll, see submitPoll): {beats, median ms} *) hostLB = Median[receipt["host_overhead_ns"][[clean[1]]]]; {Length[clean[1]], N[hostLB / 10^6]}
(* per conversations k, clean beats: {beats, median total, chip, host overhead ms} *) Table[{k, Length[clean[k]], N[Median[receipt["total_ns"][[clean[k]]]] / 10^6], N[Median[receipt["chip_ns"][[clean[k]]]] / 10^6], N[Median[receipt["host_overhead_ns"][[clean[k]]]] / 10^6]}, {k, Select[Union[receipt["rows"]], Length[clean[#]] > 0 &]}]
(* host overhead at 1 conversation by where it sits (ms, medians): outside the step, step outside the graph, graph outside record and chip, record *) With[{c = clean[1], r = receipt}, N[{Median[(r["total_ns"] - r["step_ns"])[[c]]], Median[(r["step_ns"] - r["graph_ns"])[[c]]], Median[(r["graph_ns"] - r["record_ns"] - r["step_chip_ns"])[[c]]], Median[r["record_ns"][[c]]]} / 10^6]]
