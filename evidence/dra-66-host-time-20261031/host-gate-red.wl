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
(* aien.q36-decode-timeline/1: 19 beats; integer ns per beat; host clock for host:*, period, sync, chip_ns, submit_poll; chip PTIMER for dev:*, dev_span *)
timeline = <|
  "period" -> {736278, 909992, 774279, 887415, 1275243, 3087994, 1132921, 817751, 1160506, 4204771, 2246051, 788407, 832455, 1352459, 803223, 3115162, 802599, 786502, 780567},
  "host:sched_join" -> {80, 167762, 50704, 77488, 45024, 76448, 73280, 65041, 70593, 42192, 95617, 53921, 93953, 52912, 63633, 41680, 60225, 49776, 36513},
  "host:sample" -> {14048, 4736, 4145, 3040, 7728, 5472, 33905, 6160, 47968, 16448, 56624, 10496, 15024, 5184, 8864, 6160, 7088, 6800, 6224},
  "host:sched_feed" -> {560, 320, 496, 208, 480, 736, 880, 672, 608, 608, 1057, 832, 496, 464, 736, 416, 976, 576, 512},
  "host:step_enter" -> {80, 80, 144, 80, 80, 80, 128, 96, 80, 96, 176, 80, 96, 80, 96, 80, 288, 112, 192},
  "host:embed" -> {32, 32, 32, 48, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 48, 128, 48, 32},
  "host:graph_record" -> {103985, 113905, 105472, 104897, 107953, 108353, 114721, 110321, 107793, 104721, 111664, 107601, 106321, 115361, 111377, 105665, 107553, 106849, 106129},
  "host:graph_submit_wait" -> {608197, 615509, 608502, 693958, 1108122, 2877177, 899495, 630245, 927192, 4018402, 1954929, 610293, 610997, 1167546, 613525, 2934601, 615717, 617077, 622613},
  "host:graph_exit" -> {48, 48, 48, 64, 48, 64, 48, 64, 48, 64, 32, 48, 48, 48, 32, 336, 32, 48, 48},
  "host:logits_read" -> {48, 32, 32, 48, 32, 32, 32, 48, 32, 32, 64, 32, 32, 48, 48, 32, 32, 32, 32},
  "host:step_return" -> {1360, 2544, 1984, 3840, 2448, 9648, 5088, 2432, 3392, 13408, 10384, 2352, 1984, 4928, 2160, 10752, 5760, 2336, 3936},
  "host:sched_post" -> {7840, 5024, 2720, 3744, 3296, 9952, 5312, 2640, 2768, 8768, 15472, 2720, 3472, 5856, 2720, 15392, 4800, 2848, 4336},
  "dev:experts" -> {200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200},
  "dev:gdn" -> {300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300},
  "dev:attention" -> {100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100},
  "dev:router" -> {100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100},
  "dev:head_screen" -> {300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300},
  "dev:head_recheck" -> {100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100},
  "dev:head_plain" -> {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
  "dev:token_pick" -> {200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200},
  "dev:glue" -> {800, 800, 800, 800, 800, 800, 800, 800, 800, 800, 800, 800, 800, 800, 800, 800, 800, 800, 800},
  "dev:unknown" -> {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
  "dev_span" -> {2100, 2100, 2100, 2100, 2100, 2100, 2100, 2100, 2100, 2100, 2100, 2100, 2100, 2100, 2100, 2100, 2100, 2100, 2100},
  "sync" -> {606097, 613409, 606402, 691858, 1106022, 2875077, 897395, 628145, 925092, 4016302, 1952829, 608193, 608897, 1165446, 611425, 2932501, 613617, 614977, 620513},
  "submits" -> {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
  "chip_ns" -> {22100, 22100, 22100, 22100, 22100, 22100, 22100, 22100, 22100, 22100, 22100, 22100, 22100, 22100, 22100, 22100, 22100, 22100, 22100},
  "submit_poll" -> {20000, 20000, 20000, 20000, 20000, 20000, 20000, 20000, 20000, 20000, 20000, 20000, 20000, 20000, 20000, 20000, 20000, 20000, 20000}
|>;
(* ms per beat, medians: *) N[Median /@ timeline / 10^6]
(* outside the chip + chip + sync vs the period: *) N[{Median[timeline["period"] - timeline["host:graph_submit_wait"]], Median[timeline["dev_span"]], Median[timeline["sync"]], Median[timeline["period"]]} / 10^6]
(* residual against the recorded 20.7 ms step: *) N[207/10 - Median[timeline["period"]] / 10^6]
(* DRA-66 submit + poll share per token, chip_ns - dev_span over single-submit beats: {beats, median ms} *) one = Flatten[Position[timeline["submits"], 1]]; submitPoll = Median[timeline["submit_poll"][[one]]]; {Length[one], N[submitPoll / 10^6]}
(* DRA-66 gate, sum of medians (hostLB: token receipt, receipt-1conv slice, clean 1-row beats; submitPoll: stamped timeline, single-submit beats): {host lower bound ms, submit + poll ms, sum ms, sum <= 0.5 ms} *) {N[hostLB / 10^6], N[submitPoll / 10^6], N[(hostLB + submitPoll) / 10^6], hostLB + submitPoll <= 500000}
