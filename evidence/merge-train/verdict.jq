# merge-train: cells (one JSON per arm x case, slurped) -> the lever x case verdict table.
# $s[0] = suite.json, $present[0] = {"Lnnn": "yes"|"no"|"unknown"}.
def rank: {"FAIL":6, "RED-FAIL":6, "ERROR":5, "NOT-ON-TRAIN":4, "SKIP":3, "PASS":1, "RED-OK":1, "-":0}[.] // 5;
. as $cells | $s[0] as $suite | $present[0] as $pr | ($suite.cases | map(.id)) as $ids
| {cases: $ids,
   levers: [$suite.levers[] | . as $l | ($pr[$l.id] // "unknown") as $p
     | [$ids[] | . as $cid
         | ([$suite.cases[] | select(.id == $cid)][0].levers // null) as $only
         | [$cells[] | . as $x
             | select(($x.case == $cid or $x.case == "*") and (($l.arms | index($x.arm)) != null)
                      and ($only == null or (($only | index($l.id)) != null)))
             | $x.verdict]
         | if length == 0 then "-" else max_by(rank) end] as $row
     | {id: $l.id, name: $l.name, present: $p, cells: $row,
        verdict: (if $p == "no" then "NOT-ON-TRAIN"
                  else ($row | map(select(. != "-"))
                        | if length == 0 then "-"
                          elif any(.[]; . == "FAIL" or . == "RED-FAIL") then "FAIL"
                          elif any(.[]; . == "ERROR") then "ERROR"
                          elif any(.[]; . == "NOT-ON-TRAIN") then "NOT-ON-TRAIN"
                          elif all(.[]; . == "SKIP") then "SKIP"
                          else "PASS" end)
                       + (if $p == "unknown" then "?" else "" end) end)}],
   red_controls: {ran: ([$cells[] | select(.verdict == "RED-OK" or .verdict == "RED-FAIL")] | length),
                  ok: ([$cells[] | select(.verdict == "RED-OK")] | length),
                  total: ([$suite.cases[] | select(.expect == "red" or .expect_red != null)] | length)}}
