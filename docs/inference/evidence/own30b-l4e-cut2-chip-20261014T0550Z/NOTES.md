# W-L4e cut 2: break-even back-off (small weights, GB10, 3 real layers)

HARDWARE, synthetic (3 layers; speeds are NOT the full model's). Purpose: tokens unchanged with the back-off on,
and the controls still work. own30b_spec_check --verify exact, suffix drafter, 2 runs, AIEN_OWN30B_SPEC_BREAK_EVEN=1 vs 0.
- Back-off on (be1.raw): every ON run IDENTICAL to OFF; RESULT PASS. With 3 layers a plain token costs 4.5 ms
  (measured) and the cost model's verify costs are the full model's, so the drafter rarely drafts: that is the
  measured-cost rule working, not a fault (one prompt drafted and its rounds paid).
- Back-off off (be0.raw): IDENTICAL on every run, extra-accept and act-fp8 controls caught (5 of 5).
Decision quality on full-model costs: unit test break_even_stops_drafting_where_it_loses_and_keeps_it_where_it_pays
(simulation with the W-L4d gate's measured costs: p4-like text loses >5% without back-off, <=2% with it at
27.8 ms plain, <=3% at 22.5 ms (one-list decode) over 64 tokens and <=1.5% over 256; code/JSON keeps >20% gain).
UNKNOWN until the full-model gate: real p4 ON vs OFF.
