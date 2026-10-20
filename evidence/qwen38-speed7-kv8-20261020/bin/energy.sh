#!/bin/bash
# energy.sh RUN_DIR...: per speed point: tok/s (wall and chip medians), bytes moved per token (resident weights +
# the median step's KV rows) as a fraction of 273 GB/s, and J/token = mean wall W x span / steps from the sensor log
# (~/yardsticks/sensors/<date>.csv: col 1 unix s, col 3 wall W, col 4 GPU W; ~2 s samples).
set -u; S=$HOME/yardsticks/sensors
printf "%-26s %7s %9s %9s %10s %7s %7s %7s %8s %3s\n" run ctx tok/s_wall tok/s_chip GB/tok frac273 wall_W gpu_W J/tok n
for d in "$@"; do
  f=$d/speed.json; [ -f "$f" ] || { echo "$d: no speed.json"; continue; }
  jq -r '.steps as $s | .weight_bytes_resident as $w | .points[] | [.context, .tok_s_wall_median, .tok_s_chip_median, ($w + .kv_read_bytes_median_step), .wall_ms_median, .steps_t0_unix, .steps_t1_unix, $s] | @tsv' $f |
  while read -r ctx tw tc bytes wm t0 t1 steps; do
    day=$(date -d @${t0%.*} +%F)
    read -r ww gw n < <(awk -F, -v a=$t0 -v b=$t1 '$1>=a && $1<=b {w+=$3; g+=$4; n++} END{if(n) printf "%.1f %.2f %d", w/n, g/n, n; else print "nan nan 0"}' $S/$day.csv)
    awk -v r=$(basename $d) -v c=$ctx -v tw=$tw -v tc=$tc -v by=$bytes -v wm=$wm -v ww=$ww -v gw=$gw -v n=$n -v t0=$t0 -v t1=$t1 -v s=$steps 'BEGIN{
      printf "%-26s %7d %9.2f %9.2f %10.3f %7.3f %7s %7s %8s %3d\n", substr(r,1,26), c, tw, tc, by/1e9, (by/(wm/1e3))/273e9, ww, gw, (n ? sprintf("%.2f", ww*(t1-t0)/s) : "UNMEAS"), n}'
  done
done
