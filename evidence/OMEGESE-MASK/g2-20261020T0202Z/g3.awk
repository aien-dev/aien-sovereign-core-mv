# G3 from the mask server's own log lines (serve-window.log): per request decode ms, tokens, ms/token;
# mask-on adds the constraint's own time. Row order = control, 10 off, 10 on (window.sh order).
/qwen38: (lock wait|constraint omegese)/ {
  on = ($0 ~ /constraint omegese/)
  match($0, /decode [0-9]+ tok in [0-9]+ ms/); split(substr($0, RSTART, RLENGTH), d, " "); tok = d[2]; ms = d[5]
  match($0, /wall [0-9.]+ ms\/token/); split(substr($0, RSTART, RLENGTH), w, " ")
  cu = 0; cm = 0
  if (on) { match($0, /constraint [0-9]+ us total, [0-9]+ us max step/); split(substr($0, RSTART, RLENGTH), c, " "); cu = c[2]; cm = c[5] }
  n++; mode = (n == 1) ? "control" : on ? "on" : "off"
  printf "%s\t%d\t%d\t%.1f\t%.1f\t%.2f\t%.1f\n", mode, tok, ms, w[2], ms / (tok > 1 ? tok - 1 : 1), cu / 1000 / tok, cm / 1000
  if (mode != "control") { T[mode] += tok; M[mode] += ms; C[mode] += cu }
}
END {
  for (m in T) printf "TOTAL %s: %d tokens, decode %d ms, %.1f ms per token; constraint %.1f ms total = %.2f ms per token\n", m, T[m], M[m], M[m] / T[m], C[m] / 1000, C[m] / 1000 / T[m]
}
