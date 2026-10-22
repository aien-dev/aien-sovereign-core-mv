# L371 eight readers: chip receipts (2026-10-21)

Build: `bin-6aa63303` (aien-serve-http from sc serve/q36-readers 6aa63303, libomega_gpu.a from bin-13f06473:
the batch door before L369 chip prefill, 16 prompt tokens per beat). Test server on port 18098 under q36-extra,
each run inside a spark-window with `--ledger L371`. 18091 was not touched.

- Run A `chip-all8x20480/`: one load, 8 rows x 20480 positions. Readers id on all four docs, plain id on design and code.
- Run B `chip-plain1x122880/`: one load, 1 row x 122880 positions (the plain path needs one row that holds the whole
  doc). Plain id on log and big. Window 20261021T215856Z-b73c7d.

Docs are frozen outside git in `~/handoffs/l371-docs` (`docs.SHA256SUMS`); the questions and truths are in
`questions.json`. Requests are greedy (temperature 0), thinking off, max 700 reply tokens. `ask.sh` runs one request
and writes `<doc>-<mode>.json` and `.time` (wall clock around the HTTP call).

## Time to answer

| Doc | Input tokens | Readers wall | Plain wall | Result |
|---|---|---|---|---|
| design (ROADMAP + ADRs) | 15210 | 592.8 s | 394.3 s | readers 1.50x slower (calc:4087a05b) |
| code (kv_persist.rs) | 16669 | 635.8 s | 437.6 s | readers 1.45x slower (calc:a0d25c1a) |
| log (18091 serve.log) | 54418 | 1439.2 s | 3039.4 s | readers 2.11x faster (calc:0d46714b) |
| big (logs, 117742 tok) | 117742 | 3037.4 s | no answer in 9000 s | readers more than 2.96x faster (lower bound, calc:d495fc81) |

Why (MEASURED parts, INFERRED link):
- Run A: an 8-row step costs 143.7 ms, a 1-row step 23.7 ms, 6.06x (calc:351d023d). On this door eight rows do not
  share the weight read well, so eight short reads cost more than one read of the same tokens when the doc is short.
- Run B: the plain read of the 54k log took 55.51 ms per prompt token (prefill 3022480 ms, calc:a3de129b), 18.01
  prompt tok/s (calc:352c22aa), because every beat attends over a context that grows to 54k. Each reader only holds
  6.7k to 6.9k tokens (stub dry run of the same split), so its steps stay cheap. The longer the input, the more the split pays.
- Not measured: the same runs on the ship build (L369 chip prefill measured 4.34x to 5.54x faster prompt reading at 8k to 256 tokens, BOARD 22:17Z, and
  rows riding strokes would cut the 8-row step). The ratios above belong to bin-6aa63303 only.

Map phase share of the readers wall: design 477.4 s, code 520.8 s, log 1335.0 s, big 2927.5 s (`map_ms` in
`aien_readers`).

## Quality (scored by hand against `questions.json` truths)

| Doc | Readers | Plain | Notes |
|---|---|---|---|
| design | 3.5/4 | 4/4 | readers missed "Physics 10/10": reader 4's notes hit the 400-token cap before that fact |
| code | 4/4 | 4/4 | |
| log | 3/4 | 3/4 | readers merged three server starts into two (a); plain missed the largest prompt (c), it named 5275 tokens, truth is 19889 at line 17; readers got (c) right |
| big | 12 of 14 starts found (all real), reply hit the 700-token cap | none: no answer | |

All 32 reader notes in run A hit the 400-token note cap. Fix pushed after run A (sc 21e94b0d, not chip-run yet):
the notes put the facts that answer the request first, and the connect step is told that events with different
times or ids are different events (count each once).

## Visible in the response

Every readers reply carries `aien_readers`: per part the line range, tokens, prompt tokens, note tokens, finish
reason and the notes; plus `map_ms`, `map_prompt_tokens`, `connect_prompt_tokens`. With `aien_show_notes: true`
the notes also open the reply as a collapsed `<details>` block. When streaming, progress arrives as reasoning
("Reading N tokens in n parts at once", "Notes from n readers in X s; connecting them").

## Limits

- Memory: 8 rows x 18432 and 8 x 8192 were refused by the door's memory check while 18091 was loaded
  (`server-18098-refused-8x18432.log`, `server-readers8x8192.log`); run A only fit when 18091 was down for tests.
  The live server with readers needs rows sized for the longest part plus notes.
- `server-plain1x18432.log`: one plain load ended before serving, cause UNKNOWN (not repeated).
- One run per doc and mode; no repeat runs, so no spread.
- Plain big: the request went in at 22:49:42Z (after a 27.7 s wait for the chip lock); curl's 9000 s cap (`ASK_MAX_TIME`)
  ended it at 01:19:42Z and the script then stopped the server (SIGTERM 01:19:48Z). No leave line, so no reply was
  produced. Not repeated: one more try would hold the chip's quiet lock for more than 2.5 h. The ratio is a lower bound.
- Inputs were 15k, 17k, 54k and 118k tokens (real docs), not the 8k/32k/100k in the brief. Below the threshold
  (3000 tokens) the readers id is the plain path (host test `a_short_input_on_the_readers_id_takes_the_plain_path`).

## What these receipts do not cover (review by Codex Astra, 2026-10-22)

- Build: every number above is bin-6aa63303. Not chip-run: the notes fix (21e94b0d), the ship merge
  (serve/q36-readers-ship 72bb1ee6) and L369 chip prefill. On the ship build each reader still reads its part
  through its own prefill calls, one row after another inside every beat (128-token chunks,
  `qwen36_batch.rs` prefill loop): the eight parts advance together, but their prompt reading does not share a weight
  read. Only the notes (decode) ride one weight read for all eight rows. The time gain on long inputs above comes
  from short contexts per reader, not from shared prompt reading.
- Quality after the notes fix, and readers on the ship build, are checked by the merge-train case `readers-gate`
  (sc l371/train-cases 5c88dc85: 8 readers, the final answer must join two facts from far-apart parts). That
  verdict, then the live check on 18091 after L387's swap (Hermes picks the readers id, notes visible, a tool call
  round trip), are still pending.
