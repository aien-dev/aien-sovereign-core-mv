Cell = accepted/drafted = accept rate, words delivered per round (w/r).

### forward

| prompt set | words | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---|---|---|---|---|---|
| code-golden | 30 | 15/15 = 1.000, 2.00 w/r | 20/22 = 0.909, 2.73 w/r | 24/28 = 0.857, 4.29 w/r | 27/29 = 0.931, 7.50 w/r | 29/30 = 0.967, 15.00 w/r |
| prose | 256 | 108/148 = 0.730, 1.73 w/r | 143/226 = 0.633, 2.27 w/r | 166/358 = 0.464, 2.84 w/r | 178/613 = 0.290, 3.28 w/r | 181/1159 = 0.156, 3.41 w/r |
| codelong | 256 | 119/138 = 0.862, 1.86 w/r | 158/198 = 0.798, 2.59 w/r | 192/256 = 0.750, 4.00 w/r | 210/371 = 0.566, 5.45 w/r | 222/548 = 0.405, 7.31 w/r |
| all | 542 | 242/301 = 0.804, 1.80 w/r | 321/446 = 0.720, 2.43 w/r | 382/642 = 0.595, 3.37 w/r | 415/1013 = 0.410, 4.20 w/r | 432/1737 = 0.249, 4.84 w/r |

### reverse

| prompt set | words | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---|---|---|---|---|---|
| code-golden | 28 | 14/14 = 1.000, 2.00 w/r | 18/20 = 0.900, 2.80 w/r | 22/26 = 0.846, 4.00 w/r | 25/27 = 0.926, 7.00 w/r | 27/28 = 0.964, 14.00 w/r |
| prose | 128 | 59/70 = 0.843, 1.83 w/r | 74/110 = 0.673, 2.33 w/r | 87/164 = 0.530, 3.12 w/r | 97/253 = 0.383, 4.00 w/r | 100/453 = 0.221, 4.41 w/r |
| codelong | 128 | 61/67 = 0.910, 1.91 w/r | 82/94 = 0.872, 2.72 w/r | 95/134 = 0.709, 3.76 w/r | 106/181 = 0.586, 5.57 w/r | 112/271 = 0.413, 7.53 w/r |
| all | 284 | 134/151 = 0.887, 1.88 w/r | 174/224 = 0.777, 2.54 w/r | 204/324 = 0.630, 3.46 w/r | 228/461 = 0.495, 4.81 w/r | 239/752 = 0.318, 5.92 w/r |

### 27B host reference vs the chip greedy text

- code-golden: host top-1 equals the chip word at 30 of 30 positions
- prose: host top-1 equals the chip word at 256 of 256 positions
- codelong: host top-1 equals the chip word at 255 of 256 positions

### Expert cache hit rate (generated words only)

| cache | experts | share | LRU hit (warm across prompts) | pinned hot set, held out |
|---|---|---|---|---|
| 2 GB | 723 | 2.9% | 0.451 | 0.014 |
| 4 GB | 1446 | 5.9% | 0.622 | 0.061 |
| 8 GB | 2893 | 11.8% | 0.765 | 0.145 |
| 16 GB | 5786 | 23.5% | 0.901 | 0.295 |
| 32 GB | 11573 | 47.1% | 0.952 | 0.537 |
