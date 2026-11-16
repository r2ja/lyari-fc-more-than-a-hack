# Five minutes vs the full 90 (10 Nov 2026)

Question: how well does a score on one 5-minute window, scored the way the public sample is scored, describe a model's score over the full match? Practice data only; nothing is tuned and the public window is not used.

Setup: Entry C out-of-fold averages (13-fold leave-one-match-out) on the three practice copies v5bfc, v7c and v6c, plain Viterbi 20/32. Each match is scored in full, then every non-overlapping 5-minute window (7,500 frames) inside each half is scored with the truth set to X outside the window and the prediction left whole. Script: `scripts/window_var.py`.

## Result

| copy | full-match mean | 5-min window sd | window minus own full match, 5 to 95% | windows within ±0.05 of own full match | one window per match vs full, correlation |
|---|---|---|---|---|---|
| v5bfc | 0.7826 | 0.103 | −0.160 to +0.159 | 37% | 0.65 |
| v7c | 0.7449 | 0.108 | −0.167 to +0.165 | 33% | 0.58 |
| v6c | 0.7545 | 0.110 | −0.159 to +0.176 | 40% | 0.79 |

- The full-match means reproduce Entry C's 13-fold scores.
- A 5-minute score is unbiased (mean +0.005 against its own full match) but noisy: about ±0.16 at 90%, against full-match differences between matches of sd 0.025 to 0.034 and against the +0.003 to +0.006 changes our decisions turn on.
- Same model, same match: metrica_game2 on v5bfc ranges from 0.425 to 0.922 across its windows (full match 0.786).

Consequence: a 5-minute window cannot separate systems at the scale we decide on. Our decisions use paired 13-fold full-match comparisons (about 230 five-minute windows' worth of play per copy), and the public window is never a target.
