"""How well does a 5-minute window predict a full 90? Practice copies only, Entry C out-of-fold predictions.

For each practice match: decode Entry C (Viterbi 20/32), score the full match, then score every
non-overlapping 5-minute window inside each half the way the public sample is scored (truth set to X
outside the window, prediction left whole). Prints aggregates only. Nothing is tuned.
    python scripts/window_var.py PRED_DIR   (repo root on PYTHONPATH, LYARI_DATASETS_DIR set to the copy)
Run on 10 Nov on the Entry C out-of-fold averages of v5bfc, v7c and v6c; results in docs/window-var-2026-11-10.md.
"""
import os, sys
import numpy as np
from evaluation import possession_metrics as pm
from lyari.decoders import viterbi as V
from lyari.decoders import window as W
from lyari.scoring import score as S

WIN = 5 * 60 * 25
d = sys.argv[1]
f = lambda g, p: pm.evaluate(g, p, bootstrap_n=0)["spell_f1_mean"]
full, wins, gaps = [], [], []
for m in S.master_matches("train"):
    path = os.path.join(d, m + ".csv")
    if not os.path.isfile(path):
        continue
    S.refuse_public(m)
    gt, cut = S.truth(m)
    probs = W.probs_of(path, len(gt))
    pred = V.decode(probs, 20, 32, "soft", 3.0, 0.75, 200)
    G, P = S.with_halftime(gt, cut), S.with_halftime(pred, cut)
    sf = f(G, P)
    ws = []
    for a, b in ((0, cut), (cut + 1, len(G))):
        for s in range(a, b - WIN + 1, WIN):
            g = np.full_like(G, pm.X)
            g[s:s + WIN] = G[s:s + WIN]
            if (g != pm.X).sum() == 0 or np.isin(g, [pm.H, pm.A]).sum() == 0:
                continue
            ws.append(f(g, P))
    ws = np.array(ws)
    full.append(sf); wins.append(ws); gaps.extend(ws - sf)
    print("  {:28s} full {:.3f}  5-min windows n={:2d}  mean {:.3f}  min {:.3f}  max {:.3f}".format(
        m, sf, len(ws), ws.mean(), ws.min(), ws.max()), flush=True)
allw = np.concatenate(wins); gaps = np.array(gaps)
print("ALL: matches {}  full-match mean {:.4f}  window mean {:.4f}  window sd {:.4f}".format(len(full), np.mean(full), allw.mean(), allw.std()))
print("     window minus its own full match: mean {:+.4f}  5-95% [{:+.3f}, {:+.3f}]  |diff|<=0.05 in {:.0f}% of windows".format(
    gaps.mean(), *np.percentile(gaps, [5, 95]), 100 * (np.abs(gaps) <= 0.05).mean()))
r = np.corrcoef(full, [w.mean() for w in wins])[0, 1]
one = np.array([w[len(w) // 2] for w in wins])
print("     one window per match vs full: corr {:.2f}; across matches, full-match sd {:.4f}".format(np.corrcoef(full, one)[0, 1], np.std(full)))
