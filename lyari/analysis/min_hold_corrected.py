"""min_hold_sportec with the length bias removed: the MIN_HOLD the open data supports (docs/min-hold-sportec.md).

    python -m lyari.analysis.min_hold_corrected --events ../datasets/sportec-idsse --master ../datasets/master

min_hold_sportec calls a candidate spell controlled when its team has a logged pass, shot, claim or won duel inside
it. A longer spell holds more events whether or not it was controlled: possessions that are certainly controlled
(raw-flag spells of 8 s or more that start with a turnover) show such an event in their first 2 s only about half
the time. So the share of controlled candidates is corrected per length bin:

    pi(L) = (obs(L) - eps) / (f(L) - eps)

obs = share of candidates of length L with an event, f = the same share for certainly controlled spells cut to L,
eps = the false-event floor (candidates under 0.4 s, sync and tolerance slack). MIN_HOLD is the lower edge of the
first length bin from which pi stays at or above one half. Same events, sync and tolerance as min_hold_sportec. By default only the five
Sportec practice matches (the master train split); --all-matches adds the two test matches. Aggregates only.
"""

import argparse
import glob
import json
import os
import re

import numpy as np

from lyari.analysis import min_hold_sportec as MH
from lyari.dataset import labels as L

FPS = 25
EDGES = np.array([0.4, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0])
MIDS = (EDGES[:-1] + EDGES[1:]) / 2
REF_MIN_S = 8


def one_match(npz, events_dir, tol):
    mid = re.search(r"(DFL-MAT-\w+)\.npz$", npz).group(1)
    with open(glob.glob(os.path.join(events_dir, f"DFL_02_01_matchinformation_*_{mid}.xml"))[0], encoding="utf-8") as f:
        home_id = re.search(r'HomeTeamId="([^"]*)"', f.read()).group(1)
    events = MH.read_events(glob.glob(os.path.join(events_dir, f"DFL_03_02_events_raw_*_{mid}.xml"))[0], home_id)
    z = np.load(npz)
    flag, period = z["label_source"].astype(np.int8), z["period"]
    kickoffs = {s: t for t, s, _, k in events if k == "kickoff"}
    offsets = {p: MH.sync(events, flag, period, kickoffs[p], p)[:3] for p in (1, 2)}
    ev = MH.event_frames(events, period, kickoffs, offsets)
    cands = MH.candidates(flag)
    dur = np.array([e - s for s, e, _ in cands]) / FPS
    ctrl = MH.label_candidates(cands, ev, tol)
    sp = L.spells(flag)
    refs = [(s, t) for i, (t, s, e) in enumerate(sp)
            if i and t in (L.H, L.A) and e - s >= REF_MIN_S * FPS and sp[i - 1][0] not in (t, L.D)]
    f = np.array([MH.label_candidates([(s, s + int(m * FPS), t) for s, t in refs], ev, tol).mean() for m in MIDS])
    return dict(mid=mid, dur=dur, ctrl=ctrl, f=f, nref=len(refs))


def curve(ms):
    dur = np.concatenate([m["dur"] for m in ms])
    ctrl = np.concatenate([m["ctrl"] for m in ms])
    w = np.array([m["nref"] for m in ms], float)
    f = (np.array([m["f"] for m in ms]) * w[:, None]).sum(0) / w.sum()
    eps = ctrl[dur < EDGES[0]].mean()
    bins = [(dur >= a) & (dur < b) for a, b in zip(EDGES[:-1], EDGES[1:])]
    obs = np.array([ctrl[b].mean() for b in bins])
    return dict(n=np.array([b.sum() for b in bins]), obs=obs, f=f, eps=eps,
                pi=np.clip((obs - eps) / (f - eps), 0, 1.5))


def crossing(pi):
    """Lower edge of the first length bin from which the corrected share stays >= 0.5 (keep spells >= h)."""
    for i in range(len(pi)):
        if (pi[i:] >= 0.5).all():
            return float(EDGES[i])
    return float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--events", required=True)
    ap.add_argument("--master", required=True)
    ap.add_argument("--tol", type=float, default=0.2)
    ap.add_argument("--all-matches", action="store_true", help="also the two Sportec test matches")
    ap.add_argument("--boot", type=int, default=500)
    args = ap.parse_args(argv)
    train = set(json.load(open(os.path.join(args.master, "split.json")))["train"])
    npzs = [p for p in sorted(glob.glob(os.path.join(args.master, "matches", "sportec_*.npz")))
            if args.all_matches or os.path.basename(p) in train]
    ms = [one_match(p, args.events, round(args.tol * FPS)) for p in npzs]
    c = curve(ms)
    print(f"{len(ms)} Sportec matches; false-event floor eps {c['eps']:.3f}; "
          f"certainly controlled reference spells {sum(m['nref'] for m in ms)}")
    print("length (s)       " + " ".join(f"{a:.2f}-{b:.2f}" for a, b in zip(EDGES[:-1], EDGES[1:])))
    for name in ("n", "obs", "f", "pi"):
        label = {"n": "candidates", "obs": "with an event", "f": "controlled ref", "pi": "corrected share"}[name]
        fmt = "{:9d}" if name == "n" else "{:9.2f}"
        print(f"{label:<16} " + " ".join(fmt.format(x) for x in c[name]))
    rng = np.random.default_rng(0)
    boot = [crossing(curve([ms[i] for i in rng.integers(0, len(ms), len(ms))])["pi"]) for _ in range(args.boot)]
    print(f"\ncorrected share stays >= 0.5 from {crossing(c['pi']):.2f} s "
          f"(MIN_HOLD {round(crossing(c['pi']) * FPS)} frames); bootstrap over matches 5-95 %: "
          f"{np.nanpercentile(boot, 5):.2f}-{np.nanpercentile(boot, 95):.2f} s")
    print("leave-one-match-out:", " ".join(f"{crossing(curve([q for q in ms if q is not m])['pi']):.2f}" for m in ms))


if __name__ == "__main__":
    main()
