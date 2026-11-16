"""Re-derive labels.MIN_HOLD from open data only: Sportec's per-frame flag plus its DFL event file."""

import argparse
import datetime as dt
import glob
import os
import re

import numpy as np

from evaluation import possession_metrics as pm
from lyari.dataset import labels as L

FPS = 25
SECTIONS = {"firstHalf": 1, "secondHalf": 2}
CONTROL_DUEL = {"ballClaimed", "ballControlRetained", "dribbledAround"}
SET_PIECES = ("KickOff", "ThrowIn", "FreeKick", "CornerKick", "GoalKick", "Penalty")
HOLDS = np.arange(0, 151)  # frames, 0 to 6 s

_EVENT_RE = re.compile(r'<Event ([^>]*)>(.*?)</Event>', re.S)
_TAG_RE = re.compile(r'<(\w+)((?:\s+[\w-]+="[^"]*")*)\s*/?>')
_ATTR_RE = re.compile(r'([\w-]+)="([^"]*)"')


def parse_time(s):
    return dt.datetime.fromisoformat(s).timestamp()


def read_events(path, home_id, duel_loser=True):
    """[(time_s, section or None, team 0/1, kind)] with kind 'control', 'touch' or 'kickoff'."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    out = []
    section = None
    for head, body in _EVENT_RE.findall(text):
        t = parse_time(dict(_ATTR_RE.findall(head))["EventTime"])
        for tag, attrs in _TAG_RE.findall(body):
            a = dict(_ATTR_RE.findall(attrs))
            if "GameSection" in a:
                section = SECTIONS.get(a["GameSection"], section)
            if tag == "KickOff" and "GameSection" in a:
                out.append((t, SECTIONS.get(a["GameSection"]), -1, "kickoff"))
            team_id = a.get("Team")
            if tag in ("Play", "ShotAtGoal", "BallClaiming") and team_id:
                out.append((t, section, int(team_id != home_id), "pass" if tag == "Play" else "control"))
            elif tag == "TacklingGame":
                won = a.get("WinnerResult") in CONTROL_DUEL or a.get("WinnerRole") == "withBallControl"
                out.append((t, section, int(a["WinnerTeam"] != home_id), "control" if won else "touch"))
                if duel_loser and a.get("LoserRole") == "withBallControl":
                    out.append((t, section, int(a["LoserTeam"] != home_id), "control"))
            elif tag in ("OtherBallAction", "BallDeflection") and team_id:
                out.append((t, section, int(team_id != home_id), "touch"))
    return out


def sync(events, flag, period, kickoff_t, per, shifts=np.arange(-250, 251)):
    """Frame offset for one half: kick-off time + shift that best matches pass teams to the flag."""
    idx = np.flatnonzero(period == per)
    first, n = idx[0], len(idx)
    passes = [(t, team) for t, s, team, k in events if s == per and k == "pass"]
    base = np.array([round((t - kickoff_t) * FPS) for t, _ in passes])
    team = np.array([tm for _, tm in passes])
    best = (-1.0, 0)
    for d in shifts:
        f = base + d
        ok = (f >= 0) & (f < n)
        lab = flag[first + f[ok]]
        live = lab != L.D
        agree = float((lab[live] == team[ok][live]).mean()) if live.any() else 0.0
        if agree > best[0]:
            best = (agree, int(d))
    return first, n, best[1], best[0]


def event_frames(events, period, kickoffs, offsets):
    """{kind: [(frame, team)]} in the npz's frame index."""
    out = {}
    for t, s, team, k in events:
        if s not in offsets or k == "kickoff":
            continue
        first, n, d = offsets[s]
        f = round((t - kickoffs[s]) * FPS) + d
        if 0 <= f < n:
            out.setdefault(k, []).append((first + f, team))
    return out


def candidates(flag):
    """Raw spells harmonise could revert, one round, as (start, end, team)."""
    sp = L.spells(flag)
    alive = [i for i, (t, _, _) in enumerate(sp) if t in (L.H, L.A)]
    out = []
    for k, i in enumerate(alive):
        t, s, e = sp[i]
        if i > 0 and sp[i - 1][0] == L.D:
            continue
        prev = sp[alive[k - 1]][0] if k > 0 else None
        nxt = sp[alive[k + 1]][0] if k + 1 < len(alive) else None
        after_dead = i + 1 < len(sp) and sp[i + 1][0] == L.D
        if prev is not None and prev != t and (nxt == prev or after_dead or nxt is None):
            out.append((s, e, t))
    return out


def label_candidates(cands, ev, tol):
    """Boolean per candidate: a controlled event of its team in [start - tol, end + tol)."""
    ctrl = sorted(ev.get("pass", []) + ev.get("control", []))
    fr = np.array([f for f, _ in ctrl])
    tm = np.array([t for _, t in ctrl])
    out = np.zeros(len(cands), bool)
    for j, (s, e, t) in enumerate(cands):
        lo, hi = np.searchsorted(fr, s - tol), np.searchsorted(fr, e + tol)
        out[j] = bool((tm[lo:hi] == t).any())
    return out


def event_label(flag, ev):
    """Per-frame team of the last controlled event; D where the flag is D; restarts start fresh."""
    out = flag.copy()
    ctrl = sorted(ev.get("pass", []) + ev.get("control", []))
    marks = np.full(len(flag), -1, np.int8)
    for f, t in ctrl:
        marks[f] = t
    cur = -1
    for i in range(len(flag)):
        if flag[i] == L.D:
            cur = -1
            continue
        if marks[i] >= 0:
            cur = marks[i]
        out[i] = cur if cur >= 0 else flag[i]
    return out


def balanced_acc(dur, ctrl, h):
    keep = dur >= h
    tpr = keep[ctrl].mean() if ctrl.any() else np.nan
    tnr = (~keep[~ctrl]).mean() if (~ctrl).any() else np.nan
    return 0.5 * (tpr + tnr)


def f1_keep(dur, ctrl, h):
    keep = dur >= h
    tp = (keep & ctrl).sum()
    return 2 * tp / max(keep.sum() + ctrl.sum(), 1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--events", required=True, help="folder with DFL_02_01 and DFL_03_02 xml files")
    ap.add_argument("--master", required=True, help="master folder with matches/sportec_*.npz")
    ap.add_argument("--tol", type=float, default=0.2, help="event window slack in seconds")
    ap.add_argument("--no-duel-loser", action="store_true",
                    help="do not count the duel loser's withBallControl as a controlled action")
    ap.add_argument("--no-check", action="store_true", help="skip the spellF1 check (slow)")
    args = ap.parse_args(argv)
    tol = round(args.tol * FPS)

    per_match = {}
    for npz in sorted(glob.glob(os.path.join(args.master, "matches", "sportec_*.npz"))):
        mid = re.search(r"(DFL-MAT-\w+)\.npz$", npz).group(1)
        with open(glob.glob(os.path.join(args.events, f"DFL_02_01_matchinformation_*_{mid}.xml"))[0],
                  encoding="utf-8") as f:
            info = f.read()
        home_id = re.search(r'HomeTeamId="([^"]*)"', info).group(1)
        events = read_events(glob.glob(os.path.join(args.events, f"DFL_03_02_events_raw_*_{mid}.xml"))[0],
                             home_id, duel_loser=not args.no_duel_loser)
        z = np.load(npz)
        flag, period = z["label_source"].astype(np.int8), z["period"]
        kickoffs = {s: t for t, s, _, k in events if k == "kickoff"}
        offsets, agree = {}, {}
        for per in (1, 2):
            first, n, d, a = sync(events, flag, period, kickoffs[per], per)
            offsets[per], agree[per] = (first, n, d), a
        ev = event_frames(events, period, kickoffs, offsets)
        cands = candidates(flag)
        dur = np.array([e - s for s, e, _ in cands])
        ctrl = label_candidates(cands, ev, tol)
        per_match[mid] = dict(dur=dur, ctrl=ctrl, flag=flag, period=period, ev=ev,
                              offsets={p: offsets[p][2] for p in offsets}, agree=agree)
        print(f"{mid}: sync shift {offsets[1][2]:+d}/{offsets[2][2]:+d} frames, pass-team agreement "
              f"{agree[1]:.3f}/{agree[2]:.3f}; {len(cands)} candidate spells, controlled {ctrl.mean():.2f}; "
              f"events pass {len(ev.get('pass', []))} control {len(ev.get('control', []))} "
              f"touch {len(ev.get('touch', []))}")

    dur = np.concatenate([m["dur"] for m in per_match.values()])
    ctrl = np.concatenate([m["ctrl"] for m in per_match.values()])
    print(f"\npooled: {len(dur)} candidate spells, {ctrl.sum()} controlled, {(~ctrl).sum()} touch-only")
    edges = [0, 5, 10, 15, 20, 25, 38, 50, 75, 100, 150, 10 ** 9]
    print("duration bin (s)   n    controlled share")
    for a, b in zip(edges, edges[1:]):
        m = (dur >= a) & (dur < b)
        if m.any():
            hi = "inf" if b > 10 ** 8 else f"{b / FPS:.2f}"
            print(f"  {a / FPS:5.2f}-{hi:>5}  {m.sum():5d}  {ctrl[m].mean():.3f}")

    ba = np.array([balanced_acc(dur, ctrl, h) for h in HOLDS])
    f1 = np.array([f1_keep(dur, ctrl, h) for h in HOLDS])
    hb, hf = int(HOLDS[np.nanargmax(ba)]), int(HOLDS[np.argmax(f1)])
    print(f"\nbalanced accuracy best h = {hb} frames ({hb / FPS:.2f} s), {ba.max():.3f}; "
          f"at 38: {ba[38]:.3f}; within 0.005 of best: {HOLDS[ba >= ba.max() - 0.005].min()}-"
          f"{HOLDS[ba >= ba.max() - 0.005].max()} frames")
    acc = np.array([((dur >= h) == ctrl).mean() for h in HOLDS])
    ha = int(HOLDS[np.argmax(acc)])
    print(f"accuracy best h = {ha} frames ({ha / FPS:.2f} s), {acc.max():.3f}; at 38: {acc[38]:.3f}")
    print(f"keep-F1 best h = {hf} frames ({hf / FPS:.2f} s), {f1.max():.3f}; at 38: {f1[38]:.3f}")

    print("\nper match best h (balanced accuracy) and leave-one-match-out h:")
    for mid, m in per_match.items():
        own = np.array([balanced_acc(m["dur"], m["ctrl"], h) for h in HOLDS])
        rest_d = np.concatenate([o["dur"] for k, o in per_match.items() if k != mid])
        rest_c = np.concatenate([o["ctrl"] for k, o in per_match.items() if k != mid])
        loo = int(HOLDS[np.nanargmax([balanced_acc(rest_d, rest_c, h) for h in HOLDS])])
        print(f"  {mid}: own {int(HOLDS[np.nanargmax(own)]):3d}  LOO {loo:3d}  "
              f"(own BA at LOO h {own[loo]:.3f}, at 38 {own[38]:.3f})")

    if args.no_check:
        return
    print("\ncheck: spellF1 of harmonise(flag, h) against the event label, mean over matches")
    for h in (0, 25, 38, 50, 55, 63, 75, 88, 94, 100, 125, 150):
        sc = []
        for m in per_match.values():
            gt, pr = [], []
            for p in (1, 2):
                idx = np.flatnonzero(m["period"] == p)
                ev = {k: [(f - idx[0], t) for f, t in v if m["period"][f] == p] for k, v in m["ev"].items()}
                gt += [event_label(m["flag"][idx], ev), [pm.X]]
                pr += [L.harmonise(m["flag"][idx], h), [pm.X]]
            g, q = np.concatenate(gt).astype(np.int8), np.concatenate(pr).astype(np.int8)
            sc.append(pm.spell_scores(g, q, pm.valid_runs(g))["mean_f1"])
        print(f"  h {h:3d} ({h / FPS:4.2f} s): {np.mean(sc):.4f}  [{min(sc):.3f}-{max(sc):.3f}]")


if __name__ == "__main__":
    main()
