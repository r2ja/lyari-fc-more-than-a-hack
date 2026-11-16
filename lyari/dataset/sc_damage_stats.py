"""Measure external tracking-loss statistics used in the practice simulator, from SkillCorner open data."""
import glob
import json
import os

import numpy as np
import pandas as pd

FPS = 10
ROOT = os.path.join(os.environ.get("LYARI_DATASETS_DIR") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "datasets"),
    "skillcorner-opendata", "data", "matches")


def runs(m):
    d = np.diff(np.r_[0, m.astype(np.int8), 0])
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def part1():
    agg = {k: [] for k in ["gap_s", "pgap_s", "track_s"]}
    tot = dict(frames=0, onair=0, has=0, det=0, live=0, live_det=0, dead=0, dead_det=0, slow_at=0, slow_lost=0, fast_at=0, fast_lost=0,
               pdet_sum=0.0, pframes=0, minutes=0.0)
    per_match = []
    for mdir in sorted(glob.glob(ROOT + "/*")):
        mid = os.path.basename(mdir)
        rows = []
        with open(glob.glob(mdir + "/*_tracking_extrapolated.jsonl")[0]) as f:
            for line in f:
                d = json.loads(line)
                if d.get("period") is None: continue
                b = d["ball_data"] or {}
                onair = d.get("image_corners_projection", {}).get("x_top_left") is not None
                pdet = [p.get("is_detected") for p in d.get("player_data") or []]
                rows.append((d["frame"], onair, b.get("x"), b.get("y"), b.get("is_detected"), sum(1 for x in pdet if x), len(pdet)))
        a = pd.DataFrame(rows, columns=["frame", "onair", "x", "y", "det", "npdet", "np"]).sort_values("frame")
        fr = a.frame.to_numpy(); n = len(a)
        ph = pd.read_csv(glob.glob(mdir + "/*_phases_of_play.csv")[0])
        live = np.zeros(n, bool)
        for s, e in zip(ph.frame_start, ph.frame_end):
            live |= (fr >= s) & (fr <= e)
        has = a.x.notna().to_numpy()
        det = (a.det == True).to_numpy()
        onair = a.onair.to_numpy()
        tot["frames"] += n; tot["onair"] += onair.sum(); tot["has"] += has.sum(); tot["det"] += det.sum()
        tot["live"] += live.sum(); tot["live_det"] += (live & det).sum(); tot["dead"] += (~live).sum(); tot["dead_det"] += (~live & det).sum()
        tot["minutes"] += n / FPS / 60
        s, e = runs(~det)
        agg["gap_s"] += list((e - s) / FPS)
        x, y = a.x.to_numpy(float), a.y.to_numpy(float)
        sp = np.full(n, np.nan); both = det[1:] & det[:-1]
        sp[1:][both] = np.hypot(np.diff(x), np.diff(y))[both] * FPS
        at = det[:-1]; lost = at & ~det[1:]
        for k, m in (("slow", sp[:-1] < 3), ("fast", sp[:-1] > 10)):
            tot[k + "_at"] += (at & m).sum(); tot[k + "_lost"] += (lost & m).sum()
        tot["pdet_sum"] += a.npdet.sum(); tot["pframes"] += n
        per_match.append((mid, det.mean(), (live & det).sum() / max(live.sum(), 1), (~live & det).sum() / max((~live).sum(), 1), live.mean()))

    g = np.array(agg["gap_s"])
    mins = tot["minutes"]
    print("matches", len(per_match), "minutes %.0f" % mins)
    print("ball detected share, all period frames %.3f | on-air frames share %.3f | ball xy present (incl. extrapolated) %.3f" % (tot["det"] / tot["frames"], tot["onair"] / tot["frames"], tot["has"] / tot["frames"]))
    print("ball detected in live (phases) %.3f | in dead %.3f | live share of frames %.3f" % (tot["live_det"] / tot["live"], tot["dead_det"] / tot["dead"], tot["live"] / tot["frames"]))
    print("missing runs: median %.2f s | p75 %.1f | p90 %.1f | share of missing frames in runs >10 s %.3f | runs 1-3 s per min %.3f | runs 0.2-3 s per min %.3f | runs >=0.2 s per min %.2f" % (
        np.median(g), np.quantile(g, .75), np.quantile(g, .9), g[g > 10].sum() / g.sum(), ((g >= 1) & (g <= 3)).sum() / mins, ((g >= 0.2) & (g <= 3)).sum() / mins, (g >= 0.2).sum() / mins))
    print("loss hazard slow(<3 m/s) %.4f /frame, fast(>10) %.4f /frame, ratio %.2f" % (tot["slow_lost"] / tot["slow_at"], tot["fast_lost"] / tot["fast_at"], (tot["slow_lost"] / tot["slow_at"]) / (tot["fast_lost"] / tot["fast_at"])))
    print("players detected per frame %.2f (broadcast view)" % (tot["pdet_sum"] / tot["pframes"]))
    pm = np.array([p[1:] for p in per_match])
    print("per-match ball detected: min %.3f max %.3f | live det min %.3f max %.3f | dead det min %.3f max %.3f" % (pm[:, 0].min(), pm[:, 0].max(), pm[:, 1].min(), pm[:, 1].max(), pm[:, 2].min(), pm[:, 2].max()))


def part2():
    live_gaps, live_min, slow_seen_after = [], 0.0, []
    goal = dict(at=0, lost=0); mid = dict(at=0, lost=0)
    st = dict(n=0, seen0=0, loss_t=[], reshow=0, reshow_long=0, n_long=0, refind=[], onair_dead=0, dead_frames=0)
    for mdir in sorted(glob.glob(ROOT + "/*")):
        rows = []
        with open(glob.glob(mdir + "/*_tracking_extrapolated.jsonl")[0]) as f:
            for line in f:
                d = json.loads(line)
                if d.get("period") is None: continue
                b = d["ball_data"] or {}
                rows.append((d["frame"], d["period"], b.get("x"), b.get("is_detected"), d.get("image_corners_projection", {}).get("x_top_left") is not None))
        a = pd.DataFrame(rows, columns=["frame", "period", "x", "det", "onair"]).sort_values("frame").reset_index(drop=True)
        fr, det, x, onair = a.frame.to_numpy(), (a.det == True).to_numpy(), a.x.to_numpy(float), a.onair.to_numpy()
        n = len(a)
        ph = pd.read_csv(glob.glob(mdir + "/*_phases_of_play.csv")[0]).sort_values("frame_start")
        live = np.zeros(n, bool)
        for s, e in zip(ph.frame_start, ph.frame_end):
            live |= (fr >= s) & (fr <= e)
        live_min += live.sum() / FPS / 60
        s, e = runs(~det)
        for i, j in zip(s, e):
            if i > 0 and j < n and live[i - 1:j + 1].all():
                live_gaps.append((j - i) / FPS)
        at = det[:-1] & live[:-1]; lost = at & ~det[1:]
        near = np.abs(np.nan_to_num(x[:-1])) > 52.5 - 5.5
        goal["at"] += (at & near).sum(); goal["lost"] += (lost & near).sum(); mid["at"] += (at & ~near).sum(); mid["lost"] += (lost & ~near).sum()
        ds, de = runs(~live)
        for i, j in zip(ds, de):
            if i == 0 or j >= n or (j - i) < 3 * FPS or a.period[i - 1] != a.period[min(j, n - 1)]:
                continue
            st["n"] += 1
            st["dead_frames"] += j - i; st["onair_dead"] += onair[i:j].sum()
            st["seen0"] += det[i:i + 5].any()
            if det[i]:
                k = np.flatnonzero(~det[i:j]); st["loss_t"].append(k[0] / FPS if len(k) else (j - i) / FPS)
            miss = np.flatnonzero(~det[i:j])
            if len(miss):
                later = det[i + miss[0]:j]
                re = later[-int(2 * FPS):].any()                 # seen again in the last 2 s before the restart
                st["reshow"] += re
                if j - i > 15 * FPS: st["n_long"] += 1; st["reshow_long"] += re
            if not det[j]:
                k = np.flatnonzero(det[j:j + 10 * FPS]); st["refind"].append(k[0] / FPS if len(k) else 10.0)
    g = np.array(live_gaps); g2 = g[g >= 0.2]
    lg = np.log(g2[g2 >= 1.0])
    lt = np.array(st["loss_t"]); rf = np.array(st["refind"])
    print("live minutes %.0f | live gaps >=0.2 s per live min %.2f | under 1 s per live min %.2f | gaps >=1 s: lognormal median %.2f s sigma %.2f | share >10 s %.3f" % (
        live_min, len(g2) / live_min, ((g2 < 1)).sum() / live_min, np.exp(lg.mean()), lg.std(), (g2 > 10).mean()))
    print("live loss hazard near goal line (5.5 m) / elsewhere: %.2f" % ((goal["lost"] / goal["at"]) / (mid["lost"] / mid["at"])))
    print("stoppages >=3 s: %d | ball seen in first 0.5 s %.3f | time to first loss median %.2f s (log sigma %.2f) | re-shown in last 2 s before restart %.3f (stoppages >15 s: %.3f) | after restart found within: median %.2f s | broadcast on-air during stoppages %.3f" % (
        st["n"], st["seen0"] / st["n"], np.median(lt), np.log(np.maximum(lt, 0.1)).std(), st["reshow"] / st["n"], st["reshow_long"] / max(st["n_long"], 1), np.median(rf) if len(rf) else np.nan, st["onair_dead"] / st["dead_frames"]))


def part3():
    out_runs, reapp, long_runs, mins, acc, n_live_gaps = [], [], 0, 0.0, [], 0
    for mdir in sorted(glob.glob(ROOT + "/*")):
        rows = []
        with open(glob.glob(mdir + "/*_tracking_extrapolated.jsonl")[0]) as f:
            for line in f:
                d = json.loads(line)
                if d.get("period") is None: continue
                b = d["ball_data"] or {}
                rows.append((d["frame"], b.get("x"), b.get("y"), b.get("is_detected")))
        a = pd.DataFrame(rows, columns=["frame", "x", "y", "det"]).sort_values("frame")
        det = (a.det == True).to_numpy(); x = a.x.to_numpy(float); y = a.y.to_numpy(float); n = len(a)
        mins += n / FPS / 60
        ph = pd.read_csv(glob.glob(mdir + "/*_phases_of_play.csv")[0]); fr = a.frame.to_numpy()
        live = np.zeros(n, bool)
        for s, e in zip(ph.frame_start, ph.frame_end): live |= (fr >= s) & (fr <= e)
        out = det & ((np.abs(np.nan_to_num(x)) > 52.5 + 0.5) | (np.abs(np.nan_to_num(y)) > 34 + 0.5))
        s, e = runs(out); out_runs += list((e - s) / FPS)
        s, e = runs(~det)
        long_runs += int(((e - s) / FPS > 60).sum())
        for i, j in zip(s, e):
            if i > 0 and j + 1 < n and live[i - 1:j + 2].all() and det[j] and det[j + 1]:
                reapp.append(np.hypot(x[j + 1] - x[j], y[j + 1] - y[j]) * FPS); n_live_gaps += 1
        v = np.full((n, 2), np.nan); both = det[1:] & det[:-1]
        v[1:][both] = np.c_[np.diff(x), np.diff(y)][both] * FPS
        ab = ~np.isnan(v[1:, 0]) & ~np.isnan(v[:-1, 0]) & live[1:]
        acc += list(np.hypot(*(v[1:][ab] - v[:-1][ab]).T) * FPS)
    o, r, ac = np.array(out_runs), np.array(reapp), np.array(acc)
    print("ball seen beyond a line: median run %.2f s, per min %.2f" % (np.median(o), len(o) / mins))
    print("live gap ends: %d, reappearance speed median %.1f m/s, share > 8 m/s %.3f" % (len(r), np.median(r), (r > 8).mean()))
    print("missing runs > 60 s per match %.2f (%d over %.0f min)" % (long_runs / (mins / 95), long_runs, mins))
    print("live ball acceleration at 10 fps: p50 %.1f p90 %.1f p99 %.1f m/s2" % tuple(np.quantile(ac, [.5, .9, .99])))

def part4():
    """v5be: v5b's stoppage-run constants (one missing run per stoppage, damage.py `_draw_run` / `_refind`).
    pA = share of stoppages (3 s+) with a missing run; lead = run start minus stoppage start (from 5 s before);
    lag = run end minus restart (negative when the ball is seen again before the restart); refind = in runs over
    25 s, a visible stretch of 1 s+ at least 3 s inside both ends (share, and lognormal median / sigma of its length)."""
    st = dict(n=0, with_run=0, lead=[], lag=[], n_long=0, refind=0, refind_len=[])
    pre = 5 * FPS
    for mdir in sorted(glob.glob(ROOT + "/*")):
        rows = []
        with open(glob.glob(mdir + "/*_tracking_extrapolated.jsonl")[0]) as f:
            for line in f:
                d = json.loads(line)
                if d.get("period") is None: continue
                rows.append((d["frame"], d["period"], (d["ball_data"] or {}).get("is_detected")))
        a = pd.DataFrame(rows, columns=["frame", "period", "det"]).sort_values("frame").reset_index(drop=True)
        fr, det, per = a.frame.to_numpy(), (a.det == True).to_numpy(), a.period.to_numpy()
        n = len(a)
        ph = pd.read_csv(glob.glob(mdir + "/*_phases_of_play.csv")[0]).sort_values("frame_start")
        live = np.zeros(n, bool)
        for s_, e_ in zip(ph.frame_start, ph.frame_end):
            live |= (fr >= s_) & (fr <= e_)
        ds, de = runs(~live)
        for i, j in zip(ds, de):
            if i < pre or j >= n or (j - i) < 3 * FPS or per[i - pre] != per[min(j, n - 1)]:
                continue
            st["n"] += 1
            miss = ~det[i - pre:j]
            k = np.flatnonzero(miss)
            if not len(k):
                continue
            st["with_run"] += 1
            first = k[0]
            run_start = i - pre + first
            st["lead"].append((run_start - i) / FPS)
            if not det[j]:
                f = np.flatnonzero(det[j:j + 10 * FPS])
                st["lag"].append(f[0] / FPS if len(f) else 10.0)
            else:
                m2 = np.flatnonzero(~det[run_start:j])
                st["lag"].append((run_start + m2[-1] + 1 - j) / FPS if len(m2) else 0.0)
            if j - run_start > 25 * FPS:
                st["n_long"] += 1
                inner = det[run_start + 3 * FPS:j - 3 * FPS]
                s2, e2 = runs(inner)
                L = (e2 - s2)[(e2 - s2) >= FPS]
                if len(L):
                    st["refind"] += 1
                    st["refind_len"].append(L.max() / FPS)
    ld, lg, rl = np.array(st["lead"]), np.array(st["lag"]), np.log(np.array(st["refind_len"]))
    print("stoppages >=3 s: %d | with a missing run (pA) %.3f | lead mean %.2f s sd %.2f (median %.2f) | lag mean %.2f s sd %.2f (median %.2f)" % (
        st["n"], st["with_run"] / st["n"], ld.mean(), ld.std(), np.median(ld), lg.mean(), lg.std(), np.median(lg)))
    print("runs over 25 s: %d | re-found inside (refind_p) %.3f | longest visible stretch: lognormal median %.2f s sigma %.2f" % (
        st["n_long"], st["refind"] / max(st["n_long"], 1), np.exp(rl.mean()) if len(rl) else np.nan, rl.std() if len(rl) else np.nan))


def _in_quad(px, py, quad, margin=2.0):
    """(P,) bool: points inside the convex camera footprint `quad` (4 corners in order), at least `margin` m inside."""
    inside = np.ones(len(px), bool)
    sign = None
    for i in range(4):
        x1, y1 = quad[i]
        x2, y2 = quad[(i + 1) % 4]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        length = np.hypot(x2 - x1, y2 - y1) or 1.0
        d = cross / length
        if sign is None:
            sign = 1.0 if np.nanmean(d) >= 0 else -1.0
        inside &= sign * d >= margin
    return inside


def part5():
    """v5bf: the blackout process (damage.py `_blackouts`) and the object cap, on SkillCorner. A blackout = 3 or more
    players inside the camera footprint (2 m in) undetected at once for 2 s or more (the simulator's minimum length);
    its size = the most undetected at once, so q = (size - 1) / 21 as in `_blackouts`. Rates per minute of live and of
    dead play (phases_of_play). Object cap: the most players detected at once."""
    ev, live_min, dead_min, max_det, onair_frames = [], 0.0, 0.0, 0, 0
    need = 2 * FPS
    for mdir in sorted(glob.glob(ROOT + "/*")):
        frames, miss, det_n, onair = [], [], [], []
        with open(glob.glob(mdir + "/*_tracking_extrapolated.jsonl")[0]) as f:
            for line in f:
                d = json.loads(line)
                if d.get("period") is None:
                    continue
                c = d.get("image_corners_projection") or {}
                pl = d.get("player_data") or []
                frames.append(d["frame"])
                det_n.append(sum(1 for q in pl if q.get("is_detected")))
                keys = ("top_left", "bottom_left", "bottom_right", "top_right")
                if not pl or any(c.get("x_" + k) is None for k in keys):
                    miss.append(0); onair.append(False)
                    continue
                quad = [(c["x_" + k], c["y_" + k]) for k in keys]
                px = np.array([q["x"] for q in pl], float); py = np.array([q["y"] for q in pl], float)
                dd = np.array([bool(q.get("is_detected")) for q in pl])
                inn = _in_quad(px, py, quad)
                miss.append(int((inn & ~dd).sum())); onair.append(True)
        fr = np.array(frames); miss = np.array(miss); onair = np.array(onair); n = len(fr)
        max_det = max(max_det, int(max(det_n)))
        onair_frames += int(onair.sum())
        ph = pd.read_csv(glob.glob(mdir + "/*_phases_of_play.csv")[0]).sort_values("frame_start")
        live = np.zeros(n, bool)
        for s_, e_ in zip(ph.frame_start, ph.frame_end):
            live |= (fr >= s_) & (fr <= e_)
        live_min += (live & onair).sum() / FPS / 60
        dead_min += (~live & onair).sum() / FPS / 60
        a, b = runs(onair & (miss >= 3))
        for i, j in zip(a, b):
            if j - i >= need:
                ev.append(((j - i) / FPS, int(miss[i:j].max()), bool(live[i])))
    ev = np.array(ev, dtype=object)
    L = np.array([e[0] for e in ev], float); K = np.array([e[1] for e in ev], float); lv = np.array([e[2] for e in ev], bool)
    q = (K - 1) / 21
    lg = np.log(np.clip(L, 2.0, 60.0))
    print("on-air minutes live %.0f dead %.0f | blackouts (3+ in view undetected, 2 s+): %d | per live min %.3f | per dead min %.3f | dead/live %.2f" % (
        live_min, dead_min, len(L), lv.sum() / live_min, (~lv).sum() / dead_min, ((~lv).sum() / dead_min) / (lv.sum() / live_min)))
    print("blackout length: lognormal median %.2f s sigma %.2f | size median %d, q p10 %.3f p90 %.3f | most players detected at once %d" % (
        np.exp(lg.mean()), lg.std(), int(np.median(K)), np.quantile(q, .1), np.quantile(q, .9), max_det))


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--v5bf"]:
        print("== v5bf: blackouts and the object cap")
        part5()
        raise SystemExit
    if sys.argv[1:] == ["--v5be"]:
        print("== v5be: v5b's stoppage-run constants")
        part4()
        raise SystemExit
    print("== totals, live / dead, gap lengths, speed hazard, players")
    part1()
    print("== live gaps and the stoppage ball process")
    part2()
    print("== beyond a line, reappearance speed, long outages, acceleration")
    part3()
