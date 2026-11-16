"""Import the three external providers into a common 25 Hz pitch coordinate system."""
import bz2
import glob
import json
import os
import re
import zipfile
import numpy as np
from lyari.dataset.damage import MISSING

FPS = 25
PITCH = (105.0, 68.0)
LABEL = {"H": 0, "A": 1, "D": 2}
_FRAME_RE = re.compile(rb'<Frame N="(\d+)"[^>]*? X="([-\d.]+)" Y="([-\d.]+)"')
_BALL_RE = re.compile(rb'<Frame N="(\d+)"[^>]*? X="([-\d.]+)" Y="([-\d.]+)" Z="([-\d.]+)"[^>]*?'
                      rb'BallPossession="(\d)" BallStatus="(\d)"')
_FS_RE = re.compile(rb'<FrameSet GameSection="(\w+)" MatchId="[^"]*" TeamId="([^"]*)" PersonId="([^"]*)">')
_ST_FRAME_RE = re.compile(r'matchTime="(-?\d+)" frameNumber="(\d+)" eventPeriod="(\w+)" ballStatus="(\w+)"')
_ST_LOC_RE = re.compile(r'playerId="([^"]+)" loc="\[([^,\]]+), ([^\]]+)\]"')
ST_STATUS = {"HOME": 0, "AWAY": 1, "BALLOUT": 2, "NEUTRAL": -1}
ST_PERIOD = {"FIRST_HALF": 1, "SECOND_HALF": 2}
ST_SMOOTH_SIGMA = 3.0


class Match:
    """Clean match at the source frame rate, before packing.
    px, py: (N, P) float32 metres, NaN when the player is not on the pitch / not tracked.
    seen:   (N, P) bool, True when the provider actually observed the player (else extrapolated).
    team:   (P,) 0 home / 1 away.  bx, by, bz: (N,) metres NaN when untracked.
    label:  (N,) int8 0 H, 1 A, 2 D, -1 unknown.  period: (N,) int8.  t: (N,) seconds in period."""

    def __init__(self, source, match_id, home, away, fps):
        self.source, self.match_id, self.home, self.away, self.fps = source, match_id, home, away, fps
        self.notes = {}


def fill_short_gaps(label, max_len, fill_with_prev=True):
    """Loose-ball rule: runs of unknown (-1) shorter than max_len take the previous label."""
    out = label.copy()
    n = len(out)
    i = 0
    while i < n:
        if out[i] != -1:
            i += 1
            continue
        j = i
        while j < n and out[j] == -1:
            j += 1
        if j - i <= max_len and i > 0 and fill_with_prev:
            out[i:j] = out[i - 1]
        i = j
    return out


def resample(m):
    """Resample a Match to 25 fps per period with linear interpolation of positions and
    nearest-neighbour labels / flags. Returns (period, t, px, py, seen, bx, by, bz, label)."""
    if abs(m.fps - FPS) < 1e-6:
        return m.period, m.t, m.px, m.py, m.seen, m.bx, m.by, m.bz, m.label
    outs = []
    for per in np.unique(m.period):
        idx = np.flatnonzero(m.period == per)
        t = m.t[idx]
        grid = np.arange(t[0], t[-1] + 1e-9, 1.0 / FPS)
        nn = np.clip(np.searchsorted(t, grid), 0, len(t) - 1)
        prev = np.clip(nn - 1, 0, len(t) - 1)
        choose = np.where(np.abs(t[nn] - grid) <= np.abs(t[prev] - grid), nn, prev)

        def interp(a):
            a = a[idx]
            out = np.empty((len(grid), a.shape[1]) if a.ndim == 2 else len(grid), np.float32)
            cols = range(a.shape[1]) if a.ndim == 2 else [None]
            for c in cols:
                col = a[:, c] if c is not None else a
                ok = ~np.isnan(col)
                if ok.sum() < 2:
                    v = np.full(len(grid), np.nan, np.float32)
                else:
                    v = np.interp(grid, t[ok], col[ok]).astype(np.float32)
                    # keep gaps: a grid point whose nearest source frame is missing stays missing
                    v[~ok[choose]] = np.nan
                if c is not None:
                    out[:, c] = v
                else:
                    out[:] = v
            return out

        outs.append((np.full(len(grid), per, np.int8), grid.astype(np.float32),
                     interp(m.px), interp(m.py), m.seen[idx][choose],
                     interp(m.bx), interp(m.by), interp(m.bz), m.label[idx][choose]))
    return tuple(np.concatenate([o[k] for o in outs]) for k in range(9))


def orient(px, py, bx, by, team, period, t):
    """Normalise so the home team attacks +x in every period: decided per period from the mean x
    of each team over its first five minutes."""
    px, py, bx, by = px.copy(), py.copy(), bx.copy(), by.copy()
    for per in np.unique(period):
        sel = (period == per) & (t < t[period == per].min() + 300)
        hx = np.nanmean(px[sel][:, team == 0])
        ax = np.nanmean(px[sel][:, team == 1])
        if hx > ax:  # home is on the right, so attacks left: mirror
            rows = period == per
            px[rows] *= -1
            py[rows] *= -1
            bx[rows] *= -1
            by[rows] *= -1
    return px, py, bx, by


def assign_slots(px, team):
    """Map P players to 22 slots (0-10 home, 11-21 away): a substitute takes the slot of a player
    whose last frame is before his first. Returns (P,) slot index or -1."""
    slot = np.full(px.shape[1], -1, np.int16)
    for tm, base in ((0, 0), (1, 11)):
        players = np.flatnonzero(team == tm)
        first = np.array([np.flatnonzero(~np.isnan(px[:, p]))[:1].tolist() or [np.inf] for p in players]).ravel()
        last = np.array([np.flatnonzero(~np.isnan(px[:, p]))[-1:].tolist() or [-1] for p in players]).ravel()
        busy = np.full(11, -1, float)
        for k in np.argsort(first, kind="stable"):
            if not np.isfinite(first[k]):
                continue
            free = np.flatnonzero(busy < first[k])
            if free.size == 0:
                continue  # more than 11 concurrent players: dropped
            s = int(free[0])
            busy[s] = last[k]
            slot[players[k]] = base + s
    return slot


def pack(m, seed, fps_note=""):
    """Match -> dict of arrays for np.savez_compressed, plus stats."""
    period, t, px, py, seen, bx, by, bz, label = resample(m)
    px, py, bx, by = orient(px, py, bx, by, m.team, period, t)
    slot = assign_slots(px, m.team)
    n = len(period)
    xy = np.full((n, 22, 2), np.nan, np.float32)
    seen22 = np.zeros((n, 22), bool)
    for p, s in enumerate(slot):
        if s < 0:
            continue
        ok = ~np.isnan(px[:, p])  # a substitute shares his slot with the player he replaced
        xy[ok, s, 0], xy[ok, s, 1] = px[ok, p], py[ok, p]
        seen22[ok, s] = seen[ok, p]

    def to_cm(a):
        out = np.round(a * 100.0)
        return np.where(np.isnan(out), MISSING, np.clip(out, -32000, 32000)).astype(np.int16)

    clean = dict(
        xy=to_cm(xy), vis=~np.isnan(xy[..., 0]),
        ball=to_cm(np.stack([bx, by, np.nan_to_num(bz)], -1)), ball_vis=~np.isnan(bx),
        label=label.astype(np.int8), period=period.astype(np.int8),
    )
    counts = {k: int((label == v).sum()) for k, v in LABEL.items()}
    counts["unknown"] = int((label == -1).sum())
    meta = dict(
        source=m.source, match_id=m.match_id, home=m.home, away=m.away, fps=FPS, pitch=PITCH,
        n_frames=int(n), periods=[int(x) for x in np.unique(period)], source_fps=m.fps,
        orientation="home attacks +x in every period (clean); damaged copy mirrors period 2",
        slots="clean: 0-10 home, 11-21 away; damaged: 0-22 team block 1, 23-45 team block 2",
        labels={"0": "H", "1": "A", "2": "D", "-1": "unknown"}, label_counts=counts,
        players_dropped=int((slot < 0).sum()), damage_seed=int(seed), notes=m.notes, resample=fps_note or None,
    )
    arrays = dict(
        meta=np.array(json.dumps(meta)), frame=np.arange(n, dtype=np.int32), period=clean["period"],
        t=t.astype(np.float32), label=clean["label"],
        clean_xy=clean["xy"], clean_vis=clean["vis"], clean_seen=seen22,
        clean_team=np.array([0] * 11 + [1] * 11, np.int8),
        clean_ball=clean["ball"], clean_ball_vis=clean["ball_vis"],
    )
    return arrays, dict(meta=meta)


def load_sportec(ddir, match_id):
    info = open(glob.glob(f"{ddir}/sportec-idsse/DFL_02_01_matchinformation_*_{match_id}.xml")[0], encoding="utf-8").read()
    gen = re.search(r"<General [^>]*>", info).group(0)
    attr = lambda s, k: re.search(k + r'="([^"]*)"', s).group(1)
    home_id, away_id = attr(gen, "HomeTeamId"), attr(gen, "GuestTeamId")
    m = Match("sportec", match_id, attr(gen, "HomeTeamName"), attr(gen, "GuestTeamName"), 25.0)
    team_of = {}
    for tm in re.finditer(r"<Team [^>]*>.*?</Team>", info, re.S):
        tid = attr(tm.group(0), "TeamId")
        for pl in re.finditer(r"<Player [^>]*/>", tm.group(0)):
            team_of[attr(pl.group(0), "PersonId")] = 0 if tid == home_id else 1
    raw = open(glob.glob(f"{ddir}/sportec-idsse/DFL_04_03_positions_raw_observed_*_{match_id}.xml")[0], "rb").read()
    sections = {"firstHalf": 1, "secondHalf": 2}
    per_player = {}  # (period, pid) -> (N, X, Y)
    ball = {}
    pos = 0
    while True:
        h = _FS_RE.search(raw, pos)
        if not h:
            break
        end = raw.find(b"</FrameSet>", h.end())
        block = raw[h.end():end]
        pos = end
        sec, tid, pid = h.group(1).decode(), h.group(2).decode(), h.group(3).decode()
        if sec not in sections:
            continue
        if tid == "BALL":
            ball[sections[sec]] = np.array(_BALL_RE.findall(block))
        elif pid in team_of:
            per_player[(sections[sec], pid)] = np.array(_FRAME_RE.findall(block))
    del raw
    players = sorted({pid for _, pid in per_player})
    team = np.array([team_of[p] for p in players], np.int8)
    parts = []
    for per in (1, 2):
        if per not in ball:
            continue
        b = ball[per]
        if b.size == 0:
            continue
        N = b[:, 0].astype(np.int64)
        n0, n1 = N.min(), N.max()
        n = int(n1 - n0 + 1)
        bx = np.full(n, np.nan, np.float32); by = bx.copy(); bz = bx.copy()
        poss = np.full(n, 0, np.int8); status = np.full(n, 0, np.int8)
        k = N - n0
        bx[k], by[k], bz[k] = b[:, 1].astype(np.float32), b[:, 2].astype(np.float32), b[:, 3].astype(np.float32)
        poss[k], status[k] = b[:, 4].astype(np.int8), b[:, 5].astype(np.int8)
        present = np.zeros(n, bool); present[k] = True
        label = np.where(~present, -1, np.where(status == 0, 2, np.where(poss == 1, 0, 1))).astype(np.int8)
        px = np.full((n, len(players)), np.nan, np.float32); py = px.copy()
        for j, pid in enumerate(players):
            a = per_player.get((per, pid))
            if a is None or a.size == 0:
                continue
            kk = a[:, 0].astype(np.int64) - n0
            ok = (kk >= 0) & (kk < n)
            px[kk[ok], j] = a[ok, 1].astype(np.float32)
            py[kk[ok], j] = a[ok, 2].astype(np.float32)
        parts.append((np.full(n, per, np.int8), (np.arange(n) / 25.0).astype(np.float32), px, py, bx, by, bz, label))
    m.period, m.t, m.px, m.py, m.bx, m.by, m.bz, m.label = (np.concatenate([p[i] for p in parts]) for i in range(8))
    m.seen = ~np.isnan(m.px)
    m.team = team
    m.notes = dict(label_source="per-frame BallStatus / BallPossession flags")
    return m


def _metrica_labels_from_events(events, n, frame_of):
    """events: list of dicts {team: 0/1, type, subtype, start, end} on the global frame counter.
    Rules: pass/carry/recovery/set piece/shot/challenge-won -> that team from start frame;
    ball lost keeps the loser's team until the opponent controls it (loose-ball rule);
    ball out / fault received -> D until the next set piece; before the first event -> D."""
    label = np.full(n, -1, np.int8)
    ev = sorted(events, key=lambda e: e["start"])
    cur = 2
    marks = []
    for e in ev:
        tp, st = e["type"], (e.get("subtype") or "")
        goal = tp == "SHOT" and "GOAL" in st and "OWN" not in st and "SAVED" not in st and "OFF" not in st
        if tp == "BALL OUT" or goal:
            marks.append((e["end"], 2))
        elif tp == "FAULT RECEIVED":
            marks.append((e["start"], 2))
        elif tp in ("PASS", "CARRY", "RECOVERY", "SET PIECE", "SHOT"):
            marks.append((e["start"], e["team"]))
        elif tp == "CHALLENGE" and "WON" in st:
            marks.append((e["start"], e["team"]))
        elif tp == "BALL LOST":
            marks.append((e["start"], e["team"]))
    marks.sort()
    prev = 0
    for f, lab in marks:
        f = int(frame_of(f))
        label[prev:f] = cur
        cur, prev = lab, max(f, 0)
    label[prev:] = cur
    return label


def load_metrica_csv(ddir, game):
    import pandas as pd
    base = f"{ddir}/metrica-sample-data/data/Sample_Game_{game}/Sample_Game_{game}_"
    frames = {}
    cols_all = []
    for side, tm in (("Home", 0), ("Away", 1)):
        df = pd.read_csv(base + f"RawTrackingData_{side}_Team.csv", skiprows=2)
        names = [c for c in df.columns[3:-2] if not c.startswith("Unnamed")]
        for nm in names:
            i = list(df.columns).index(nm)
            cols_all.append((tm, nm, df.iloc[:, i].to_numpy(np.float32), df.iloc[:, i + 1].to_numpy(np.float32)))
        frames[side] = df
    df = frames["Home"]
    n = len(df)
    m = Match("metrica", f"game{game}", "Home", "Away", 25.0)
    m.period = df["Period"].to_numpy(np.int8)
    m.t = df["Time [s]"].to_numpy(np.float32)
    m.t = m.t - np.array([m.t[m.period == p].min() for p in m.period])
    m.px = np.stack([(c[2] - 0.5) * PITCH[0] for c in cols_all], 1)
    m.py = np.stack([-(c[3] - 0.5) * PITCH[1] for c in cols_all], 1)
    m.team = np.array([c[0] for c in cols_all], np.int8)
    m.seen = ~np.isnan(m.px)
    m.bx = (df["Ball"].to_numpy(np.float32) - 0.5) * PITCH[0]
    m.by = -(df.iloc[:, list(df.columns).index("Ball") + 1].to_numpy(np.float32) - 0.5) * PITCH[1]
    m.bz = np.full(n, np.nan, np.float32)
    ev = pd.read_csv(base + "RawEventsData.csv")
    events = [dict(team=0 if r["Team"] == "Home" else 1, type=r["Type"], subtype=str(r["Subtype"]),
                   start=int(r["Start Frame"]), end=int(r["End Frame"]) if r["End Frame"] else int(r["Start Frame"]))
              for _, r in ev.iterrows()]
    m.label = _metrica_labels_from_events(events, n, lambda f: f - 1)
    m.notes = dict(label_source="derived from the event file (pass/carry chains, ball out -> D)")
    return m


def smooth_nan(a, period, sigma=ST_SMOOTH_SIGMA):
    """Gaussian smoothing along time, per period, ignoring NaNs (normalised convolution); frames that
    were NaN stay NaN. SoccerTrack positions are rounded to 0.01 of the pitch (1.05 m in x, 0.68 m in
    y); unsmoothed, a 0.16 s velocity would carry ~3 m/s of rounding noise."""
    r = int(3 * sigma)
    w = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    out = np.full_like(a, np.nan)
    for per in np.unique(period):
        idx = np.flatnonzero(period == per)
        x = a[idx].astype(np.float64)
        ok = ~np.isnan(x)
        v = np.where(ok, x, 0.0)
        num = np.zeros_like(v)
        den = np.zeros_like(v)
        n = len(idx)
        for k, wk in zip(range(-r, r + 1), w):
            src = slice(max(0, k), n + min(0, k))
            dst = slice(max(0, -k), n - max(0, k))
            num[dst] += wk * v[src]
            den[dst] += wk * ok[src]
        out[idx] = np.where(ok, num / np.maximum(den, 1e-12), np.nan)
    return out.astype(a.dtype)


def st_labels(status):
    """ballStatus -> 0 H / 1 A / 2 D. NEUTRAL (ball live, nobody in control) keeps the team that had
    it, the competition's control rule; a NEUTRAL run right after a dead ball takes the next team."""
    lab = np.array([ST_STATUS[s] for s in status], np.int8)
    out = lab.copy()
    for order in (range(len(lab)), range(len(lab) - 1, -1, -1)):   # previous team first, then next team
        team = -1
        for i in order:
            if lab[i] == 2:
                team = -1
            elif lab[i] >= 0:
                team = lab[i]
            elif out[i] == -1 and team >= 0:
                out[i] = team
    out[out == -1] = 2   # NEUTRAL between two dead balls: nobody had it
    return out


def load_soccertrack(ddir, match_id):
    d = f"{ddir}/soccertrack-v2/raw/{match_id}"
    md = open(f"{d}/{match_id}_tracker_box_metadata.xml", encoding="utf-8").read()
    attr = lambda s, k: re.search(k + r'="([^"]*)"', s).group(1)
    pitch = re.search(r"<pitch [^>]*>", md).group(0)
    W, H = float(attr(pitch, "width")), float(attr(pitch, "height"))
    teams = re.findall(r"<team id=[^>]*>", md)
    side = {attr(t, "id"): attr(t, "side") for t in teams}
    name = {attr(t, "side"): attr(t, "nameEn") or attr(t, "name") for t in teams}
    pid_team = {attr(p, "id"): (0 if side[attr(p, "teamId")] == "home" else 1) for p in re.findall(r"<player id=[^>]*>", md)}
    pids = sorted(pid_team, key=lambda p: (pid_team[p], p))
    col = {p: i for i, p in enumerate(pids)}
    P = len(pids)
    m = Match("soccertrack", str(match_id), name.get("home", "home"), name.get("away", "away"), 25.0)
    per, t, status, px, py, bx, by = [], [], [], [], [], [], []
    x = y = None
    keep = False
    with open(f"{d}/{match_id}_tracker_box_data.xml", encoding="utf-8") as f:
        for line in f:
            if "<frame " in line:
                mt, _, ep, bs = _ST_FRAME_RE.search(line).groups()
                keep = ep in ST_PERIOD
                if keep:
                    per.append(ST_PERIOD[ep]); t.append(int(mt) / 1000.0); status.append(bs)
                    x = np.full(P, np.nan, np.float32); y = x.copy()
                    bx.append(np.nan); by.append(np.nan)
            elif keep and ("<player " in line or "<ball " in line):
                mm = _ST_LOC_RE.search(line)
                if not mm:
                    continue
                pid, lx, ly = mm.groups()
                try:
                    fx, fy = (float(lx) - 0.5) * W, (0.5 - float(ly)) * H
                except ValueError:
                    continue
                if "<ball " in line:
                    bx[-1], by[-1] = fx, fy
                elif pid in col:
                    x[col[pid]], y[col[pid]] = fx, fy
            elif keep and "</frame>" in line:
                px.append(x); py.append(y)
    m.period = np.array(per, np.int8)
    tt = np.array(t, np.float32)
    m.t = tt - np.array([tt[m.period == p].min() for p in m.period], np.float32)
    m.px = smooth_nan(np.stack(px), m.period)
    m.py = smooth_nan(np.stack(py), m.period)
    m.seen = ~np.isnan(m.px)
    m.bx = smooth_nan(np.array(bx, np.float32)[:, None], m.period)[:, 0]
    m.by = smooth_nan(np.array(by, np.float32)[:, None], m.period)[:, 0]
    m.bz = np.full(len(per), np.nan, np.float32)
    m.team = np.array([pid_team[p] for p in pids], np.int8)
    m.label = st_labels(status)
    m.notes = dict(label_source="per-frame ballStatus (HOME / AWAY / BALLOUT = D); NEUTRAL keeps the team in control "
                                "(the next team when it follows a dead ball)",
                   positions="loc is a fraction of the pitch rounded to 0.01; x = (loc_x - 0.5) * width, "
                             "y = (0.5 - loc_y) * height; Gaussian-smoothed in time, sigma 3 frames, to undo the rounding",
                   licence="SoccerTrack v2, CC BY 4.0 (atomscott/soccertrack-v2)",
                   status_counts={k: int(sum(s == k for s in status)) for k in ST_STATUS})
    return m
