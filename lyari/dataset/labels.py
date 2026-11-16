"""One possession rule for every source: the competition's."""

import numpy as np

MIN_HOLD = 38  # frames at 25 fps = 1.5 s
H, A, D = 0, 1, 2


def spells(label):
    """[(label, start, end_exclusive)] over a label array."""
    change = np.flatnonzero(np.diff(label) != 0)
    starts = np.r_[0, change + 1]
    ends = np.r_[change + 1, len(label)]
    return [(int(label[s]), int(s), int(e)) for s, e in zip(starts, ends)]


def harmonise(label, min_hold=MIN_HOLD, max_rounds=10):
    """Per-frame labels (0 H, 1 A, 2 D, -1 unknown) -> labels under the competition's control rule."""
    out = np.asarray(label).astype(np.int8).copy()
    for _ in range(max_rounds):
        sp = spells(out)
        alive = [i for i, (t, _, _) in enumerate(sp) if t in (H, A)]
        changed = False
        for k, i in enumerate(alive):
            t, s, e = sp[i]
            if e - s >= min_hold:
                continue
            if i > 0 and sp[i - 1][0] == D:
                continue                                  # restart: the owner is right
            prev = sp[alive[k - 1]][0] if k > 0 else None
            nxt = sp[alive[k + 1]][0] if k + 1 < len(alive) else None
            after_dead = i + 1 < len(sp) and sp[i + 1][0] == D
            if prev is not None and prev != t and (nxt == prev or after_dead or nxt is None):
                out[s:e] = prev
                changed = True
        if not changed:
            break
    return out


STOPPAGE_GAP = 200   # frames (8 s): A4's longest interpolated ball gap; longer gaps are stoppage evidence


def stoppage_gaps(label, ball_vis, min_frames=STOPPAGE_GAP):
    """Metrica: labels come from events, which miss some stoppages, while Metrica's tracking drops
    the ball whenever it is out of play. Live frames inside a run of >= min_frames with no ball in
    the source tracking become D. Measured on all three Metrica games (18 Oct): inside such runs the
    players move at 1.0-1.25 m/s, like labelled dead ball (1.0-1.1), against 2.2 m/s in live play;
    shorter gaps move like live play. Uses the tracking only, never a possession label."""
    out = np.asarray(label).astype(np.int8).copy()
    gap = ~np.asarray(ball_vis, bool)
    edges = np.flatnonzero(np.diff(np.r_[0, gap.astype(np.int8), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a >= min_frames:
            seg = out[a:b]
            seg[(seg == H) | (seg == A)] = D
    return out


def possession_stats(label, fps=25):
    """(possessions per live minute, mean spell seconds, dead share) for a label array."""
    sp = [x for x in spells(label) if x[0] in (H, A)]
    live = float(((label == H) | (label == A)).sum()) / fps / 60.0
    if not sp or live == 0:
        return 0.0, 0.0, float((label == D).mean())
    return len(sp) / live, float(np.mean([e - s for _, s, e in sp])) / fps, float((label == D).mean())


def fill_short_ball_gaps(ball, ball_vis, max_frames):
    """(ball, ball_vis) with ball-missing runs of at most `max_frames` frames filled by a straight line
    between the two visible ends. Longer runs, and runs at either end of the match, stay missing.
    Metrica's export drops the ball for a few frames at touches and event frames; those blips are the
    source's own artefact, not tracking loss (120-153 per game, 79-82 % followed by a possession
    change within 1 s against 5 % at random), so the damage simulator fills them before it decides
    where the ball goes missing. Works on a copy; the clean arrays in the file are never touched."""
    ball = np.asarray(ball, np.float64).copy()
    vis = np.asarray(ball_vis, bool).copy()
    d = np.diff(np.r_[1, vis.astype(np.int8), 1])
    starts, ends = np.flatnonzero(d == -1), np.flatnonzero(d == 1)
    for a, b in zip(starts, ends):
        if b - a <= max_frames and a > 0 and b < len(vis):
            w = (np.arange(a, b) - (a - 1)) / float(b - (a - 1))
            ball[a:b] = ball[a - 1] + w[:, None] * (ball[b] - ball[a - 1])
            vis[a:b] = True
    return ball, vis
