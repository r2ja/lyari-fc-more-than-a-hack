"""Generic tracking-quirk detectors and a cleaner for the damaged-layout tracking. numpy only."""

import numpy as np

FPS = 25
FROZEN_S = 10.0          # a real player never stands within 1 cm of the same spot for 10 s
FROZEN_TOL_M = 0.01
BALL_MAX_MS = 40.0       # faster than any shot (~35 m/s record): a jump, not a kick
SPARSE_PLAYERS = 6       # fewer tracked players than this: the tracking has no picture of the game
SPARSE_S = 5.0
BOX_X, BOX_Y = 36.0, 20.16   # penalty box: 16.5 m from a 52.5 m half-length, 40.32 m wide
DEEP_S = 60.0            # a slot sitting in the OTHER team's box for a minute, barely moving
DEEP_SPEED_MS = 0.5


def _runs(mask):
    """[(start, end)] of True runs, end exclusive."""
    m = np.r_[False, np.asarray(mask, bool), False]
    a = np.flatnonzero(~m[:-1] & m[1:])
    b = np.flatnonzero(m[:-1] & ~m[1:])
    return list(zip(a.tolist(), b.tolist()))


def frozen_mask(xy, min_s=FROZEN_S, tol=FROZEN_TOL_M):
    """(N, S) True where a slot's position has not moved more than tol for at least min_s (the whole run is marked)."""
    n, s, _ = xy.shape
    out = np.zeros((n, s), bool)
    step = np.full((n, s), np.inf, np.float32)
    step[1:] = np.nanmax(np.abs(np.diff(xy, axis=0)), axis=2)
    still = step <= tol                                  # NaN steps (absent) compare False
    for k in range(s):
        for a, b in _runs(still[:, k]):
            if b - a + 1 >= min_s * FPS:
                out[max(a - 1, 0):b, k] = True           # include the frame the stillness starts from
    return out


def ball_spike_mask(ball, max_ms=BALL_MAX_MS):
    """(N,) True on a ball position reached from the previous visible one faster than max_ms (one-frame jumps)."""
    b = np.asarray(ball)[:, :2]
    vis = np.flatnonzero(~np.isnan(b[:, 0]))
    out = np.zeros(len(b), bool)
    if len(vis) < 2:
        return out
    d = np.hypot(*(b[vis[1:]] - b[vis[:-1]]).T)
    dt = (vis[1:] - vis[:-1]) / FPS
    fast = d / dt > max_ms
    # a spike jumps away and back: mark the frame that is far from BOTH neighbours, not the return step
    for i in np.flatnonzero(fast):
        j = i + 1                                         # index into vis of the suspicious frame
        if j + 1 < len(vis):
            back = np.hypot(*(b[vis[j + 1]] - b[vis[j]])) / ((vis[j + 1] - vis[j]) / FPS)
            if back > max_ms:
                out[vis[j]] = True
    return out


def sparse_runs(xy, period, min_players=SPARSE_PLAYERS, min_s=SPARSE_S):
    """[(start, end)] inside the halves where fewer than min_players slots are tracked for at least min_s."""
    count = (~np.isnan(xy[..., 0])).sum(1)
    mask = (count < min_players) & (np.asarray(period) > 0)
    return [(a, b) for a, b in _runs(mask) if b - a >= min_s * FPS]


def deep_stationary_runs(xy, period, min_s=DEEP_S, max_speed=DEEP_SPEED_MS):
    """[(slot, start, end)]: a slot of one team block sitting almost still inside the box its team ATTACKS for at
    least min_s (where only the other team's keeper stands for that long): a misassigned keeper or a ghost.
    Each team's attacking direction per half is read from its block's mean x (no labels)."""
    period = np.asarray(period)
    out = []
    with np.errstate(invalid="ignore"):
        v = np.full(xy.shape[:2], np.nan, np.float32)
        v[1:] = np.hypot(*(np.diff(xy, axis=0)).transpose(2, 0, 1)) * FPS
    for half in (1, 2):
        rows = np.flatnonzero(period == half)
        if not len(rows):
            continue
        with np.errstate(invalid="ignore"):
            cx = [np.nanmean(xy[rows, blk, 0]) for blk in (slice(0, 23), slice(23, 46))]
        for block, (lo, hi) in enumerate(((0, 23), (23, 46))):
            defends_negative = cx[block] < cx[1 - block]
            attack_sign = 1.0 if defends_negative else -1.0      # the goal this block attacks is at attack_sign * 52.5
            for k in range(lo, hi):
                x, y = xy[rows, k, 0], xy[rows, k, 1]
                with np.errstate(invalid="ignore"):
                    inside = (attack_sign * x > BOX_X) & (np.abs(y) < BOX_Y) & (np.nan_to_num(v[rows, k], nan=0.0) < max_speed * 3)
                for a, b in _runs(inside):
                    if b - a < min_s * FPS:
                        continue
                    if np.nanmedian(v[rows[a:b], k]) < max_speed:
                        out.append((k, int(rows[a]), int(rows[b - 1]) + 1))
    return out


GOAL_NEAR_M = 6.0        # a keeper stands within this of his goal line most of the time
MATE_NEAR_M = 3.0        # ... and no other-block player is this close (else it is a corner crowd, not a lone keeper)


def wrong_block_keeper_frames(xy, period):
    """(frames, slot-frames): per-frame count of slots standing alone within GOAL_NEAR_M of the goal their OWN block"""
    period = np.asarray(period)
    frames = np.zeros(len(xy), bool)
    total = 0
    for half in (1, 2):
        rows = np.flatnonzero(period == half)
        if not len(rows):
            continue
        with np.errstate(invalid="ignore"):
            cx = [np.nanmean(xy[rows, blk, 0]) for blk in (slice(0, 23), slice(23, 46))]
        for block, (lo, hi) in enumerate(((0, 23), (23, 46))):
            attack_sign = 1.0 if cx[block] < cx[1 - block] else -1.0
            olo, ohi = (23, 46) if block == 0 else (0, 23)
            x = xy[rows, lo:hi, 0]
            with np.errstate(invalid="ignore"):
                deep = (attack_sign * x > 52.5 - GOAL_NEAR_M) & (np.abs(xy[rows, lo:hi, 1]) < 20.16)
            for k in np.flatnonzero(deep.any(0)):
                f = np.flatnonzero(deep[:, k])
                d = np.hypot(xy[rows[f], None, lo + k, 0] - xy[rows[f], olo:ohi, 0],
                             xy[rows[f], None, lo + k, 1] - xy[rows[f], olo:ohi, 1])
                with np.errstate(invalid="ignore"):
                    alone = ~(np.nan_to_num(d, nan=np.inf) < MATE_NEAR_M).any(1)
                frames[rows[f[alone]]] = True
                total += int(alone.sum())
    return int(frames.sum()), total


def report(damaged, period):
    """Aggregates of every quirk on the graded halves. Counts only, no positions or frame numbers."""
    xy, ball = damaged["xy"], damaged["ball"]
    live = np.asarray(period) > 0
    fz = frozen_mask(xy)
    spikes = ball_spike_mask(ball)
    vis_ball = ~np.isnan(np.asarray(ball)[:, 0]) & live
    return dict(
        frozen_slot_frames=int(fz[live].sum()),
        frozen_runs=sum(len([r for r in _runs(fz[:, k]) if live[r[0]]]) for k in range(fz.shape[1])),
        deep_stationary_runs=len(deep_stationary_runs(xy, period)),
        sparse_runs=len(sparse_runs(xy, period)),
        sparse_frames=int(sum(b - a for a, b in sparse_runs(xy, period))),
        wrong_block_keeper_frames=wrong_block_keeper_frames(xy, period)[0],
        ball_spikes=int((spikes & live).sum()),
        ball_spike_share=float((spikes & live).sum() / max(vis_ball.sum(), 1)),
    )


def reassign_parked(xy, period):
    """Move each run found by deep_stationary_runs (a slot parked in the box its own block attacks: in practice the
    other team's keeper filed under the wrong team) into a slot of the other block that is free for the whole run.
    No free slot: the run is blanked instead. Returns (new xy, runs moved, runs blanked)."""
    xy = xy.copy()
    moved = blanked = 0
    for k, a, b in deep_stationary_runs(xy, period):
        other = range(23, 46) if k < 23 else range(0, 23)
        free = [j for j in other if np.isnan(xy[a:b, j, 0]).all()]
        if free:
            xy[a:b, free[0]] = xy[a:b, k]
            moved += 1
        else:
            blanked += 1
        xy[a:b, k] = np.nan
    return xy, moved, blanked


def clean(damaged, frozen=True, spikes=True, period=None):
    """A copy of the damaged dict with frozen slot positions and one-frame ball spikes set missing (NaN), as if the
    tracker had lost them, and (with period) parked other-team keepers moved to the other block. Labels and every
    other array are untouched. Frozen runs go first, so a frozen ghost is blanked, never moved."""
    out = dict(damaged)
    if frozen or period is not None:
        xy = damaged["xy"].copy()
        if frozen:
            xy[frozen_mask(xy)] = np.nan
        if period is not None:
            xy, _, _ = reassign_parked(xy, period)
        out["xy"] = xy
    if spikes:
        ball = np.array(damaged["ball"], copy=True)
        ball[ball_spike_mask(ball)] = np.nan
        out["ball"] = ball
        if "ball_vis" in damaged:
            out["ball_vis"] = ~np.isnan(ball[:, 0])
    return out
