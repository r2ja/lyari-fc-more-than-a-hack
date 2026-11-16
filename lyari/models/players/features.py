"""A3: per-frame features from the damaged tracking, for a players-only model (issue #6).

The damaged copy has 46 anonymous slots (23 per team block) that get renumbered whenever a track
breaks, so nothing may depend on slot identity. Each team block is treated as a *set*: per-player
features that the model pools (mean and max over present players), plus team-level shape
features computed here. The ball is optional (off by default: the point of A3 is to work without
it) and, when on, enters only as (x, y, present) with the A4 repair confidence. With the A4 repair
(train.py --ball-repair) the ball arrives as (x, y, confidence): gaps up to 8 s are filled by a
straight line and the confidence falls with the gap length.

Coordinates are metres on the damaged copy (home block attacks +x in period 1, mirrored in
period 2; the copy may be flipped). Velocities are central differences over 0.16 s, like the
measurements in lyari/models/rules. numpy only; torch is only needed by train.py.
"""

import numpy as np

FPS = 25
HALF_X, HALF_Y = 52.5, 34.0
PER_PLAYER = 6          # x, y, vx, vy, speed, present
BALL_REL = 4            # with ball_rel: dx, dy and distance to the ball, closing speed (before `present`)
TEAM_SHAPE = 12         # see team_shape()
EXTRA = False           # --extra-clues: 4 more columns on the ball vector (see extra_clues); set by train/load_model
EXTRA_DIM = 4
CONTROL = False         # --control-physics: 3 more ball-relative columns per player (see control_physics); set by set_control
CONTROL_DIM = 3
N_SLOTS = 23


def velocities(xy, h=2, vmax=12.0):
    """Central differences over 2h frames; NaN where either end is missing, and NaN above vmax m/s
    (a slot reused by another player, or an identity swap, makes a jump that is not a velocity)."""
    v = np.full_like(xy, np.nan)
    v[h:-h] = (xy[2 * h:] - xy[:-2 * h]) / (2 * h / FPS)
    with np.errstate(invalid="ignore"):
        bad = np.sqrt((v ** 2).sum(-1)) > vmax
    v[bad] = np.nan
    return v


def ball_relative(xy, v, ball):
    """(N, 23, 4): each player's offset to the ball (/10 m, clipped), distance (/10 m, clipped) and
    speed toward the ball (/12 m/s). Zeros where the player or the ball is missing."""
    ok = ~np.isnan(xy[..., 0]) & ~np.isnan(ball[:, None, 0])
    d = np.where(ok[..., None], ball[:, None, :] - xy, 0.0)
    dist = np.sqrt((d ** 2).sum(-1))
    u = d / np.maximum(dist, 1e-6)[..., None]
    okv = ok & ~np.isnan(v[..., 0])
    closing = np.where(okv, (np.nan_to_num(v) * u).sum(-1), 0.0)
    out = np.stack([np.clip(d[..., 0] / 10.0, -3, 3), np.clip(d[..., 1] / 10.0, -3, 3),
                    np.where(ok, np.clip(dist / 10.0, 0, 3), 0.0), np.clip(closing, -12, 12) / 12.0], axis=-1)
    return np.where(ok[..., None], out, 0.0).astype(np.float32)


def set_control(on):
    """Turn the control-physics inputs on or off (train.py flag, predict.py config). Changes BALL_REL."""
    global CONTROL, BALL_REL
    CONTROL = bool(on)
    BALL_REL = 4 + (CONTROL_DIM if CONTROL else 0)


def control_physics(xy, v, ball, h=6):
    """(N, 23, 3) control physics: does the ball move *with* this player?
    Ball velocity minus player velocity (x, y, /12 m/s, clipped) and a with-ball score
    exp(-distance / 2.5 m) * exp(-|relative velocity| / 4 m/s): high when the ball is at a player's feet and
    travels with him (control), small for a ball flying past or a brief contact. Both velocities are central
    differences over 0.48 s (h=6): over 0.16 s, touches and tracker shake hide the match (on damaged open data a
    player within 2 m of the ball moves with it at a median 4.4 m/s relative speed over 0.16 s if his team has the
    ball, 5.8 if not; over 0.48 s 2.0 vs 3.8). Scales set on the damaged open data: the score passes 0.3 on 38% of
    live frames with the ball seen and the higher team is the team in possession on 93.5% of them.
    `v` is unused (kept for the per_player signature). Zeros where anything is missing."""
    vp = velocities(xy, h=h)
    vb = velocities(ball[:, None, :], h=h, vmax=40.0)[:, 0]
    ok = ~np.isnan(xy[..., 0]) & ~np.isnan(vp[..., 0]) & ~np.isnan(vb[:, None, 0])
    dv = np.where(ok[..., None], vb[:, None, :] - np.nan_to_num(vp), 0.0)
    rel = np.sqrt((dv ** 2).sum(-1))
    dist = np.sqrt(((np.nan_to_num(ball[:, None, :]) - np.nan_to_num(xy)) ** 2).sum(-1))
    with_ball = np.exp(-dist / 2.5) * np.exp(-rel / 4.0)
    out = np.stack([np.clip(dv[..., 0], -12, 12) / 12.0, np.clip(dv[..., 1], -12, 12) / 12.0, with_ball], axis=-1)
    return np.where(ok[..., None], out, 0.0).astype(np.float32)


def per_player(xy, v, ball=None):
    """(N, 23, 6) features for one team block: normalised x, y, vx, vy, speed, present. With `ball`
    (N, 2), (N, 23, 10): the four ball-relative features are inserted before `present`."""
    present = ~np.isnan(xy[..., 0])
    x = np.where(present, xy[..., 0] / HALF_X, 0.0)
    y = np.where(present, xy[..., 1] / HALF_Y, 0.0)
    ok = present & ~np.isnan(v[..., 0])
    vx = np.where(ok, np.clip(v[..., 0], -12, 12) / 12.0, 0.0)
    vy = np.where(ok, np.clip(v[..., 1], -12, 12) / 12.0, 0.0)
    speed = np.sqrt(vx ** 2 + vy ** 2)
    base = [x, y, vx, vy, speed]
    if ball is not None:
        rel = ball_relative(xy, v, ball)
        if CONTROL:
            rel = np.concatenate([rel, control_physics(xy, v, ball)], axis=-1)
        base += [rel[..., k] for k in range(BALL_REL)]
    return np.stack(base + [present.astype(np.float32)], axis=-1).astype(np.float32)


def team_shape(xy, v):
    """(N, 12) shape features for one team block: count, centroid x/y, spread x/y, front and back
    line x, width, mean velocity x/y, mean speed, share moving toward +x."""
    present = ~np.isnan(xy[..., 0])
    n = present.sum(axis=1).astype(np.float32)
    safe = np.maximum(n, 1.0)
    x = np.where(present, xy[..., 0], 0.0)
    y = np.where(present, xy[..., 1], 0.0)
    cx = x.sum(1) / safe
    cy = y.sum(1) / safe
    sx = np.sqrt((np.where(present, (xy[..., 0] - cx[:, None]) ** 2, 0.0)).sum(1) / safe)
    sy = np.sqrt((np.where(present, (xy[..., 1] - cy[:, None]) ** 2, 0.0)).sum(1) / safe)
    front = np.where(present, xy[..., 0], -np.inf).max(1)
    back = np.where(present, xy[..., 0], np.inf).min(1)
    front = np.where(np.isfinite(front), front, 0.0)
    back = np.where(np.isfinite(back), back, 0.0)
    ymax = np.where(present, xy[..., 1], -np.inf).max(1)
    ymin = np.where(present, xy[..., 1], np.inf).min(1)
    width = np.where(np.isfinite(ymax) & np.isfinite(ymin), ymax - ymin, 0.0)
    okv = present & ~np.isnan(v[..., 0])
    nv = np.maximum(okv.sum(1), 1.0)
    vx = np.where(okv, v[..., 0], 0.0).sum(1) / nv
    vy = np.where(okv, v[..., 1], 0.0).sum(1) / nv
    sp = np.where(okv, np.sqrt(v[..., 0] ** 2 + v[..., 1] ** 2), 0.0).sum(1) / nv
    fwd = np.where(okv, (v[..., 0] > 0.5), 0.0).sum(1) / nv
    out = np.stack([n / 11.0, cx / HALF_X, cy / HALF_Y, sx / 20.0, sy / 20.0, front / HALF_X, back / HALF_X,
                    width / 68.0, np.clip(vx, -8, 8) / 8.0, np.clip(vy, -8, 8) / 8.0, np.clip(sp, 0, 8) / 8.0, fwd], axis=1)
    return out.astype(np.float32)


def _attack_sign(blocks):
    """+1 if block 0 defends the -x goal over this stretch, else -1: the team whose deepest player stands at
    lower x most of the time (goalkeepers), by majority over the frames. Needs no metadata."""
    lo = [np.where(np.isnan(b[..., 0]), np.inf, b[..., 0]).min(1) for b in blocks]
    ok = np.isfinite(lo[0]) & np.isfinite(lo[1])
    if not ok.any():
        return 1.0
    return 1.0 if np.mean(lo[0][ok] < lo[1][ok]) >= 0.5 else -1.0


def extra_clues(blocks, ball=None):
    """(N, 4): goal-side share of each block (players nearer their own goal than their nearest opponent) and"""
    s = _attack_sign(blocks)
    goals = [np.array([-52.5 * s, 0.0]), np.array([52.5 * s, 0.0])]
    gs, pr = [], []
    for j in (0, 1):
        p, o = blocks[j], blocks[1 - j]
        d = np.linalg.norm(p[:, :, None, :] - o[:, None, :, :], axis=-1)            # (N, 23, 23)
        dn = np.where(np.isnan(d), np.inf, d)
        k = dn.argmin(2)
        near = np.take_along_axis(o, k[..., None].repeat(2, -1), axis=1)
        own = np.linalg.norm(p - goals[j], axis=-1)
        opp_own = np.linalg.norm(near - goals[j], axis=-1)
        present = ~np.isnan(p[..., 0])
        ok = present & np.isfinite(dn.min(2))
        cnt = np.maximum(present.sum(1), 1)
        gs.append(np.where(ok, own < opp_own, False).sum(1) / cnt)
        pr.append(np.minimum((dn < 3.0).sum(2).max(1), 4) / 4.0)
    out = np.stack([gs[0], gs[1], pr[0], pr[1]], axis=1)
    return np.nan_to_num(out).astype(np.float32)


def frame_features(damaged, use_ball=False, ball=None, ball_conf=None, ball_rel=False):
    """A load_master() 'damaged' dict -> dict of arrays:
        players (N, 2, 23, 6)   per-player features per team block
        teams   (N, 2*12 + 6)   team shape features for both blocks, plus relative terms
        ball    (N, 4)          x, y, present, confidence (zeros when use_ball is False)
    """
    xy = damaged["xy"]
    v = velocities(xy)
    blocks = [xy[:, :N_SLOTS], xy[:, N_SLOTS:]]
    vb = [v[:, :N_SLOTS], v[:, N_SLOTS:]]
    bxy = None
    if ball_rel:
        bxy = (damaged["ball"] if ball is None else ball)[:, :2].astype(np.float64)
    players = np.stack([per_player(blocks[0], vb[0], bxy), per_player(blocks[1], vb[1], bxy)], axis=1)
    s0, s1 = team_shape(blocks[0], vb[0]), team_shape(blocks[1], vb[1])
    rel = np.stack([s0[:, 1] - s1[:, 1], s0[:, 2] - s1[:, 2], s0[:, 5] - s1[:, 6], s1[:, 5] - s0[:, 6],
                    s0[:, 8] - s1[:, 8], s0[:, 10] - s1[:, 10]], axis=1).astype(np.float32)
    teams = np.concatenate([s0, s1, rel], axis=1)
    n = len(xy)
    if use_ball:
        b = damaged["ball"] if ball is None else ball
        present = ~np.isnan(b[:, 0])
        if ball_conf is None and b.shape[1] >= 3:          # A4-repaired ball: (x, y, confidence)
            ball_conf = np.nan_to_num(b[:, 2]).astype(np.float32)
        conf = np.ones(n, np.float32) if ball_conf is None else ball_conf
        ballf = np.stack([np.where(present, b[:, 0] / HALF_X, 0.0), np.where(present, b[:, 1] / HALF_Y, 0.0),
                          present.astype(np.float32), np.where(present, conf, 0.0)], axis=1).astype(np.float32)
    else:
        ballf = np.zeros((n, 4), np.float32)
    if EXTRA:
        ballf = np.concatenate([ballf, extra_clues(blocks)], axis=1)
    return dict(players=players, teams=teams, ball=ballf)


def flip_blocks(feats):
    """Swap the two team blocks (home/away augmentation). Relative terms change sign."""
    players = feats["players"][:, ::-1].copy()
    t = feats["teams"]
    r = t[:, 2 * TEAM_SHAPE:]
    rel = np.stack([-r[:, 0], -r[:, 1], r[:, 3], r[:, 2], -r[:, 4], -r[:, 5]], axis=1)   # front/back terms swap
    teams = np.concatenate([t[:, TEAM_SHAPE:2 * TEAM_SHAPE], t[:, :TEAM_SHAPE], rel], axis=1)
    ball = feats["ball"].copy()
    if ball.shape[1] >= 4 + EXTRA_DIM:                      # extra clues: per-block columns swap too
        ball[:, [4, 5, 6, 7]] = ball[:, [5, 4, 7, 6]]
    return dict(players=players, teams=teams, ball=ball)
