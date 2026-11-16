"""A4: fill the gaps in a tracked ball, offline (issue #7).

The decoder runs after the match, so every gap has a last-seen point before it and a first-seen
point after it. Measured against the hidden clean ball on the train split (README), a straight
line between those two points beats every cleverer story we tried: a cubic through the end
velocities (the 0.3 m tracking noise ruins the velocities), following the player who was near
both ends (worse even when an oracle picks the best player), and a ballistic arc (no better than
the chord it reduces to). Passes travel straight; dribbles do not last; the chord wins.

So: linear interpolation for gaps up to `long_gap_s`, with a confidence that falls with gap
length the way the measured error does; longer gaps are left missing, because the ball is usually
off the pitch or off camera during a stoppage and the gap length itself is evidence of D.

    ball, conf, src = repair(ball_xy, players_xy)      # (N, 2), (N,), (N,) int8
numpy only; nothing here is tuned on possession labels.
"""

import numpy as np

FPS = 25
SEEN, INTERP, MISSING = 0, 1, 2
SOURCE_NAMES = ("seen", "interp", "missing")

DEFAULTS = dict(
    long_gap_s=8.0,   # gaps longer than this stay missing
    # measured mean error of the chord by gap length (train split, v2): 0.3 m under 1 s, 1 m at
    # 1-2 s, 2.7 m at 2-4 s, 5 m at 4-8 s. Confidence = 1 / (1 + error / 1 m).
    error_m_per_s=0.65,
)


def runs(mask):
    """[(start, end_exclusive)] of True runs."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def repair(ball, players=None, params=None):
    """ball: (N, 2) metres, NaN where missing. players is accepted for interface stability and
    unused. Returns (ball_repaired (N, 2), confidence (N,), source (N,) int8)."""
    p = dict(DEFAULTS, **(params or {}))
    n = len(ball)
    out = ball.copy()
    conf = np.where(np.isnan(ball[:, 0]), 0.0, 1.0)
    src = np.where(np.isnan(ball[:, 0]), MISSING, SEEN).astype(np.int8)
    long_ = int(p["long_gap_s"] * FPS)
    for a, b in runs(np.isnan(ball[:, 0])):
        length = b - a
        if a == 0 or b == n or length > long_:
            continue
        p0, p1 = ball[a - 1], ball[b]
        s = np.arange(1, length + 1) / (length + 1)
        out[a:b] = p0 + s[:, None] * (p1 - p0)
        err = p["error_m_per_s"] * (length / FPS)
        conf[a:b] = 1.0 / (1.0 + err)
        src[a:b] = INTERP
    return out, conf, src


def repair_match(damaged, params=None):
    """Convenience: a load_master() 'damaged' dict -> (ball, conf, src)."""
    return repair(damaged["ball"], damaged["xy"], params)
