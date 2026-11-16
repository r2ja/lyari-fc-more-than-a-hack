"""Three-state possession Viterbi decoder."""
import numpy as np
from evaluation import possession_metrics as pm

CODES = np.array([pm.H, pm.A, pm.D], np.int8)
EPS = 1e-3


def run_lengths(mask):
    """For every frame, the length of the run of True it sits in (0 where False)."""
    n = len(mask)
    out = np.zeros(n, np.int32)
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        out[a:b] = b - a
    return out


def emissions(probs, rule="soft", missing_lr=3.0, conf=0.75, dead_run=200):
    """(N, 3) log-emission scores for states H, A, D.

    organisers rule: a frame that names a team is right with probability `conf` (the baseline's
    frame accuracy, about 0.7), so it is worth log(conf / (1 - conf)) nats, not log(1 / eps)."""
    probs = np.asarray(probs, dtype=np.float64)
    nan = np.isnan(probs).any(axis=1)
    e = np.log(np.clip(np.nan_to_num(probs, nan=1.0 / 3), EPS, 1.0))
    e[nan] = 0.0
    if rule == "organisers":
        seen = ~nan & (probs[:, :2].sum(axis=1) > 0)
        team = probs[seen, 1] > probs[seen, 0]            # True where the vote is A
        e[seen] = np.log(1 - conf)
        e[np.flatnonzero(seen), team.astype(int)] = np.log(conf)
        e[seen, 2] = np.log(EPS)                          # a player has the ball: not dead
        missing = ~nan & ~seen
        e[missing] = 0.0                                  # no ball: no team evidence
        # a ball missing for a long run is a stoppage; short gaps are normal in play. Only runs of
        # at least dead_run frames carry dead evidence, spread over the run so it does not pile up
        long = missing & (run_lengths(missing) >= dead_run)
        e[long, 2] = np.log(missing_lr)
        e[long, :2] = -np.log(missing_lr)
    return e


def viterbi(logemit, switch, dead, dead_out=None):
    """Most likely state path. logemit (N, 3); costs in nats. `dead` is the cost of entering D from a team,
    `dead_out` (default: the same) the cost of leaving it: rule-based teams make leaving cheaper, since a restart is
    a sharp event (a kick) while a stoppage needs sustained calm."""
    n = len(logemit)
    dead_out = dead if dead_out is None else dead_out
    cost = np.array([[0.0, -switch, -dead],
                     [-switch, 0.0, -dead],
                     [-dead_out, -dead_out, 0.0]])
    score = logemit[0].copy()
    back = np.zeros((n, 3), np.int8)
    for t in range(1, n):
        cand = score[:, None] + cost                     # cand[i, j]: come from i, go to j
        back[t] = np.argmax(cand, axis=0)
        score = cand[back[t], np.arange(3)] + logemit[t]
    path = np.empty(n, np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return CODES[path]


def decode(probs, switch=20.0, dead=10.0, rule="soft", missing_lr=3.0, conf=0.75, dead_run=200):
    return viterbi(emissions(probs, rule, missing_lr, conf, dead_run), switch, dead)
