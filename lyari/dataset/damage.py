"""Training-time damage for the 13 external practice matches. Distances are in metres."""
import json
import warnings

import numpy as np
from lyari.dataset import fingerprint as FP, labels as L

FPS = 25
MISSING = np.int16(-32768)
N_SLOTS = 46
SLOTS_PER_TEAM = N_SLOTS // 2
PITCH_X, PITCH_Y = 105.0, 68.0
GHOST = -2

DEFAULTS = {'A_min_spell_s': 3.0,
 'B_air_m': 1.0,
 'B_contact_m': 1.2,
 'B_goal_m': 5.5,
 'B_long_median_s': 15.0,
 'B_long_share': 0.0,
 'B_long_sigma': 0.6,
 'B_median_s': 2.8,
 'B_min_s': 0.0,
 'B_mult_air': 3.0,
 'B_mult_contact': 1.5,
 'B_mult_far': 1.0,
 'B_mult_goal': 9.0,
 'B_mult_out': 6.0,
 'B_mult_slow': 1.9,
 'B_mult_speed': 1.0,
 'B_per_live_min': 0.8,
 'B_sigma': 0.92,
 'B_slow_ms': 2.0,
 'B_speed_ms': 8.0,
 'C_max_frames': 12,
 'C_mean_frames': 5.0,
 'C_per_live_min': 0.0,
 'D_max_s': 240.0,
 'D_median_s': 100.0,
 'D_min_s': 60.0,
 'D_sigma': 0.5,
 'F_per_live_min': 0.0,
 'ball_contact_m': 1.5,
 'ball_drift_m': 0.07,
 'ball_drift_tau_s': 0.15,
 'ball_noise_m': 0.0,
 'ball_smooth_causal': 0,
 'ball_smooth_s': 0.11,
 'ball_stick_other': 0.005,
 'ball_stick_rate': 0.02,
 'blackout_dead_mult': 1.0,
 'blackout_far_beta': 0.0,
 'blackout_median_s': 4.07,
 'blackout_offset_s': 0.5,
 'blackout_per_min': 0.041,
 'blackout_q_hi': 0.381,
 'blackout_q_lo': 0.095,
 'blackout_sigma': 0.47,
 'buffer_s': 1.0,
 'camera_setback_m': 30.0,
 'crowd_radius_m': 1.5,
 'damage_version': 5,
 'dup_h': 0.00025,
 'dup_mean_s': 0.4,
 'dup_offset_m': 0.8,
 'edge_min_frames': 2,
 'far_y_m': 15.0,
 'fill_s': 1.0,
 'flip_prob': 0.5,
 'fov_bias_m': 5.0,
 'fov_p_out': 0.9,
 'fov_tau_s': 1.5,
 'fov_width_m': 110.0,
 'ghost_dead_mult': 4.0,
 'ghost_max_s': 60.0,
 'ghost_median_s': 8.0,
 'ghost_pair_share': 0.08,
 'ghost_per_min': 0.0,
 'ghost_sigma': 0.8,
 'interp_gap_frames': 12,
 'interp_p': 0.5,
 'lag_max_s': 6.0,
 'lag_mean_s': -1.3,
 'lag_min_s': -4.0,
 'lag_sd_s': 3.9,
 'lead_max_s': 3.0,
 'lead_mean_s': 0.94,
 'lead_min_s': -5.0,
 'lead_sd_s': 3.4,
 'loss_v5': 1,
 'max_objects': 22,
 'mean_count_target': 20.35,
 'occ_crowd_a': 4.0,
 'occ_far_beta': 0.7,
 'occ_far_len_beta': 0.0,
 'occ_h0': 0.0003,
 'occ_long_max_s': 60.0,
 'occ_long_median_s': 6.0,
 'occ_long_share': 0.1,
 'occ_long_sigma': 1.0,
 'occ_median_s': 0.7,
 'occ_sigma': 0.9,
 'occ_trunc_s': 8.0,
 'pA': 0.98,
 'pD': 0.0,
 'p_relink': 0.5,
 'player_drift_m': 0.085,
 'player_drift_tau_s': 1.0,
 'player_noise_m': 0.0,
 'present_target': 0.596,
 'present_tol': 0.01,
 'randomize': True,
 'reacq_median_s': 0.4,
 'reacq_sigma': 0.8,
 'refind_margin_s': 3.0,
 'refind_median_s': 4.4,
 'refind_min_s': 25.0,
 'refind_p': 0.64,
 'refind_sigma': 0.63,
 'relink_long_p': 0.0,
 'relink_max_s': 60.0,
 'silent_crowd_s': 8.0,
 'silent_open_s': 40.0,
 'snap_min_s': 2.0,
 'still_hide_p': 0.0,
 'still_min_s': 1.0,
 'still_ms': 0.5,
 'swap_prob_per_encounter': 0.08,
 'swap_radius_m': 1.0,
 'v5be': 1,
 'v5bf': 1,
 'zombie_max_s': 1.0,
 'zombie_min_s': 0.2,
 'zombie_p': 0.2}

RANGES = {'B_median_s': None,
 'B_per_live_min': None,
 'B_sigma': None,
 'C_per_live_min': None,
 'ball_noise_m': (0.0, 0.01),
 'ball_stick_rate': (0.01, 0.04),
 'blackout_dead_mult': None,
 'blackout_median_s': None,
 'blackout_per_min': None,
 'buffer_s': (0.6, 1.2),
 'dup_h': (0.0001, 0.0005),
 'fill_s': (0.5, 1.5),
 'ghost_per_min': None,
 'lag_mean_s': None,
 'lead_mean_s': None,
 'mean_count_target': None,
 'occ_crowd_a': (2.0, 6.0),
 'occ_far_beta': (0.3, 1.0),
 'occ_h0': (0.00015, 0.0005),
 'pA': None,
 'pD': None,
 'p_relink': (0.3, 0.7),
 'player_noise_m': (0.0, 0.01),
 'present_target': None,
 'refind_p': None,
 'silent_crowd_s': (5.0, 12.0),
 'silent_open_s': (25.0, 60.0)}

DRIFT_RANGES = {'B_mult_goal': None,
 'B_mult_slow': None,
 'ball_drift_m': None,
 'ball_drift_tau_s': (0.1, 0.25),
 'ball_smooth_s': None,
 'player_drift_m': None,
 'player_drift_tau_s': (0.7, 2.0)}

SCALED = ('B_mult_speed', 'B_mult_air', 'B_mult_out', 'B_mult_contact')
SCALED_FIXED = ('B_mult_speed',)


def draw_params(rng, params):
    """Use the original random stream, including draws reserved for fixed external settings."""
    p = dict(params)
    drawn = {}
    for key, bounds in RANGES.items():
        u = rng.random()
        if bounds is None or not p["randomize"]:
            continue
        lo, hi = bounds
        v = np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))) if key == "dup_h" else lo + u * (hi - lo)
        p[key] = drawn[key] = float(v)
    for key in SCALED:
        u = rng.random()
        if key not in SCALED_FIXED and p["randomize"]:
            p[key] = drawn[key] = float(p[key] * np.exp(np.log(0.5) + u * (np.log(2.0) - np.log(0.5))))
    for key, bounds in DRIFT_RANGES.items():
        u = rng.random()
        if bounds is not None and p["randomize"]:
            lo, hi = bounds
            p[key] = drawn[key] = float(lo + u * (hi - lo))
    return p, drawn


def _runs(mask):
    """[(start, end_exclusive)] of the maximal True runs of a 1-D mask."""
    d = np.diff(np.r_[0, np.asarray(mask, np.int8), 0])
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def _lognormal(rng, median, sigma, size=None):
    return rng.lognormal(np.log(median), sigma, size)


def _frames(seconds):
    return int(round(seconds * FPS))


def _mean_filter(x, w):
    k = np.ones(w) / w
    return np.convolve(x, k, "same")


def _ema(x, tau_frames):
    """Exponential moving average of a 1-D array (NaN-free), forward pass."""
    a = 1.0 / max(tau_frames, 1.0)
    out = np.empty_like(x, dtype=np.float64)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = out[i - 1] + a * (x[i] - out[i - 1])
    return out


def _gauss_kernel(tau_frames):
    half = int(np.ceil(4 * tau_frames))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / tau_frames) ** 2)
    return k, half


def _fft_valid(x, k):
    """'valid' convolution of x (M, ...) with the 1-D kernel k along axis 0: M - len(k) + 1 rows."""
    m = x.shape[0] + len(k) - 1
    size = 1 << int(np.ceil(np.log2(m)))
    kf = np.fft.rfft(k, size).reshape((-1,) + (1,) * (x.ndim - 1))
    full = np.fft.irfft(np.fft.rfft(x, size, axis=0) * kf, size, axis=0)
    return full[len(k) - 1: x.shape[0]]


def _drift(rng, shape, amp, tau_frames):
    """Slowly wandering position error: Gaussian-filtered white noise, standard deviation `amp` (metres)
    per axis, time scale `tau_frames`. Its frame-to-frame change is about amp / tau^2, i.e. millimetres."""
    if amp <= 0:
        return np.zeros(shape)
    k, half = _gauss_kernel(max(tau_frames, 1.0))
    k = k / np.sqrt((k * k).sum())
    white = rng.normal(0.0, 1.0, (shape[0] + 2 * half,) + tuple(shape[1:]))
    return amp * _fft_valid(white, k)


def _smooth_runs(x, vis, tau_frames, causal=False):
    """Gaussian smoothing of x (N, d) inside its visible runs (normalised convolution: a gap never
    pulls the track towards zero, the edges of a run use only that run's frames). causal=True keeps only
    the past half of the kernel, like a tracker's online filter: the first frame after a gap is the raw
    detection instead of an average pulled towards where the ball goes next."""
    if tau_frames <= 0:
        return x
    k, half = _gauss_kernel(tau_frames)
    if causal:
        k = k.copy()
        k[:half] = 0.0
    w = vis.astype(np.float64)
    pad = lambda a: np.concatenate([np.zeros((half,) + a.shape[1:]), a, np.zeros((half,) + a.shape[1:])])
    num = _fft_valid(pad(np.where(vis[:, None], x, 0.0)), k)
    den = _fft_valid(pad(w), k)
    out = num / np.maximum(den, 1e-12)[:, None]
    return np.where(vis[:, None], out, x)


def _crowded(xy, vis, radius):
    """(N, P) bool: player within radius of another player (any team). xy in metres, NaN when missing."""
    n, p, _ = xy.shape
    out = np.zeros((n, p), bool)
    r2 = radius * radius
    for a in range(0, n, 8192):
        b = min(n, a + 8192)
        blk = xy[a:b]
        d = blk[:, :, None, :] - blk[:, None, :, :]
        d2 = (d * d).sum(-1)
        d2[:, np.arange(p), np.arange(p)] = np.inf
        d2 = np.where(np.isnan(d2), np.inf, d2)
        out[a:b] = (d2 < r2).any(-1) & vis[a:b]
    return out


def _nearest(xy, ball):
    """(distance, index) of the player nearest the ball per frame (inf / -1 where undefined)."""
    n = len(ball)
    dist = np.full(n, np.inf)
    idx = np.full(n, -1, np.int64)
    for a in range(0, n, 8192):
        b = min(n, a + 8192)
        d = xy[a:b] - ball[a:b, None, :2]
        d2 = np.where(np.isnan(d).any(-1), np.inf, (d * d).sum(-1))
        k = d2.argmin(axis=1)
        m = d2[np.arange(b - a), k]
        ok = np.isfinite(m)
        dist[a:b] = np.where(ok, np.sqrt(np.where(ok, m, 0.0)), np.inf)
        idx[a:b] = np.where(ok, k, -1)
    return dist, idx


def dead_proxy(ball, bvis):
    """Kinematic stoppage spells for a source with no dead-ball label: the ball nearly still for
    2 s or more, or off the pitch, or missing in the source for 8 s or more. Label-free."""
    n = len(ball)
    b, v = L.fill_short_ball_gaps(ball[:, :2], bvis, FPS)
    still = np.zeros(n, bool)
    if v.sum() > 10:
        vel = np.gradient(np.where(v[:, None], b, np.nan), axis=0) * FPS
        sp = np.linalg.norm(vel, axis=1)
        sp = np.where(np.isnan(sp), 0.0, sp)
        still = (_mean_filter(sp, 7) < 0.6) & v
    out = (np.abs(b[:, 0]) > PITCH_X / 2 + 0.3) | (np.abs(b[:, 1]) > PITCH_Y / 2 + 0.3)
    m = still | (out & v)
    for a, e in _runs(~v):
        if e - a >= 200:
            m[a:e] = True
    m = _mean_filter(m.astype(float), 25) > 0.5
    for a, e in _runs(~m):                      # close holes shorter than 3 s
        if 0 < a and e < n and e - a < 75:
            m[a:e] = True
    for a, e in _runs(m):                       # drop spells shorter than 1.5 s
        if e - a < 38:
            m[a:e] = False
    return _runs(m)


def _draw_run(rng, p, s, e):
    """A stoppage-anchored run for the dead spell [s, e): start and end around it, asymmetric."""
    lead = float(np.clip(rng.normal(p["lead_mean_s"], p["lead_sd_s"]), p["lead_min_s"], p["lead_max_s"]))
    lag = float(np.clip(rng.normal(p["lag_mean_s"], p["lag_sd_s"]), p["lag_min_s"], p["lag_max_s"]))
    return max(0, s + _frames(lead)), e + _frames(lag)


def _run_mask(n, a, b, refound):
    m = np.zeros(n, bool)
    m[a:min(n, b)] = True
    if refound is not None:
        m[refound[0]:refound[1]] = False
    return m


def _refind(rng, p, a, b):
    """The tracker re-finds the placed ball once inside a long stoppage run: (start, end) or None."""
    if b - a <= _frames(p["refind_min_s"]) or rng.random() >= p["refind_p"]:
        return None
    length = int(_lognormal(rng, p["refind_median_s"], p["refind_sigma"]) * FPS)
    margin = _frames(p["refind_margin_s"])
    lo, hi = a + margin, b - margin - length
    if hi <= lo:
        return None
    st = int(rng.integers(lo, hi))
    return st, st + length


def ball_episodes(ball, bvis, label, xy, dead, p, rng, far=None):
    """Mask of frames where the damaged ball is missing, plus what produced it.

    ball (N, 3) metres with NaN where absent (already mirrored, short source gaps filled), bvis (N,),
    dead = [(start, end)] of the stoppages, xy (N, 22, 2) players in metres."""
    n = len(ball)
    inplay = label != 2
    live_min = float(inplay.sum()) / FPS / 60.0
    info = dict(A=0, B=0, C=0, D=0, source_gaps=0)

    # source gaps: every edge moved so it never sits on a label change (Metrica drops the ball one
    # frame after the restart); the end can only be found later, the start only lost earlier
    moved = np.zeros(n, bool)
    gaps = [(a, b) for a, b in _runs(~bvis)]
    for a, b in gaps:
        lead = float(np.clip(rng.normal(p["lead_mean_s"], p["lead_sd_s"]), p["lead_min_s"], p["lead_max_s"]))
        lag = float(np.clip(rng.normal(p["lag_mean_s"], p["lag_sd_s"]), p["lag_min_s"], p["lag_max_s"]))
        a2 = a - max(p["edge_min_frames"], _frames(abs(lead))) if a > 0 else 0
        b2 = b + max(p["edge_min_frames"], _frames(abs(lag))) if b < n else n
        moved[max(0, a2):min(n, b2)] = True
    info["source_gaps"] = len(gaps)
    in_live_gaps = sum(1 for a, _ in gaps if inplay[min(a, n - 1)])

    # A: one run per stoppage
    eligible = [(s, e) for s, e in dead if e - s > _frames(p["A_min_spell_s"])]
    a_runs = {}
    for s, e in eligible:
        if rng.random() < p["pA"]:
            a, b = _draw_run(rng, p, s, e)
            a_runs[(s, e)] = (a, b, _refind(rng, p, a, b))
    # D: one rare long outage
    mask_d = np.zeros(n, bool)
    if dead and rng.random() < p["pD"]:
        long_spells = [(s, e) for s, e in dead if e - s >= _frames(45.0)]
        if long_spells:
            s, e = max(long_spells, key=lambda x: x[1] - x[0])
            length = float(np.clip(_lognormal(rng, p["D_median_s"], p["D_sigma"]), p["D_min_s"], p["D_max_s"]))
            end = min(s + _frames(length), e + _frames(rng.uniform(0, 20.0)))
            mask_d[s:min(n, end)] = True
            info["D"] = 1

    def build_a():
        m = np.zeros(n, bool)
        for a, b, rf in a_runs.values():
            m |= _run_mask(n, a, b, rf)
        return m

    mask_a = build_a()
    base = moved | mask_a | mask_d

    # B: in-play losses, placed by weight and counted exactly
    cand = np.flatnonzero(inplay & bvis & ~base)
    b_mask = np.zeros(n, bool)
    if len(cand) > 10:
        speed = np.linalg.norm(np.gradient(ball[:, :2], axis=0), axis=1) * FPS
        speed = np.where(np.isnan(speed), 0.0, speed)
        z = np.nan_to_num(ball[:, 2]) if ball.shape[1] > 2 else np.zeros(n)
        out = (np.abs(ball[:, 0]) > PITCH_X / 2) | (np.abs(ball[:, 1]) > PITCH_Y / 2)
        dist, _ = _nearest(xy, ball)
        mult = np.ones(n)
        mult[speed > p["B_speed_ms"]] *= p["B_mult_speed"]
        mult[z > p["B_air_m"]] *= p["B_mult_air"]
        mult[out & ~np.isnan(ball[:, 0])] *= p["B_mult_out"]
        mult[dist < p["B_contact_m"]] *= p["B_mult_contact"]
        mult[speed < p["B_slow_ms"]] *= p["B_mult_slow"]
        mult[np.abs(np.nan_to_num(ball[:, 0])) > PITCH_X / 2 - p["B_goal_m"]] *= p["B_mult_goal"]
        if far is not None:
            mult[far] *= p["B_mult_far"]
        n_b = int(round(max(0.0, p["B_per_live_min"] * live_min - in_live_gaps)))
        n_b = min(n_b, len(cand) // 2)
        if n_b > 0:
            w = mult[cand]
            starts = rng.choice(cand, size=n_b, replace=False, p=w / w.sum())
            longer = rng.random(n_b) < p["B_long_share"]
            length = np.where(longer, _lognormal(rng, p["B_long_median_s"], p["B_long_sigma"], n_b),
                              _lognormal(rng, p["B_median_s"], p["B_sigma"], n_b))
            for st, ln in zip(starts, length):
                b_mask[st:st + max(2, int(max(ln, p["B_min_s"]) * FPS))] = True     # B_min_s: v7 (0 before)
            info["B"] = n_b
    # F (v6): extra in-play losses on the far touchline, where the ball is small and often hidden
    if far is not None and p["F_per_live_min"] > 0:
        cand_f = np.flatnonzero(inplay & bvis & ~base & ~b_mask & far)
        n_f = min(int(round(p["F_per_live_min"] * live_min)), len(cand_f) // 2)
        if n_f > 0:
            wf = mult[cand_f] if len(cand) > 10 else np.ones(len(cand_f))   # same speed / contact weights as B
            starts = rng.choice(cand_f, size=n_f, replace=False, p=wf / wf.sum())
            for st, ln in zip(starts, _lognormal(rng, p["B_median_s"], p["B_sigma"], n_f)):
                b_mask[st:st + max(2, int(max(ln, p["B_min_s"]) * FPS))] = True
            info["F"] = n_f
    # C: blips
    c_mask = np.zeros(n, bool)
    cand_c = np.flatnonzero(inplay & bvis & ~base & ~b_mask)
    n_c = min(int(round(p["C_per_live_min"] * live_min)), len(cand_c) // 2)
    if n_c > 0:
        starts = rng.choice(cand_c, size=n_c, replace=False)
        length = np.clip(rng.geometric(1.0 / p["C_mean_frames"], n_c), 1, p["C_max_frames"])
        for st, ln in zip(starts, length):
            c_mask[st:st + ln] = True
        info["C"] = n_c


    s_mask = np.zeros(n, bool)
    if p["still_hide_p"] > 0:
        sp = np.linalg.norm(np.gradient(ball[:, :2], axis=0), axis=1) * FPS
        still = _mean_filter((np.nan_to_num(sp, nan=99.0) < p["still_ms"]).astype(float), 13) > 0.99
        for a, e in _runs(still & bvis):
            # stretches that touch live play only (stoppages are the A runs' job: Sportec freezes its ball through
            # them, 45 % of its frames are still), hidden whole so no edge falls on the live/dead change
            if not inplay[a:e].any():
                continue
            if e - a >= _frames(p["still_min_s"]) and rng.random() < p["still_hide_p"]:
                s_mask[a:max(e, a + _frames(p["snap_min_s"]))] = True
        info["S"] = int(s_mask.sum())
    gone = base | b_mask | c_mask | s_mask
    # presence correction, two-sided, on the stoppage runs only (never on live frames)
    target, tol = p["present_target"], p["present_tol"]
    visible = float((bvis & ~gone).mean())
    if visible > target + tol:                        # too much ball: add runs to stoppages not yet covered
        todo = [sp for sp in eligible if sp not in a_runs]
        for i in rng.permutation(len(todo)):
            if visible <= target + tol:
                break
            s, e = todo[i]
            a, b = _draw_run(rng, p, s, e)
            a_runs[(s, e)] = (a, b, _refind(rng, p, a, b))
            gone = base | build_a() | b_mask | c_mask | s_mask
            visible = float((bvis & ~gone).mean())
    elif visible < target - tol:                      # too little: give runs back
        keys = list(a_runs)
        for i in rng.permutation(len(keys)):
            if visible >= target - tol:
                break
            del a_runs[keys[int(i)]]
            gone = moved | build_a() | mask_d | b_mask | c_mask | s_mask
            visible = float((bvis & ~gone).mean())
    info["A"] = len(a_runs)
    info["pA_effective"] = len(a_runs) / max(len(eligible), 1)
    info["eligible_stoppages"] = len(eligible)
    return gone, info


def _occlusion(vis, crowded, ratio, h0, p, rng):
    """(N, 22) bool occlusion gaps: hazard grows with crowding and with distance from the camera."""
    n, pl = vis.shape
    h = h0 * (1.0 + p["occ_crowd_a"] * crowded) * ratio ** p["occ_far_beta"]
    h = np.where(vis, h, 0.0)
    st = np.argwhere(rng.random((n, pl)) < h)
    k = len(st)
    if k == 0:
        return np.zeros((n, pl), bool)
    long = rng.random(k) < p["occ_long_share"]
    ln = np.where(long, np.minimum(_lognormal(rng, p["occ_long_median_s"], p["occ_long_sigma"], k), p["occ_long_max_s"]),
                  np.minimum(_lognormal(rng, p["occ_median_s"], p["occ_sigma"], k), p["occ_trunc_s"]))
    if p["occ_far_len_beta"] > 0:                       # v6: a far player, once lost, stays lost longer
        ln = ln * ratio[st[:, 0], st[:, 1]] ** p["occ_far_len_beta"]
    d = np.zeros((n + 1, pl), np.int32)
    np.add.at(d, (st[:, 0], st[:, 1]), 1)
    np.add.at(d, (np.minimum(st[:, 0] + np.maximum(1, (ln * FPS).astype(int)), n), st[:, 1]), -1)
    return np.cumsum(d[:-1], axis=0) > 0


def _fov(xy, vis, dball, p, rng):
    """(N, 22) bool: players outside a ball-following window along x are missing for the stretch they
    stay outside (plus a re-acquisition delay). Off when the window is as wide as the pitch."""
    n, pl = vis.shape
    out = np.zeros((n, pl), bool)
    if p["fov_width_m"] >= PITCH_X:
        return out
    bx = dball[:, 0].copy()
    ok = ~np.isnan(bx)
    if not ok.any():
        return out
    bx = np.interp(np.arange(n), np.flatnonzero(ok), bx[ok])
    centre = _ema(bx, p["fov_tau_s"] * FPS)
    steps = int(np.ceil(n / (10 * FPS)))
    bias = np.repeat(rng.normal(0, p["fov_bias_m"], steps), 10 * FPS)[:n]
    centre = centre + _ema(bias, 10 * FPS)
    outside = np.abs(xy[..., 0] - centre[:, None]) > p["fov_width_m"] / 2
    for s in range(pl):
        for a, b in _runs(outside[:, s] & vis[:, s]):
            if rng.random() < p["fov_p_out"]:
                out[a:min(n, b + _frames(_lognormal(rng, p["reacq_median_s"], p["reacq_sigma"]))), s] = True
    return out


def _blackouts(xy, vis, dead_mask, p, seed, scale=1.0, ratio=None):
    """(N, 22) bool: now and then a seed player and a cluster of nearby players disappear together.
    The uniform draws that place events are fixed by `seed`, and each event draws its own details
    from a stream keyed by (seed, frame), so raising `scale` only ADDS events: the mean count can be
    corrected smoothly without reshuffling the ones already there."""
    n, pl = vis.shape
    black = np.zeros((n, pl), bool)
    u = np.random.default_rng(seed).random(n)
    rate = scale * p["blackout_per_min"] / 60.0 / FPS * np.where(dead_mask, p["blackout_dead_mult"], 1.0)
    for f in np.flatnonzero(u < rate):
        rng = np.random.default_rng((seed, int(f)))
        present = np.flatnonzero(vis[f])
        if len(present) == 0:
            continue
        if p["blackout_far_beta"] > 0 and ratio is not None:      # v6: clusters far from the camera more often
            w = ratio[f, present] ** p["blackout_far_beta"]
            seed_player = int(rng.choice(present, p=w / w.sum()))
        else:
            seed_player = int(rng.choice(present))
        q = rng.uniform(p["blackout_q_lo"], p["blackout_q_hi"])
        k = 1 + int(rng.binomial(21, q))
        d = np.linalg.norm(xy[f] - xy[f, seed_player], axis=-1)
        order = np.argsort(np.where(np.isnan(d), np.inf, d))[:k]
        length = _frames(float(np.clip(_lognormal(rng, p["blackout_median_s"], p["blackout_sigma"]), 2.0, 60.0)))
        for s in order:
            a = f + _frames(rng.uniform(0, p["blackout_offset_s"]))
            black[a:a + length, s] = True
    return black


def _fragments(s, vis_s, present_s, crowded_s, p, rng, interp):
    """Track fragments of clean player s: (start, end, s, presence) tuples, plus zombie starts.

    A dropout shorter than the buffer keeps the same track with probability p_relink (and, up to
    12 frames, is bridged by the true positions with probability interp_p); a longer one always
    starts a new track. Silent identity breaks happen without any dropout, more in crowds."""
    n = len(vis_s)
    present = present_s.copy()
    relinked = np.zeros(n, bool)
    lost = vis_s & ~present
    buffer = _frames(p["buffer_s"])
    for a, b in _runs(lost):
        if a == 0 or b >= n or not present[a - 1] or not present[b]:
            continue
        if b - a < buffer and rng.random() < p["p_relink"]:
            relinked[b] = True
            if b - a <= p["interp_gap_frames"] and rng.random() < p["interp_p"]:
                present[a:b] = True
                interp[a:b, s] = True
        elif p["relink_long_p"] > 0 and b - a <= _frames(p["relink_max_s"]) and rng.random() < p["relink_long_p"]:
            relinked[b] = True                         # v6: the tracker keeps the identity across the dropout
    resume = present & ~np.r_[False, present[:-1]]
    hz = np.where(crowded_s, 1.0 / (p["silent_crowd_s"] * FPS), 1.0 / (p["silent_open_s"] * FPS))
    cut = present & (rng.random(n) < hz) & ~resume
    starts = np.flatnonzero((resume & ~relinked) | cut)
    zombies = [int(f) for f in np.flatnonzero(cut & crowded_s) if rng.random() < p["zombie_p"]]
    frags = []
    if starts.size:
        ends = np.r_[starts[1:], n]
        for a, b in zip(starts, ends):
            seg = present[a:b]
            if not seg.any():
                continue
            last = a + int(np.flatnonzero(seg)[-1]) + 1
            frags.append((int(a), last, s, seg[: last - a]))
    return frags, zombies


def _ghost_objects(xy, vis, crowded, dead_mask, count, cam, zombies, p, rng):
    """Extra tracked objects: [(start, end, block, clean column or -2, positions (len, 2), kind)]."""
    n = len(vis)
    out = []
    bench_block = int(rng.integers(0, 2))
    # touchline bursts: bystanders picked up as players, mostly in stoppages
    rate = p["ghost_per_min"] / 60.0 / FPS * np.where(dead_mask, p["ghost_dead_mult"], 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        centroid_x = np.nanmean(np.where(vis, xy[..., 0], np.nan), axis=1)
    for f in np.flatnonzero(rng.random(n) < rate):
        k = 1 + int(rng.random() < p["ghost_pair_share"])      # real graded frames hold at most 25 objects: strays come one or two at a time
        cx = centroid_x[f] if np.isfinite(centroid_x[f]) else 0.0
        for _ in range(k):
            side = cam if rng.random() < 0.7 else -cam
            life = min(_lognormal(rng, p["ghost_median_s"], p["ghost_sigma"]), p["ghost_max_s"])
            length = max(2, _frames(life))
            b = min(n, f + length)
            t = np.arange(b - f)
            x0 = float(np.clip(cx + rng.uniform(-35, 35), -PITCH_X / 2 - 5, PITCH_X / 2 + 5))
            y0 = side * (PITCH_Y / 2 + rng.uniform(0.0, 5.0))
            if rng.random() < 0.6:                       # jogging along the line, reversing now and then
                v = rng.uniform(1.5, 3.0) * rng.choice([-1.0, 1.0])
                vel = np.full(len(t), v)
                for c in np.cumsum(_frames(1) * rng.integers(10, 21, size=max(1, len(t) // (10 * FPS) + 2))):
                    if c < len(t):
                        vel[c:] *= -1.0
                px = x0 + np.cumsum(vel) / FPS
                py = np.full(len(t), y0)
            else:                                        # standing about
                px = x0 + rng.normal(0, 0.3, len(t))
                py = y0 + rng.normal(0, 0.3, len(t))
            block = bench_block if side == cam else 1 - bench_block
            out.append((int(f), int(b), block * SLOTS_PER_TEAM, GHOST, np.stack([px, py], axis=1), "ghost"))
    # duplicate boxes on crowded players
    hz = np.where(crowded & vis, p["dup_h"], 0.0)
    for f, s in np.argwhere(rng.random(hz.shape) < hz):
        length = max(2, int(rng.geometric(1.0 / (p["dup_mean_s"] * FPS))))
        b = min(n, f + length)
        pos = xy[f:b, s] + rng.normal(0, p["dup_offset_m"], 2)
        out.append((int(f), int(b), (0 if s < 11 else 1) * SLOTS_PER_TEAM, int(s), pos, "dup"))
    # zombies: an old track keeps coasting for a moment after the tracker started a new one
    for f, s in zombies:
        length = max(2, _frames(rng.uniform(p["zombie_min_s"], p["zombie_max_s"])))
        b = min(n, f + length)
        if f < 3 or not np.isfinite(xy[f - 1, s]).all() or not np.isfinite(xy[f - 3, s]).all():
            continue
        v = (xy[f - 1, s] - xy[f - 3, s]) * FPS / 2.0
        pos = xy[f - 1, s] + v[None, :] * (np.arange(1, b - f + 1)[:, None] / FPS)
        out.append((int(f), int(b), (0 if s < 11 else 1) * SLOTS_PER_TEAM, GHOST, pos, "zombie"))
    return out


def damage(clean, seed, params=None):
    """Apply the external-data v5bf simulator to a clean 25 Hz practice match."""
    options = {} if params is None else dict(params)
    if set(options) - {"randomize"}:
        raise ValueError("The release implements v5bf; randomize is its only switch.")
    p = dict(DEFAULTS, **options)
    kids = np.random.SeedSequence(int(seed)).spawn(12)
    streams = [np.random.default_rng(c) for c in kids]
    r_par, r_flip, r_drift, r_ball, r_bpos, r_occ, r_fov, r_black, r_brk, r_ghost, r_swap, r_noise = streams
    p, drawn = draw_params(r_par, p)

    n = clean["xy"].shape[0]
    xy = clean["xy"].astype(np.float32) / 100.0
    vis = clean["vis"].copy()
    xy[~vis] = np.nan
    ball = clean["ball"].astype(np.float64) / 100.0
    bvis = clean["ball_vis"].copy()
    ball[~bvis] = np.nan
    label = clean["label"].copy()
    period = clean["period"]
    clean_count = vis.sum(axis=1)

    # 1. flip home/away
    flip = bool(r_flip.random() < p["flip_prob"])
    if flip:
        order = np.r_[np.arange(11, 22), np.arange(0, 11)]
        xy, vis = xy[:, order], vis[:, order]
        label = np.where(label == 0, 1, np.where(label == 1, 0, label)).astype(np.int8)

    # 2. teams swap ends in period 2 (clean data is normalised, raw data is not)
    mirror = (period == 2)
    xy[mirror] *= -1
    ball[mirror, :2] *= -1

    ball_src, bvis_src = ball.copy(), bvis.copy()
    # 3. ball: fill the export's own short blips on a local copy, then decide where it goes missing
    ball, bvis = L.fill_short_ball_gaps(ball, bvis, _frames(p["fill_s"]))
    ball = np.where(bvis[:, None], ball, np.nan)
    dead = _runs(label == 2)
    proxy = False
    if not dead:
        dead, proxy = dead_proxy(ball, bvis), True
    cam_side = 1.0 if np.random.default_rng(kids[5]).random() < 0.5 else -1.0   # r_occ's first draw, read early
    far = cam_side * np.nan_to_num(ball[:, 1]) < -p["far_y_m"]
    gone, binfo = ball_episodes(ball, bvis, label, xy, dead, p, r_ball, far)
    visible_ball = bvis & ~gone
    # v5 drift streams (the spare stream: nothing else moves); ball first, then the 22 players
    ball_drift = _drift(r_drift, (n, 2), p["ball_drift_m"], p["ball_drift_tau_s"] * FPS)
    player_drift = _drift(r_drift, (n, 22, 2), p["player_drift_m"], p["player_drift_tau_s"] * FPS)
    dball = ball[:, :2] + ball_drift + r_bpos.normal(0, p["ball_noise_m"], (n, 2))
    dball[~visible_ball] = np.nan
    near_d, near_k = _nearest(xy, ball)
    contact = near_d < p["ball_contact_m"]
    stick = visible_ball & (r_bpos.random(n) < np.where(contact, p["ball_stick_rate"], p["ball_stick_other"]))
    for f in np.flatnonzero(stick & (near_k >= 0)):
        dball[f] = xy[f, near_k[f]]
    # v5: the tracker's output filter smooths everything it exports, its glitches included
    dball = _smooth_runs(dball, visible_ball, p["ball_smooth_s"] * FPS, causal=bool(p["ball_smooth_causal"]))
    dead_mask = np.zeros(n, bool)
    for a, b in dead:
        dead_mask[a:b] = True

    # 4. players: correlated dropout
    crowded = _crowded(xy, vis, p["crowd_radius_m"])
    cam = 1.0 if r_occ.random() < 0.5 else -1.0
    cam_y = cam * (PITCH_Y / 2 + p["camera_setback_m"])
    dist = np.hypot(xy[..., 0], xy[..., 1] - cam_y)
    ratio = np.clip(np.nan_to_num(dist, nan=p["camera_setback_m"]) / p["camera_setback_m"], 1.0, None)
    fov = _fov(xy, vis, dball, p, r_fov)
    target_deficit = float(clean_count.mean()) - p["mean_count_target"]
    corrected = clean_count.mean() >= 21.8             # complete sources only; incomplete ones keep their own deficit
    occ_seed = int(r_occ.integers(0, 2 ** 62))
    black_seed = int(r_black.integers(0, 2 ** 62))
    occ = _occlusion(vis, crowded, ratio, p["occ_h0"], p, np.random.default_rng(occ_seed))
    scale = 1.0
    for _ in range(8):                                  # the blackout rate carries the correction (clusters, not scatter)
        black = _blackouts(xy, vis, dead_mask, p, black_seed, scale, ratio)
        present = vis & ~occ & ~fov & ~black
        deficit = float(clean_count.mean() - present.sum(axis=1).mean())
        if not corrected or abs(deficit - target_deficit) < 0.1:
            break
        scale = float(np.clip(scale * (1.0 + 0.8 * (target_deficit - deficit) / max(deficit, 0.3)), 0.3, 4.0))
    p_occ_h0 = scale                                    # recorded as the blackout scale actually used

    # 5. tracks: fragments, then slots handed out in time order, like a tracker spawning track ids
    interp = np.zeros((n, 22), bool)
    fragments, zombies = [], []
    for s in range(22):
        f, z = _fragments(s, vis[:, s], present[:, s], crowded[:, s], p, r_brk, interp)
        fragments += f
        zombies += [(zf, s) for zf in z]
    intervals = [[] for _ in range(N_SLOTS)]
    src_slot = np.full((n, 22), -1, np.int16)
    count = np.zeros(n, np.int16)
    n_frag = 0
    for a, last, s, seg in sorted(fragments, key=lambda f: (f[0], f[2])):
        block = 0 if s < 11 else SLOTS_PER_TEAM
        free = [k for k in range(block, block + SLOTS_PER_TEAM) if all(e <= a for _, e in intervals[k][-1:])]
        if not free:
            continue                                   # no free slot: this fragment is lost (rare)
        slot = int(r_brk.choice(free))
        intervals[slot].append((a, last))
        src_slot[a:last, s] = np.where(seg, slot, -1)
        count[a:last] += seg
        n_frag += 1

    # 6. identity swaps between teammates
    n_swap = 0
    for team in (0, 1):
        cols = range(11 * team, 11 * team + 11)
        for i in cols:
            for j in cols:
                if j <= i:
                    continue
                d = xy[:, i] - xy[:, j]
                close = ((d * d).sum(-1) < p["swap_radius_m"] ** 2)
                close &= (src_slot[:, i] >= 0) & (src_slot[:, j] >= 0)
                for f in np.flatnonzero(close & ~np.r_[False, close[:-1]]):
                    if r_swap.random() >= p["swap_prob_per_encounter"]:
                        continue
                    si, sj = src_slot[f, i], src_slot[f, j]
                    ei = f + np.argmax(np.r_[src_slot[f:, i] != si, True])
                    ej = f + np.argmax(np.r_[src_slot[f:, j] != sj, True])
                    e = min(ei, ej)
                    src_slot[f:e, i], src_slot[f:e, j] = sj, si
                    n_swap += 1

    # 7. write the player slots
    dxy = np.full((n, N_SLOTS, 2), np.nan, np.float32)
    slot_src = np.full((n, N_SLOTS), -1, np.int8)
    rows, cols = np.nonzero(src_slot >= 0)
    noise = r_noise.normal(0, p["player_noise_m"], (rows.size, 2))
    noise[interp[rows, cols]] = 0.0                     # a bridged gap follows the true track
    dxy[rows, src_slot[rows, cols]] = xy[rows, cols] + noise + player_drift[rows, cols]
    slot_src[rows, src_slot[rows, cols]] = order[cols] if flip else cols

    # 8. ghosts and duplicates: more objects than players, in free slots, never above 25 at once
    extras = _ghost_objects(xy, vis, crowded, dead_mask, count, cam, zombies, p, r_ghost)
    n_ghost = n_dup = n_zomb = 0
    for a, b, block, src, pos, kind in sorted(extras, key=lambda e: e[0]):
        if (count[a:b] + 1).max() >= p["max_objects"] + 1:
            continue
        free = [k for k in range(block, block + SLOTS_PER_TEAM)
                if all(e <= a or s >= b for s, e in intervals[k])]
        if not free:
            continue
        slot = int(r_ghost.choice(free))
        intervals[slot].append((a, b))
        ok = np.isfinite(pos).all(axis=1)
        rr = np.arange(a, b)[ok]
        if not len(rr):
            continue
        dxy[rr, slot] = pos[ok]
        slot_src[rr, slot] = (order[src] if flip else src) if src >= 0 else GHOST
        count[rr] += 1
        n_ghost += kind == "ghost"
        n_dup += kind == "dup"
        n_zomb += kind == "zombie"

    def to_cm(a):
        out = np.round(a * 100.0)
        out = np.where(np.isnan(out), MISSING, np.clip(out, -32000, 32000))
        return out.astype(np.int16)

    dmg_vis = slot_src != -1
    fp = FP.fingerprint_arrays(~np.isnan(dball[:, 0]), dmg_vis.sum(axis=1), clean_count)
    inplay = label != 2
    stats = dict(
        seed=int(seed), flip=flip, n_frames=int(n), damage_version=int(p["damage_version"]), draw=drawn,
        ball_missing_clean=float((~clean["ball_vis"]).mean()),
        ball_missing_damaged=float(np.isnan(dball[:, 0]).mean()),
        ball_missing_damaged_in_play=float(np.isnan(dball[inplay, 0]).mean()) if inplay.any() else None,
        ball_missing_damaged_dead=float(np.isnan(dball[~inplay, 0]).mean()) if (~inplay).any() else None,
        dead_proxy=proxy, ball_runs=binfo, blackout_scale=float(p_occ_h0),
        player_frames_clean=int(vis.sum()), player_frames_damaged=int((slot_src >= 0).sum()),
        fragments=int(n_frag), swaps=int(n_swap), ghosts=int(n_ghost), duplicates=int(n_dup), zombies=int(n_zomb),
        fingerprint=fp,
        params={k: (v if isinstance(v, (bool, str)) else float(v)) for k, v in p.items()},
    )
    return dict(
        dmg_xy=to_cm(dxy),
        dmg_vis=dmg_vis,
        dmg_slot_src=slot_src,
        dmg_ball=to_cm(dball),
        dmg_ball_vis=~np.isnan(dball[:, 0]),
        dmg_label=label.astype(np.int8),
        dmg_flip=np.array(flip),
        dmg_stats=np.array(json.dumps(stats)),
    ), stats
