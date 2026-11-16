"""Aggregate diagnostics for simulated practice matches."""
import numpy as np

FPS = 25
BUCKET_EDGES_S = (0.5, 2.0, 10.0)


def check_contract(stats):
    """Require scalar diagnostics or short numeric vectors."""
    for key, value in stats.items():
        if isinstance(value, (bool, int, float, str)) or value is None:
            continue
        if isinstance(value, (list, tuple)) and len(value) <= 24 and all(isinstance(x, (int, float)) for x in value):
            continue
        raise ValueError(f"Unexpected aggregate diagnostic: {key}")


def run_lengths(mask):
    """Lengths (frames) of the maximal runs of True in a 1-D boolean mask."""
    d = np.diff(np.r_[0, np.asarray(mask, np.int8), 0])
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def _quantiles(x, qs=(0.5, 0.9)):
    return [float(np.quantile(x, q)) for q in qs] if len(x) else [float("nan")] * len(qs)


def ball_stats(visible, fps=FPS):
    """Ball visibility aggregates from a per-frame boolean (True = ball present)."""
    visible = np.asarray(visible, bool)
    n = len(visible)
    miss = run_lengths(~visible) / fps
    seen = run_lengths(visible) / fps
    minutes = n / fps / 60.0
    out = dict(frames=int(n), present=float(visible.mean()), missing_runs=int(len(miss)),
               missing_runs_per_min=float(len(miss) / max(minutes, 1e-9)))
    if len(miss):
        med, p90 = _quantiles(miss)
        edges = (0,) + BUCKET_EDGES_S + (1e9,)
        buckets = [float(np.mean((miss >= a) & (miss < b))) for a, b in zip(edges[:-1], edges[1:])]
        out.update(missing_mean_s=float(miss.mean()), missing_median_s=med, missing_p90_s=p90,
                   missing_max_s=float(miss.max()), missing_buckets=buckets)
    if len(seen):
        out.update(seen_mean_s=float(seen.mean()), seen_median_s=float(np.median(seen)))
    return out


def count_stats(count, clean_count=None):
    """Players-per-frame aggregates from an integer count per frame."""
    count = np.asarray(count)
    out = dict(players_mean=float(count.mean()), players_median=float(np.median(count)),
               players_min=int(count.min()), players_max=int(count.max()),
               p_lt22=float((count < 22).mean()), p_lt21=float((count < 21).mean()),
               p_lt16=float((count < 16).mean()), p_gt22=float((count > 22).mean()))
    if clean_count is not None:
        clean_count = np.asarray(clean_count)
        deficit = clean_count - count
        out.update(clean_mean=float(clean_count.mean()), deficit_mean=float(deficit.mean()),
                   p_deficit_ge1=float((deficit >= 1).mean()), p_deficit_ge2=float((deficit >= 2).mean()),
                   p_deficit_ge7=float((deficit >= 7).mean()),
                   retention=float(count.sum() / max(clean_count.sum(), 1)))
    return out


def fingerprint_arrays(ball_visible, count, clean_count=None, fps=FPS):
    out = {}
    out.update(ball_stats(ball_visible, fps))
    out.update(count_stats(count, clean_count))
    check_contract(out)
    return out
