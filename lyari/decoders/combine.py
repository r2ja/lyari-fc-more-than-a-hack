"""Ways of combining several models' H/A/D probabilities into one. numpy only."""

import numpy as np

EPS = 1e-6


def combine(probs, how="mean"):
    """probs: list of (N, 3) arrays (or an (M, N, 3) array) -> (N, 3)."""
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 3 or p.shape[-1] != 3:
        raise ValueError("expected (M, N, 3), got {}".format(p.shape))
    if how == "mean":
        return p.mean(0)
    if how == "geometric":
        logp = np.log(np.clip(p, EPS, 1.0)).mean(0)
        out = np.exp(logp - logp.max(1, keepdims=True))
        out /= out.sum(1, keepdims=True)
        out[np.isnan(p).any(0).any(1)] = np.nan
        return out
    raise ValueError("unknown combination {!r}".format(how))
