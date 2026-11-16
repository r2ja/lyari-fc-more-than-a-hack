"""Load probabilities aligned to the practice reference."""
import numpy as np
from evaluation import possession_metrics as pm
from lyari.scoring import score as S

CODES = np.array([pm.H, pm.A, pm.D], np.int8)


def probs_of(path, n_expected):
    """Any prediction file -> (N, 3) probabilities (labels become one-hot, X becomes NaN)."""
    codes, reason = S.read_prediction(path, n_expected, decoder=lambda p: p)
    if reason:
        raise ValueError("{}: {}".format(path, reason))
    if codes.ndim == 2:
        return codes
    probs = np.zeros((len(codes), 3))
    alive = codes < pm.X
    probs[np.flatnonzero(alive), codes[alive]] = 1.0
    probs[~alive] = np.nan
    return probs
