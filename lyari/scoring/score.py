"""Read practice predictions and external possession references."""
import glob
import json
import os

import numpy as np
import pandas as pd
from evaluation import possession_metrics as pm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE_OF_MASTER = np.array([pm.H, pm.A, pm.D, pm.X], np.int8)
PROB_COLUMNS = ("p_h", "p_a", "p_d")


def refuse_public(match):
    if match == "BAR-ATH":
        raise ValueError("This analysis accepts external practice matches only.")


def master_dir():
    base = os.environ.get("LYARI_DATASETS_DIR") or os.path.join(os.path.dirname(ROOT), "datasets")
    return os.path.join(base, "master")


def master_matches(split=None):
    """Match names in the master dataset, optionally only those of one split ('train' / 'test')."""
    folder = master_dir()
    names = sorted(os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(os.path.join(folder, "matches", "*.npz")))
    if split:
        with open(os.path.join(folder, "split.json")) as handle:
            members = {os.path.splitext(os.path.basename(f))[0] for f in json.load(handle)[split]}
        names = [n for n in names if n in members]
    return names


def truth(match, which="damaged"):
    """(codes, cut) for one match: organiser label codes, and the row index where half 2 starts.

    BAR-ATH: the public CSV as is (X outside the window). Master: dmg_label or label, with -1 -> X.
    `cut` is None for BAR-ATH, whose X-filled halftime already separates the halves.
    """
    refuse_public(match)
    path = os.path.join(master_dir(), "matches", match + ".npz")
    if not os.path.isfile(path):
        raise IOError("no master match {} (looked in {})".format(match, os.path.dirname(path)))
    with np.load(path) as f:
        period = f["period"]
        if which == "damaged":
            label = f["dmg_label"]
        elif which == "clean":
            label = f["label"]
        else:  # "source": the provider's own label before the v2 control rule, in the damaged copy's orientation
            label = (f["label_source"] if "label_source" in f else f["label"]).copy()
            if bool(f["dmg_flip"]):
                h, a = label == 0, label == 1
                label[h], label[a] = 1, 0
    starts = np.flatnonzero(np.diff(period) != 0) + 1
    if len(starts) != 1:
        raise ValueError("{}: expected two periods, found boundaries at {}".format(match, starts))
    return CODE_OF_MASTER[label.astype(np.int64)], int(starts[0])


def with_halftime(codes, cut):
    """Insert one X between the halves. The scorer splits halves on X (valid_runs), and the master
    files keep only in-play halves back to back, so without it a possession could cross halftime."""
    if cut is None:
        return codes
    return np.concatenate((codes[:cut], [pm.X], codes[cut:])).astype(np.int8)


def argmax_decoder(probs):
    """Per-frame most likely label. Ties go H > A > D (argmax order). NaN rows -> X."""
    out = np.array([pm.H, pm.A, pm.D], np.int8)[np.argmax(np.nan_to_num(probs, nan=-1.0), axis=1)]
    out[np.isnan(probs).any(axis=1)] = pm.X
    return out


def read_prediction(path, n_expected, decoder=argmax_decoder):
    """(codes, reason). Accepts frame,label or frame,p_H,p_A,p_D. Exactly one of the two is None.

    Strict in the same way as the organisers' validate_submission: every frame 0..n-1 in order,
    no blanks, nothing outside the allowed values.
    """
    try:
        table = pd.read_csv(path)
    except Exception as error:
        return None, "could not read: {}".format(error)
    table.columns = [str(c).strip().lower() for c in table.columns]
    if "frame" not in table.columns:
        return None, "no 'frame' column (got {})".format(list(table.columns))

    if "label" in table.columns:
        reason = pm.validate_submission(table, n_expected)
        return (None, reason) if reason else (pm.label_codes(table), None)

    if not all(c in table.columns for c in PROB_COLUMNS):
        return None, "need 'label' or 'p_H,p_A,p_D' columns (got {})".format(list(table.columns))
    if len(table) != n_expected:
        return None, "row count {} != {}".format(len(table), n_expected)
    if not np.array_equal(table["frame"].values, np.arange(n_expected)):
        return None, "'frame' column is not 0..{} in order".format(n_expected - 1)
    probs = table[list(PROB_COLUMNS)].to_numpy(dtype=np.float64)
    finite = probs[~np.isnan(probs).any(axis=1)]
    if (finite < 0).any() or (finite > 1).any():
        return None, "probabilities outside [0, 1]"
    return decoder(probs), None
