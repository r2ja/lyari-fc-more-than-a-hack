"""The two file formats every stage shares. One place, so no stage invents its own."""

import os

import numpy as np
import pandas as pd

PROB_HEADER = ("frame", "p_H", "p_A", "p_D")
LABELS = np.array(["H", "A", "D", "X"])  # index = the organisers' label code (H 0, A 1, D 2, X 3)


def _ensure_folder(path):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)


def write_probs(path, probs):
    """(N, 3) array of p_H, p_A, p_D -> frame,p_H,p_A,p_D. NaN rows are written as NaN (= X)."""
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != 3:
        raise ValueError("probs must be (N, 3), got {}".format(probs.shape))
    _ensure_folder(path)
    table = pd.DataFrame(probs, columns=PROB_HEADER[1:])
    table.insert(0, "frame", np.arange(len(probs)))
    table.to_csv(path, index=False, float_format="%.6g")


def write_labels(path, labels):
    """Labels as H/A/D/X strings or organiser codes 0..3 -> frame,label."""
    labels = np.asarray(labels)
    if labels.dtype.kind in "iu":
        labels = LABELS[labels]
    _ensure_folder(path)
    # newline="\n": the same bytes on Windows and Linux, so a submission's sha256 is portable
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))


def read_probs(path):
    """frame,p_H,p_A,p_D -> (N, 3) float array. For validated reading use lyari.scoring.score."""
    table = pd.read_csv(path)
    table.columns = [str(c).strip() for c in table.columns]
    return table[list(PROB_HEADER[1:])].to_numpy(dtype=np.float64)
