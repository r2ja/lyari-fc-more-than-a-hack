"""Sample model inputs at a lower rate; return probabilities on the original grid.

Features are computed at 25 Hz BEFORE sampling, so velocities, ball repair and
camera augmentation retain their original physical units and time constants.
"""

import numpy as np


def check_stride(stride):
    if isinstance(stride, (bool, np.bool_)) or not isinstance(stride, (int, np.integer)) or stride < 1:
        raise ValueError("stride must be a positive integer")
    return int(stride)


def _interpolate(values, source_times, output_times):
    """Column-wise linear interpolation, holding the nearest endpoint at the edges."""
    return np.column_stack([np.interp(output_times, source_times, col) for col in values.T]).astype(np.float32)


def _normalise(prob):
    prob = np.clip(prob, 0.0, 1.0)
    return prob / np.maximum(prob.sum(axis=1, keepdims=True), 1e-12)


def predict_strided(predict, model, feats, device, stride=1, phase_average=False, **kwargs):
    """Run the existing predictor on decimated features and restore all original rows.

    Single phase uses offsets 0, stride, 2*stride, ... . With phase averaging, run
    offsets 0..stride-1 and average corresponding coarse bins BEFORE upsampling.
    A full bin is timestamped at its centre (2 for stride5), not its first frame:
    this avoids a two-frame timing bias. A partial final bin uses only available
    phases and their mean timestamp. Edge frames hold the nearest bin's value.

    Boundary displacements are converted from sampled-frame units to original
    frame units. Average/interpolate their absolute target times, then subtract
    each output time; this also handles off-grid frames and endpoint holds.
    Optional path probabilities follow the same grid. No labels are used.
    """
    stride = check_stride(stride)
    if stride == 1:
        return predict(model, feats, device, **kwargs)
    n = len(feats["teams"])
    if any(len(v) != n for v in feats.values()):
        raise ValueError("feature arrays must have identical frame counts")
    if n == 0:
        return predict(model, feats, device, **kwargs)
    phases = range(min(stride, n)) if phase_average else (0,)
    bins = (n + stride - 1) // stride
    count = np.zeros(bins, np.float64)
    time_sum = np.zeros(bins, np.float64)
    totals = None
    is_tuple = False
    for offset in phases:
        times = np.arange(offset, n, stride, dtype=np.float64)
        result = predict(model, {k: v[offset::stride] for k, v in feats.items()}, device, **kwargs)
        is_tuple = isinstance(result, tuple)
        parts = result if is_tuple else (result,)
        if totals is None:
            totals = [None if p is None else np.zeros((bins, p.shape[1]), np.float64) for p in parts]
        m = len(times)
        count[:m] += 1
        time_sum[:m] += times
        for i, part in enumerate(parts):
            if part is None:
                continue
            values = part.astype(np.float64)
            if kwargs.get("boundary", False) and i == 1:
                values[:, 1] = times + values[:, 1] * stride
            totals[i][:m] += values
    source_times = time_sum / count
    output_times = np.arange(n, dtype=np.float64)
    output = []
    for i, total in enumerate(totals):
        if total is None:
            output.append(None)
            continue
        values = _interpolate(total / count[:, None], source_times, output_times)
        if i == 0:
            values = _normalise(values)
        elif kwargs.get("boundary", False) and i == 1:
            values[:, 0] = np.clip(values[:, 0], 0.0, 1.0)
            values[:, 1] -= output_times
        else:  # optional path: conditional player distribution and state distribution
            values[:, :46] = _normalise(values[:, :46])
            values[:, 46:] = _normalise(values[:, 46:])
        output.append(values)
    return tuple(output) if is_tuple else output[0]
