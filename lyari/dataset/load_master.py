"""Read one master-dataset match. numpy only.

    from load_master import load_match
    m = load_match("master/matches/sportec_DFL-MAT-J03WMX.npz")
    m["clean"]["xy"]      # (N, 22, 2) float32 metres, NaN when absent
    m["damaged"]["xy"]    # (N, 46, 2) float32 metres, NaN when absent  <- what the competition looks like
    m["label"]            # (N,) int8  0 H, 1 A, 2 D, -1 unknown         (clean sides)
    m["damaged"]["label"] # same, but swapped if the damaged copy flipped home and away

    python load_master.py master/matches/sportec_DFL-MAT-J03WMX.npz    # prints a summary
"""
import json
import sys

import numpy as np

MISSING = -32768


def _m(a):
    out = a.astype(np.float32) / 100.0
    out[a == MISSING] = np.nan
    return out


def load_match(path):
    with np.load(path) as f:
        meta = json.loads(str(f["meta"]))
        return dict(
            meta=meta, frame=f["frame"], period=f["period"], t=f["t"], label=f["label"],
            label_source=f["label_source"] if "label_source" in f else f["label"],   # v2: the source's own label
            clean=dict(xy=_m(f["clean_xy"]), vis=f["clean_vis"], seen=f["clean_seen"], team=f["clean_team"],
                       ball=_m(f["clean_ball"]), ball_vis=f["clean_ball_vis"]),
            damaged=dict(xy=_m(f["dmg_xy"]), vis=f["dmg_vis"], slot_src=f["dmg_slot_src"],
                         team=np.array([0] * 23 + [1] * 23, np.int8),
                         ball=_m(f["dmg_ball"]), ball_vis=f["dmg_ball_vis"], label=f["dmg_label"],
                         flip=bool(f["dmg_flip"]), stats=json.loads(str(f["dmg_stats"]))),
        )


def speeds(xy, fps=25):
    """(N, S) speed in m/s from positions, NaN where either frame is missing."""
    v = np.full(xy.shape[:2], np.nan, np.float32)
    d = np.diff(xy, axis=0)
    v[1:] = np.sqrt((d * d).sum(-1)) * fps
    return v


if __name__ == "__main__":
    m = load_match(sys.argv[1])
    meta = m["meta"]
    lab = m["label"]
    print(f"{meta['source']} {meta['match_id']}: {meta['home']} v {meta['away']}  {meta['n_frames']} frames @ {meta['fps']} fps")
    print("labels", meta["label_counts"])
    print("clean ball present %.2f, damaged ball present %.2f" % (m["clean"]["ball_vis"].mean(), m["damaged"]["ball_vis"].mean()))
    print("clean players/frame %.1f, damaged players/frame %.1f" % (m["clean"]["vis"].sum(1).mean(), m["damaged"]["vis"].sum(1).mean()))
    st = m["damaged"]["stats"]
    print("damage: flip", st["flip"], "fragments", st["fragments"], "swaps", st["swaps"])
