"""Build the v5bf tracking arrays and possession labels for one practice match."""
import json
import numpy as np
from lyari.dataset import labels as L
from lyari.dataset.damage import damage

VERSION = 5


def upgrade_arrays(arrays, min_hold=L.MIN_HOLD, damage_params=None, seed_offset=0):
    """v1 arrays (as written by build_master_dataset.pack) -> (v2 arrays, meta, damage stats)."""
    meta = json.loads(str(arrays["meta"]))
    source_label = arrays["label_source"] if "label_source" in arrays else arrays["label"]
    base = source_label
    if meta.get("source") == "metrica":   # v3: event-derived labels miss stoppages the tracking shows
        base = L.stoppage_gaps(source_label, arrays["clean_ball_vis"])
        meta["label_fix"] = "metrica: live frames in source ball gaps >= {} frames -> D".format(L.STOPPAGE_GAP)
    label = L.harmonise(base, min_hold)
    clean = dict(xy=arrays["clean_xy"], vis=arrays["clean_vis"], ball=arrays["clean_ball"],
                 ball_vis=arrays["clean_ball_vis"], label=label, period=arrays["period"])
    dmg, dstats = damage(clean, int(meta["damage_seed"]) + int(seed_offset), damage_params)
    rate, mean_s, dead = L.possession_stats(label)
    src_rate, src_mean, _ = L.possession_stats(source_label)
    meta.update(version=VERSION, label_rule="control: spells under {} frames that interrupt the other team revert".format(min_hold),
                label_counts={"H": int((label == 0).sum()), "A": int((label == 1).sum()), "D": int((label == 2).sum()),
                              "unknown": int((label == -1).sum())},
                possession=dict(per_live_minute=round(rate, 2), mean_spell_s=round(mean_s, 1), dead_share=round(dead, 3),
                                source_per_live_minute=round(src_rate, 2), source_mean_spell_s=round(src_mean, 1)))
    out = {k: v for k, v in arrays.items() if not k.startswith("dmg_")}
    out.update(dmg)
    out["label"] = label
    out["label_source"] = source_label.astype(np.int8)
    out["meta"] = np.array(json.dumps(meta))
    return out, meta, dstats
