"""Generate Lyari FC's frame-level possession file using the ten Entry D checkpoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

import games
from lyari import io
from lyari.dataset.load_master import load_match
from lyari.decoders.combine import combine
from lyari.decoders import viterbi
from lyari.models.players import features
from lyari.models.players.predict import load_model
from lyari.models.players.train import predict_match, repaired_ball
from lyari.pipeline import epts, quirks, submit

ROOT = Path(__file__).resolve().parents[1]
# The second half's graded period begins before its kickoff, as specified in the organiser FAQ.
SECOND_KICKOFF = 113742


def load_manifest():
    manifest = json.loads((ROOT / "weights/manifest.json").read_text(encoding="utf-8"))
    for member in manifest["models"]:
        path = ROOT / "weights" / member["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != member["sha256"]:
            raise ValueError(f"Checkpoint checksum mismatch for {member['name']}")
    return manifest


def predict_member(path, tracking):
    model, config = load_model(path)
    data = dict(tracking)
    if config.get("ball_repair"):
        data["ball"] = repaired_ball(data["ball"])
    frames = features.frame_features(data, use_ball=config["use_ball"], ball_rel=config.get("ball_rel", False))
    torch.manual_seed(0)
    probs, _ = predict_match(model, frames, torch.device("cpu"), boundary=True)
    return probs.astype(np.float64)


def run(data, work, output, manifest):
    work.mkdir(parents=True, exist_ok=True)
    meta, raw = games.require(["atd_meta", "atd_raw"], data_dir=data)
    rows = games.GAMES[games.GAME]["n_video_frames"]
    path = submit.translate(meta, raw, games.half_video_blocks(games.GAME), rows, str(work))
    match = load_match(path)
    period = match["period"]
    tracking = match["damaged"]
    if manifest["clean_tracking"]:
        tracking = quirks.clean(tracking, period=period)
    probs = combine([predict_member(ROOT / "weights" / member["file"], tracking)
                     for member in manifest["models"]], "mean")
    decoder = manifest["decoder"]
    codes = np.full(len(period), 3, dtype=np.int64)
    for half in (1, 2):
        indices = np.flatnonzero(period == half)
        start, end = int(indices[0]), int(indices[-1]) + 1
        if half == 2 and start < SECOND_KICKOFF < end:
            codes[start:SECOND_KICKOFF] = 2
            start = SECOND_KICKOFF
        codes[start:end] = viterbi.decode(probs[start:end], decoder["switch"], decoder["dead"], rule="soft")
    io.write_labels(str(output), codes)
    submit.validate(str(output), rows)
    return submit.sha256(str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="root of the organisers' bundle, the folder containing tactical/ (or set MTAH_DATA_DIR)")
    parser.add_argument("--work", type=Path, default=ROOT.parent / "runs" / "final", help="local work folder outside this checkout")
    parser.add_argument("--check", action="store_true", help="run the complete pipeline twice and compare file bytes")
    args = parser.parse_args()
    work = args.work.resolve()
    if epts.inside_repo(str(work)):
        parser.error("--work must be outside this repository")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest = load_manifest()
    output = Path(submit.check_out(submit.default_out()))
    digest = run(args.data, work / "run1", output, manifest)
    if args.check:
        again = run(args.data, work / "run2", work / "run2" / output.name, manifest)
        if again != digest:
            raise SystemExit("Reproducibility check failed: the two output checksums differ.")
        print("Two runs produced identical bytes.")
    print(f"sha256 {digest}  {output}")


if __name__ == "__main__":
    main()
