"""Prepare the 13 open training matches or reproduce the final ten training recipes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from lyari.dataset import providers
from lyari.dataset.prepare import upgrade_arrays


def recipe():
    return json.loads(Path(__file__).with_name("recipe.json").read_text(encoding="utf-8"))


def clean_arrays(name, raw):
    source, game = name.split("_", 1)
    if source == "sportec":
        match = providers.load_sportec(raw, game)
    elif source == "metrica":
        match = providers.load_metrica_csv(raw, int(game.removeprefix("game")))
    else:
        match = providers.load_soccertrack(raw, game)
    filename = name + ".npz"
    seed = (20261020 * 1000003 + sum(ord(char) * (i + 1) for i, char in enumerate(filename))) % (2 ** 31)
    return providers.pack(match, seed)[0]


def prepare(args, config):
    dest = args.data.resolve()
    if dest.exists():
        raise SystemExit("Choose a new output folder for the prepared training copy.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        master = Path(tmp) / "prepared" / "master"
        (master / "matches").mkdir(parents=True)
        matched = {}
        for name in config["matches"]:
            filename = name + ".npz"
            if args.raw is not None:
                arrays = clean_arrays(name, args.raw)
            else:
                with np.load(args.clean_master / "matches" / filename) as old:
                    arrays = {k: old[k] for k in old.files if not k.startswith("dmg_")}
            out, meta, _ = upgrade_arrays(arrays)
            np.savez_compressed(master / "matches" / filename, **out)
            matched[filename] = {"file":filename,"meta":meta}
        (master / "manifest.json").write_text(json.dumps({"complete": True,"damage_overrides":
            {"damage_version":5,"loss_v5":1,"v5be":1,"v5bf":1}, "matches":matched}),encoding="utf-8")
        (master / "split.json").write_text(json.dumps({"train":list(matched),"test":[]}),encoding="utf-8")
        master.parent.rename(dest)


def check_data(root, config):
    master = root / "master"
    split = json.loads((master / "split.json").read_text(encoding="utf-8-sig"))
    manifest = json.loads((master / "manifest.json").read_text(encoding="utf-8-sig"))
    names = config["matches"]
    if (not manifest.get("complete") or manifest.get("damage_overrides") !=
            {"damage_version":5,"loss_v5":1,"v5be":1,"v5bf":1} or
            len(split["train"]) != len(names) or {Path(n).stem for n in split["train"]} != set(names)):
        raise ValueError("Expected the complete v5bf copy and the 13 registered training matches.")
    expected = json.loads(Path(__file__).with_name("v5bf_hashes.json").read_text(encoding="utf-8"))["matches"]
    for name in names:
        path = master / "matches" / (name + ".npz")
        if not path.is_file():
            raise ValueError(f"Missing practice match {name}")
        if content_hash(path) != expected[name]:
            raise ValueError(f"Practice match {name} differs from the v5bf copy the final models were trained on")


def content_hash(path):
    """SHA-256 over every array except dmg_stats (diagnostic text), in sorted key order."""
    digest = hashlib.sha256()
    with np.load(path) as arrays:
        for key in sorted(arrays.files):
            if key == "dmg_stats":
                continue
            value = np.ascontiguousarray(arrays[key])
            digest.update(f"{key}|{value.dtype.str}|{value.shape}|".encode())
            digest.update(value.tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(required=True, dest="action")
    prep = actions.add_parser("prepare", help="build the external v5bf practice copy")
    origins = prep.add_mutually_exclusive_group(required=True)
    origins.add_argument("--raw", type=Path, help="parent of the three provider directories")
    origins.add_argument("--clean-master", type=Path, help="an existing clean practice master folder with matches/")
    prep.add_argument("--data", type=Path, required=True, help="new output folder")
    check = actions.add_parser("check", help="check prepared practice inputs")
    check.add_argument("--data", type=Path, required=True)
    train = actions.add_parser("train", help="train the ten final models, in manifest order")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--out", type=Path, required=True, help="separate directory for new checkpoints")
    args = parser.parse_args()
    config = recipe()
    if args.action == "prepare":
        prepare(args, config)
        return
    check_data(args.data, config)
    if args.action == "check":
        print("Verified the 13 practice inputs: content matches the v5bf copy used for the final models.")
        return
    import torch
    for member, spec in config["members"].items():
        out = args.out.resolve() / member
        out.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-m", "lyari.models.players.train", "--out", str(out), "--seed",
                   str(spec["seed"]), *config["schedule"], *config["shared"], *spec["flags"], "--no-val", "--resume"]
        subprocess.run(command, check=True, env={**os.environ, "LYARI_DATASETS_DIR":str(args.data.resolve())})
        ckpt = torch.load(out / "model.pt", map_location="cpu", weights_only=True)
        if ckpt["epoch"] != 30 or set(ckpt["train"]) != set(config["matches"]) or ckpt["val"]:
            raise RuntimeError(f"Incomplete final checkpoint for {member}")


if __name__ == "__main__":
    main()
