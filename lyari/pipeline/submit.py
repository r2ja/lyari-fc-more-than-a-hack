"""EPTS translation and the organisers’ submission-format check."""
import hashlib
import os
import subprocess
import sys
from lyari.pipeline import epts

REPO = epts.REPO
TEAM = "Lyari FC"
RESULTS = os.path.join(REPO, "evaluation", "results")


def halves_from_blocks(blocks):
    """games.half_video_blocks rows (0-based video frames) -> EPTS frame-id halves (1-based)."""
    return tuple((first + 1, last + 1) for _, first, last in blocks)


def check_halves(meta_halves, halves):
    """The metadata's half markers, when it has them, must match the organisers' blocks."""
    if meta_halves is None:
        return
    got = tuple(tuple(int(v) for v in h) for h in meta_halves)
    if len(got) == 1 and got[0][0] <= halves[0][0] and got[0][1] >= halves[-1][1]:

        print("translate: metadata has one period {} covering both halves; halves from games.py".format(got[0]))
        return
    if got != tuple(halves):
        raise ValueError("half markers in the metadata {} differ from games.py's {}; the frame "
                         "alignment is in doubt, stopping".format(got, tuple(halves)))


def translate(meta, raw, blocks, n_frames, work):
    halves = halves_from_blocks(blocks)
    check_halves(epts.read_metadata(meta)["halves"], halves)
    arrays = epts.translate(meta, raw, halves)
    if len(arrays["frame"]) != n_frames:
        raise ValueError("tracking has {} frames, the submission needs {}".format(
            len(arrays["frame"]), n_frames))
    path = os.path.join(work, "match.npz")
    epts.write(arrays, path)
    print("translate: frames {}  in halves {}  ball present {:.3f}".format(
        len(arrays["frame"]), int((arrays["period"] > 0).sum()), arrays["dmg_ball_vis"].mean()))
    return path


def default_out(team=TEAM):
    if not team or os.path.basename(team) != team or team in (".", ".."):
        raise ValueError("team must be a plain file name, got {!r}".format(team))
    return os.path.join(RESULTS, team + ".csv")


def check_out(path):
    """A submission is frame-level: inside the repo it may only go to the gitignored"""
    path = os.path.abspath(path)
    if epts.inside_repo(path) and os.path.normcase(os.path.dirname(path)) != os.path.normcase(RESULTS):
        raise ValueError("inside the repo the submission may only go to evaluation/results/ "
                         "(gitignored): {}".format(path))
    return path


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def validate(path, rows):
    """The organisers' validate_submission.py on this exact file; raises when it fails."""
    script = os.path.join(REPO, "evaluation", "validate_submission.py")
    path = os.path.abspath(path)  # the validator runs from the repo root
    done = subprocess.run([sys.executable, script, path, "--rows", str(rows)], cwd=REPO,
                          capture_output=True, text=True)
    if done.returncode != 0:
        raise ValueError("validate_submission.py rejected {}:\n{}".format(path, done.stdout + done.stderr))
    return done.stdout
