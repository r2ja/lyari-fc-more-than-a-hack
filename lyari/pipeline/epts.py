"""EPTS tracking (the competition's ATD format) -> one .npz in the master-dataset layout."""

import argparse
import io
import json
import os
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from lyari.dataset.load_master import MISSING

FPS = 25
SLOTS_PER_TEAM = 23
HALF_MARKERS = ("first_half_start", "first_half_end", "second_half_start", "second_half_end")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _text(node, path):
    found = node.find(path)
    return None if found is None or found.text is None else found.text.strip()


def _parse(meta_path):
    """XML root with namespaces dropped, so a default xmlns does not hide every tag."""
    root = ET.parse(meta_path).getroot()
    for node in root.iter():
        if isinstance(node.tag, str) and "}" in node.tag:
            node.tag = node.tag.split("}", 1)[1]
    return root


def read_metadata(meta_path):
    """Teams, players, pitch, frame rate, half markers and the line layouts of an EPTS metadata file."""
    root = _parse(meta_path)
    teams = [(t.get("id"), _text(t, "Name") or t.get("id")) for t in root.iter("Team")]
    team_of = {p.get("id"): p.get("teamId") for p in root.iter("Player")}
    channel = {c.get("id"): (c.get("playerId"), c.get("channelId")) for c in root.iter("PlayerChannel")}
    params = {_text(p, "Name"): _text(p, "Value") for p in root.iter("ProviderParameter")}
    halves = tuple((int(params[a]), int(params[b])) for a, b in zip(HALF_MARKERS[::2], HALF_MARKERS[1::2])
                   if params.get(a) and params.get(b))
    field = root.find(".//FieldSize")
    pitch = (float(_text(field, "Width")), float(_text(field, "Height")))
    rate = _text(root, ".//FrameRate")
    if rate is None:
        raise ValueError("no FrameRate in the metadata")
    fps = float(rate)

    layouts = []
    for spec in root.iter("DataFormatSpecification"):
        players_reg, ball_reg = [r for r in spec if r.tag == "SplitRegister"][:2]
        groups = []
        for group in players_reg:
            refs = [channel[r.get("playerChannelId")] for r in group.iter("PlayerChannelRef")]
            pid = refs[0][0]
            kinds = [kind for _, kind in refs]
            groups.append((pid, kinds.index("x"), kinds.index("y")))
        pids = [g[0] for g in groups]
        if len(set(pids)) != len(pids):
            raise ValueError("a player is listed twice in the layout starting at frame {}".format(spec.get("startFrame")))
        ball = [r.get("channelId") for r in ball_reg.iter("BallChannelRef")]
        layouts.append(dict(start=int(spec.get("startFrame")), end=int(spec.get("endFrame")),
                            groups=groups, ball=(ball.index("x"), ball.index("y"))))
    layouts.sort(key=lambda l: l["start"])
    for before, after in zip(layouts, layouts[1:]):
        if after["start"] <= before["end"]:
            raise ValueError("layouts overlap: frames {}-{} and {}-{}".format(
                before["start"], before["end"], after["start"], after["end"]))
    return dict(teams=teams, team_of=team_of, pitch=pitch, fps=fps, halves=halves or None, layouts=layouts)


def check_halves(halves):
    """Exactly two halves, each start <= end, the second after the first."""
    if halves is None or len(halves) != 2:
        raise ValueError("need two halves (metadata markers or halves=((s1, e1), (s2, e2))), got {}".format(halves))
    (s1, e1), (s2, e2) = halves
    if not (s1 <= e1 < s2 <= e2):
        raise ValueError("halves must be ordered and disjoint, got {}".format(halves))
    return halves


def assign_slots(meta):
    """player id -> damaged slot: first team 0-22, second team 23-45, in order of appearance."""
    team_ids = [t for t, _ in meta["teams"]]
    slot, used = {}, [0, 0]
    for layout in meta["layouts"]:
        for pid, _, _ in layout["groups"]:
            if pid in slot:
                continue
            block = team_ids.index(meta["team_of"][pid])
            if used[block] == SLOTS_PER_TEAM:
                raise ValueError("team {} has more than {} tracks; the 46-slot layout cannot hold them"
                                 .format(team_ids[block], SLOTS_PER_TEAM))
            slot[pid] = block * SLOTS_PER_TEAM + used[block]
            used[block] += 1
    return slot


MERGE_GAP = 50          # frames: a track missing for up to 2 s keeps its slot when slots are packed


def presence_runs(present, merge_gap=MERGE_GAP):
    """(start, end) inclusive row ranges where a track is present, gaps up to merge_gap merged."""
    idx = np.flatnonzero(present)
    if len(idx) == 0:
        return []
    cut = np.flatnonzero(np.diff(idx) > merge_gap + 1)
    starts = np.r_[idx[0], idx[cut + 1]]
    ends = np.r_[idx[cut], idx[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


def pack_team(tracks, present, block, merge_gap=MERGE_GAP):
    """Tracks of one team -> slots block*23 .. block*23+22. With 23 tracks or fewer, each track keeps
    one slot (order of appearance). With more (the tracker re-acquires people under new ids), each
    presence run gets a slot that is free for its whole run, preferring the track's previous slot.
    Runs merge gaps up to `merge_gap` frames; if that leaves no free slot it retries with exact
    presence, and a run that still finds none (more than 23 of the team tracked at once, e.g. a
    duplicated track) is dropped and counted.
    Returns (list of (track column, first row, last row, slot), number of dropped runs)."""
    base = block * SLOTS_PER_TEAM
    if len(tracks) <= SLOTS_PER_TEAM:
        return [(c, 0, present.shape[0] - 1, base + i) for i, c in enumerate(tracks)], 0
    for gap in (merge_gap, 0):
        runs = sorted((a, b, c) for c in tracks for a, b in presence_runs(present[:, c], gap))
        free_after = [-1] * SLOTS_PER_TEAM          # last row each slot is busy
        last_slot, out, dropped = {}, [], 0
        for a, b, c in runs:
            prev = last_slot.get(c)
            if prev is not None and free_after[prev] < a:
                k = prev
            else:
                free = [i for i in range(SLOTS_PER_TEAM) if free_after[i] < a]
                if not free:
                    dropped += 1
                    continue
                k = min(free, key=lambda i: free_after[i])
            free_after[k] = b
            last_slot[c] = k
            out.append((c, a, b, base + k))
        if dropped == 0 or gap == 0:
            return out, dropped
    return out, dropped


def _point(group, ix, iy, fid, what):
    """(x, y) from one comma group; an empty group is missing, a short one is an error."""
    if group == "" or group == ",":
        return np.nan, np.nan
    vals = group.split(",")
    if len(vals) <= max(ix, iy):
        raise ValueError("frame {}: {} has {} fields, the layout needs {}".format(fid, what, len(vals), max(ix, iy) + 1))
    try:
        return tuple(float(v) if v else np.nan for v in (vals[ix], vals[iy]))
    except ValueError:
        raise ValueError("frame {}: {} has a field that is not a number".format(fid, what)) from None


def translate(meta_path, raw_path, halves=None):
    """EPTS pair -> dict of arrays with the master .npz keys. `halves` = ((start, end), (start,
    end)) in frame ids overrides the metadata markers."""
    meta = read_metadata(meta_path)
    if meta["fps"] != FPS:
        raise ValueError("frame rate {} in the metadata; only {} fps is supported".format(meta["fps"], FPS))
    halves = check_halves(halves or meta["halves"])
    team_ids = [t for t, _ in meta["teams"]]
    columns = []                                   # every track id, in order of appearance
    for layout in meta["layouts"]:
        for pid, _, _ in layout["groups"]:
            if pid not in columns:
                columns.append(pid)
    col = {pid: i for i, pid in enumerate(columns)}
    length, width = meta["pitch"]
    plans = [(l["start"], l["end"], np.array([col[p] for p, _, _ in l["groups"]]),
              [(x, y) for _, x, y in l["groups"]], l["ball"]) for l in meta["layouts"]]

    frame_ids, rows_xy, rows_ball = [], [], []
    k = 0
    with open(raw_path) as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(":")
            if len(parts) != 3 or not parts[0].isdigit():
                raise ValueError("line {}: expected frame_id:players:ball".format(number))
            fid, players, ball = int(parts[0]), parts[1], parts[2]
            while plans[k][1] < fid and k + 1 < len(plans):
                k += 1
            start, end, slots, picks, (bx, by) = plans[k]
            if not start <= fid <= end:
                raise ValueError("frame {} is outside every DataFormatSpecification".format(fid))
            groups = players.split(";")
            if len(groups) != len(slots):
                raise ValueError("frame {}: {} player groups, the layout lists {} player groups".format(
                    fid, len(groups), len(slots)))
            xy = np.full((len(columns), 2), np.nan)
            for s, (ix, iy), group in zip(slots, picks, groups):
                xy[s] = _point(group, ix, iy, fid, "a player")
            frame_ids.append(fid)
            rows_xy.append(xy)
            rows_ball.append(_point(ball, bx, by, fid, "the ball"))

    frame_ids = np.array(frame_ids)
    n = len(frame_ids)
    if n == 0 or not np.array_equal(frame_ids, np.arange(1, n + 1)):
        raise ValueError("frame ids must be contiguous 1..N (row i = frame_id - 1)")
    tracks_xy = np.array(rows_xy)
    ball = np.array(rows_ball)
    for a in (tracks_xy, ball):  # a point is there only when both coordinates are finite
        lost = ~np.isfinite(a).all(-1)
        a[lost] = np.nan
        a[..., 0] = (a[..., 0] - 0.5) * length
        a[..., 1] = -(a[..., 1] - 0.5) * width
    present = ~np.isnan(tracks_xy[..., 0])
    in_halves = np.zeros(n, bool)
    for first, last in halves:
        in_halves[(frame_ids >= first) & (frame_ids <= last)] = True
    present &= in_halves[:, None]          # outside the halves nothing is graded or modelled: no players kept
    xy = np.full((n, 2 * SLOTS_PER_TEAM, 2), np.nan)
    n_tracks, max_at_once, dropped_runs = [], [], []
    for block, tid in enumerate(team_ids[:2]):
        tracks = [col[p] for p in columns if meta["team_of"].get(p) == tid]
        n_tracks.append(len(tracks))
        max_at_once.append(int(present[:, tracks].sum(1).max()) if tracks else 0)
        packed, lost = pack_team(tracks, present, block)
        dropped_runs.append(lost)
        for c, a, b, k in packed:
            seg = tracks_xy[a:b + 1, c]
            keep = present[a:b + 1, c]
            xy[a:b + 1][keep, k] = seg[keep]

    period = np.zeros(n, np.int8)
    t = np.full(n, np.nan, np.float32)
    for per, (first, last) in enumerate(halves, start=1):
        rows = (frame_ids >= first) & (frame_ids <= last)
        if not rows.any():
            raise ValueError("half {} ({}-{}) has no frame in the file".format(per, first, last))
        period[rows] = per
        t[rows] = (frame_ids[rows] - first) / FPS

    def to_cm(a):
        out = np.round(a * 100.0)
        return np.where(np.isnan(out), MISSING, np.clip(out, -32000, 32000)).astype(np.int16)

    unknown = np.full(n, -1, np.int8)
    info = dict(source="epts", match_id=None, home=meta["teams"][0][1], away=meta["teams"][1][1], fps=FPS,
                pitch=[length, width], n_frames=int(n), periods=[1, 2], source_fps=meta["fps"],
                orientation="raw: as in the tracking file (teams swap ends at half-time)",
                slots="damaged: 0-22 first metadata team, 23-45 second; one slot per track, or per presence run when a team has more than 23 tracks; clean side empty",
                tracks_per_team=n_tracks, max_present_per_team=max_at_once, dropped_runs_per_team=dropped_runs,
                labels={"0": "H", "1": "A", "2": "D", "-1": "unknown"},
                label_counts={"H": 0, "A": 0, "D": 0, "unknown": int(n)}, players_dropped=0,
                damage_seed=None, resample=None, halves=[list(h) for h in halves],
                notes=dict(row="row i = frame_id - 1; period 0 outside the halves"))
    return dict(
        meta=np.array(json.dumps(info)), frame=np.arange(n, dtype=np.int32), period=period, t=t, label=unknown,
        clean_xy=np.full((n, 22, 2), MISSING, np.int16), clean_vis=np.zeros((n, 22), bool),
        clean_seen=np.zeros((n, 22), bool), clean_team=np.array([0] * 11 + [1] * 11, np.int8),
        clean_ball=np.full((n, 3), MISSING, np.int16), clean_ball_vis=np.zeros(n, bool),
        dmg_xy=to_cm(xy), dmg_vis=~np.isnan(xy[..., 0]), dmg_slot_src=np.full((n, 46), -1, np.int8),
        dmg_ball=to_cm(ball), dmg_ball_vis=~np.isnan(ball[:, 0]), dmg_label=unknown.copy(),
        dmg_flip=np.array(False), dmg_stats=np.array(json.dumps(dict(flip=False, fragments=None, swaps=None))),
    )


def write(arrays, path):
    """np.savez_compressed with every machine-specific zip field fixed (timestamp, creating
    system, file mode), so the same arrays give the same bytes on any OS."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, value in arrays.items():
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asanyarray(value), allow_pickle=False)
            entry = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.create_system = 3
            entry.external_attr = 0o644 << 16
            archive.writestr(entry, buffer.getvalue())


def inside_repo(path):
    """True when `path` is in this repo. Case-insensitive on Windows; another drive is outside."""
    out, repo = (os.path.normcase(os.path.abspath(p)) for p in (path, REPO))
    try:
        return os.path.commonpath([out, repo]) == repo
    except ValueError:  # different drives
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("meta", help="EPTS metadata XML")
    ap.add_argument("raw", help="EPTS raw tracking text")
    ap.add_argument("--out", required=True, help=".npz path, outside the repo")
    ap.add_argument("--halves", help="s1,e1,s2,e2 frame ids, overriding the metadata markers")
    args = ap.parse_args(argv)
    out = os.path.abspath(args.out)
    if inside_repo(out):
        raise ValueError("--out must be outside the repo (frame-level file): {}".format(out))
    halves = None
    if args.halves:
        v = [int(x) for x in args.halves.split(",")]
        halves = ((v[0], v[1]), (v[2], v[3]))
    arrays = translate(args.meta, args.raw, halves)
    write(arrays, out)

    print("frames {}  in halves {}  ball present {:.3f}  objects/frame in halves {:.1f}".format(
        len(arrays["frame"]), int((arrays["period"] > 0).sum()), arrays["dmg_ball_vis"].mean(),
        arrays["dmg_vis"][arrays["period"] > 0].sum(1).mean()))
    info = json.loads(str(arrays["meta"]))
    print("teams in metadata order: block 0 {!r}, block 1 {!r}; track ids per team {}; most present at once {}; track runs dropped {}".format(
        info["home"], info["away"], info["tracks_per_team"], info["max_present_per_team"], info["dropped_runs_per_team"]))


if __name__ == "__main__":
    main()
