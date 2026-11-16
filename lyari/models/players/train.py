"""A3: train the players-only possession model on the master dataset (issue #6)."""

import argparse
import json
import os
import warnings
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from lyari.dataset.load_master import load_match
from lyari.models.players import features as FE
from lyari.models.players.model import PossessionNet
from lyari.models.players.temporal_sampling import check_stride, predict_strided
from lyari.scoring import score as S

DISP_REACH = 12            # sampled frames: +/- 0.48 s at stride1, +/- 2.4 s at stride5


def change_frames(y):
    """Indices where the label changes between two known frames. y: (T,) numpy or torch."""
    if isinstance(y, torch.Tensor):
        d = (y[1:] != y[:-1]) & (y[1:] >= 0) & (y[:-1] >= 0)
        return torch.nonzero(d, as_tuple=False)[:, 0] + 1
    d = (y[1:] != y[:-1]) & (y[1:] >= 0) & (y[:-1] >= 0)
    return np.flatnonzero(d) + 1


def short_possessions(label, period, max_frames):
    """Return complete short team-possession spans within a single period."""
    y, per = np.asarray(label), np.asarray(period)
    cut = np.flatnonzero((y[1:] != y[:-1]) | (per[1:] != per[:-1])) + 1
    starts, ends = np.r_[0, cut], np.r_[cut, len(y)]
    out = []
    for k in range(1, len(starts) - 1):
        a, b = starts[k], ends[k]
        if y[a] not in (0, 1) or b - a > max_frames:
            continue
        if per[a - 1] != per[a] or per[b] != per[a] or y[a - 1] < 0 or y[b] < 0:
            continue                                      # cut by a period edge or an unknown label: not complete
        out.append((a + b) // 2)
    return np.asarray(out, np.int64)


def spell_lengths(label):
    """(N,) float32: for every frame, the length in frames of the label run it sits in (over the FULL match, so a
    spell cut by a training window keeps its true length). Used by --spell-balance and --spell-weight."""
    y = np.asarray(label)
    cut = np.flatnonzero(y[1:] != y[:-1]) + 1
    starts, ends = np.r_[0, cut], np.r_[cut, len(y)]
    return np.repeat((ends - starts).astype(np.float32), ends - starts)


def spell_weights(Y, L, alpha, ref_frames):
    """(B, T) loss weights that count every possession spell about equally, like the metric does (spell F1 counts
    a 2 s spell as much as a 60 s one, while a per-frame loss weights them by length). H/A frames get
    (ref / spell length) ** alpha, clipped to [0.2, 8]; dead-ball and unknown frames keep weight 1."""
    w = torch.clamp((ref_frames / L.clamp_min(1.0)) ** alpha, 0.2, 8.0)
    return torch.where((Y == 0) | (Y == 1), w, torch.ones_like(w))


POSSESSION_WEIGHTS = {"linear": (8.0, 1.0, 0.25, 4.0), "sqrt": (5.0, 0.5, 0.5, 2.0)}   # ref s, power, clip


def possession_weights(Y, L, kind, fps=25.0):
    """(B, T) --spell-weight loss weights: every possession counts about equally, live (H/A) frames only. linear =
    clip(8 s / length, 0.25, 4) (FCB-Hacks), sqrt = clip(sqrt(5 s / length), 0.5, 2) (jekkiechan). The live weights
    are renormalised to mean 1 over the batch's live frames and dead-ball / unknown frames keep 1, so unlike
    --spell-balance (ledger row 22, which made a long possession frame weigh 0.2 against a dead frame's 1) the dead
    vs live balance of the loss, and with it the decoder costs, stays where it was."""
    live = (Y == 0) | (Y == 1)
    if kind == "none" or not live.any():
        return torch.ones(Y.shape, device=Y.device)
    ref, power, lo, hi = POSSESSION_WEIGHTS[kind]
    w = torch.clamp((ref * fps / L.clamp_min(1.0)) ** power, lo, hi)
    w = w / w[live].mean()
    return torch.where(live, w, torch.ones_like(w))


def transition_loss(logits, Y, eps=1e-6):
    """Change-vs-stable loss between consecutive frames (jekkiechan's transition surrogate). Two edges: dead/live
    from p_D, and H/A from p_H / (p_H + p_A) on pairs where both frames are live. The model's chance of a change
    between t and t+1 is p_t (1 - p_t+1) + (1 - p_t) p_t+1; BCE against the true change, averaged within the change
    pairs and within the stable pairs, then the two halves, so the rare changes weigh as much as the stable run.
    Pairs touching an unknown frame are ignored."""
    p = torch.softmax(logits, dim=-1)
    a, b = Y[:, :-1], Y[:, 1:]
    known = (a >= 0) & (b >= 0)
    pd = p[..., 2]
    q = p[..., 0] / (p[..., 0] + p[..., 1]).clamp_min(eps)
    total = logits.sum() * 0.0
    for x, valid, target in ((pd, known, (a == 2) != (b == 2)),
                             (q, known & (a != 2) & (b != 2), a != b)):
        change = (x[:, :-1] * (1 - x[:, 1:]) + (1 - x[:, :-1]) * x[:, 1:]).clamp(eps, 1 - eps)
        bce = -torch.where(target, change.log(), (1 - change).log())
        groups = [bce[valid & g] for g in (target, ~target)]
        groups = [g.mean() for g in groups if g.numel()]
        if groups:
            total = total + sum(groups) / len(groups)
    return total


def boundary_targets(Y, sigma):
    """(heat (B,T), disp (B,T), disp_mask (B,T)) from labels (B,T) with -100 = unknown.
    heat: max over changes of exp(-(t-c)^2 / 2 sigma^2); disp: (c - t) / DISP_REACH for the nearest
    change within DISP_REACH frames, else masked. Unknown frames carry no target."""
    B, T = Y.shape
    t = torch.arange(T, device=Y.device, dtype=torch.float32)
    heat = torch.zeros(B, T, device=Y.device)
    disp = torch.zeros(B, T, device=Y.device)
    mask = torch.zeros(B, T, device=Y.device, dtype=torch.bool)
    for b in range(B):
        c = change_frames(Y[b]).float()
        if len(c) == 0:
            continue
        diff = c[None, :] - t[:, None]                          # (T, C)
        heat[b] = torch.exp(-0.5 * (diff / sigma) ** 2).max(dim=1).values
        k = diff.abs().argmin(dim=1)
        near = diff[torch.arange(T), k]
        mask[b] = near.abs() <= DISP_REACH
        disp[b] = near / DISP_REACH
    known = Y != -100
    return heat * known, disp * known, mask & known


def state_targets(Y, who):
    """(B,T) hold (0) / flight (1) / dead (2) / -100: the labelled team's nearest player within
    2 m holds (possessor target found), a live frame without one is in flight, D is dead."""
    out = torch.full_like(Y, -100)
    out[(Y == 0) | (Y == 1)] = 1
    out[who >= 0] = 0
    out[Y == 2] = 2
    return out


def change_weights(Y, weight, reach=50):
    """(B, T) loss weights: 1 + weight within `reach` frames of a label change in the window, so the
    seconds around a turnover, where A3 errs most, count more. Labels are the training labels."""
    if weight <= 0:
        return None
    y = Y.clone()
    change = torch.zeros_like(y, dtype=torch.float32)
    diff = (y[:, 1:] != y[:, :-1]) & (y[:, 1:] >= 0) & (y[:, :-1] >= 0)
    change[:, 1:] = diff.float()
    k = torch.ones(1, 1, 2 * reach + 1, device=Y.device)
    near = F.conv1d(change[:, None], k, padding=reach)[:, 0] > 0
    return 1.0 + weight * near.float()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device_of(name):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)
    if dev.type == "cuda" and getattr(torch.version, "hip", None) and not os.environ.get("LYARI_MIOPEN"):
        # AMD (ROCm on Windows): MIOpen's GRU builds a dropout kernel that needs rocrand headers the Windows SDK lacks
        # (HIPRTC_ERROR_COMPILATION, miopenStatusUnknownError). PyTorch's own GRU kernels do the same maths without
        # MIOpen. Nvidia is unaffected. LYARI_MIOPEN=1 turns MIOpen back on.
        torch.backends.cudnn.enabled = False
    return dev


def load_features(match, use_ball, ball_rel=False, ball_repair=False):
    """Full-match features (for validation / prediction) and labels."""
    feats, label, _ = load_match_raw(match, use_ball, ball_rel, ball_repair=ball_repair)
    return feats, label


def hide_unseen(m):
    """Blank damaged slots whose source player was not actually seen by the provider (extrapolated tracks in
    SkillCorner fill off-camera players with estimates; `clean_seen` says which). The result looks
    like raw broadcast tracking: missing players instead of guessed ones. Training data prep only."""
    if int(m["meta"].get("version", 0)) < 3:
        raise SystemExit("--seen-only needs master v3: dmg_slot_src was stored in flipped column order before v3 (audit F4)")
    src = m["damaged"]["slot_src"]
    seen = m["clean"]["seen"]
    rows, cols = np.nonzero(src >= 0)
    unseen = np.zeros(src.shape, bool)
    unseen[rows, cols] = ~seen[rows, src[rows, cols]]
    xy = m["damaged"]["xy"].copy()
    xy[unseen] = np.nan
    return dict(m["damaged"], xy=xy), float(unseen.sum() / max((src >= 0).sum(), 1))


def repaired_ball(ball):
    """A4: (N, 2) damaged ball -> (N, 3) x, y, confidence, gaps up to 8 s filled by a straight line.
    Offline and label-free (lyari/models/rules/ball_repair.py)."""
    from lyari.models.rules import ball_repair as R
    rb, conf, _ = R.repair(ball[:, :2].astype(np.float64))
    return np.concatenate([rb, conf[:, None]], axis=1).astype(np.float32)


def subset_matches(names, n, key=""):
    """Learning curve: n of the train matches, the same for every training seed and nested (the 3 are inside the 6,
    the 6 inside the 9). Sources take turns (largest first) so each size keeps the source mix; within a source the
    order is a fixed shuffle keyed on `key` (the held-out match), so different folds draw different subsets."""
    import zlib
    rng = np.random.default_rng(zlib.crc32(("max-train|" + key).encode()))
    groups = {}
    for m in sorted(names):
        groups.setdefault(m.split("_")[0], []).append(m)
    for g in groups.values():
        rng.shuffle(g)
    order = []
    while any(groups.values()):
        for src in sorted(groups, key=lambda k: (-len(groups[k]), k)):
            if groups[src]:
                order.append(groups[src].pop(0))
    return sorted(order[:n])


def load_match_raw(match, use_ball, ball_rel=False, seen_only=False, need_feats=True, ball_repair=False, clean_ball=False, root=None):
    """(features, label, raw) where raw holds what training-time augmentation needs: the damaged
    xy and ball, the period, and a camera centre (the true ball in the damaged frame, gaps filled
    with the players' centroid) for the broadcast-visibility augmentation. need_feats=False skips
    the full-match features (training windows are featurised on the fly) to save memory."""
    m = load_match(os.path.join(root or S.master_dir(), "matches", match + ".npz"))
    if seen_only:
        m["damaged"], _ = hide_unseen(m)
    if clean_ball:          # control: the source's true ball on every frame it exists (no simulated gaps)
        tb = m["clean"]["ball"][:, :2].astype(np.float32).copy()
        tb[m["period"] == 2] *= -1                # the damaged copy mirrors period 2
        m["damaged"] = dict(m["damaged"], ball=tb)
    elif ball_repair:
        m["damaged"] = dict(m["damaged"], ball=repaired_ball(m["damaged"]["ball"]))
    feats = FE.frame_features(m["damaged"], use_ball=use_ball, ball_rel=ball_rel) if need_feats else None
    label = m["damaged"]["label"].astype(np.int64)
    label[label < 0] = -100                       # ignored by the loss
    period = m["period"]
    centre = m["clean"]["ball"][:, :2].copy()
    centre[period == 2] *= -1                     # the damaged copy mirrors period 2
    xy = m["damaged"]["xy"]
    with warnings.catch_warnings():                # frames with no player tracked (broadcast cut-aways)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        centroid = np.nanmean(xy, axis=1)
    gap = np.isnan(centre[:, 0])
    centre[gap] = centroid[gap]
    centre = np.nan_to_num(centre)
    true_ball = m["clean"]["ball"][:, :2].astype(np.float32)   # NaN where the source has no ball
    true_ball[period == 2] *= -1
    raw = dict(xy=xy.astype(np.float32), ball=m["damaged"]["ball"].astype(np.float32), period=period, centre=centre.astype(np.float32),
               true_ball=true_ball, spell_len=spell_lengths(label))
    return feats, label, raw


POSSESSOR_RADIUS = 2.0     # m: the labelled team's nearest player counts as on the ball only this close


def possessor_target(xy, true_ball, y, radius=POSSESSOR_RADIUS):
    """(T,) int64: slot 0..45 of the player on the ball, -100 where there is no target.

    On a frame labelled H (block 0) or A (block 1), the possessor is that block's present player
    nearest the true ball, if within `radius`. Frames that are dead, unlabelled, without a true
    ball or without a player close enough get no target. Training signal only."""
    n = FE.N_SLOTS
    d = np.linalg.norm(xy - true_ball[:, None, :], axis=-1)
    d = np.where(np.isnan(d), np.inf, d)
    out = np.full(len(y), -100, np.int64)
    for block in (0, 1):
        sub = d[:, block * n:(block + 1) * n]
        k = sub.argmin(axis=1)
        ok = (y == block) & (sub[np.arange(len(y)), k] < radius)
        out[ok] = block * n + k[ok]
    return out


def split_train_val(matches):
    """Hold out the last train match of each source for validation."""
    by_source = {}
    for name in matches:
        by_source.setdefault(name.split("_")[0], []).append(name)
    val = [v[-1] for v in by_source.values() if len(v) > 1]
    train = [n for n in matches if n not in val]
    return train, val


def ema(x, tau_frames):
    """Exponential moving average along time (camera pan lag)."""
    out = np.empty_like(x)
    a = 1.0 / max(tau_frames, 1.0)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = out[i - 1] + a * (x[i] - out[i - 1])
    return out


class Windows:
    """Random windows over matches, augmented on the raw tracking and featurised afterwards, so"""

    def __init__(self, items, window, rng, augment=True, use_ball=False, p_camera=0.6, p_count=0.5, ball_rel=False,
                 workers=0, p_transition=0.0, p_noball=0.0, stride=1, p_short=0.0, short_max_s=5.0, fps=25.0):
        self.stride = check_stride(stride)
        if self.stride > 1:
            # Keep raw tracking at 25 Hz for augmentation/velocities. Spell lengths
            # and transition sampling, like the loss targets, use sampled frames.
            items = [(f, y, dict(raw, spell_len=np.repeat(spell_lengths(y[::self.stride]),
                                                        self.stride)[:len(y)])) for f, y, raw in items]
        self.items, self.window, self.rng, self.augment = items, window, rng, augment
        self.use_ball, self.p_camera, self.p_count, self.ball_rel = use_ball, p_camera, p_count, ball_rel
        self.p_transition, self.p_noball = p_transition, p_noball
        self.changes = [change_frames(label[::self.stride]) for _, label, _ in items]
        # --p-short: that share of training windows is centred on a complete possession of <= short_max_s, cycling
        # through each match's short possessions in a fixed shuffled order (every one is seen before any repeats).
        # The choice is made on the main thread from self.rng, so --workers batches do not depend on thread timing;
        # at p_short 0 no random number is drawn and every other arm's windows are unchanged.
        self.p_short = float(p_short)
        self.shorts, self._short_pos = [], []
        if self.p_short > 0:
            mx = int(round(short_max_s * fps / self.stride))
            for _, label, raw in items:
                c = short_possessions(label[::self.stride], np.asarray(raw["period"])[::self.stride], mx)
                self.shorts.append(c[self.rng.permutation(len(c))])
                self._short_pos.append(0)
        self.last_who = []       # possessor targets of the last batch, one (T,) array per window
        self.last_len = []       # full-match spell length of every frame of the last batch (for --spell-balance / --spell-weight)
        # workers > 0: each window gets its own generator seeded from self.rng (so batches do not
        # depend on thread timing), windows are built by a thread pool, and the next batch is built
        # in the background while the GPU trains on the current one. workers = 0: the original
        # sequential path, unchanged.
        self.workers = int(workers)
        self._pool = self._ahead = self._next = None
        if self.workers > 0:
            from concurrent.futures import ThreadPoolExecutor
            self._pool = ThreadPoolExecutor(self.workers)
            self._ahead = ThreadPoolExecutor(1)

    def _short_job(self):
        """(match index, sampled-frame centre) of the next short-possession window, or None for an ordinary window."""
        if not (self.augment and self.p_short > 0 and self.rng.random() < self.p_short):
            return None
        have = [i for i, c in enumerate(self.shorts) if len(c)]
        if not have:
            return None
        i = have[int(self.rng.integers(len(have)))]
        c = int(self.shorts[i][self._short_pos[i] % len(self.shorts[i])])
        self._short_pos[i] += 1
        return i, c

    def _short_start(self, raw, n, centre):
        """Window start (sampled frames) centred on `centre`, moved to stay inside the centre's period."""
        per = np.asarray(raw["period"])[::self.stride][:n]
        same = np.flatnonzero(per == per[centre])
        lo, hi = int(same[0]), int(same[-1]) + 1
        return int(np.clip(centre - self.window // 2, lo, max(lo, hi - self.window)))

    def _window(self, raw, label, rng=None, changes=None, with_slice=False, centre=None):
        """(xy, ball, y, who) of one training window, plus the window's slice when with_slice is True.
        centre: a sampled frame to centre the window on (--p-short) instead of a random or change-centred start."""
        n = (len(label) + self.stride - 1) // self.stride
        rng = self.rng if rng is None else rng
        centred = centre is None and self.augment and changes is not None and len(changes) > 0 and rng.random() < self.p_transition
        for _ in range(10 if centre is None else 0):          # stay inside one half
            if centred:                                       # a label change somewhere in the middle 60 %
                c = int(changes[rng.integers(len(changes))])
                a = int(np.clip(c - rng.integers(int(0.2 * self.window), int(0.8 * self.window)), 0, max(0, n - self.window)))
            else:
                a = int(rng.integers(0, max(1, n - self.window)))
            sl = slice(a * self.stride, (a + self.window) * self.stride)
            per = raw["period"][sl]
            if per.min() == per.max():
                break
        if centre is not None:
            a = self._short_start(raw, n, centre)
            sl = slice(a * self.stride, (a + self.window) * self.stride)
        xy, ball, y = raw["xy"][sl].copy(), raw["ball"][sl].copy(), label[sl].copy()
        centre = raw["centre"][sl].copy()
        true_ball = raw["true_ball"][sl].copy() if "true_ball" in raw else np.full((len(y), 2), np.nan, np.float32)
        if not self.augment:
            out = (xy, ball, y, possessor_target(xy, true_ball, y))
            return out + (sl,) if with_slice else out
        if rng.random() < 0.5:                                # swap home/away
            xy = np.concatenate([xy[:, FE.N_SLOTS:], xy[:, :FE.N_SLOTS]], axis=1)
            y = np.where(y == 0, 1, np.where(y == 1, 0, y))
        if rng.random() < 0.5:                                # mirror x
            xy[..., 0] *= -1; ball[:, 0] *= -1; centre[:, 0] *= -1; true_ball[:, 0] *= -1
        if rng.random() < 0.5:                                # mirror y
            xy[..., 1] *= -1; ball[:, 1] *= -1; centre[:, 1] *= -1; true_ball[:, 1] *= -1
        who = possessor_target(xy, true_ball, y)
        if rng.random() < self.p_noball:                      # whole window without a ball: team from the players alone
            ball[:] = np.nan
        if rng.random() < self.p_camera:                      # broadcast camera window
            cam = ema(centre, 25.0)
            half_w, half_d = rng.uniform(17.5, 30.0), rng.uniform(20.0, 34.0)
            outside = (np.abs(xy[..., 0] - cam[:, None, 0]) > half_w) | (np.abs(xy[..., 1] - cam[:, None, 1]) > half_d)
            xy[outside] = np.nan
            hidden = who >= 0
            hidden[hidden] = np.isnan(xy[np.flatnonzero(hidden), who[hidden], 0])
            who[hidden] = -100                                # possessor off camera: no target
        return (xy, ball, y, who, sl) if with_slice else (xy, ball, y, who)

    def _one(self, seed, job=None):
        rng = np.random.default_rng(seed)
        i = int(rng.integers(len(self.items))) if job is None else job[0]
        _, label, raw = self.items[i]
        xy, ball, y, who, length = self._window_len(raw, label, rng, self.changes[i], None if job is None else job[1])
        f = FE.frame_features(dict(xy=xy, ball=ball), use_ball=self.use_ball, ball_rel=self.ball_rel)
        teams = f["teams"]
        if self.augment and rng.random() < self.p_count:
            teams = teams.copy(); teams[:, 0] = 0.0; teams[:, FE.TEAM_SHAPE] = 0.0
        s = slice(None, None, self.stride)
        return f["players"][s], teams[s], f["ball"][s], y[s], who[s], length[s]

    def _window_len(self, raw, label, rng=None, changes=None, centre=None):
        """_window plus the full-match spell length of each frame of the window. The slice comes back as a return
        value, never through shared state, so it is safe with --workers (windows built on several threads)."""
        xy, ball, y, who, sl = self._window(raw, label, rng, changes, with_slice=True, centre=centre)
        length = raw["spell_len"][sl].copy() if "spell_len" in raw else np.ones(len(y), np.float32)
        return xy, ball, y, who, length

    def _make(self, seeds, jobs=None):
        out = list(self._pool.map(self._one, seeds, jobs if jobs is not None else [None] * len(seeds)))
        return (torch.from_numpy(np.stack([o[0] for o in out])), torch.from_numpy(np.stack([o[1] for o in out])),
                torch.from_numpy(np.stack([o[2] for o in out])), torch.from_numpy(np.stack([o[3] for o in out])),
                [o[4] for o in out], [o[5] for o in out])

    def _draw(self, size):
        """Seeds for one batch, plus the --p-short jobs when it is on (drawn after the seeds, on the main thread)."""
        seeds = self.rng.integers(0, 2 ** 63, size)
        if self.p_short <= 0:
            return (seeds,)
        return seeds, [self._short_job() for _ in range(size)]

    def batch(self, size):
        if self.workers > 0:
            if self._next is None or self._next[0] != size:
                self._next = (size, self._ahead.submit(self._make, *self._draw(size)))
            P, T, B, Y, W, Ls = self._next[1].result()
            self._next = (size, self._ahead.submit(self._make, *self._draw(size)))
            self.last_who, self.last_len = W, Ls
            return P, T, B, Y
        P, T, B, Y, W, Ls = [], [], [], [], [], []
        for _ in range(size):
            job = self._short_job() if self.p_short > 0 else None
            i = int(self.rng.integers(len(self.items))) if job is None else job[0]
            _, label, raw = self.items[i]
            xy, ball, y, who, length = self._window_len(raw, label, changes=self.changes[i],
                                                        centre=None if job is None else job[1])
            s = slice(None, None, self.stride)
            W.append(who[s])
            Ls.append(length[s])
            f = FE.frame_features(dict(xy=xy, ball=ball), use_ball=self.use_ball, ball_rel=self.ball_rel)
            teams = f["teams"]
            if self.augment and self.rng.random() < self.p_count:
                teams = teams.copy(); teams[:, 0] = 0.0; teams[:, FE.TEAM_SHAPE] = 0.0
            P.append(f["players"][s]); T.append(teams[s]); B.append(f["ball"][s]); Y.append(y[s])
        self.last_who, self.last_len = W, Ls
        return (torch.from_numpy(np.stack(P)), torch.from_numpy(np.stack(T)), torch.from_numpy(np.stack(B)),
                torch.from_numpy(np.stack(Y)))


@torch.no_grad()
def predict_match(model, feats, device, chunk=2000, overlap=250, boundary=False, path=False, stride=1, phase_average=False):
    """Full-match logits -> (N, 3) probabilities, run in overlapping chunks. With boundary=True
    also (N, 2): p_B and the displacement in frames, from the boundary heads. With path=True (a model
    with the possessor and state heads) also (N, 49): P(each of the 46 slots holds the ball | someone
    holds it) and P(hold, flight, dead), the inputs of the possession-path decoder (lyari/models/path)."""
    stride = check_stride(stride)
    if stride > 1:
        return predict_strided(predict_match, model, feats, device, stride=stride, phase_average=phase_average,
                               chunk=chunk, overlap=overlap, boundary=boundary, path=path)
    model.eval()
    n = len(feats["teams"])
    out = np.zeros((n, 5), np.float32)
    want_path = path and getattr(model, "possessor", False) and getattr(model, "state_aux", False)
    pth = np.zeros((n, 49), np.float32) if want_path else None
    weight = np.zeros(n, np.float32)
    starts = list(range(0, max(1, n - overlap), chunk - overlap))
    for a in starts:
        b = min(n, a + chunk)
        if b == a:
            break
        p = torch.from_numpy(feats["players"][a:b][None]).to(device)
        t = torch.from_numpy(feats["teams"][a:b][None]).to(device)
        bl = torch.from_numpy(feats["ball"][a:b][None]).to(device)
        res = model(p, t, bl, aux=want_path, extra=True)
        prob = F.softmax(res["logits"][0], dim=-1).cpu().numpy()
        w = np.ones(b - a, np.float32)
        w[:overlap // 2] = 0.5 if a > 0 else 1.0
        w[-overlap // 2:] = 0.5 if b < n else 1.0
        out[a:b, :3] += prob * w[:, None]
        if "boundary" in res:
            out[a:b, 3] += torch.sigmoid(res["boundary"][0]).cpu().numpy() * w
            out[a:b, 4] += res["disp"][0].cpu().numpy() * DISP_REACH * w
        if want_path:
            pth[a:b, :46] += F.softmax(res["who"][0], dim=-1).cpu().numpy() * w[:, None]
            pth[a:b, 46:] += F.softmax(res["state"][0], dim=-1).cpu().numpy() * w[:, None]
        weight[a:b] += w
        if b == n:
            break
    out = out / np.maximum(weight, 1e-6)[:, None]
    res = [out[:, :3]]
    if boundary:
        res.append(out[:, 3:])
    if path:
        res.append(pth / np.maximum(weight, 1e-6)[:, None] if want_path else None)
    return tuple(res) if len(res) > 1 else res[0]


def predict_by_period(model, feats, period, device, **kw):
    """predict_match on each run of constant period separately (half 1, half-time, half 2, ...), so no inference
    window crosses half-time; training windows never do (they are drawn inside one played period). Same outputs as
    predict_match, concatenated back in frame order."""
    period = np.asarray(period)
    n = len(period)
    cuts = (np.flatnonzero(np.diff(period) != 0) + 1).tolist()
    bounds = [0] + cuts + [n]
    parts = [predict_match(model, {k: v[a:b] for k, v in feats.items()}, device, **kw)
             for a, b in zip(bounds[:-1], bounds[1:])]
    if isinstance(parts[0], tuple):
        return tuple(None if parts[0][i] is None else np.concatenate([q[i] for q in parts]) for i in range(len(parts[0])))
    return np.concatenate(parts)


def write_path(path, pth):
    """<match>_path.npz next to the probabilities file: (N, 49) float16, slots 0-45 then hold, flight, dead."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, path=pth.astype(np.float16))


def write_boundary(path, bnd):
    """frame,p_B,disp next to the probabilities file."""
    import pandas as pd
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    table = pd.DataFrame({"frame": np.arange(len(bnd)), "p_B": bnd[:, 0], "disp": bnd[:, 1]})
    table.to_csv(path, index=False, float_format="%.5g")


def frame_macro_f1(prob, label):
    pred = prob.argmax(1)
    ok = label >= 0
    f1 = []
    for k in range(3):
        tp = ((pred == k) & (label == k) & ok).sum()
        fp = ((pred == k) & (label != k) & ok).sum()
        fn = ((pred != k) & (label == k) & ok).sum()
        f1.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps", type=int, default=200, help="batches per epoch")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--window", type=int, default=500, help="sampled frames per window (20 s at stride1; 100 s at stride5)")
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth input/label; targets and transition weights use sampled frames")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--use-ball", action="store_true")
    ap.add_argument("--p-camera", type=float, default=0.6, help="share of training windows with the broadcast camera mask")
    ap.add_argument("--p-noball", type=float, default=0.0, help="share of training windows with the ball blanked throughout")
    ap.add_argument("--extra-dirs", default="", help="comma list of dataset folders (each holding master/) with the same train "
                    "matches under other damage draws; they are added to the training items, validation stays on the main copy")
    ap.add_argument("--no-val", action="store_true",
                    help="train on every match with no validation hold-out and keep the final epoch (the recipe the 13-fold "
                         "out-of-fold scores measure); without it, a run with no --holdout keeps 3 matches aside to pick the epoch")
    ap.add_argument("--ball-rel", action="store_true",
                    help="give every player his offset, distance and closing speed to the ball (before pooling)")
    ap.add_argument("--change-weight", type=float, default=0.0,
                    help="extra loss weight within 50 sampled frames of a label change (0 = plain cross-entropy)")
    ap.add_argument("--possessor", type=float, default=0.0,
                    help="weight of the auxiliary who-has-the-ball loss (0 = no possessor head)")
    ap.add_argument("--sources", default="", help="comma list to restrict train sources (e.g. sportec,metrica)")
    ap.add_argument("--clean-ball", action="store_true",
                    help="control: train on the true ball (no simulated ball gaps); use with --p-camera 0. Validation stays damaged")
    ap.add_argument("--ball-repair", action="store_true",
                    help="feed the A4-repaired ball (gaps up to 8 s filled, with a confidence) instead of the raw damaged ball")
    ap.add_argument("--seen-only", default="", help="comma list of sources whose estimated (not seen) player positions are hidden, when a source provides an observed flag")
    ap.add_argument("--encoder", default="set", choices=["set", "attn"],
                    help="set: each team pooled on its own (v1-v16); attn: cross-team attention over all players + ball (v17)")
    ap.add_argument("--attn-layers", type=int, default=1)
    ap.add_argument("--temporal", default="gru", choices=["gru", "tcn"],
                    help="gru: bidirectional GRU over time (all models so far); tcn: dilated temporal convolutions (MS-TCN style)")
    ap.add_argument("--tcn-layers", type=int, default=10, help="dilated layers per TCN stage (receptive field 2^layers frames)")
    ap.add_argument("--tcn-stages", type=int, default=1, help="TCN stages; stages after the first refine the H/A/D probabilities")
    ap.add_argument("--boundary", type=float, default=0.0, help="weight of the boundary head loss (0 = no head)")
    ap.add_argument("--disp", type=float, default=0.0, help="weight of the displacement head loss")
    ap.add_argument("--aux-start", type=float, default=0.4, help="boundary / displacement losses start after this fraction of epochs")
    ap.add_argument("--sigma-start", type=float, default=12.0, help="boundary target sigma (frames) at aux start")
    ap.add_argument("--sigma-end", type=float, default=4.0, help="boundary target sigma (frames) at the last epoch")
    ap.add_argument("--p-transition", type=float, default=0.0, help="share of training windows centred on a label change")
    ap.add_argument("--p-short", type=float, default=0.0,
                    help="share of training windows centred on a complete possession of <= --short-max-s, cycling through "
                         "each match's short possessions (arm v5bsp; 0 = off, every other arm unchanged)")
    ap.add_argument("--short-max-s", type=float, default=5.0, help="longest possession (s) --p-short centres on")
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay per step for the weights used at validation and saved (0 = off)")
    ap.add_argument("--extra-clues", action="store_true",
                    help="4 more inputs: goal-side share and pressure per team (positions only)")
    ap.add_argument("--control-physics", action="store_true",
                    help="with --ball-rel: 3 more inputs per player, ball velocity relative to the player and a with-ball score")
    ap.add_argument("--max-train", type=int, default=0,
                    help="learning curve: train on only this many of the train matches (nested, source-balanced; 0 = all)")
    ap.add_argument("--spell-balance", type=float, default=0.0,
                    help="weight H/A frames by (ref / spell length) ** this, so each possession counts about equally as in the "
                         "metric (0 = off, 1 = fully balanced)")
    ap.add_argument("--spell-ref-s", type=float, default=10.0, help="spell length (s) that keeps weight 1 under --spell-balance")
    ap.add_argument("--spell-weight", default="none", choices=["none", "linear", "sqrt"],
                    help="possession-equal weights on live frames only, mean 1 over them (linear 8 s / length in "
                         "[0.25, 4], sqrt sqrt(5 s / length) in [0.5, 2]); dead frames keep 1")
    ap.add_argument("--transition-loss", type=float, default=0.0,
                    help="weight of the group-balanced change-vs-stable loss between consecutive frames (0 = off)")
    ap.add_argument("--state-aux", type=float, default=0.0, help="weight of the hold / flight / dead auxiliary loss")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=0,
                    help="threads that build training windows, with the next batch built while the GPU trains (0 = original sequential path)")
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--smoke", action="store_true", help="tiny run on CPU to check the plumbing")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/last.pt if it exists")
    ap.add_argument("--init", default="", help="start from this model.pt's weights, e.g. a self-supervised pretraining "
                                               "(lyari.models.players.pretrain); same architecture flags, fresh optimiser")
    ap.add_argument("--holdout", default="", help="out-of-fold: train on every other train match, keep the final epoch, "
                                                  "write this match's probabilities to <out>/<match>.csv")
    args = ap.parse_args()
    if args.stride < 1:
        ap.error("--stride must be positive")
    FE.EXTRA = bool(args.extra_clues)
    if args.control_physics and not args.ball_rel:
        ap.error("--control-physics needs --ball-rel")
    FE.set_control(args.control_physics)
    if args.smoke:
        args.epochs, args.steps, args.batch, args.window, args.device = 2, 10, 4, 250, "cpu"
    set_seed(args.seed)
    device = device_of(args.device)
    os.makedirs(args.out, exist_ok=True)

    matches = S.master_matches("train")
    if args.sources:
        keep = set(args.sources.split(","))
        matches = [m for m in matches if m.split("_")[0] in keep]
    if args.smoke:
        matches = matches[:2]
    if args.holdout:
        if args.holdout not in matches:
            raise SystemExit("holdout {} is not in the train split".format(args.holdout))
        train_names, val_names = [m for m in matches if m != args.holdout], [args.holdout]
    elif args.no_val:           # the shipped model trained like the out-of-fold ones: every match, the final epoch
        train_names, val_names = list(matches), []
    else:
        train_names, val_names = split_train_val(matches)
    if args.max_train and args.max_train < len(train_names):
        train_names = subset_matches(train_names, args.max_train, args.holdout)
    print("device", device, "| train", len(train_names), "matches | val", val_names)
    t0 = time.time()
    seen_only = set(filter(None, args.seen_only.split(",")))
    train_items = [load_match_raw(m, args.use_ball, args.ball_rel, m.split("_")[0] in seen_only, need_feats=False,
                                  ball_repair=args.ball_repair, clean_ball=args.clean_ball) for m in train_names]
    for extra in filter(None, args.extra_dirs.split(",")):     # the same matches under other damage draws
        root = os.path.join(extra, "master")
        train_items += [load_match_raw(m, args.use_ball, args.ball_rel, m.split("_")[0] in seen_only, need_feats=False,
                                       ball_repair=args.ball_repair, clean_ball=args.clean_ball, root=root) for m in train_names]
    if args.extra_dirs:
        print("train items {} ({} matches x {} damage draws)".format(len(train_items), len(train_names), 1 + len(args.extra_dirs.split(","))))
    val_items = [(m, *load_features(m, args.use_ball, args.ball_rel, args.ball_repair)) for m in val_names]
    print("features loaded in %.0f s" % (time.time() - t0))

    model = PossessionNet(hidden=args.hidden, per_player=FE.PER_PLAYER + (FE.BALL_REL if args.ball_rel else 0),
                          possessor=args.possessor > 0, encoder=args.encoder, attn_layers=args.attn_layers,
                          boundary=args.boundary > 0 or args.disp > 0, state_aux=args.state_aux > 0,
                          ball_dim=4 + (FE.EXTRA_DIM if FE.EXTRA else 0), temporal=args.temporal,
                          tcn_layers=args.tcn_layers, tcn_stages=args.tcn_stages).to(device)
    print("parameters", sum(p.numel() for p in model.parameters()))
    if args.init:                                  # fine-tune: start from a trained model's weights, fresh optimiser
        model.load_state_dict(torch.load(args.init, map_location=device)["state"])
        print("initialised from", args.init)
    ema_model = None
    if args.ema > 0:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema))
    eval_model = lambda: ema_model.module if ema_model is not None else model
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * args.steps)
    rng = np.random.default_rng(args.seed)
    windows = Windows(train_items, args.window, rng, use_ball=args.use_ball, p_camera=args.p_camera, ball_rel=args.ball_rel,
                      workers=args.workers, p_transition=args.p_transition, p_noball=args.p_noball, stride=args.stride, p_short=args.p_short, short_max_s=args.short_max_s)
    best, log, start = -1.0, [], 1
    last_path = os.path.join(args.out, "last.pt")
    run_id = dict(train=train_names, val=val_names, holdout=args.holdout, hidden=args.hidden, use_ball=args.use_ball,
                  window=args.window, seed=args.seed, epochs=args.epochs, steps=args.steps, batch=args.batch,
                  change_weight=args.change_weight, ball_rel=args.ball_rel, possessor=args.possessor,
                  seen_only=sorted(seen_only), ball_repair=args.ball_repair, clean_ball=args.clean_ball, boundary=args.boundary, disp=args.disp,
                  aux_start=args.aux_start, sigma=[args.sigma_start, args.sigma_end], p_transition=args.p_transition,
                  ema=args.ema, state_aux=args.state_aux, spell_balance=args.spell_balance, spell_ref_s=args.spell_ref_s)
    if args.p_short > 0:                           # runs without the sampler keep their resume identity
        run_id.update(p_short=args.p_short, short_max_s=args.short_max_s)
    if args.stride != 1:                           # old stride1 checkpoints keep their resume identity
        run_id.update(stride=args.stride)
    if args.encoder != "set":                      # keep old run ids (and their --resume checkpoints) unchanged
        run_id.update(encoder=args.encoder, attn_layers=args.attn_layers)
    if args.control_physics:
        run_id.update(control_physics=True)
    if args.max_train:
        run_id.update(max_train=args.max_train)
    if args.spell_weight != "none" or args.transition_loss > 0:
        run_id.update(spell_weight=args.spell_weight, transition_loss=args.transition_loss)
    if args.init:
        run_id.update(init=os.path.basename(os.path.dirname(os.path.abspath(args.init))))
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device)
        if ck.get("run") != run_id:    # a checkpoint from another fold or config (shared --out): never continue it
            raise SystemExit("--resume: {} belongs to a different run (other holdout, matches or settings); "
                             "rerun without --resume or use a fresh --out".format(last_path))
        model.load_state_dict(ck["state"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        best, log, start = ck["best"], ck["log"], ck["epoch"] + 1
        rng = np.random.default_rng(args.seed + start)
        windows = Windows(train_items, args.window, rng, use_ball=args.use_ball, p_camera=args.p_camera, ball_rel=args.ball_rel,
                          workers=args.workers, p_transition=args.p_transition, p_noball=args.p_noball, stride=args.stride, p_short=args.p_short, short_max_s=args.short_max_s)
        if ema_model is not None and "ema" in ck:
            ema_model.load_state_dict(ck["ema"])
        print("resumed after epoch", ck["epoch"])
    for epoch in range(start, args.epochs + 1):
        model.train()
        losses = []
        te = time.time()
        heads_on = (args.boundary > 0 or args.disp > 0) and epoch > args.aux_start * args.epochs
        frac = min(1.0, max(0.0, (epoch - args.aux_start * args.epochs) / max(1.0, args.epochs * (1 - args.aux_start))))
        sigma = args.sigma_start + (args.sigma_end - args.sigma_start) * frac
        for _ in range(args.steps):
            P, T, B, Y = (x.to(device) for x in windows.batch(args.batch))
            Wt = torch.from_numpy(np.stack(windows.last_who)).to(device)
            res = model(P, T, B, aux=args.possessor > 0, extra=True)
            logits = res["logits"]
            if args.possessor > 0:
                who = res["who"]
                aux = F.cross_entropy(who.reshape(-1, who.shape[-1]), Wt.reshape(-1), ignore_index=-100) \
                    if (Wt >= 0).any() else logits.sum() * 0.0
            w = change_weights(Y, args.change_weight)
            if args.spell_balance > 0:
                Lt = torch.from_numpy(np.stack(windows.last_len)).to(device)
                sb = spell_weights(Y, Lt, args.spell_balance, args.spell_ref_s * 25.0 / args.stride)
                w = sb if w is None else w * sb
            if args.spell_weight != "none":
                Lt = torch.from_numpy(np.stack(windows.last_len)).to(device)
                pw = possession_weights(Y, Lt, args.spell_weight, fps=25.0 / args.stride)
                w = pw if w is None else w * pw
            if w is None:
                loss = F.cross_entropy(logits.reshape(-1, 3), Y.reshape(-1), ignore_index=-100)
            else:
                per = F.cross_entropy(logits.reshape(-1, 3), Y.reshape(-1), ignore_index=-100, reduction="none")
                valid = (Y.reshape(-1) != -100).float()
                wf = w.reshape(-1) * valid
                loss = (per * wf).sum() / wf.sum().clamp_min(1.0)
            for sl in res.get("stage_logits", []):        # earlier MS-TCN stages: plain cross-entropy
                loss = loss + F.cross_entropy(sl.reshape(-1, 3), Y.reshape(-1), ignore_index=-100)
            if args.possessor > 0:
                loss = loss + args.possessor * aux
            if heads_on:
                heat, dtarget, dmask = boundary_targets(Y, sigma)
                known = (Y != -100).float()
                if args.boundary > 0:
                    pos = (heat * known).sum() / known.sum().clamp_min(1.0)
                    pw = torch.clamp(1.0 / pos.clamp_min(1e-3) - 1.0, 1.0, 20.0)
                    bce = F.binary_cross_entropy_with_logits(res["boundary"], heat, reduction="none")
                    bce = bce * (1.0 + (pw - 1.0) * heat) * known
                    loss = loss + args.boundary * bce.sum() / known.sum().clamp_min(1.0)
                if args.disp > 0 and dmask.any():
                    loss = loss + args.disp * (res["disp"] - dtarget).abs()[dmask].mean()
            if args.transition_loss > 0:
                loss = loss + args.transition_loss * transition_loss(logits, Y)
            if args.state_aux > 0:
                st = state_targets(Y, Wt)
                loss = loss + args.state_aux * F.cross_entropy(res["state"].reshape(-1, 3), st.reshape(-1), ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if ema_model is not None:
                ema_model.update_parameters(model)
            losses.append(loss.item())
        vals = [frame_macro_f1(predict_match(eval_model(), f, device, stride=args.stride), y) for _, f, y in val_items] or [float("nan")]
        val = float(np.nanmean(vals))
        log.append(dict(epoch=epoch, loss=float(np.mean(losses)), val_macro_f1=val, seconds=round(time.time() - te, 1)))
        print("epoch %2d  loss %.4f  val frame macroF1 %.4f  (%.0f s)" % (epoch, np.mean(losses), val, time.time() - te))
        if args.holdout or val > best or np.isnan(val):      # out-of-fold: the final epoch, no selection on the holdout
            best = val
            torch.save(dict(state=eval_model().state_dict(), config=dict(hidden=args.hidden, use_ball=args.use_ball,
                            window=args.window, stride=args.stride, seed=args.seed, change_weight=args.change_weight, ball_rel=args.ball_rel,
                            possessor=args.possessor, seen_only=sorted(seen_only), sources=args.sources,
                            ball_repair=args.ball_repair, clean_ball=args.clean_ball, encoder=args.encoder, attn_layers=args.attn_layers,
                            boundary=args.boundary > 0 or args.disp > 0,
                            state_aux=args.state_aux > 0, ema=args.ema, extra_clues=bool(args.extra_clues),
                            control_physics=bool(args.control_physics), max_train=args.max_train,
                            temporal=args.temporal, tcn_layers=args.tcn_layers, tcn_stages=args.tcn_stages), val_macro_f1=val, epoch=epoch,
                            train=train_names, val=val_names), os.path.join(args.out, "model.pt"))
        torch.save(dict(state=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), best=best, log=log,
                        epoch=epoch, run=run_id, **({"ema": ema_model.state_dict()} if ema_model is not None else {})), last_path)
    json.dump(log, open(os.path.join(args.out, "train_log.json"), "w"), indent=1)
    print("best val frame macroF1 %.4f -> %s" % (best, os.path.join(args.out, "model.pt")))
    if args.holdout:
        from lyari import io
        name, feats, _ = val_items[0]
        torch.manual_seed(0)
        final = eval_model().to("cpu")
        if final.boundary:
            prob, bnd, pth = predict_match(final, feats, torch.device("cpu"), boundary=True, path=True, stride=args.stride)
            write_boundary(os.path.join(args.out, name + "_boundary.csv"), bnd)
        else:
            prob, pth = predict_match(final, feats, torch.device("cpu"), path=True, stride=args.stride)
        if pth is not None:
            write_path(os.path.join(args.out, name + "_path.npz"), pth)
        io.write_probs(os.path.join(args.out, name + ".csv"), prob.astype(np.float64))
        json.dump(dict(run_id, final_epoch=epoch, final_val_macro_f1=val), open(os.path.join(args.out, name + ".json"), "w"), indent=1)
        print("out-of-fold probabilities ->", os.path.join(args.out, name + ".csv"))


if __name__ == "__main__":
    main()
