"""A3 network: a set encoder per team block, then a bidirectional GRU over time, then H/A/D per frame."""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from lyari.models.players.features import PER_PLAYER, TEAM_SHAPE


class SetEncoder(nn.Module):
    def __init__(self, width=32, per_player=PER_PLAYER):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(per_player, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())
        self.width = width

    def forward(self, players, per_player=False):    # (B, T, 23, 6)
        present = players[..., -1:]                  # (B, T, 23, 1)
        h = self.mlp(players) * present
        count = present.sum(dim=2).clamp(min=1.0)
        mean = h.sum(dim=2) / count
        mx = (h + (present - 1.0) * 1e4).max(dim=2).values   # absent players cannot win the max
        mx = torch.where(present.sum(dim=2) > 0, mx, torch.zeros_like(mx))
        pooled = torch.cat([mean, mx], dim=-1)       # (B, T, 2 * width)
        return (pooled, h) if per_player else pooled


class CrossTeamAttention(nn.Module):
    """Per-player MLP (as SetEncoder), then transformer layers over the 46 player tokens + 1 ball token."""

    def __init__(self, width=32, per_player=PER_PLAYER, layers=1, heads=2, ball_dim=4, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(per_player, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())
        self.team = nn.Parameter(torch.zeros(2, width))           # tags block 0 / block 1
        self.ball = nn.Sequential(nn.Linear(ball_dim, width), nn.ReLU())
        layer = nn.TransformerEncoderLayer(width, heads, dim_feedforward=2 * width, dropout=dropout,
                                           batch_first=True, norm_first=True)
        self.attn = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.width = width
        self.chunk = 4096          # frames per attention call; in training each chunk is recomputed in the
                                   # backward pass (checkpointing), so GPU memory stays near one chunk's worth

    def _attend(self, tok, pad):
        outs = []
        for a in range(0, tok.shape[0], self.chunk):
            t, m = tok[a:a + self.chunk], pad[a:a + self.chunk]
            if self.training and torch.is_grad_enabled():
                outs.append(checkpoint(self.attn, t, None, m, use_reentrant=False))
            else:
                outs.append(self.attn(t, src_key_padding_mask=m))
        return torch.cat(outs, dim=0)

    def forward(self, players, ball):              # (B, T, 2, 23, F), (B, T, 4)
        B, T, _, N, F = players.shape
        present = torch.cat([players[:, :, 0, :, -1], players[:, :, 1, :, -1]], dim=2)   # (B, T, 46)
        h = self.mlp(players)                                                            # (B, T, 2, 23, W)
        h = (h + self.team[None, None, :, None, :]).reshape(B, T, 2 * N, -1)
        tok = torch.cat([h, self.ball(ball)[:, :, None, :]], dim=2).reshape(B * T, 2 * N + 1, -1)
        pad = torch.cat([present < 0.5, torch.zeros_like(present[..., :1], dtype=torch.bool)], dim=2)
        out = self._attend(tok, pad.reshape(B * T, -1)).reshape(B, T, 2 * N + 1, -1)
        pl = out[:, :, :2 * N] * present[..., None]                                      # absent players -> 0
        pooled = []
        for k in range(2):
            hk, pk = pl[:, :, k * N:(k + 1) * N], present[:, :, k * N:(k + 1) * N, None]
            count = pk.sum(dim=2).clamp(min=1.0)
            mx = (hk + (pk - 1.0) * 1e4).max(dim=2).values
            mx = torch.where(pk.sum(dim=2) > 0, mx, torch.zeros_like(mx))
            pooled.append(torch.cat([hk.sum(dim=2) / count, mx], dim=-1))
        return pooled[0], pooled[1], out[:, :, -1], pl[:, :, :N], pl[:, :, N:]


class DilatedResidual(nn.Module):
    def __init__(self, width, dilation, dropout):
        super().__init__()
        self.conv = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
        self.out = nn.Conv1d(width, width, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                            # (B, C, T)
        return x + self.drop(self.out(torch.relu(self.conv(x))))


class TCNStage(nn.Module):
    """1x1 in, dilated residual layers at `width`, then LayerNorm and a projection to `out_dim` (keeps the
    heads' input on the same scale as the GRU's)."""

    def __init__(self, in_dim, width, layers, dropout, out_dim):
        super().__init__()
        self.inp = nn.Conv1d(in_dim, width, 1)
        self.layers = nn.ModuleList([DilatedResidual(width, 2 ** i, dropout) for i in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, out_dim)

    def forward(self, x):                            # (B, T, in) -> (B, T, out_dim)
        h = self.inp(x.transpose(1, 2))
        for layer in self.layers:
            h = layer(h)
        return torch.tanh(self.out(self.norm(h.transpose(1, 2))))


class PossessionNet(nn.Module):
    def __init__(self, set_width=32, hidden=128, layers=2, dropout=0.2, per_player=PER_PLAYER, possessor=False,
                 encoder="set", attn_layers=1, boundary=False, state_aux=False, ball_dim=4,
                 temporal="gru", tcn_layers=10, tcn_stages=1):
        super().__init__()
        self.temporal = temporal
        self.encoder = encoder
        if encoder == "attn":
            self.enc = CrossTeamAttention(set_width, per_player, layers=attn_layers, ball_dim=ball_dim)
        else:
            self.enc = SetEncoder(set_width, per_player)
        in_dim = 2 * (2 * set_width) + (2 * TEAM_SHAPE + 6) + ball_dim + (set_width if encoder == "attn" else 0)
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        if temporal == "tcn":                        # same output width as the bidirectional GRU (2 * hidden)
            self.tcn = TCNStage(hidden, hidden, tcn_layers, dropout, 2 * hidden)
            self.refine = nn.ModuleList([TCNStage(3, hidden // 2, tcn_layers, dropout, 2 * hidden) for _ in range(tcn_stages - 1)])
            self.refine_head = nn.ModuleList([nn.Linear(2 * hidden, 3) for _ in range(tcn_stages - 1)])
        else:
            self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                              dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(2 * hidden, 3)
        self.possessor = possessor
        if possessor:                                # auxiliary: which player has the ball (training signal only)
            self.who = nn.Sequential(nn.Linear(set_width + 2 * hidden, 64), nn.ReLU(), nn.Linear(64, 1))
        self.boundary, self.state_aux = boundary, state_aux
        if boundary:
            self.bnd = nn.Linear(2 * hidden, 2)      # [boundary logit, displacement / 12 frames]
        if state_aux:
            self.state = nn.Linear(2 * hidden, 3)    # hold / flight / dead logits

    def forward(self, players, teams, ball, aux=False, extra=False):   # (B,T,2,23,6), (B,T,30), (B,T,4)
        if self.encoder == "attn":
            e0, e1, b, h0, h1 = self.enc(players, ball)
            x = torch.cat([e0, e1, teams, ball, b], dim=-1)
        else:
            e0, h0 = self.enc(players[:, :, 0], per_player=True)
            e1, h1 = self.enc(players[:, :, 1], per_player=True)
            x = torch.cat([e0, e1, teams, ball], dim=-1)
        x = self.proj(x)
        stage_logits = []
        if self.temporal == "tcn":
            x = self.tcn(x)
            logits = self.head(x)
            for stage, head in zip(self.refine, self.refine_head):   # MS-TCN refinement on the probabilities
                stage_logits.append(logits)
                x = stage(torch.softmax(logits, dim=-1))
                logits = head(x)
        else:
            x, _ = self.gru(x)
            logits = self.head(x)                    # (B, T, 3) logits for H, A, D
        if extra:                                    # the boundary / state heads, as a dict
            out = {"logits": logits, "hidden": x}     # hidden: the temporal features (self-supervised heads)
            if stage_logits:
                out["stage_logits"] = stage_logits   # earlier MS-TCN stages, trained with the same loss
            if self.boundary:
                bd = self.bnd(x)
                out["boundary"], out["disp"] = bd[..., 0], bd[..., 1]
            if self.state_aux:
                out["state"] = self.state(x)
            if aux and self.possessor:
                out["who"] = self._who(players, h0, h1, x)
            return out
        if not aux:
            return logits
        return logits, self._who(players, h0, h1, x)

    def _who(self, players, h0, h1, x):
        h = torch.cat([h0, h1], dim=2)               # (B, T, 46, set_width), zero for absent players
        ctx = x[:, :, None, :].expand(-1, -1, h.shape[2], -1)
        who = self.who(torch.cat([h, ctx], dim=-1))[..., 0]       # (B, T, 46)
        present = torch.cat([players[:, :, 0, :, -1], players[:, :, 1, :, -1]], dim=2)
        return who + (present - 1.0) * 1e4           # absent players cannot be the possessor
