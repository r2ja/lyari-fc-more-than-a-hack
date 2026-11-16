"""Load the inference architecture and its trained parameters."""
import torch
from lyari.models.players.model import PossessionNet


def load_model(path):
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt["config"]
    from lyari.models.players import features as FE
    FE.EXTRA = bool(cfg.get("extra_clues", False))
    FE.set_control(cfg.get("control_physics", False))
    model = PossessionNet(hidden=cfg["hidden"], per_player=FE.PER_PLAYER + (FE.BALL_REL if cfg.get("ball_rel") else 0),
                          possessor=cfg.get("possessor", 0) > 0, encoder=cfg.get("encoder", "set"),
                          attn_layers=cfg.get("attn_layers", 1), boundary=bool(cfg.get("boundary", False)),
                          state_aux=bool(cfg.get("state_aux", False)),
                          ball_dim=4 + (FE.EXTRA_DIM if FE.EXTRA else 0), temporal=cfg.get("temporal", "gru"),
                          tcn_layers=cfg.get("tcn_layers", 10), tcn_stages=cfg.get("tcn_stages", 1))
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model, cfg
