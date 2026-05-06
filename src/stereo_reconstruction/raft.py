import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from utils import get_device


RAFT_REPO_PATH = "third-party/RAFT-Stereo"
RAFT_WEIGHTS_PATH = "third-party/RAFT-Stereo/models/raftstereo-middlebury.pth"


def load_raft_model(repo_path: str, weights_path: str, device: str):
    sys.path.insert(0, repo_path)
    sys.path.insert(0, os.path.join(repo_path, "core"))
    from raft_stereo import RAFTStereo

    class Args:
        hidden_dims         = [128] * 3
        context_dims        = [128] * 3
        corr_implementation = "reg"
        shared_backbone     = False
        corr_levels         = 4
        corr_radius         = 4
        n_downsample        = 2
        slow_fast_gru       = False
        n_gru_layers        = 3
        mixed_precision     = False
        context_norm        = "batch"

    model = RAFTStereo(Args())

    state = torch.load(weights_path, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]

    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)

    model.eval()
    model = model.to(device)
    return model


def run_raft(model, left_gray: np.ndarray, right_gray: np.ndarray,
             device: str, iters: int = 32) -> np.ndarray:
    def to_tensor(img_gray):
        img = img_gray
        if img.dtype != np.uint8:
            img = img.astype(np.float32)
            if img.max() <= 1.5:
                img = img * 255.0
            img = np.clip(img, 0, 255).astype(np.uint8)

        rgb = np.stack([img] * 3, axis=-1)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float()
        return t.unsqueeze(0).to(device)

    img1 = to_tensor(left_gray)
    img2 = to_tensor(right_gray)

    h, w = left_gray.shape
    pad_h = (-h) % 32
    pad_w = (-w) % 32

    img1 = F.pad(img1, [0, pad_w, 0, pad_h], mode="replicate")
    img2 = F.pad(img2, [0, pad_w, 0, pad_h], mode="replicate")

    with torch.no_grad():
        _, flow_up = model(img1, img2, iters=iters, test_mode=True)

    disparity = -flow_up[0, 0].detach().cpu().numpy()
    disparity = disparity[:h, :w]

    return disparity.astype(np.float32)


def initialize_raft(
    repo_path: str = RAFT_REPO_PATH,
    weights_path: str = RAFT_WEIGHTS_PATH,
):
    device = get_device()
    model = load_raft_model(repo_path, weights_path, device)

    return model, device
