import cv2
import numpy as np
from pathlib import Path
import torch
import json


DEFAULT_CAMERA_PARAMS = {
    "fx": 452.67,
    "fy": 452.67,
    "cx": 320.0,
    "cy": 240.0,
    "baseline_m": 0.2432,
    "width": 640,
    "height": 480,
}


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_camera_params(pair_id: str, data_root: Path, calib_dir: Path = None) -> dict:
    pair_num = pair_id.split("_")[1]
    calib_root = calib_dir or data_root / "rocks" / "calib"
    with (calib_root / f"pair_{pair_num}.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def find_right_path(pair_id: str, right_dir: Path) -> Path:
    pair_num = pair_id.split("_")[1]
    return sorted(right_dir.glob(f"pair_{pair_num}_R_*"))[0]


def load_stereo_pair(pair_id: str, data_root: Path, params: dict):
    left_dir = data_root / "rocks" / "left"
    right_dir = data_root / "rocks" / "right"

    left_path = sorted(left_dir.glob(f"{pair_id}.*"))[0]
    right_path = find_right_path(pair_id, right_dir)

    left_bgr = cv2.imread(str(left_path))
    right_bgr = cv2.imread(str(right_path))

    width = params.get("width")
    height = params.get("height")
    if width is not None and height is not None:
        width = int(width)
        height = int(height)
        if left_bgr.shape[1] != width or left_bgr.shape[0] != height:
            left_bgr = cv2.resize(left_bgr, (width, height))
        if right_bgr.shape[1] != width or right_bgr.shape[0] != height:
            right_bgr = cv2.resize(right_bgr, (width, height))

    left_gray = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)

    return left_gray, right_gray, left_rgb


def disparity_to_depth(disparity: np.ndarray, params: dict) -> np.ndarray:
    fx = params["fx"]
    baseline = params["baseline_m"]
    depth = np.where(np.isfinite(disparity) & (disparity > 0), fx * baseline / disparity, np.nan)
    return depth.astype(np.float32)
