import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import *


def load_disparity(data_root: Path, method: str, rock_id: str) -> np.ndarray:
    path = data_root / method.lower() / rock_id / "disparity.npy"
    disparity = np.load(path)
    return disparity.astype(np.float32)


def warp_right_to_left(right_gray: np.ndarray, disparity: np.ndarray):
    h, w = right_gray.shape
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    map_x = xs - disparity.astype(np.float32)
    map_y = ys

    warped = cv2.remap(
        right_gray,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.nan,
    )

    in_bounds = np.isfinite(disparity) & (disparity > 0) & (map_x >= 0) & (map_x <= w - 1)
    return warped, in_bounds


def gradient_magnitude(img: np.ndarray):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def edge_alignment_score(left_gray: np.ndarray, disparity: np.ndarray, valid: np.ndarray):
    img_grad = gradient_magnitude(left_gray)
    disp_clean = np.where(valid, disparity, np.nan)

    disp_for_grad = np.nan_to_num(disp_clean, nan=0.0)
    disp_grad = gradient_magnitude(disp_for_grad)

    mask = valid & np.isfinite(img_grad) & np.isfinite(disp_grad)
    if mask.sum() < 100:
        return np.nan

    a = img_grad[mask].reshape(-1)
    b = disp_grad[mask].reshape(-1)

    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])


def smoothness_score(disparity: np.ndarray, valid: np.ndarray):
    disp = np.where(valid, disparity, np.nan)

    dx = np.abs(np.diff(disp, axis=1))
    dy = np.abs(np.diff(disp, axis=0))

    dx = dx[np.isfinite(dx)]
    dy = dy[np.isfinite(dy)]

    vals = np.concatenate([dx, dy])
    if vals.size == 0:
        return np.nan

    return float(np.nanmedian(vals))


def evaluate_method(method: str, rock_id: str, disparity: np.ndarray, left_gray: np.ndarray, right_gray: np.ndarray, params: dict):
    h, w = left_gray.shape

    if disparity.shape != left_gray.shape:
        disparity = cv2.resize(disparity, (w, h), interpolation=cv2.INTER_LINEAR)

    valid_disp = np.isfinite(disparity) & (disparity > 0.1)
    depth = disparity_to_depth(disparity, params)
    valid_depth = np.isfinite(depth) & (depth > 0.05) & (depth < 50.0)

    warped_right, reproj_valid = warp_right_to_left(right_gray, disparity)
    reproj_mask = valid_disp & reproj_valid & np.isfinite(warped_right)

    abs_error = np.abs(left_gray - warped_right)
    reproj_mae = float(np.nanmean(abs_error[reproj_mask])) if reproj_mask.sum() > 0 else np.nan

    disp_valid_values = disparity[valid_disp]
    depth_valid_values = depth[valid_depth]

    row = {
        "rock_id": rock_id,
        "method": method,
        "valid_disparity_fraction": float(valid_disp.mean()),
        "valid_depth_fraction_0p05_50m": float(valid_depth.mean()),
        "reprojection_valid_fraction": float(reproj_mask.mean()),
        "photometric_mae": reproj_mae,
        "edge_alignment_corr": edge_alignment_score(left_gray, disparity, valid_disp),
        "median_abs_disparity_gradient": smoothness_score(disparity, valid_disp),
        "disp_median": float(np.nanmedian(disp_valid_values)) if disp_valid_values.size else np.nan,
        "depth_median": float(np.nanmedian(depth_valid_values)) if depth_valid_values.size else np.nan,
    }

    return row, warped_right, abs_error, reproj_mask


def save_eval_plot(out_path: Path, rock_id: str, left_rgb: np.ndarray, rows: dict, artifacts: dict):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), gridspec_kw={"hspace": 0.001})

    axes[0, 0].imshow(left_rgb)
    axes[0, 0].set_title("Left image", fontsize=18)
    axes[0, 0].axis("off")

    methods = ["SGBM", "RAFT"]

    all_disp = []
    for method in methods:
        disp = artifacts[method]["disparity"]
        valid = np.isfinite(disp) & (disp > 0.1)
        if valid.any():
            all_disp.append(disp[valid])

    if all_disp:
        all_disp = np.concatenate(all_disp)
        vmin = float(np.percentile(all_disp, 2))
        vmax = float(np.percentile(all_disp, 98))
    else:
        vmin, vmax = 0.0, 1.0

    for i, method in enumerate(methods):
        disp = artifacts[method]["disparity"]
        warped = artifacts[method]["warped"]
        abs_error = artifacts[method]["abs_error"]
        mask = artifacts[method]["mask"]
        row = rows[method]

        axes[i, 1].imshow(disp, cmap="magma", vmin=vmin, vmax=vmax)
        axes[i, 1].set_title(f"{method} disparity", fontsize=18)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(warped, cmap="gray", vmin=0, vmax=1)
        axes[i, 2].set_title(f"{method} right warped to left", fontsize=18)
        axes[i, 2].axis("off")

        err_vis = np.where(mask, abs_error, np.nan)
        axes[i, 3].imshow(err_vis, cmap="inferno", vmin=0, vmax=0.5)
        axes[i, 3].set_title(
            f"{method} abs error\nMAE={row['photometric_mae']:.4f}, valid={row['valid_disparity_fraction']:.2f}",
            fontsize=18,
        )
        axes[i, 3].axis("off")

    axes[1, 0].imshow(left_rgb)
    axes[1, 0].set_title("Left image", fontsize=18)
    axes[1, 0].axis("off")

    plt.suptitle(f"Stereo evaluation: {rock_id}", fontsize=18)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()


def evaluate_pair(rock_id: str, data_root: Path, params: dict):
    left_gray, right_gray, left_rgb = load_stereo_pair(rock_id, data_root, params)

    rows = {}
    artifacts = {}

    for method in ["SGBM", "RAFT"]:
        disparity = load_disparity(data_root, method, rock_id)
        row, warped, abs_error, mask = evaluate_method(
            method=method,
            rock_id=rock_id,
            disparity=disparity,
            left_gray=left_gray,
            right_gray=right_gray,
            params=params,
        )
        rows[method] = row
        artifacts[method] = {
            "disparity": disparity,
            "warped": warped,
            "abs_error": abs_error,
            "mask": mask,
        }

    out_path = data_root / "eval_sgbm_vs_raft" / f"{rock_id}_eval.png"
    save_eval_plot(out_path, rock_id, left_rgb, rows, artifacts)

    return [rows["SGBM"], rows["RAFT"]]


def find_rock_ids(data_root: Path):
    left_dir = data_root / "rocks" / "left"
    return sorted(p.stem for p in left_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"])


def summarize_results(df: pd.DataFrame):
    metrics = [
        "valid_disparity_fraction",
        "valid_depth_fraction_0p05_50m",
        "reprojection_valid_fraction",
        "photometric_mae",
        "edge_alignment_corr",
        "median_abs_disparity_gradient",
        "disp_median",
        "depth_median",
    ]

    summary = df.groupby("method")[metrics].mean()
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate SGBM vs RAFT stereo outputs without ground-truth depth.")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--rock_id", type=str, default=None)
    parser.add_argument("--camera_calib_dir", type=str, default=None, help="Directory containing per-pair calibration JSON files.")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    calib_dir = Path(args.camera_calib_dir) if args.camera_calib_dir else None

    if args.rock_id:
        rock_ids = [args.rock_id]
    else:
        rock_ids = find_rock_ids(data_root)

    all_rows = []

    for i, rock_id in enumerate(rock_ids, 1):
        print(f"[{i}/{len(rock_ids)}] evaluating {rock_id}")
        params = load_camera_params(rock_id, data_root, calib_dir)
        rows = evaluate_pair(
            rock_id=rock_id,
            data_root=data_root,
            params=params,
        )
        all_rows.extend(rows)

    out_dir = data_root / "eval_sgbm_vs_raft"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(all_rows)
    csv_path = out_dir / "sgbm_vs_raft_metrics.csv"
    summary_path = out_dir / "sgbm_vs_raft_summary.csv"

    df.to_csv(csv_path, index=False)
    summary = summarize_results(df)
    summary.to_csv(summary_path)


if __name__ == "__main__":
    main()