import argparse
from pathlib import Path
import cv2
import numpy as np

from benchmark_utils import (
    evaluate_curiosity_disparity_benchmark,
    evaluate_curiosity_meshes_benchmark,
    evaluate_middlebury_benchmark,
    write_csv,
)
from raft import RAFT_REPO_PATH, RAFT_WEIGHTS_PATH, load_raft_model, run_raft
from utils import get_device


MIDDLEBURY_ROOT = Path("data/middlebury_stereo")
CURIOSITY_ROOT = Path("data/rocks")
DATA_ROOT = Path("data")
OUT_DIR = Path("data/benchmark/stereo_raft")

RAFT_ITERS = 32
MAX_WIDTH = 960


def run_raft_bgr(model, left_bgr, right_bgr, device, iters):
    left_gray = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)

    pred = run_raft(model, left_gray, right_gray, device, iters)

    pred = pred.astype(np.float32)
    pred[~np.isfinite(pred)] = np.nan
    pred[pred <= 0] = np.nan

    return pred
def parse_args():
    parser = argparse.ArgumentParser(description="RAFT benchmark on Middlebury and Curiosity stereo pairs.")
    parser.add_argument("--middlebury_root", type=str, default=str(MIDDLEBURY_ROOT))
    parser.add_argument("--curiosity_root", type=str, default=str(CURIOSITY_ROOT))
    parser.add_argument("--data_root", type=str, default=str(DATA_ROOT))
    parser.add_argument("--out_dir", type=str, default=str(OUT_DIR))
    parser.add_argument("--raft_repo", type=str, default=RAFT_REPO_PATH)
    parser.add_argument("--raft_weights", type=str, default=RAFT_WEIGHTS_PATH)
    parser.add_argument("--raft_iters", type=int, default=RAFT_ITERS)
    parser.add_argument("--max_width", type=int, default=MAX_WIDTH)
    return parser.parse_args()


def main():
    global MIDDLEBURY_ROOT
    global CURIOSITY_ROOT
    global DATA_ROOT
    global OUT_DIR
    global MAX_WIDTH

    args = parse_args()

    MIDDLEBURY_ROOT = Path(args.middlebury_root)
    CURIOSITY_ROOT = Path(args.curiosity_root)
    DATA_ROOT = Path(args.data_root)
    OUT_DIR = Path(args.out_dir)
    MAX_WIDTH = args.max_width
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    model = load_raft_model(args.raft_repo, args.raft_weights, device)
    middlebury_rows = evaluate_middlebury_benchmark(
        middlebury_root=MIDDLEBURY_ROOT,
        out_dir=OUT_DIR,
        max_width=MAX_WIDTH,
        method="raft",
        method_label="RAFT",
        run_disparity=lambda left, right: run_raft_bgr(model, left, right, device, args.raft_iters),
        extra_row_fields={"raft_iters": args.raft_iters},
    )
    curiosity_disparity_rows = evaluate_curiosity_disparity_benchmark(
        curiosity_root=CURIOSITY_ROOT,
        out_dir=OUT_DIR,
        max_width=MAX_WIDTH,
        method="raft",
        method_label="RAFT",
        run_disparity=lambda left, right: run_raft_bgr(model, left, right, device, args.raft_iters),
        extra_row_fields={"raft_iters": args.raft_iters},
    )
    curiosity_mesh_rows = evaluate_curiosity_meshes_benchmark(
        curiosity_root=CURIOSITY_ROOT,
        data_root=DATA_ROOT,
        out_dir=OUT_DIR,
        methods=["raft"],
        per_scene_filename="curiosity_raft_mesh_metrics_per_scene.csv",
        summary_filename="curiosity_raft_mesh_metrics_summary.csv",
        log_label="CURIOSITY RAFT MESH EVALUATION",
    )

    all_rows = middlebury_rows + curiosity_disparity_rows + curiosity_mesh_rows
    write_csv(OUT_DIR / "raft_benchmark_results.csv", all_rows)


if __name__ == "__main__":
    main()
