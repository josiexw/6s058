import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from alpha import reconstruct_alpha_shape
from poisson import reconstruct_poisson
from eval import evaluate_one
from utils import (
    DEFAULT_CAMERA_PARAMS,
    clean_point_cloud,
    depth_to_points,
    find_rock_ids,
    load_depth,
    make_point_cloud,
    save_mesh,
    save_point_cloud,
)


GAN_MODULE_PATH = Path(__file__).resolve().parent / "3dgan.py"
GAN_SPEC = importlib.util.spec_from_file_location("terrain_3dgan", GAN_MODULE_PATH)
GAN_MODULE = importlib.util.module_from_spec(GAN_SPEC)
GAN_SPEC.loader.exec_module(GAN_MODULE)
generate_depth_conditioned_3dgan_mesh = GAN_MODULE.generate_depth_conditioned_3dgan_mesh


METHODS = ["sgbm", "raft"]
KEEP_RATIOS = [0.50, 0.25]
RECON_METHODS = ["alpha_shape", "poisson", "diff_fill"]
DATA_ROOT = Path("data")
OUT_ROOT = DATA_ROOT / "ablation" / "completion_sparsity"
TMP_ROOT = OUT_ROOT / "tmp"
N_SAMPLE = 5000
TAU_FRAC = 0.01
PER_SCENE_FIELDS = [
    "depth_method",
    "rock_id",
    "recon_method",
    "keep_ratio",
    "status",
    "num_components",
    "largest_component_fraction",
    "visible_coverage_at_tau",
    "mesh_novelty_at_tau",
    "normalized_chamfer_mean",
    "error",
]
SUMMARY_FIELDS = [
    "depth_method",
    "recon_method",
    "keep_ratio",
    "num_attempted",
    "num_successful",
    "success_rate",
    "num_components_mean",
    "largest_component_fraction_mean",
    "visible_coverage_at_tau_mean",
    "mesh_novelty_at_tau_mean",
    "normalized_chamfer_mean_mean",
]
NUMERIC_FIELDS = [
    "num_components",
    "largest_component_fraction",
    "visible_coverage_at_tau",
    "mesh_novelty_at_tau",
    "normalized_chamfer_mean",
]


def stable_seed(*parts) -> int:
    text = "::".join(str(part) for part in parts)
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)


def mask_depth(depth: np.ndarray, keep_ratio: float, seed: int):
    masked = depth.copy()
    valid = np.isfinite(masked) & (masked > 0)
    valid_idx = np.flatnonzero(valid)

    rng = np.random.default_rng(seed)
    keep_count = max(1, int(round(len(valid_idx) * keep_ratio)))
    keep_idx = rng.choice(valid_idx, size=keep_count, replace=False)
    keep_mask = np.zeros(masked.size, dtype=bool)
    keep_mask[keep_idx] = True
    keep_mask = keep_mask.reshape(masked.shape)

    masked[valid & ~keep_mask] = np.nan
    return masked


def original_visible_point_cloud(depth: np.ndarray):
    points = depth_to_points(depth, DEFAULT_CAMERA_PARAMS)
    pcd = make_point_cloud(points)
    return clean_point_cloud(pcd)


def masked_point_cloud(depth: np.ndarray):
    points = depth_to_points(depth, DEFAULT_CAMERA_PARAMS)
    return clean_point_cloud(make_point_cloud(points))


def reconstruct_mesh(recon_method: str, depth: np.ndarray, rock_id: str):
    pcd = masked_point_cloud(depth)

    if recon_method == "alpha_shape":
        mesh, _ = reconstruct_alpha_shape(pcd)
        return pcd, mesh

    if recon_method == "poisson":
        mesh = reconstruct_poisson(pcd)
        return pcd, mesh

    if recon_method == "diff_fill":
        mesh = generate_depth_conditioned_3dgan_mesh(
            depth=depth,
            rock_id=rock_id,
            resolution=96,
            fill_iterations=250,
            smooth_iterations=12,
            terrain_noise_frac=0.015,
            max_face_depth_jump_frac=0.08,
            observed_smooth_blend=0.25,
        )
        return pcd, mesh

    raise ValueError(f"Unknown reconstruction method: {recon_method}")


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    summary = {}

    for row in rows:
        key = (row["depth_method"], row["recon_method"], row["keep_ratio"])
        summary.setdefault(key, []).append(row)

    out_rows = []
    for key, group in sorted(summary.items()):
        depth_method, recon_method, keep_ratio = key
        out = {
            "depth_method": depth_method,
            "recon_method": recon_method,
            "keep_ratio": keep_ratio,
            "num_attempted": len(group),
            "num_successful": sum(1 for row in group if row["status"] == "ok"),
        }
        out["success_rate"] = out["num_successful"] / out["num_attempted"]

        for field in NUMERIC_FIELDS:
            vals = [float(row[field]) for row in group if row["status"] == "ok"]
            out[f"{field}_mean"] = float(np.mean(vals)) if vals else float("nan")

        out_rows.append(out)

    return out_rows


def parse_worker_result(result: subprocess.CompletedProcess[str]):
    if result.returncode != 0:
        return None

    for line in reversed(result.stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def failure_row(depth_method: str, rock_id: str, recon_method: str, keep_ratio: float, result):
    error = (result.stderr or "").strip()
    if result.returncode == 0:
        error = (
            f"Missing JSON worker output. stdout={result.stdout!r} stderr={result.stderr!r}"
        )
    elif not error:
        error = (result.stdout or "").strip()

    return {
        "depth_method": depth_method,
        "rock_id": rock_id,
        "recon_method": recon_method,
        "keep_ratio": keep_ratio,
        "status": "failed",
        "error": error,
    }


def run(depth_method: str, rock_id: str, keep_ratio: float, recon_method: str):
    depth = load_depth(DATA_ROOT, depth_method, rock_id)
    reference_pcd = original_visible_point_cloud(depth)
    masked_depth = mask_depth(
        depth,
        keep_ratio=keep_ratio,
        seed=stable_seed(depth_method, rock_id, keep_ratio),
    )

    ref_dir = TMP_ROOT / depth_method / rock_id / f"keep_{int(keep_ratio * 100)}"
    ref_dir.mkdir(parents=True, exist_ok=True)
    visible_path = ref_dir / "reference_visible_points.ply"
    save_point_cloud(visible_path, reference_pcd)
    mesh_path = ref_dir / f"{recon_method}.ply"

    _, mesh = reconstruct_mesh(recon_method, masked_depth, rock_id)
    save_mesh(mesh_path, mesh)
    metrics = evaluate_one(
        mesh_path=mesh_path,
        visible_path=visible_path,
        n_sample=N_SAMPLE,
        tau_frac=TAU_FRAC,
    )
    return {
        "depth_method": depth_method,
        "rock_id": rock_id,
        "recon_method": recon_method,
        "keep_ratio": keep_ratio,
        "status": "ok",
        **metrics,
    }


def run_batch():
    rows = []
    failures = []
    per_scene_path = OUT_ROOT / "completion_ablation_per_scene.csv"
    summary_path = OUT_ROOT / "completion_ablation_summary.csv"
    failures_path = OUT_ROOT / "completion_ablation_failures.csv"
    jobs_completed = 0
    for depth_method in METHODS:
        rock_ids = find_rock_ids(DATA_ROOT, depth_method)
        for keep_ratio in KEEP_RATIOS:
            for rock_id in rock_ids:
                for recon_method in RECON_METHODS:
                    print(
                        f"{depth_method} {rock_id} keep={keep_ratio:.2f} {recon_method}",
                        flush=True,
                    )
                    result = subprocess.run(
                        [sys.executable, __file__],
                        env={
                            **os.environ,
                            "ABLATION_MODE": "worker",
                            "ABLATION_DEPTH_METHOD": depth_method,
                            "ABLATION_ROCK_ID": rock_id,
                            "ABLATION_KEEP_RATIO": str(keep_ratio),
                            "ABLATION_RECON_METHOD": recon_method,
                        },
                        capture_output=True,
                        text=True,
                    )
                    parsed = parse_worker_result(result)
                    if parsed is not None:
                        rows.append(parsed)
                    else:
                        failed = failure_row(
                            depth_method, rock_id, recon_method, keep_ratio, result
                        )
                        rows.append(failed)
                        failures.append(failed)
                    jobs_completed += 1
                    if jobs_completed % 50 == 0:
                        write_csv(per_scene_path, rows, PER_SCENE_FIELDS)
                        write_csv(summary_path, summarize(rows), SUMMARY_FIELDS)
                        if failures:
                            write_csv(failures_path, failures, PER_SCENE_FIELDS)

    write_csv(per_scene_path, rows, PER_SCENE_FIELDS)
    write_csv(summary_path, summarize(rows), SUMMARY_FIELDS)

    if failures:
        write_csv(failures_path, failures, PER_SCENE_FIELDS)


if __name__ == "__main__":
    if os.environ.get("ABLATION_MODE") == "worker":
        row = run(
            depth_method=os.environ["ABLATION_DEPTH_METHOD"],
            rock_id=os.environ["ABLATION_ROCK_ID"],
            keep_ratio=float(os.environ["ABLATION_KEEP_RATIO"]),
            recon_method=os.environ["ABLATION_RECON_METHOD"],
        )
        print(json.dumps(row))
    else:
        run_batch()
