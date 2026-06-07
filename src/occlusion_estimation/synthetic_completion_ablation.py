import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parent))

from alpha import reconstruct_alpha_shape
from eval import evaluate_one
from poisson import reconstruct_poisson
from utils import (
    DEFAULT_CAMERA_PARAMS,
    clean_point_cloud,
    depth_to_points,
    make_point_cloud,
    save_mesh,
    save_point_cloud,
)


GAN_MODULE_PATH = Path(__file__).resolve().parent / "3dgan.py"
GAN_SPEC = importlib.util.spec_from_file_location("terrain_3dgan", GAN_MODULE_PATH)
GAN_MODULE = importlib.util.module_from_spec(GAN_SPEC)
GAN_SPEC.loader.exec_module(GAN_MODULE)
completed_depth_to_mesh = GAN_MODULE.completed_depth_to_mesh
generate_depth_conditioned_3dgan_mesh = GAN_MODULE.generate_depth_conditioned_3dgan_mesh

SURFACES = ["ridge", "crater", "ripples", "step"]
OCCLUSIONS = ["center_hole", "diagonal_shadow", "right_band"]
RECON_METHODS = ["alpha_shape", "poisson", "diff_fill"]
DATA_ROOT = Path("data")
OUT_ROOT = DATA_ROOT / "ablation" / "synthetic_completion"
TMP_ROOT = OUT_ROOT / "artifacts"
WIDTH = DEFAULT_CAMERA_PARAMS["width"]
HEIGHT = DEFAULT_CAMERA_PARAMS["height"]
GT_SAMPLE_POINTS = 30000
N_SAMPLE = 5000
TAU_FRAC = 0.01
PER_SCENE_FIELDS = [
    "surface_id",
    "occlusion_id",
    "recon_method",
    "status",
    "num_components",
    "largest_component_fraction",
    "gt_coverage_at_tau",
    "mesh_novelty_at_tau",
    "normalized_chamfer_mean",
    "error",
]
SUMMARY_FIELDS = [
    "recon_method",
    "num_attempted",
    "num_successful",
    "success_rate",
    "num_components_mean",
    "largest_component_fraction_mean",
    "gt_coverage_at_tau_mean",
    "mesh_novelty_at_tau_mean",
    "normalized_chamfer_mean_mean",
]
NUMERIC_FIELDS = [
    "num_components",
    "largest_component_fraction",
    "gt_coverage_at_tau",
    "mesh_novelty_at_tau",
    "normalized_chamfer_mean",
]


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    summary = {}

    for row in rows:
        key = row["recon_method"]
        summary.setdefault(key, []).append(row)

    out_rows = []
    for recon_method, group in sorted(summary.items()):
        out = {
            "recon_method": recon_method,
            "num_attempted": len(group),
            "num_successful": sum(1 for row in group if row["status"] == "ok"),
        }
        out["success_rate"] = out["num_successful"] / out["num_attempted"]

        for field in NUMERIC_FIELDS:
            vals = [float(row[field]) for row in group if row["status"] == "ok"]
            out[f"{field}_mean"] = float(np.mean(vals)) if vals else float("nan")

        out_rows.append(out)

    return out_rows


def grid_coordinates(height: int, width: int):
    rows = np.arange(height, dtype=np.float64)
    cols = np.arange(width, dtype=np.float64)
    grid_rows, grid_cols = np.meshgrid(rows, cols, indexing="ij")
    x = (grid_cols - DEFAULT_CAMERA_PARAMS["cx"]) / DEFAULT_CAMERA_PARAMS["fx"]
    y = -(grid_rows - DEFAULT_CAMERA_PARAMS["cy"]) / DEFAULT_CAMERA_PARAMS["fy"]
    return x, y, grid_rows, grid_cols


def synthetic_depth(surface_id: str, height: int, width: int):
    x, y, _, _ = grid_coordinates(height, width)

    if surface_id == "ridge":
        depth = (
            1.55
            + 0.18 * np.sin(5.0 * x)
            + 0.08 * y
            + 0.10 * np.exp(-((x + 0.22) ** 2 / 0.03 + (y - 0.10) ** 2 / 0.05))
        )
    elif surface_id == "crater":
        radius2 = (x + 0.05) ** 2 + (y - 0.08) ** 2
        depth = (
            1.70
            + 0.05 * x
            - 0.04 * y
            - 0.22 * np.exp(-radius2 / 0.05)
            + 0.08 * np.exp(-radius2 / 0.012)
        )
    elif surface_id == "ripples":
        depth = (
            1.45
            + 0.10 * np.sin(7.0 * x) * np.cos(5.5 * y)
            + 0.06 * np.sin(3.0 * (x + y))
            + 0.05 * np.exp(-((x - 0.18) ** 2 / 0.02 + (y + 0.18) ** 2 / 0.04))
        )
    elif surface_id == "step":
        depth = (
            1.60
            + 0.12 * np.tanh(8.0 * (x - 0.05))
            - 0.06 * y
            + 0.05 * np.sin(4.0 * y)
            + 0.04 * np.exp(-((x + 0.30) ** 2 / 0.04 + y**2 / 0.08))
        )
    else:
        raise ValueError(f"Unknown surface: {surface_id}")

    return np.clip(depth.astype(np.float32), 0.8, 2.5)


def occlusion_mask(occlusion_id: str, height: int, width: int):
    x, y, _, _ = grid_coordinates(height, width)

    if occlusion_id == "center_hole":
        occluded = ((x + 0.10) / 0.24) ** 2 + ((y - 0.02) / 0.30) ** 2 < 1.0
    elif occlusion_id == "diagonal_shadow":
        occluded = (y > 0.45 * x + 0.03) & (x > -0.28)
    elif occlusion_id == "right_band":
        occluded = x > 0.18
    else:
        raise ValueError(f"Unknown occlusion: {occlusion_id}")

    return ~occluded


def mask_depth(depth: np.ndarray, keep_mask: np.ndarray):
    masked = depth.copy()
    masked[~keep_mask] = np.nan
    return masked


def synthetic_point_cloud(depth: np.ndarray, clean: bool):
    points = depth_to_points(depth, DEFAULT_CAMERA_PARAMS)
    pcd = make_point_cloud(points)
    if clean:
        pcd = clean_point_cloud(pcd)
    return pcd


def ground_truth_mesh(depth: np.ndarray):
    _, _, grid_rows, grid_cols = grid_coordinates(depth.shape[0], depth.shape[1])
    return completed_depth_to_mesh(
        depth.astype(np.float64),
        grid_rows,
        grid_cols,
        DEFAULT_CAMERA_PARAMS,
        max_face_depth_jump_frac=1.0,
    )


def ground_truth_point_cloud(mesh: o3d.geometry.TriangleMesh, n_points: int):
    pcd = mesh.sample_points_uniformly(number_of_points=n_points)
    return pcd


def reconstruct_mesh(recon_method: str, depth: np.ndarray, scene_id: str):
    observed_pcd = synthetic_point_cloud(depth, clean=True)

    if len(observed_pcd.points) == 0:
        raise RuntimeError("Observed synthetic point cloud is empty after masking.")

    if recon_method == "alpha_shape":
        mesh, _ = reconstruct_alpha_shape(observed_pcd)
        return observed_pcd, mesh

    if recon_method == "poisson":
        mesh = reconstruct_poisson(observed_pcd)
        return observed_pcd, mesh

    if recon_method == "diff_fill":
        mesh = generate_depth_conditioned_3dgan_mesh(
            depth=depth,
            rock_id=scene_id,
            resolution=96,
            fill_iterations=250,
            smooth_iterations=12,
            terrain_noise_frac=0.015,
            max_face_depth_jump_frac=0.08,
            observed_smooth_blend=0.25,
        )
        return observed_pcd, mesh

    raise ValueError(f"Unknown reconstruction method: {recon_method}")


def evaluate_mesh(mesh_path: Path, visible_path: Path):
    metrics = evaluate_one(
        mesh_path=mesh_path,
        visible_path=visible_path,
        n_sample=N_SAMPLE,
        tau_frac=TAU_FRAC,
    )
    return {
        "num_components": metrics["num_components"],
        "largest_component_fraction": metrics["largest_component_fraction"],
        "gt_coverage_at_tau": metrics["visible_coverage_at_tau"],
        "mesh_novelty_at_tau": metrics["mesh_novelty_at_tau"],
        "normalized_chamfer_mean": metrics["normalized_chamfer_mean"],
    }


def run(surface_id: str, occlusion_id: str, recon_method: str):
    scene_id = f"{surface_id}__{occlusion_id}"
    out_dir = TMP_ROOT / surface_id / occlusion_id / recon_method
    out_dir.mkdir(parents=True, exist_ok=True)

    depth = synthetic_depth(surface_id, HEIGHT, WIDTH)
    keep_mask = occlusion_mask(occlusion_id, HEIGHT, WIDTH)
    partial_depth = mask_depth(depth, keep_mask)

    reference_mesh = ground_truth_mesh(depth)
    reference_points = ground_truth_point_cloud(reference_mesh, GT_SAMPLE_POINTS)
    reference_points_path = out_dir / "reference_points.ply"
    save_point_cloud(reference_points_path, reference_points)
    np.save(out_dir / "full_depth.npy", depth)
    np.save(out_dir / "partial_depth.npy", partial_depth)
    save_mesh(out_dir / "reference_mesh.ply", reference_mesh)

    observed_pcd, mesh = reconstruct_mesh(recon_method, partial_depth, scene_id)
    observed_points_path = out_dir / "observed_points.ply"
    mesh_path = out_dir / f"{recon_method}.ply"
    save_point_cloud(observed_points_path, observed_pcd)
    save_mesh(mesh_path, mesh)

    metrics = evaluate_mesh(mesh_path, reference_points_path)
    return {
        "surface_id": surface_id,
        "occlusion_id": occlusion_id,
        "recon_method": recon_method,
        "status": "ok",
        **metrics,
    }


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


def failure_row(surface_id: str, occlusion_id: str, recon_method: str, result):
    error = (result.stderr or "").strip()
    if result.returncode == 0:
        error = (
            f"Missing JSON worker output. stdout={result.stdout!r} stderr={result.stderr!r}"
        )
    elif not error:
        error = (result.stdout or "").strip()

    return {
        "surface_id": surface_id,
        "occlusion_id": occlusion_id,
        "recon_method": recon_method,
        "status": "failed",
        "error": error,
    }


def run_batch():
    rows = []
    failures = []
    per_scene_path = OUT_ROOT / "synthetic_completion_per_scene.csv"
    summary_path = OUT_ROOT / "synthetic_completion_summary.csv"
    failures_path = OUT_ROOT / "synthetic_completion_failures.csv"
    jobs_completed = 0

    for surface_id in SURFACES:
        for occlusion_id in OCCLUSIONS:
            for recon_method in RECON_METHODS:
                label = f"{surface_id} {occlusion_id} {recon_method}"
                print(f"Running synthetic completion ablation: {label}", flush=True)
                result = subprocess.run(
                    [sys.executable, __file__],
                    env={
                        **os.environ,
                        "SYNTHETIC_ABLATION_MODE": "worker",
                        "SYNTHETIC_SURFACE_ID": surface_id,
                        "SYNTHETIC_OCCLUSION_ID": occlusion_id,
                        "SYNTHETIC_RECON_METHOD": recon_method,
                    },
                    capture_output=True,
                    text=True,
                )
                parsed = parse_worker_result(result)
                if parsed is not None:
                    rows.append(parsed)
                else:
                    failed = failure_row(surface_id, occlusion_id, recon_method, result)
                    rows.append(failed)
                    failures.append(failed)
                jobs_completed += 1
                if jobs_completed % 12 == 0:
                    write_csv(per_scene_path, rows, PER_SCENE_FIELDS)
                    write_csv(summary_path, summarize(rows), SUMMARY_FIELDS)
                    if failures:
                        write_csv(failures_path, failures, PER_SCENE_FIELDS)

    write_csv(per_scene_path, rows, PER_SCENE_FIELDS)
    write_csv(summary_path, summarize(rows), SUMMARY_FIELDS)

    if failures:
        write_csv(failures_path, failures, PER_SCENE_FIELDS)


if __name__ == "__main__":
    if os.environ.get("SYNTHETIC_ABLATION_MODE") == "worker":
        row = run(
            surface_id=os.environ["SYNTHETIC_SURFACE_ID"],
            occlusion_id=os.environ["SYNTHETIC_OCCLUSION_ID"],
            recon_method=os.environ["SYNTHETIC_RECON_METHOD"],
        )
        print(json.dumps(row))
    else:
        run_batch()
