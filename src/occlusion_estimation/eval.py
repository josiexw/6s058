import csv
import math
from pathlib import Path
import numpy as np
import open3d as o3d


RECON_METHODS = {
    "alpha_shape": {
        "display_name": "alpha_shape",
        "mesh_name": "alpha_mesh.ply",
    },
    "poisson": {
        "display_name": "poisson",
        "mesh_name": "poisson_mesh.ply",
    },
    "3dgan": {
        "display_name": "3dgan",
        "mesh_name": "3dgan_mesh.ply",
    },
}


METHODS = ["sgbm", "raft"]
DATA_ROOT = Path("data")
GEOMETRY_ROOT = DATA_ROOT / "geometry_completion"
OUT_DIR = DATA_ROOT / "geometry_completion" / "eval"
N_SAMPLE = 5000
TAU_FRAC = 0.01


def load_mesh(path: Path):
    mesh = o3d.io.read_triangle_mesh(str(path))
    return mesh


def load_visible_points(path: Path):
    pcd = o3d.io.read_point_cloud(str(path))
    points = np.asarray(pcd.points)
    return points[np.isfinite(points).all(axis=1)]


def point_bbox_diag(points):
    extent = points.max(axis=0) - points.min(axis=0)
    return float(np.linalg.norm(extent))


def connected_component_stats(mesh):
    _, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    num_components = int(len(cluster_n_triangles))
    largest_component_fraction = float(cluster_n_triangles.max() / len(mesh.triangles))
    return num_components, largest_component_fraction


def sample_mesh_points(mesh, n_points):
    pcd = mesh.sample_points_uniformly(number_of_points=n_points)
    return np.asarray(pcd.points)


def nearest_neighbor_distances(src_points, dst_points):
    src_points = np.asarray(src_points, dtype=np.float64)
    dst_points = np.asarray(dst_points, dtype=np.float64)
    dst_pcd = o3d.geometry.PointCloud()
    dst_pcd.points = o3d.utility.Vector3dVector(dst_points)
    tree = o3d.geometry.KDTreeFlann(dst_pcd)

    dists = []

    for p in src_points:
        _, _, dist2 = tree.search_knn_vector_3d(p, 1)
        dists.append(math.sqrt(float(dist2[0])))

    return np.asarray(dists, dtype=np.float64)


def mean_distance(dists):
    dists = np.asarray(dists, dtype=np.float64)
    dists = dists[np.isfinite(dists)]

    if len(dists) == 0:
        return float("nan")

    return float(np.mean(dists))


def fraction_leq(dists, threshold):
    dists = np.asarray(dists, dtype=np.float64)
    dists = dists[np.isfinite(dists)]

    if len(dists) == 0 or not np.isfinite(threshold):
        return float("nan")

    return float(np.mean(dists <= threshold))


def fraction_gt(dists, threshold):
    dists = np.asarray(dists, dtype=np.float64)
    dists = dists[np.isfinite(dists)]

    if len(dists) == 0 or not np.isfinite(threshold):
        return float("nan")

    return float(np.mean(dists > threshold))


def evaluate_one(mesh_path: Path, visible_path: Path, n_sample: int, tau_frac: float):
    mesh = load_mesh(mesh_path)
    visible_points = load_visible_points(visible_path)

    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    visible_diag = point_bbox_diag(visible_points)

    n_sample_actual = min(n_sample, max(1000, len(mesh.triangles)))
    mesh_sampled_points = sample_mesh_points(mesh, n_sample_actual)

    visible_to_mesh_sample = nearest_neighbor_distances(visible_points, mesh_sampled_points)
    mesh_sample_to_visible = nearest_neighbor_distances(mesh_sampled_points, visible_points)

    v2m_mean = mean_distance(visible_to_mesh_sample)
    m2v_mean = mean_distance(mesh_sample_to_visible)

    chamfer_mean = float(v2m_mean + m2v_mean)

    num_components, largest_component_fraction = connected_component_stats(mesh)

    if visible_diag > 0:
        normalized_chamfer_mean = chamfer_mean / visible_diag
        tau = tau_frac * visible_diag
    else:
        normalized_chamfer_mean = float("nan")
        tau = float("nan")

    visible_coverage_at_tau = fraction_leq(visible_to_mesh_sample, tau)
    mesh_novelty_at_tau = fraction_gt(mesh_sample_to_visible, tau)

    return {
        "mesh_vertices": len(mesh.vertices),
        "mesh_triangles": len(mesh.triangles),

        "num_components": num_components,
        "largest_component_fraction": largest_component_fraction,

        "visible_coverage_at_tau": visible_coverage_at_tau,
        "mesh_novelty_at_tau": mesh_novelty_at_tau,

        "normalized_chamfer_mean": normalized_chamfer_mean,
    }


def find_jobs(geometry_root: Path):
    jobs = []
    missing = []

    for method in METHODS:
        for recon_folder, info in RECON_METHODS.items():
            display_name = info["display_name"]
            mesh_name = info["mesh_name"]
            recon_dir = geometry_root / recon_folder / method

            if not recon_dir.exists():
                missing.append({
                    "depth_method": method,
                    "recon_method": display_name,
                    "recon_folder": recon_folder,
                    "rock_id": "",
                    "mesh_path": "",
                    "visible_path": "",
                    "error": f"Missing reconstruction directory: {recon_dir}",
                })
                continue

            for rock_dir in sorted(recon_dir.iterdir()):
                if not rock_dir.is_dir():
                    continue

                rock_id = rock_dir.name

                mesh_path = rock_dir / mesh_name
                visible_path = geometry_root / "poisson" / method / rock_id / "visible_points.ply"

                if not mesh_path.exists():
                    missing.append({
                        "depth_method": method,
                        "recon_method": display_name,
                        "recon_folder": recon_folder,
                        "rock_id": rock_id,
                        "mesh_path": str(mesh_path),
                        "visible_path": str(visible_path),
                        "error": f"Missing mesh file: {mesh_name}",
                    })
                    continue

                if not visible_path.exists():
                    missing.append({
                        "depth_method": method,
                        "recon_method": display_name,
                        "recon_folder": recon_folder,
                        "rock_id": rock_id,
                        "mesh_path": str(mesh_path),
                        "visible_path": str(visible_path),
                        "error": "Missing visible_points.ply in poisson output",
                    })
                    continue

                jobs.append({
                    "depth_method": method,
                    "recon_method": display_name,
                    "recon_folder": recon_folder,
                    "rock_id": rock_id,
                    "mesh_path": mesh_path,
                    "visible_path": visible_path,
                })

    return jobs, missing


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    extra_fields = set()
    for row in rows:
        extra_fields.update(row.keys())

    for field in sorted(extra_fields):
        if field not in fieldnames:
            fieldnames.append(field)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, attempted_counts):
    groups = {}

    for row in rows:
        key = (row["depth_method"], row["recon_method"])
        groups.setdefault(key, []).append(row)

    summary_rows = []

    numeric_fields = [
        "mesh_vertices",
        "mesh_triangles",

        "num_components",
        "largest_component_fraction",

        "visible_coverage_at_tau",
        "mesh_novelty_at_tau",

        "normalized_chamfer_mean",
    ]

    all_keys = sorted(set(list(groups.keys()) + list(attempted_counts.keys())))

    for depth_method, recon_method in all_keys:
        group_rows = groups.get((depth_method, recon_method), [])
        num_attempted = attempted_counts.get((depth_method, recon_method), len(group_rows))
        num_successful = len(group_rows)

        if num_attempted > 0:
            success_rate = num_successful / num_attempted
        else:
            success_rate = float("nan")

        out = {
            "depth_method": depth_method,
            "recon_method": recon_method,
            "num_attempted": num_attempted,
            "num_successful": num_successful,
            "num_failed_or_skipped": num_attempted - num_successful,
            "success_rate": success_rate,
        }

        folders_used = sorted(set(row.get("recon_folder", "") for row in group_rows))
        out["recon_folders_used"] = ";".join(folders_used)

        for field in numeric_fields:
            vals = []

            for row in group_rows:
                val = float(row[field])
                if np.isfinite(val):
                    vals.append(val)

            if vals:
                vals = np.asarray(vals, dtype=np.float64)
                out[f"{field}_mean"] = float(np.mean(vals))
            else:
                out[f"{field}_mean"] = float("nan")

        summary_rows.append(out)

    return summary_rows


def count_attempts(jobs, missing):
    attempted_counts = {}

    for job in jobs:
        key = (job["depth_method"], job["recon_method"])
        attempted_counts[key] = attempted_counts.get(key, 0) + 1

    for row in missing:
        rock_id = row.get("rock_id", "")
        if rock_id == "":
            continue

        key = (row["depth_method"], row["recon_method"])
        attempted_counts[key] = attempted_counts.get(key, 0) + 1

    return attempted_counts


def main():
    jobs, missing = find_jobs(GEOMETRY_ROOT)
    attempted_counts = count_attempts(jobs, missing)
    rows = []
    failures = list(missing)

    for i, job in enumerate(jobs, start=1):
        depth_method = job["depth_method"]
        recon_method = job["recon_method"]
        recon_folder = job["recon_folder"]
        rock_id = job["rock_id"]

        print(f"[{i}/{len(jobs)}] Evaluating {depth_method}/{recon_method}/{rock_id}")

        try:
            metrics = evaluate_one(
                mesh_path=job["mesh_path"],
                visible_path=job["visible_path"],
                n_sample=N_SAMPLE,
                tau_frac=TAU_FRAC,
            )

            row = {
                "depth_method": depth_method,
                "recon_method": recon_method,
                "recon_folder": recon_folder,
                "rock_id": rock_id,
                "mesh_path": str(job["mesh_path"]),
                "visible_path": str(job["visible_path"]),
            }

            row.update(metrics)
            rows.append(row)

        except Exception as e:
            print(f"Skipping {depth_method}/{recon_method}/{rock_id}: {e}")
            failures.append({
                "depth_method": depth_method,
                "recon_method": recon_method,
                "recon_folder": recon_folder,
                "rock_id": rock_id,
                "mesh_path": str(job["mesh_path"]),
                "visible_path": str(job["visible_path"]),
                "error": str(e),
            })

    per_mesh_csv = OUT_DIR / "geometry_completion_per_mesh.csv"
    summary_csv = OUT_DIR / "geometry_completion_summary.csv"
    failures_csv = OUT_DIR / "geometry_completion_failures.csv"

    write_csv(per_mesh_csv, rows)

    summary_rows = summarize(rows, attempted_counts)
    write_csv(summary_csv, summary_rows)

    if failures:
        write_csv(failures_csv, failures)


if __name__ == "__main__":
    main()
