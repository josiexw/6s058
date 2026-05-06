import os
from pathlib import Path
import numpy as np
import open3d as o3d
import subprocess
import sys

from utils import *


METHODS = ["sgbm", "raft"]
DATA_ROOT = Path("data")
NORMAL_RADIUS_FRAC = 0.04
NORMAL_MAX_NN = 30

POISSON_DEPTH = 7
POISSON_SCALE = 1.2
DENSITY_QUANTILE = 0.02
CROP_EXPANSION_FRAC = 0.15


def estimate_normals(pcd: o3d.geometry.PointCloud):
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = np.asarray(bbox.get_extent(), dtype=np.float64)
    diag = float(np.linalg.norm(extent))
    radius = NORMAL_RADIUS_FRAC * diag

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=NORMAL_MAX_NN,
        )
    )

    pcd.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, 0.0]))
    return pcd


def crop_mesh_to_input_bounds(mesh: o3d.geometry.TriangleMesh, pcd: o3d.geometry.PointCloud):
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = np.asarray(bbox.get_extent(), dtype=np.float64)
    expansion = CROP_EXPANSION_FRAC * extent

    min_bound = np.asarray(bbox.min_bound) - expansion
    max_bound = np.asarray(bbox.max_bound) + expansion

    crop_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    cropped = mesh.crop(crop_box)

    if len(cropped.vertices) == 0 or len(cropped.triangles) == 0:
        raise RuntimeError("Cropping removed the entire Poisson mesh.")

    return cropped


def reconstruct_poisson(pcd: o3d.geometry.PointCloud):
    pcd = estimate_normals(pcd)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=POISSON_DEPTH,
        scale=POISSON_SCALE,
        linear_fit=False,
    )

    densities = np.asarray(densities)
    density_threshold = np.quantile(densities, DENSITY_QUANTILE)
    remove_mask = densities < density_threshold

    mesh.remove_vertices_by_mask(remove_mask)
    mesh = crop_mesh_to_input_bounds(mesh, pcd)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    return mesh


def process_rock(data_root: Path, method: str, rock_id: str):
    out_dir = data_root / "geometry_completion" / "poisson" / method / rock_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Poisson] Processing {method}/{rock_id}")

    depth = load_depth(data_root, method, rock_id)
    points = depth_to_points(depth, DEFAULT_CAMERA_PARAMS)
    pcd = make_point_cloud(points)
    pcd = clean_point_cloud(pcd)

    mesh = reconstruct_poisson(pcd)

    save_point_cloud(out_dir / "visible_points.ply", pcd)
    save_mesh(out_dir / "poisson_mesh.ply", mesh)


def main():
    method = os.environ.get("POISSON_METHOD")
    rock_id = os.environ.get("POISSON_ROCK_ID")

    if method is not None and rock_id is not None:
        process_rock(DATA_ROOT, method, rock_id)
        return

    for method in METHODS:
        if not (DATA_ROOT / method).exists():
            print(f"Skipping {method}: missing {DATA_ROOT / method}")
            continue

        rock_ids = find_rock_ids(DATA_ROOT, method)
        print(f"Found {len(rock_ids)} rocks under {DATA_ROOT / method}")

        for rock_id in rock_ids:
            try:
                subprocess.run(
                    [sys.executable, __file__],
                    env={
                        **os.environ,
                        "POISSON_METHOD": method,
                        "POISSON_ROCK_ID": rock_id,
                    },
                    stderr=subprocess.DEVNULL,
                )
            except Exception as e:
                print(f"Skipping {method}/{rock_id}: {e}")


if __name__ == "__main__":
    main()
