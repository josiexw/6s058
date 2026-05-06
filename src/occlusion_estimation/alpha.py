from pathlib import Path
import numpy as np
import open3d as o3d

from utils import *


METHODS = ["sgbm", "raft"]
DATA_ROOT = Path("data")
ALPHA_FRAC = 0.02
CROP_EXPANSION_FRAC = 0.15


def compute_alpha(pcd: o3d.geometry.PointCloud):
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = np.asarray(bbox.get_extent(), dtype=np.float64)
    diag = float(np.linalg.norm(extent))

    if not np.isfinite(diag) or diag <= 1e-8:
        raise RuntimeError("Invalid point-cloud bounding box.")

    alpha = ALPHA_FRAC * diag

    if not np.isfinite(alpha) or alpha <= 1e-8:
        raise RuntimeError("Invalid alpha value.")

    return alpha


def crop_mesh_to_input_bounds(mesh: o3d.geometry.TriangleMesh, pcd: o3d.geometry.PointCloud):
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = np.asarray(bbox.get_extent(), dtype=np.float64)
    expansion = CROP_EXPANSION_FRAC * extent

    min_bound = np.asarray(bbox.min_bound) - expansion
    max_bound = np.asarray(bbox.max_bound) + expansion

    crop_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    cropped = mesh.crop(crop_box)

    if len(cropped.vertices) == 0 or len(cropped.triangles) == 0:
        raise RuntimeError("Cropping removed the entire alpha-shape mesh.")

    return cropped


def reconstruct_alpha_shape(pcd: o3d.geometry.PointCloud):
    alpha = compute_alpha(pcd)
    tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd,
        alpha,
        tetra_mesh,
        pt_map,
    )

    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise RuntimeError("Alpha-shape reconstruction produced an empty mesh.")

    mesh = crop_mesh_to_input_bounds(mesh, pcd)

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()

    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise RuntimeError("Final alpha-shape mesh is empty.")

    return mesh, alpha


def process_rock(data_root: Path, method: str, rock_id: str):
    out_dir = data_root / "geometry_completion" / "alpha_shape" / method / rock_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Alpha] Processing {method}/{rock_id}")

    depth = load_depth(data_root, method, rock_id)
    points = depth_to_points(depth, DEFAULT_CAMERA_PARAMS)
    pcd = make_point_cloud(points)
    pcd = clean_point_cloud(pcd)
    mesh, alpha = reconstruct_alpha_shape(pcd)

    save_point_cloud(out_dir / "visible_points.ply", pcd)
    save_mesh(out_dir / "alpha_mesh.ply", mesh)
    np.save(out_dir / "alpha.npy", np.array([alpha], dtype=np.float32))


def main():
    for method in METHODS:
        if not (DATA_ROOT / method).exists():
            print(f"Skipping {method}: missing {DATA_ROOT / method}")
            continue

        rock_ids = find_rock_ids(DATA_ROOT, method)
        print(f"Found {len(rock_ids)} rocks under {DATA_ROOT / method}")

        for rock_id in rock_ids:
            try:
                process_rock(DATA_ROOT, method, rock_id)
            except Exception as e:
                print(f"Skipping {method}/{rock_id}: {e}")


if __name__ == "__main__":
    main()
