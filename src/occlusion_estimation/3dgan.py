import hashlib
from pathlib import Path
import numpy as np
import open3d as o3d

from utils import *


METHODS = ["sgbm", "raft"]
DATA_ROOT = Path("data")
DEFAULT_GRID_RESOLUTION = 96
DEFAULT_FILL_ITERATIONS = 250
DEFAULT_SMOOTH_ITERATIONS = 12
DEFAULT_TERRAIN_NOISE_FRAC = 0.015
DEFAULT_MAX_FACE_DEPTH_JUMP_FRAC = 0.08
DEFAULT_OBSERVED_SMOOTH_BLEND = 0.25


def _seed_from_rock_id(rock_id: str) -> int:
    digest = hashlib.sha1(rock_id.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16)


def _box_blur(values: np.ndarray, known_mask: np.ndarray = None, iterations: int = 1):
    result = values.astype(np.float64, copy=True)

    for _ in range(iterations):
        padded = np.pad(result, 1, mode="edge")
        blurred = (
            padded[:-2, :-2]
            + padded[:-2, 1:-1]
            + padded[:-2, 2:]
            + padded[1:-1, :-2]
            + padded[1:-1, 1:-1]
            + padded[1:-1, 2:]
            + padded[2:, :-2]
            + padded[2:, 1:-1]
            + padded[2:, 2:]
        ) / 9.0

        if known_mask is None:
            result = blurred
        else:
            result = np.where(known_mask, result, blurred)

    return result


def _nearest_fill(height: np.ndarray, known_mask: np.ndarray):
    filled = height.astype(np.float64, copy=True)

    if known_mask.all():
        return filled

    try:
        from scipy.ndimage import distance_transform_edt

        _, indices = distance_transform_edt(~known_mask, return_indices=True)
        return filled[indices[0], indices[1]]
    except ImportError:
        pass

    global_mean = float(np.nanmean(filled[known_mask]))
    filled[~known_mask] = global_mean

    return filled


def _diffusion_complete_heightmap(
    height: np.ndarray,
    known_mask: np.ndarray,
    fill_iterations: int,
    smooth_iterations: int,
):
    completed = _nearest_fill(height, known_mask)

    for _ in range(fill_iterations):
        padded = np.pad(completed, 1, mode="edge")
        diffused = (
            padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
        ) / 4.0
        completed = np.where(known_mask, height, diffused)

    completed = _box_blur(completed, known_mask=known_mask, iterations=smooth_iterations)
    completed = np.where(known_mask, height, completed)

    return completed


def _fractal_terrain_noise(shape, rng, amplitude: float):
    if amplitude <= 0:
        return np.zeros(shape, dtype=np.float64)

    noise = np.zeros(shape, dtype=np.float64)
    weight_sum = 0.0

    for scale, weight in [(4, 0.55), (8, 0.30), (16, 0.15)]:
        coarse_shape = (
            max(2, int(np.ceil(shape[0] / scale)) + 1),
            max(2, int(np.ceil(shape[1] / scale)) + 1),
        )
        coarse = rng.normal(0.0, 1.0, size=coarse_shape)
        expanded = np.kron(coarse, np.ones((scale, scale), dtype=np.float64))
        expanded = expanded[: shape[0], : shape[1]]
        expanded = _box_blur(expanded, iterations=max(1, scale // 2))

        noise += weight * expanded
        weight_sum += weight

    if weight_sum > 0:
        noise /= weight_sum

    noise -= float(np.mean(noise))
    std = float(np.std(noise))
    if std > 1e-8:
        noise /= std

    return amplitude * noise


def depth_to_partial_grid(depth: np.ndarray, resolution: int):
    if depth.ndim != 2:
        raise ValueError(f"Depth must be 2D, got shape {depth.shape}")

    h, w = depth.shape
    out_h = int(resolution)
    out_w = max(2, int(round(resolution * w / h)))

    rows = np.linspace(0, h - 1, out_h)
    cols = np.linspace(0, w - 1, out_w)
    row_idx = np.clip(np.rint(rows).astype(np.int64), 0, h - 1)
    col_idx = np.clip(np.rint(cols).astype(np.int64), 0, w - 1)

    sampled = depth[np.ix_(row_idx, col_idx)].astype(np.float64)
    valid_mask = np.isfinite(sampled) & (sampled > MIN_DEPTH) & (sampled < MAX_DEPTH)

    if int(np.count_nonzero(valid_mask)) < MIN_POINTS // 4:
        raise RuntimeError("Too few valid depth cells for terrain-conditioned 3D-GAN completion.")

    partial = np.full_like(sampled, np.nan, dtype=np.float64)
    partial[valid_mask] = sampled[valid_mask]

    grid_rows, grid_cols = np.meshgrid(rows, cols, indexing="ij")

    return partial, valid_mask, grid_rows, grid_cols


def completed_depth_to_mesh(
    completed_depth: np.ndarray,
    grid_rows: np.ndarray,
    grid_cols: np.ndarray,
    params: dict,
    max_face_depth_jump_frac: float,
):
    rows, cols = completed_depth.shape

    z = completed_depth.astype(np.float64)
    x = (grid_cols - params["cx"]) * z / params["fx"]
    y = -(grid_rows - params["cy"]) * z / params["fy"]

    vertices = np.stack(
        [
            x.reshape(-1),
            y.reshape(-1),
            z.reshape(-1),
        ],
        axis=1,
    )

    depth_values = z.reshape(-1)
    finite_depth = depth_values[np.isfinite(depth_values)]
    depth_scale = float(np.percentile(finite_depth, 95) - np.percentile(finite_depth, 5))
    max_face_depth_jump = max(0.25, max_face_depth_jump_frac * max(depth_scale, 1e-6))

    faces = []
    for row in range(rows - 1):
        for col in range(cols - 1):
            v00 = row * cols + col
            v01 = v00 + 1
            v10 = (row + 1) * cols + col
            v11 = v10 + 1

            tri_a = [v00, v10, v01]
            tri_b = [v01, v10, v11]

            if np.ptp(depth_values[tri_a]) <= max_face_depth_jump:
                faces.append(tri_a)

            if np.ptp(depth_values[tri_b]) <= max_face_depth_jump:
                faces.append(tri_b)

    if not faces:
        raise RuntimeError("Depth discontinuity filtering removed all 3D-GAN terrain faces.")

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32))
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()

    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise RuntimeError("Terrain-conditioned 3D-GAN depth mesh is empty.")

    return mesh


def generate_depth_conditioned_3dgan_mesh(
    depth: np.ndarray,
    rock_id: str,
    resolution: int,
    fill_iterations: int,
    smooth_iterations: int,
    terrain_noise_frac: float,
    max_face_depth_jump_frac: float,
    observed_smooth_blend: float,
):
    partial, known_mask, grid_rows, grid_cols = depth_to_partial_grid(depth, resolution)

    completed = _diffusion_complete_heightmap(
        partial,
        known_mask,
        fill_iterations=fill_iterations,
        smooth_iterations=smooth_iterations,
    )

    known_depth = partial[known_mask]
    depth_scale = float(np.percentile(known_depth, 95) - np.percentile(known_depth, 5))
    noise_amplitude = max(depth_scale, 1e-6) * terrain_noise_frac
    rng = np.random.default_rng(_seed_from_rock_id(rock_id))
    noise = _fractal_terrain_noise(completed.shape, rng, noise_amplitude)
    noise = _box_blur(noise, iterations=2)
    completed = np.where(known_mask, partial, completed + noise)
    completed = np.clip(completed, MIN_DEPTH, MAX_DEPTH)

    smoothed = _box_blur(completed, iterations=max(1, smooth_iterations // 2))
    blend = float(np.clip(observed_smooth_blend, 0.0, 1.0))
    completed = np.where(known_mask, (1.0 - blend) * completed + blend * smoothed, smoothed)
    completed = np.clip(completed, MIN_DEPTH, MAX_DEPTH)

    return completed_depth_to_mesh(
        completed,
        grid_rows,
        grid_cols,
        DEFAULT_CAMERA_PARAMS,
        max_face_depth_jump_frac=max_face_depth_jump_frac,
    )


def process_rock(
    data_root: Path,
    method: str,
    rock_id: str,
    resolution: int,
    fill_iterations: int,
    smooth_iterations: int,
    terrain_noise_frac: float,
    max_face_depth_jump_frac: float,
    observed_smooth_blend: float,
):
    out_dir = data_root / "geometry_completion" / "3dgan" / method / rock_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Terrain 3D-GAN] Processing {method}/{rock_id}")

    depth = load_depth(data_root, method, rock_id)
    points = depth_to_points(depth, DEFAULT_CAMERA_PARAMS)
    pcd = make_point_cloud(points)
    pcd = clean_point_cloud(pcd)

    mesh = generate_depth_conditioned_3dgan_mesh(
        depth,
        rock_id=rock_id,
        resolution=resolution,
        fill_iterations=fill_iterations,
        smooth_iterations=smooth_iterations,
        terrain_noise_frac=terrain_noise_frac,
        max_face_depth_jump_frac=max_face_depth_jump_frac,
        observed_smooth_blend=observed_smooth_blend,
    )

    save_point_cloud(out_dir / "visible_points.ply", pcd)
    save_mesh(out_dir / "3dgan_mesh.ply", mesh)


def main():
    for method in METHODS:
        if not (DATA_ROOT / method).exists():
            print(f"Skipping {method}: missing {DATA_ROOT / method}")
            continue

        rock_ids = find_rock_ids(DATA_ROOT, method)

        for rock_id in rock_ids:
            try:
                process_rock(
                    DATA_ROOT,
                    method,
                    rock_id,
                    resolution=DEFAULT_GRID_RESOLUTION,
                    fill_iterations=DEFAULT_FILL_ITERATIONS,
                    smooth_iterations=DEFAULT_SMOOTH_ITERATIONS,
                    terrain_noise_frac=DEFAULT_TERRAIN_NOISE_FRAC,
                    max_face_depth_jump_frac=DEFAULT_MAX_FACE_DEPTH_JUMP_FRAC,
                    observed_smooth_blend=DEFAULT_OBSERVED_SMOOTH_BLEND,
                )
            except Exception as e:
                print(f"Skipping {method}/{rock_id}: {e}")


if __name__ == "__main__":
    main()
