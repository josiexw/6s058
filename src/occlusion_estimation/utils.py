import numpy as np
from pathlib import Path
import open3d as o3d


DEFAULT_CAMERA_PARAMS = {
    "fx": 452.67,
    "fy": 452.67,
    "cx": 320.0,
    "cy": 240.0,
    "baseline_m": 0.2432,
    "width": 640,
    "height": 480,
}

STRIDE = 2
MIN_DEPTH = 0.05
MAX_DEPTH = 10.0
MIN_POINTS = 200
MAX_POINTS = 30000
VOXEL_SIZE_FRAC = 0.006
OUTLIER_NB_NEIGHBORS = 20
OUTLIER_STD_RATIO = 2.0


def clean_point_cloud(pcd: o3d.geometry.PointCloud):
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = np.asarray(bbox.get_extent(), dtype=np.float64)
    diag = float(np.linalg.norm(extent))
    voxel_size = VOXEL_SIZE_FRAC * diag

    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    pcd, ind = pcd.remove_statistical_outlier(
        nb_neighbors=OUTLIER_NB_NEIGHBORS,
        std_ratio=OUTLIER_STD_RATIO,
    )

    points = np.asarray(pcd.points)

    if len(points) > MAX_POINTS:
        rng = np.random.default_rng(0)
        keep = rng.choice(len(points), size=MAX_POINTS, replace=False)
        points = points[keep]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    points = np.asarray(pcd.points)
    points = np.unique(points, axis=0)
    cleaned = o3d.geometry.PointCloud()
    cleaned.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    return cleaned


def find_rock_ids(data_root: Path, method: str):
    method_dir = data_root / method
    rock_ids = []

    for rock_dir in sorted(method_dir.iterdir()):
        if (rock_dir / "depth.npy").exists():
            rock_ids.append(rock_dir.name)

    return rock_ids


def load_depth(data_root: Path, method: str, rock_id: str):
    depth_path = data_root / method / rock_id / "depth.npy"
    depth = np.load(depth_path).astype(np.float32)
    return depth


def depth_to_points(depth: np.ndarray, params: dict):
    h, w = depth.shape

    fx = params["fx"]
    fy = params["fy"]
    cx = params["cx"]
    cy = params["cy"]

    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    xs = xs[::STRIDE, ::STRIDE]
    ys = ys[::STRIDE, ::STRIDE]
    z = depth[::STRIDE, ::STRIDE]

    valid = np.isfinite(z) & (z > MIN_DEPTH) & (z < MAX_DEPTH)

    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    z = z[valid].astype(np.float32)
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    points = np.stack([x, -y, z], axis=1).astype(np.float32)

    return points


def make_point_cloud(points: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    return pcd


def save_mesh(path: Path, mesh: o3d.geometry.TriangleMesh):
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(path), mesh)


def save_point_cloud(path: Path, pcd: o3d.geometry.PointCloud):
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), pcd)