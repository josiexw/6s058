from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


METHODS = ["sgbm", "raft"]
DATA_ROOT = Path("data")
OUT_DIR = Path("data/geometry_completion/comparisons")
NUM_PER_METHOD = 10

RECON_METHODS = {
    "poisson": {
        "title": "Poisson",
        "mesh_name": "poisson_mesh.ply",
    },
    "alpha_shape": {
        "title": "Alpha",
        "mesh_name": "alpha_mesh.ply",
    },
    "3dgan": {
        "title": "3D-GAN",
        "mesh_name": "3dgan_mesh.ply",
    },
}


def collect_candidates(data_root: Path, methods):
    candidates_by_method = {method: [] for method in methods}

    for method in methods:
        rock_ids = set()

        for recon_method, recon_config in RECON_METHODS.items():
            root = data_root / "geometry_completion" / recon_method / method
            if not root.exists():
                continue

            for rock_dir in root.iterdir():
                if not rock_dir.is_dir():
                    continue

                mesh_path = rock_dir / recon_config["mesh_name"]
                if mesh_path.exists():
                    rock_ids.add(rock_dir.name)

        for rock_id in sorted(rock_ids):
            visible_paths = [
                data_root / "geometry_completion" / recon_method / method / rock_id / "visible_points.ply"
                for recon_method in RECON_METHODS
            ]
            item = {
                "method": method,
                "rock_id": rock_id,
                "visible_path": next((path for path in visible_paths if path.exists()), visible_paths[0]),
            }

            for recon_method, recon_config in RECON_METHODS.items():
                item[f"{recon_method}_path"] = (
                    data_root
                    / "geometry_completion"
                    / recon_method
                    / method
                    / rock_id
                    / recon_config["mesh_name"]
                )

            candidates_by_method[method].append(item)

    return candidates_by_method


def load_point_cloud(path: Path):
    pcd = o3d.io.read_point_cloud(str(path))
    return np.asarray(pcd.points)


def load_mesh(path: Path):
    mesh = o3d.io.read_triangle_mesh(str(path))
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    return verts, tris


def compute_global_bounds(point_sets, mesh_sets):
    mins = []
    maxs = []

    for pts in point_sets:
        if pts is not None and len(pts) > 0:
            mins.append(pts.min(axis=0))
            maxs.append(pts.max(axis=0))

    for mesh in mesh_sets:
        if mesh is not None:
            verts, _ = mesh
            if len(verts) > 0:
                mins.append(verts.min(axis=0))
                maxs.append(verts.max(axis=0))

    if not mins:
        return None

    min_xyz = np.min(np.stack(mins, axis=0), axis=0)
    max_xyz = np.max(np.stack(maxs, axis=0), axis=0)

    center = 0.5 * (min_xyz + max_xyz)
    extent = max_xyz - min_xyz
    radius = 0.5 * np.max(extent)

    if radius <= 0:
        radius = 1.0

    return center, radius


def setup_ax(ax, title, center, radius):
    ax.set_title(title)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=20, azim=-60)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


def plot_point_cloud(ax, pts):
    if pts is None or len(pts) == 0:
        ax.text(
            0.5,
            0.5,
            0.5,
            "Missing",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.5)


def plot_mesh(ax, mesh):
    if mesh is None:
        ax.text(
            0.5,
            0.5,
            0.5,
            "Missing",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    verts, tris = mesh

    ax.plot_trisurf(
        verts[:, 0],
        verts[:, 1],
        verts[:, 2],
        triangles=tris,
        linewidth=0.05,
        antialiased=True,
        shade=True,
    )


def save_comparison(item, out_dir: Path):
    visible_pts = load_point_cloud(item["visible_path"])
    meshes = {
        recon_method: load_mesh(item[f"{recon_method}_path"])
        for recon_method in RECON_METHODS
    }

    bounds = compute_global_bounds(
        [visible_pts],
        list(meshes.values()),
    )

    if bounds is None:
        print(f"Skipping {item['method']}/{item['rock_id']}: nothing to render")
        return False

    center, radius = bounds

    num_panels = 1 + len(RECON_METHODS)
    fig = plt.figure(figsize=(5 * num_panels, 5))

    axes = [
        fig.add_subplot(1, num_panels, panel_idx, projection="3d")
        for panel_idx in range(1, num_panels + 1)
    ]

    setup_ax(axes[0], "Visible Points", center, radius)
    plot_point_cloud(axes[0], visible_pts)

    for ax, (recon_method, recon_config) in zip(axes[1:], RECON_METHODS.items()):
        setup_ax(ax, recon_config["title"], center, radius)
        plot_mesh(ax, meshes[recon_method])

    fig.suptitle(f"{item['method']} | {item['rock_id']}", fontsize=14)
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{item['method']}__{item['rock_id']}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return True


def main():
    candidates_by_method = collect_candidates(DATA_ROOT, METHODS)
    selected = []

    for method in METHODS:
        method_candidates = candidates_by_method[method]
        num_available = len(method_candidates)
        num_selected = min(NUM_PER_METHOD, num_available)
        selected.extend(method_candidates[:num_selected])

    for item in selected:
        save_comparison(item, OUT_DIR)


if __name__ == "__main__":
    main()