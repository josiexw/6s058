from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from utils import clean_mesh, load_mesh


METHODS = ["sgbm", "raft"]
INPUT_ROOT = Path("data/geometry_completion/printable")
PRINTABLE_BLUE = (0.10, 0.42, 0.78, 1.0)
PREVIEW_VOXEL_RESOLUTION = 40
AXIS_TITLE_FONTSIZE = 22
FIGURE_TITLE_FONTSIZE = 24
AXIS_ZOOM = 0.88
VIEW_ELEVATION = 20
VIEW_AZIMUTH = 60
STRATEGIES = [
    ("poisson_only", "Poisson"),
    ("alpha_only", "Alpha"),
    ("hybrid", "Hybrid"),
    ("3dgan", "3D-GAN"),
]


def set_axes_equal(ax, bounds):
    mins = bounds[0]
    maxs = bounds[1]
    centers = (mins + maxs) / 2.0
    spans = maxs - mins
    radius = max(spans) * AXIS_ZOOM / 2.0

    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def add_mesh_to_axis(ax, mesh: trimesh.Trimesh, bounds, elev: float, azim: float, title: str):
    vertices = mesh.vertices
    faces = mesh.faces
    triangles = vertices[faces]

    collection = Poly3DCollection(
        triangles,
        linewidths=0.05,
        edgecolors="k",
        alpha=1.0,
    )
    collection.set_facecolor(PRINTABLE_BLUE)

    ax.add_collection3d(collection)

    set_axes_equal(ax, bounds)

    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(title, fontsize=AXIS_TITLE_FONTSIZE, pad=10)


def add_missing_to_axis(ax, title: str):
    ax.text(
        0.5,
        0.5,
        0.5,
        "Missing",
        ha="center",
        va="center",
        transform=ax.transAxes,
    )
    ax.set_axis_off()
    ax.set_title(title, fontsize=AXIS_TITLE_FONTSIZE, pad=10)


def make_preview_mesh(mesh: trimesh.Trimesh):
    bounds = mesh.bounds
    longest_side = float(np.max(bounds[1] - bounds[0]))
    pitch = longest_side / PREVIEW_VOXEL_RESOLUTION
    voxels = mesh.voxelized(pitch=pitch).fill()
    preview = voxels.marching_cubes
    return clean_mesh(preview)


def render_mesh_comparison(meshes, png_path: Path, elev: float, azim: float, title: str):
    preview_meshes = []

    for strategy, label, mesh in meshes:
        if mesh is None:
            preview_meshes.append((strategy, label, None))
        else:
            preview_meshes.append((strategy, label, make_preview_mesh(mesh)))

    available_meshes = [mesh for _, _, mesh in preview_meshes if mesh is not None]
    bounds_min = np.min(np.stack([mesh.bounds[0] for mesh in available_meshes], axis=0), axis=0)
    bounds_max = np.max(np.stack([mesh.bounds[1] for mesh in available_meshes], axis=0), axis=0)
    bounds = np.stack([bounds_min, bounds_max], axis=0)

    fig = plt.figure(figsize=(5 * len(preview_meshes), 5), dpi=180)

    for idx, (_, label, mesh) in enumerate(preview_meshes, start=1):
        ax = fig.add_subplot(1, len(preview_meshes), idx, projection="3d")

        if mesh is None:
            add_missing_to_axis(ax, label)
        else:
            add_mesh_to_axis(ax, mesh, bounds, elev=elev, azim=azim, title=label)

    fig.suptitle(title, fontsize=FIGURE_TITLE_FONTSIZE, y=0.98)

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.84, wspace=0.02)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path, bbox_inches="tight", pad_inches=0.01, transparent=False)
    plt.close(fig)
    crop_png_whitespace(png_path)


def crop_png_whitespace(png_path: Path, margin: int = 12):
    try:
        from PIL import Image, ImageChops
    except ImportError:
        return

    image = Image.open(png_path).convert("RGB")
    background = Image.new("RGB", image.size, (255, 255, 255))
    diff = ImageChops.difference(image, background)
    bbox = diff.getbbox()

    if bbox is None:
        return

    left = max(0, bbox[0] - margin)
    upper = max(0, bbox[1] - margin)
    right = min(image.size[0], bbox[2] + margin)
    lower = min(image.size[1], bbox[3] + margin)

    image.crop((left, upper, right, lower)).save(png_path)


def collect_printable_groups(input_root: Path, method: str):
    method_root = input_root / method
    if not method_root.exists():
        raise FileNotFoundError(f"Missing method folder: {method_root}")

    groups = []
    for rock_dir in sorted(method_root.iterdir()):
        if not rock_dir.is_dir():
            continue

        obj_paths = {
            strategy: rock_dir / f"printable_{strategy}.obj"
            for strategy, _ in STRATEGIES
        }

        if any(path.exists() for path in obj_paths.values()):
            groups.append((rock_dir.name, obj_paths))

    return groups


def display_title(method: str, rock_name: str):
    pair_id = rock_name.split("_L_")[0]
    return f"{method} | {pair_id}"


def main():
    for method in METHODS:
        groups = collect_printable_groups(INPUT_ROOT, method)
        for rock_name, obj_paths in groups:
            print(f"Rendering {method}/{rock_name}", flush=True)
            meshes = []
            for strategy, label in STRATEGIES:
                obj_path = obj_paths[strategy]
                mesh = load_mesh(obj_path) if obj_path.exists() else None
                meshes.append((strategy, label, mesh))

            png_path = INPUT_ROOT / method / rock_name / "printable_comparison.png"
            render_mesh_comparison(
                meshes,
                png_path,
                elev=VIEW_ELEVATION,
                azim=VIEW_AZIMUTH,
                title=display_title(method, rock_name),
            )


if __name__ == "__main__":
    main()