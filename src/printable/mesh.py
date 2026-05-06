from pathlib import Path
import trimesh

from utils import *


METHODS = ["sgbm", "raft"]
STRATEGIES = ["poisson_only", "alpha_only", "hybrid", "3dgan"]
MAX_PRINTABLES_PER_METHOD_STRATEGY = 5
DATA_ROOT = Path("data")
OUTPUT_ROOT = Path("data/geometry_completion/printable")
TARGET_RESOLUTION = 220


def make_hybrid_printable_mesh(poisson_mesh, alpha_mesh, target_resolution):
    pitch = combined_mesh_pitch([poisson_mesh, alpha_mesh], target_resolution)
    combined = trimesh.util.concatenate([poisson_mesh, alpha_mesh])
    combined = repair_mesh(combined)
    return voxelized_printable_mesh(combined, pitch)


def collect_rocks(data_root: Path, method: str, strategy: str):
    if strategy == "hybrid":
        poisson_root = data_root / "geometry_completion" / "poisson" / method
        alpha_root = data_root / "geometry_completion" / "alpha_shape" / method

        rocks = []

        for rock_dir in sorted(poisson_root.iterdir()):
            if not rock_dir.is_dir():
                continue

            poisson_path = rock_dir / "poisson_mesh.ply"
            alpha_path = alpha_root / rock_dir.name / "alpha_mesh.ply"

            if poisson_path.exists() and alpha_path.exists():
                rocks.append(rock_dir.name)

        return rocks
    elif strategy in ["poisson_only", "alpha_only"]:
        recon_method = "poisson" if strategy == "poisson_only" else "alpha_shape"
        recon_root = data_root / "geometry_completion" / recon_method / method

        rocks = []

        for rock_dir in sorted(recon_root.iterdir()):
            if not rock_dir.is_dir():
                continue

            mesh_path = rock_dir / f"{recon_method.split('_')[0]}_mesh.ply"

            if mesh_path.exists():
                rocks.append(rock_dir.name)

        return rocks
    elif strategy == "3dgan":
        recon_root = data_root / "geometry_completion" / "3dgan" / method

        rocks = []

        for rock_dir in sorted(recon_root.iterdir()):
            if not rock_dir.is_dir():
                continue

            mesh_path = rock_dir / "3dgan_mesh.ply"

            if mesh_path.exists():
                rocks.append(rock_dir.name)

        return rocks
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def process_one(data_root, output_root, method, rock_name, strategy, target_resolution):
    if strategy == "hybrid":
        poisson_path = (
            data_root
            / "geometry_completion"
            / "poisson"
            / method
            / rock_name
            / "poisson_mesh.ply"
        )

        alpha_path = (
            data_root
            / "geometry_completion"
            / "alpha_shape"
            / method
            / rock_name
            / "alpha_mesh.ply"
        )

        poisson_mesh = load_mesh(poisson_path)
        alpha_mesh = load_mesh(alpha_path)
        printable_mesh = make_hybrid_printable_mesh(poisson_mesh, alpha_mesh, target_resolution)
    elif strategy in ["poisson_only", "alpha_only"]:
        recon_method = "poisson" if strategy == "poisson_only" else "alpha_shape"
        mesh_path = (
            data_root
            / "geometry_completion"
            / recon_method
            / method
            / rock_name
            / f"{recon_method.split('_')[0]}_mesh.ply"
        )

        mesh = load_mesh(mesh_path)
        printable_mesh = voxelized_printable_mesh(mesh, mesh_pitch(mesh, target_resolution))
    elif strategy == "3dgan":
        mesh_path = (
            data_root
            / "geometry_completion"
            / "3dgan"
            / method
            / rock_name
            / "3dgan_mesh.ply"
        )

        mesh = load_mesh(mesh_path)
        printable_mesh = voxelized_printable_mesh(mesh, mesh_pitch(mesh, target_resolution))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    out_dir = output_root / method / rock_name
    out_dir.mkdir(parents=True, exist_ok=True)

    obj_path = out_dir / f"printable_{strategy}.obj"
    printable_mesh.export(obj_path)

    if not printable_mesh.is_watertight:
        raise RuntimeError(f"Exported mesh is not watertight: {obj_path}")


def main():
    failures = []

    for strategy in STRATEGIES:
        for method in METHODS:
            rocks = collect_rocks(DATA_ROOT, method, strategy)[:MAX_PRINTABLES_PER_METHOD_STRATEGY]

            print(f"\nProcessing {method} with strategy {strategy}: {len(rocks)} rocks")

            for rock_name in rocks:
                try:
                    process_one(
                        data_root=DATA_ROOT,
                        output_root=OUTPUT_ROOT,
                        method=method,
                        rock_name=rock_name,
                        strategy=strategy,
                        target_resolution=TARGET_RESOLUTION,
                    )
                except Exception as e:
                    print(f"ERROR {method}/{rock_name} [{strategy}]: {e}")
                    failures.append((method, rock_name, strategy, str(e)))

    if failures:
        raise RuntimeError(f"{len(failures)} mesh export(s) failed.")


if __name__ == "__main__":
    main()