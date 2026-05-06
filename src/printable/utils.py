import numpy as np
import trimesh
from pathlib import Path


def remove_duplicate_faces_safe(mesh: trimesh.Trimesh):
    faces_sorted = np.sort(mesh.faces, axis=1)
    _, unique_idx = np.unique(faces_sorted, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    mesh.update_faces(unique_idx)
    mesh.remove_unreferenced_vertices()
    return mesh


def remove_degenerate_faces_safe(mesh: trimesh.Trimesh):
    nondegenerate = mesh.nondegenerate_faces()
    mesh.update_faces(nondegenerate)
    mesh.remove_unreferenced_vertices()
    return mesh


def clean_mesh(mesh: trimesh.Trimesh):
    mesh.remove_unreferenced_vertices()
    mesh = remove_duplicate_faces_safe(mesh)
    mesh = remove_degenerate_faces_safe(mesh)
    mesh.process(validate=True)
    return mesh


def repair_mesh(mesh: trimesh.Trimesh):
    mesh = clean_mesh(mesh)
    trimesh.repair.fix_inversion(mesh)
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fill_holes(mesh)
    mesh = clean_mesh(mesh)
    return mesh


def load_mesh(path: Path):
    mesh = trimesh.load(path, force="mesh", process=False)

    if isinstance(mesh, trimesh.Scene):
        meshes = [
            g for g in mesh.geometry.values()
            if isinstance(g, trimesh.Trimesh) and len(g.vertices) > 0 and len(g.faces) > 0
        ]
        mesh = trimesh.util.concatenate(meshes)

    return clean_mesh(mesh)


def keep_largest_component(mesh: trimesh.Trimesh):
    parts = mesh.split(only_watertight=False)
    largest = max(parts, key=lambda m: len(m.faces))
    return clean_mesh(largest)


def mesh_pitch(mesh: trimesh.Trimesh, target_resolution: int):
    bounds_min = np.asarray(mesh.bounds[0], dtype=np.float64)
    bounds_max = np.asarray(mesh.bounds[1], dtype=np.float64)
    longest_side = float(np.max(bounds_max - bounds_min))
    return longest_side / target_resolution


def combined_mesh_pitch(meshes, target_resolution: int):
    bounds_min = np.min(np.stack([mesh.bounds[0] for mesh in meshes], axis=0), axis=0)
    bounds_max = np.max(np.stack([mesh.bounds[1] for mesh in meshes], axis=0), axis=0)
    longest_side = float(np.max(bounds_max - bounds_min))
    return longest_side / target_resolution


def voxelized_printable_mesh(mesh: trimesh.Trimesh, pitch: float):
    voxels = mesh.voxelized(pitch=pitch)
    printable = voxels.fill().marching_cubes
    printable = repair_mesh(printable)
    printable = keep_largest_component(printable)
    return repair_mesh(printable)