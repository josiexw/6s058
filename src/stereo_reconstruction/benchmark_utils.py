import csv
from pathlib import Path
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh


IMAGE_LEFT = "im0.png"
IMAGE_RIGHT = "im1.png"
GT_NAMES = ["disp0.pfm", "disp1.pfm"]

MIDDLEBURY_METRICS = [
    "valid_prediction_ratio",
    "mae",
    "rmse",
    "bad_1",
    "bad_4",
]

CURIOSITY_DISPARITY_METRICS = ["valid_disparity_ratio"]

MESH_METRICS = [
    "poisson_mesh_vertices",
    "poisson_mesh_faces",
    "poisson_mesh_components",
    "poisson_mesh_largest_component_ratio",
    "alpha_mesh_vertices",
    "alpha_mesh_faces",
    "alpha_mesh_components",
    "alpha_mesh_largest_component_ratio",
    "printable_hybrid_vertices",
    "printable_hybrid_faces",
    "printable_hybrid_components",
    "printable_hybrid_largest_component_ratio",
]


def read_pfm(path):
    with open(path, "rb") as f:
        header = f.readline().decode("latin-1").rstrip()
        dims = f.readline().decode("latin-1").strip()
        while dims.startswith("#"):
            dims = f.readline().decode("latin-1").strip()

        width, height = map(int, dims.split())
        scale = float(f.readline().decode("latin-1").strip())
        endian = "<" if scale < 0 else ">"

        data = np.fromfile(f, endian + "f")
        channels = 3 if header == "PF" else 1
        shape = (height, width, channels) if channels == 3 else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)

        if channels == 3:
            data = data[:, :, 0]

        return data.astype(np.float32)


def pair_index_from_name(path):
    name = Path(path).name
    parts = name.split("_")
    return parts[1]


def discover_middlebury_scenes(root):
    root = Path(root)
    scenes = []

    for scene_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        left = scene_dir / IMAGE_LEFT
        right = scene_dir / IMAGE_RIGHT
        gt = next(scene_dir / name for name in GT_NAMES if (scene_dir / name).exists())

        scenes.append({
            "name": scene_dir.relative_to(root).as_posix(),
            "left": left,
            "right": right,
            "gt": gt,
        })

    return scenes


def discover_curiosity_pairs(root):
    root = Path(root)
    left_dir = root / "left"
    right_dir = root / "right"

    left_files = sorted(list(left_dir.glob("pair_*_L_*.jpg")) + list(left_dir.glob("pair_*_L_*.png")))
    right_files = sorted(list(right_dir.glob("pair_*_R_*.jpg")) + list(right_dir.glob("pair_*_R_*.png")))

    left_map = {}
    right_map = {}

    for path in left_files:
        idx = pair_index_from_name(path)
        left_map[idx] = path

    for path in right_files:
        idx = pair_index_from_name(path)
        right_map[idx] = path

    scenes = []
    for idx in sorted(set(left_map) & set(right_map)):
        left = left_map[idx]
        right = right_map[idx]
        scenes.append({
            "name": f"pair_{idx}",
            "pair_index": idx,
            "rock_name": left.stem,
            "left": left,
            "right": right,
        })

    return scenes


def resize_middlebury_for_eval(left, right, gt, max_width):
    if left.shape[1] <= max_width:
        return left, right, gt, 1.0

    scale = max_width / left.shape[1]
    new_w = int(round(left.shape[1] * scale))
    new_h = int(round(left.shape[0] * scale))

    left = cv2.resize(left, (new_w, new_h), interpolation=cv2.INTER_AREA)
    right = cv2.resize(right, (new_w, new_h), interpolation=cv2.INTER_AREA)
    gt = cv2.resize(gt, (new_w, new_h), interpolation=cv2.INTER_NEAREST) * scale

    return left, right, gt, scale


def resize_curiosity_for_eval(left, right, max_width):
    if left.shape[1] <= max_width:
        return left, right, 1.0

    scale = max_width / left.shape[1]
    new_w = int(round(left.shape[1] * scale))
    new_h = int(round(left.shape[0] * scale))

    left = cv2.resize(left, (new_w, new_h), interpolation=cv2.INTER_AREA)
    right = cv2.resize(right, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return left, right, scale


def compute_middlebury_metrics(pred, gt):
    valid_gt = np.isfinite(gt) & (gt > 0)

    valid_pred = np.isfinite(pred) & (pred > 0)
    eval_mask = valid_gt & valid_pred

    total_gt = int(valid_gt.sum())
    total_eval = int(eval_mask.sum())

    if total_eval == 0:
        return {
            "valid_gt_pixels": total_gt,
            "evaluated_pixels": 0,
            "valid_prediction_ratio": 0.0,
            "mae": np.nan,
            "rmse": np.nan,
            "bad_1": np.nan,
            "bad_4": np.nan,
        }

    err = np.abs(pred[eval_mask] - gt[eval_mask])

    return {
        "valid_gt_pixels": total_gt,
        "evaluated_pixels": total_eval,
        "valid_prediction_ratio": total_eval / total_gt,
        "mae": float(np.mean(err)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bad_1": float(np.mean(err > 1.0)),
        "bad_4": float(np.mean(err > 4.0)),
    }


def compute_low_texture_mask(left):
    gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY).astype(np.float32)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx ** 2 + gy ** 2)

    return grad <= np.percentile(grad, 25)


def compute_curiosity_disparity_metrics(pred):
    h, w = pred.shape[:2]
    total_pixels = h * w

    valid_pred = np.isfinite(pred) & (pred > 0)
    valid_count = int(valid_pred.sum())

    return {
        "total_pixels": total_pixels,
        "valid_disparity_pixels": valid_count,
        "valid_disparity_ratio": valid_count / total_pixels,
    }


def mesh_paths_for(data_root, method, rock_name):
    return {
        "poisson_mesh": data_root / "geometry_completion" / "poisson" / method / rock_name / "poisson_mesh.ply",
        "alpha_mesh": data_root / "geometry_completion" / "alpha_shape" / method / rock_name / "alpha_mesh.ply",
        "printable_hybrid": (
            data_root / "geometry_completion" / "printable" / method / rock_name / "printable_hybrid.obj"
        ),
    }


def compute_one_mesh_metrics(mesh_path, prefix):
    mesh = trimesh.load(mesh_path, force="mesh")

    parts = mesh.split(only_watertight=False)
    num_components = len(parts)
    largest_faces = max(len(p.faces) for p in parts)
    largest_component_ratio = largest_faces / len(mesh.faces)

    return {
        f"{prefix}_vertices": int(len(mesh.vertices)),
        f"{prefix}_faces": int(len(mesh.faces)),
        f"{prefix}_components": int(num_components),
        f"{prefix}_largest_component_ratio": float(largest_component_ratio),
    }


def compute_mesh_metrics(data_root, method, rock_name):
    metrics = {
        "mesh_method": method,
        "rock_name": rock_name,
    }

    for prefix, path in mesh_paths_for(data_root, method, rock_name).items():
        metrics.update(compute_one_mesh_metrics(path, prefix))

    return metrics


def normalize_for_vis(x, pmin=2, pmax=98):
    y = x.copy().astype(np.float32)
    valid = np.isfinite(y)

    lo = np.nanpercentile(y[valid], pmin)
    hi = np.nanpercentile(y[valid], pmax)

    y = (y - lo) / (hi - lo)
    y = np.clip(y, 0, 1)
    y[~valid] = 0

    return (y * 255).astype(np.uint8)


def save_middlebury_visualization(left, gt, pred, out_path, title, method_label):
    eval_mask = np.isfinite(gt) & (gt > 0) & np.isfinite(pred) & (pred > 0)
    err = np.full_like(gt, np.nan, dtype=np.float32)
    err[eval_mask] = np.abs(pred[eval_mask] - gt[eval_mask])

    gt_vis = normalize_for_vis(gt)
    pred_vis = normalize_for_vis(pred)
    err_vis = normalize_for_vis(err, pmin=0, pmax=95)

    left_rgb = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=160)

    axes[0].imshow(left_rgb)
    axes[0].set_title("Left image")
    axes[0].axis("off")

    axes[1].imshow(gt_vis, cmap="magma")
    axes[1].set_title("Ground truth disparity")
    axes[1].axis("off")

    axes[2].imshow(pred_vis, cmap="magma")
    axes[2].set_title(f"{method_label} disparity")
    axes[2].axis("off")

    axes[3].imshow(err_vis, cmap="inferno")
    axes[3].set_title("Absolute error")
    axes[3].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_curiosity_visualization(left, right, pred, out_path, title, method_label):
    pred_vis = normalize_for_vis(pred)
    valid_pred = np.isfinite(pred) & (pred > 0)

    left_rgb = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)

    valid_vis = valid_pred.astype(np.uint8) * 255
    low_texture_vis = compute_low_texture_mask(left).astype(np.uint8) * 255

    fig, axes = plt.subplots(1, 5, figsize=(20, 4), dpi=160)

    axes[0].imshow(left_rgb)
    axes[0].set_title("Left image")
    axes[0].axis("off")

    axes[1].imshow(right_rgb)
    axes[1].set_title("Right image")
    axes[1].axis("off")

    axes[2].imshow(pred_vis, cmap="magma")
    axes[2].set_title(f"{method_label} disparity")
    axes[2].axis("off")

    axes[3].imshow(valid_vis, cmap="gray")
    axes[3].set_title("Valid disparity mask")
    axes[3].axis("off")

    axes[4].imshow(low_texture_vis, cmap="gray")
    axes[4].set_title("Low-texture mask")
    axes[4].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_csv(path, rows):
    all_fields = []
    for row in rows:
        for key in row.keys():
            if key not in all_fields:
                all_fields.append(key)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, metric_names):
    summary = {}

    for metric in metric_names:
        values = np.array(
            [r[metric] for r in rows if metric in r and np.isfinite(r[metric])],
            dtype=np.float32,
        )

        summary[metric] = np.nan if values.size == 0 else float(values.mean())

    summary["num_scenes"] = len(rows)
    return summary


def evaluate_middlebury_benchmark(
    middlebury_root,
    out_dir,
    max_width,
    method,
    method_label,
    run_disparity,
    extra_row_fields=None,
):
    middlebury_out = out_dir / "middlebury"
    vis_dir = middlebury_out / "visualizations"

    middlebury_out.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    scenes = discover_middlebury_scenes(middlebury_root)

    rows = []

    print(f"MIDDLEBURY {method_label} EVALUATION")
    print(f"Found {len(scenes)} Middlebury scenes")

    for i, scene in enumerate(scenes):
        print(f"[Middlebury {i + 1}/{len(scenes)}] {scene['name']}")

        left = cv2.imread(str(scene["left"]), cv2.IMREAD_COLOR)
        right = cv2.imread(str(scene["right"]), cv2.IMREAD_COLOR)
        gt = read_pfm(scene["gt"])
        left, right, gt, scale = resize_middlebury_for_eval(left, right, gt, max_width)
        pred = run_disparity(left, right)
        metrics = compute_middlebury_metrics(pred, gt)

        row = {
            "dataset": "middlebury",
            "scene": scene["name"],
            "method": method,
            "scale": scale,
            **(extra_row_fields or {}),
            **metrics,
        }

        rows.append(row)

        safe_name = scene["name"].replace("/", "__")
        np.save(middlebury_out / f"{safe_name}_{method}_disparity.npy", pred)
        save_middlebury_visualization(
            left,
            gt,
            pred,
            vis_dir / f"{safe_name}_{method}.png",
            f"{scene['name']} - Middlebury {method_label}",
            method_label,
        )

    summary = summarize(rows, MIDDLEBURY_METRICS)
    summary_row = {
        "dataset": "middlebury",
        "method": method,
        **summary,
    }

    write_csv(middlebury_out / f"middlebury_{method}_per_scene.csv", rows)
    write_csv(middlebury_out / f"middlebury_{method}_summary.csv", [summary_row])
    return rows


def evaluate_curiosity_disparity_benchmark(
    curiosity_root,
    out_dir,
    max_width,
    method,
    method_label,
    run_disparity,
    extra_row_fields=None,
):
    curiosity_out = out_dir / "curiosity"
    vis_dir = curiosity_out / "visualizations"

    curiosity_out.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    scenes = discover_curiosity_pairs(curiosity_root)

    rows = []

    print(f"CURIOSITY {method_label} DISPARITY EVALUATION")
    print(f"Found {len(scenes)} Curiosity stereo pairs")

    for i, scene in enumerate(scenes):
        print(f"[Curiosity {i + 1}/{len(scenes)}] {scene['name']}")

        left = cv2.imread(str(scene["left"]), cv2.IMREAD_COLOR)
        right = cv2.imread(str(scene["right"]), cv2.IMREAD_COLOR)
        left, right, scale = resize_curiosity_for_eval(left, right, max_width)
        pred = run_disparity(left, right)
        metrics = compute_curiosity_disparity_metrics(pred)

        row = {
            "dataset": "curiosity",
            "scene": scene["name"],
            "pair_index": scene["pair_index"],
            "rock_name": scene["rock_name"],
            "method": method,
            "scale": scale,
            "left_path": str(scene["left"]),
            "right_path": str(scene["right"]),
            **(extra_row_fields or {}),
            **metrics,
        }

        rows.append(row)

        safe_name = scene["name"].replace("/", "__")
        np.save(curiosity_out / f"{safe_name}_{method}_disparity.npy", pred)
        save_curiosity_visualization(
            left,
            right,
            pred,
            vis_dir / f"{safe_name}_{method}.png",
            f"{scene['name']} - Curiosity {method_label}",
            method_label,
        )

    summary = summarize(rows, CURIOSITY_DISPARITY_METRICS)
    summary_row = {
        "dataset": "curiosity",
        "method": method,
        **summary,
    }

    write_csv(curiosity_out / f"curiosity_{method}_per_scene.csv", rows)
    write_csv(curiosity_out / f"curiosity_{method}_summary.csv", [summary_row])
    return rows


def evaluate_curiosity_meshes_benchmark(
    curiosity_root,
    data_root,
    out_dir,
    methods,
    per_scene_filename,
    summary_filename,
    log_label,
):
    mesh_out = out_dir / "curiosity_meshes"
    mesh_out.mkdir(parents=True, exist_ok=True)
    scenes = discover_curiosity_pairs(curiosity_root)[:5]
    rows = []
    print(log_label)

    for scene in scenes:
        rock_name = scene["rock_name"]

        for method in methods:
            print(f"Evaluating mesh metrics for {method}/{rock_name}")
            try:
                row = {
                    "dataset": "curiosity_meshes",
                    "scene": scene["name"],
                    "pair_index": scene["pair_index"],
                    "rock_name": rock_name,
                    "method": method,
                }

                row.update(compute_mesh_metrics(data_root, method, rock_name))
                rows.append(row)
            except Exception as e:
                print(f"Skipping {method}/{rock_name}: {e}")

    summaries = []

    for method in methods:
        method_rows = [r for r in rows if r["method"] == method]
        summary = summarize(method_rows, MESH_METRICS)
        summaries.append({
            "dataset": "curiosity_meshes",
            "method": method,
            **summary,
        })

    write_csv(mesh_out / per_scene_filename, rows)
    write_csv(mesh_out / summary_filename, summaries)
    return rows
