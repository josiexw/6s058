from pathlib import Path

from benchmark_utils import (
    evaluate_curiosity_disparity_benchmark,
    evaluate_curiosity_meshes_benchmark,
    evaluate_middlebury_benchmark,
    write_csv,
)
from sgbm import run_sgbm


MIDDLEBURY_ROOT = Path("data/middlebury_stereo")
CURIOSITY_ROOT = Path("data/rocks")
DATA_ROOT = Path("data")
OUT_DIR = Path("data/benchmark/stereo_sgbm")

MAX_WIDTH = 960

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    middlebury_rows = evaluate_middlebury_benchmark(
        middlebury_root=MIDDLEBURY_ROOT,
        out_dir=OUT_DIR,
        max_width=MAX_WIDTH,
        method="sgbm",
        method_label="SGBM",
        run_disparity=run_sgbm,
    )
    curiosity_disparity_rows = evaluate_curiosity_disparity_benchmark(
        curiosity_root=CURIOSITY_ROOT,
        out_dir=OUT_DIR,
        max_width=MAX_WIDTH,
        method="sgbm",
        method_label="SGBM",
        run_disparity=run_sgbm,
    )
    curiosity_mesh_rows = evaluate_curiosity_meshes_benchmark(
        curiosity_root=CURIOSITY_ROOT,
        data_root=DATA_ROOT,
        out_dir=OUT_DIR,
        methods=["sgbm", "raft"],
        per_scene_filename="curiosity_mesh_metrics_per_scene.csv",
        summary_filename="curiosity_mesh_metrics_summary.csv",
        log_label="CURIOSITY MESH EVALUATION",
    )

    all_rows = middlebury_rows + curiosity_disparity_rows + curiosity_mesh_rows
    write_csv(OUT_DIR / "sgbm_benchmark_results.csv", all_rows)


if __name__ == "__main__":
    main()
