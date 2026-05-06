import argparse
from utils import *
from sgbm import run_sgbm_wls
from raft import initialize_raft, run_raft


def save_disparity_png(disparity: np.ndarray, out_path: Path):
    disp_vis = np.where(np.isfinite(disparity) & (disparity > 0), disparity, 0)
    disp_norm = cv2.normalize(disp_vis, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imwrite(str(out_path), cv2.applyColorMap(disp_norm, cv2.COLORMAP_MAGMA))


def process_pair(rock_id: str, data_root: Path, params: dict, raft_model, raft_device):
    print(f"Processing {rock_id}")
    left_gray, right_gray, _ = load_stereo_pair(rock_id, data_root, params)

    for method_name, run_fn in [
        ("SGBM", lambda: run_sgbm_wls(left_gray, right_gray)),
        ("RAFT", lambda: run_raft(raft_model, left_gray, right_gray, raft_device)),
    ]:
        disparity = run_fn()
        depth = disparity_to_depth(disparity, params)

        out_dir = data_root / method_name.lower() / rock_id
        out_dir.mkdir(parents=True, exist_ok=True)

        np.save(out_dir / "disparity.npy", disparity)
        np.save(out_dir / "depth.npy", depth)
        save_disparity_png(disparity, out_dir / "disparity.png")


def main():
    parser = argparse.ArgumentParser(description="Mars rock stereo reconstruction pipeline.")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--rock_id", type=str, default=None)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    left_dir = data_root / "rocks" / "left"

    rock_ids = [args.rock_id] if args.rock_id else sorted(
        p.stem for p in left_dir.iterdir() if p.suffix.lower() in (".jpg", ".png", ".jpeg")
    )

    raft_model, raft_device = initialize_raft()

    for rock_id in rock_ids:
        params = DEFAULT_CAMERA_PARAMS
        process_pair(
            rock_id,
            data_root,
            params,
            raft_model,
            raft_device,
        )


if __name__ == "__main__":
    main()
