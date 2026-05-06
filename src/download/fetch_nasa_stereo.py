import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from PIL import Image
from tqdm import tqdm

DOWNLOAD_DIR = Path("data/rocks")
MAX_IMAGE_SIZE = 512
NUM_PAIRS = 100
THREADS = 8
API_BASE = "https://mars.nasa.gov/api/v1/raw_image_items/"
DEFAULT_FX = 452.67
DEFAULT_FY = 452.67
DEFAULT_CX = 320.0
DEFAULT_CY = 240.0
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
BASELINE_M = 0.2432


def fetch_page(page: int, sol_min: int, sol_max: int) -> List[Dict]:
    params = {
        "order": "sol asc",
        "per_page": 100,
        "page": page,
        "condition_2": "msl:mission",
        "condition_3": f"{sol_min}:sol:gte",
        "condition_4": f"{sol_max}:sol:lte",
    }
    r = requests.get(API_BASE, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("items", [])


def collect_pairs(sol_min: int = 100, sol_max: int = 2000) -> List[Tuple[Dict, Dict]]:
    print(f"Fetching NAV image metadata for sols {sol_min}-{sol_max}...")

    left_map: Dict[str, Dict] = {}   # spacecraft_clock -> photo
    right_map: Dict[str, Dict] = {}

    page = 0
    with tqdm(desc="Fetching pages", unit="page") as pbar:
        while True:
            items = fetch_page(page, sol_min, sol_max)
            if not items:
                break
            for item in items:
                imageid = item.get("imageid", "")
                url = item.get("https_url", "")
                if not url or item.get("is_thumbnail"):
                    continue
                clock = str(item.get("spacecraft_clock", ""))
                if imageid.startswith("NLA"):
                    left_map[clock] = item
                elif imageid.startswith("NRA"):
                    right_map[clock] = item
            page += 1
            pbar.update(1)
            pbar.set_postfix(left=len(left_map), right=len(right_map))
            time.sleep(0.1)

            common = set(left_map) & set(right_map)
            if len(common) >= NUM_PAIRS:
                break

    common = sorted(set(left_map) & set(right_map))[:NUM_PAIRS]
    print(f"Found {len(common)} matched stereo pairs")
    return [(left_map[k], right_map[k]) for k in common]


def download_and_resize(url: str, path: Path) -> Tuple[int, int]:
    r = requests.get(url, timeout=60, stream=True)
    r.raise_for_status()
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    with Image.open(tmp) as img:
        img = img.convert("RGB")
        img.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
        img.save(path, quality=90, optimize=True)
        width, height = img.size
    tmp.unlink(missing_ok=True)
    return width, height


def write_calibration(path: Path, width: int, height: int) -> None:
    scale_x = width / DEFAULT_WIDTH
    scale_y = height / DEFAULT_HEIGHT
    params = {
        "fx": DEFAULT_FX * scale_x,
        "fy": DEFAULT_FY * scale_y,
        "cx": DEFAULT_CX * scale_x,
        "cy": DEFAULT_CY * scale_y,
        "baseline_m": BASELINE_M,
        "width": width,
        "height": height,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
        f.write("\n")


def download_pair(args):
    idx, left, right, left_dir, right_dir, calib_dir = args
    left_id  = left["imageid"]
    right_id = right["imageid"]
    left_path = left_dir / f"pair_{idx:04d}_L_{left_id}.jpg"
    right_path = right_dir / f"pair_{idx:04d}_R_{right_id}.jpg"

    width, height = download_and_resize(left["https_url"], left_path)
    download_and_resize(right["https_url"], right_path)
    write_calibration(calib_dir / f"pair_{idx:04d}.json", width, height)
    return idx


def main():
    left_dir  = DOWNLOAD_DIR / "left"
    right_dir = DOWNLOAD_DIR / "right"
    calib_dir = DOWNLOAD_DIR / "calib"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)
    calib_dir.mkdir(parents=True, exist_ok=True)

    pairs = collect_pairs(sol_min=100, sol_max=2000)

    if not pairs:
        print("No pairs found.")
        return

    print(f"\nDownloading {len(pairs)} pairs with {THREADS} threads...")
    args = [(idx+1, l, r, left_dir, right_dir, calib_dir) for idx, (l, r) in enumerate(pairs)]

    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        futures = [executor.submit(download_pair, a) for a in args]
        with tqdm(total=len(futures), desc="Downloading", unit="pair") as pbar:
            for future in as_completed(futures):
                future.result()
                pbar.update(1)

    print(f"\nDone: {len(pairs)} stereo pairs saved to {DOWNLOAD_DIR}")


if __name__ == "__main__":
    main()