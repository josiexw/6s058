import subprocess
import sys
from pathlib import Path


SCRIPTS = ["poisson.py", "alpha.py", "3dgan.py"]


def run_script(script_path: Path):
    cmd = [sys.executable, str(script_path)]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, stderr=subprocess.DEVNULL)
    if result.returncode != 0:
        print(f"Skipping {script_path.stem}: exited with code {result.returncode}")


def main():
    script_dir = Path(__file__).resolve().parent

    for script_name in SCRIPTS:
        script_path = script_dir / script_name
        try:
            run_script(script_path)
        except Exception as e:
            print(f"\nERROR: {e}")


if __name__ == "__main__":
    main()
