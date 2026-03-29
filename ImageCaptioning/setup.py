import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


def get_conda_env():
    env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if not env or env == "base":
        print("activate a conda env first: conda activate <your-env>")
        sys.exit(1)
    return env


def get_conda_python(env_name):
    result = subprocess.run(["conda", "info", "--base"], capture_output=True, text=True, check=True)
    conda_root = Path(result.stdout.strip())
    python = conda_root / "envs" / env_name / ("python.exe" if sys.platform == "win32" else "bin/python")
    return str(python) if python.exists() else sys.executable


def check_and_install(python, import_name, install_name, channel="pytorch"):
    ok = subprocess.run([python, "-c", f"import {import_name}"], capture_output=True).returncode == 0
    if ok:
        print(f"{import_name} already installed")
    else:
        print(f"installing {import_name}...")
        env_name = get_conda_env()
        if channel == "pip":
            subprocess.run(["conda", "run", "-n", env_name, "pip", "install", install_name], check=True)
        else:
            subprocess.run(["conda", "install", "-n", env_name, "-c", channel, install_name, "-y"], check=True)



def download_with_progress(url, dest):
    def reporthook(count, block_size, total_size):
        if total_size > 0:
            pct = min(count * block_size * 100 / total_size, 100)
            bar = int(pct / 2)
            print(f"\r  [{'=' * bar}{' ' * (50 - bar)}] {pct:.1f}%", end="", flush=True)
    urllib.request.urlretrieve(url, dest, reporthook)
    print()


def main():
    env_name = get_conda_env()
    print(f"conda env: {env_name}")

    python = get_conda_python(env_name)
    check_and_install(python, "torchvision", "torchvision", channel="pytorch")
    print("packages good")

    script_dir = Path(__file__).resolve().parent
    data_dir = (script_dir / "data").resolve()

    if (data_dir / "Images").is_dir() and (data_dir / "captions.txt").is_file():
        print(f"data found at {data_dir}")
        return

    print("downloading Flickr8k...")
    data_dir.mkdir(parents=True, exist_ok=True)

    url = "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr8k.zip"
    tmp_zip = data_dir / "flickr8k.zip"

    try:
        download_with_progress(url, tmp_zip)
    except Exception as e:
        print(f"\ndownload failed: {e}")
        print("grab it manually: https://www.kaggle.com/datasets/adityajn105/flickr8k")
        print(f"put Images/ and captions.txt in {data_dir}")
        sys.exit(1)

    print("extracting...")
    with zipfile.ZipFile(tmp_zip, "r") as zf:
        zf.extractall(data_dir)

    nested = data_dir / "flickr8k"
    if nested.is_dir():
        shutil.move(str(nested / "Images"), str(data_dir / "Images"))
        shutil.move(str(nested / "captions.txt"), str(data_dir / "captions.txt"))
        shutil.rmtree(nested)

    tmp_zip.unlink()
    print(f"done! data at {data_dir}")


if __name__ == "__main__":
    main()
