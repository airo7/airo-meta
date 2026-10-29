"""Fetch the third-party robot models from DeepMind's MuJoCo Menagerie into third_party/menagerie (git-ignored).

uv run scripts/fetch_models.py
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

REPO = "https://github.com/google-deepmind/mujoco_menagerie.git"
COMMIT = "f3b6013629c7b94f8ee93fd977b161fa8350f9bc"
MODELS = ("universal_robots_ur5e", "robotiq_2f85", "unitree_g1")
DEST = Path(__file__).resolve().parent.parent / "third_party" / "menagerie"


def main():
    if all((DEST / m).is_dir() for m in MODELS):
        print(f"already in {DEST}")
        return
    with tempfile.TemporaryDirectory() as tmp:
        git = lambda *a: subprocess.run(["git", "-C", tmp, *a], check=True, capture_output=True)
        git("init", "-q")
        git("remote", "add", "origin", REPO)
        git("sparse-checkout", "set", "--no-cone", *(f"/{m}/" for m in MODELS), "/LICENSE")
        git("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", COMMIT)
        git("checkout", "-q", "FETCH_HEAD")
        DEST.mkdir(parents=True, exist_ok=True)
        for m in MODELS:
            shutil.rmtree(DEST / m, ignore_errors=True)
            shutil.copytree(Path(tmp) / m, DEST / m)
        shutil.copy(Path(tmp) / "LICENSE", DEST / "LICENSE")
    print(f"{', '.join(MODELS)} -> {DEST}")


if __name__ == "__main__":
    main()
