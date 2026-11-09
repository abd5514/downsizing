"""Download FL3D from Kaggle.

FL3D is frame-level driver drowsiness data built from the NITYMED night-time
driving videos: 53,331 labelled frames across alert / microsleep / yawning,
with a face box per frame.

Kaggle has two credential formats in the wild and the client reads either:

  ~/.kaggle/access_token   newer single-line token, starts with KGAT_
  ~/.kaggle/kaggle.json    older username + key pair
  KAGGLE_API_TOKEN         environment variable, same value as access_token

This script never asks for or handles the credential itself - it only checks
that one is in place before shelling out to the official client.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

DATASET = "matjazmuc/frame-level-driver-drowsiness-detection-fl3d"
DEST = Path("data/fl3d")

ACCESS_TOKEN = Path.home() / ".kaggle" / "access_token"
KAGGLE_JSON = Path.home() / ".kaggle" / "kaggle.json"

SETUP_HELP = f"""No Kaggle credential found.

Expected one of:
  {ACCESS_TOKEN}
  {KAGGLE_JSON}
  the KAGGLE_API_TOKEN environment variable

To create one: kaggle.com -> avatar -> Settings -> API -> Create New Token,
then follow the command Kaggle shows you. Tighten the permissions afterwards:

  chmod 600 ~/.kaggle/access_token

Also open the dataset page once while signed in and accept its terms, or the
download returns 403 even with a valid token:
  https://www.kaggle.com/datasets/{DATASET}
"""


def find_credential() -> Path | None:
    if os.environ.get("KAGGLE_API_TOKEN"):
        return None  # present, but not a file - nothing to permission-check
    for path in (ACCESS_TOKEN, KAGGLE_JSON):
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def have_credential() -> bool:
    return bool(os.environ.get("KAGGLE_API_TOKEN")) or any(
        p.exists() and p.stat().st_size > 0 for p in (ACCESS_TOKEN, KAGGLE_JSON)
    )


def kaggle_command() -> list[str]:
    """Locate the kaggle CLI.

    It is installed into this project's venv, and the venv's bin directory is
    not on PATH when the interpreter is invoked by full path (which is how the
    README tells you to run everything), so a bare "kaggle" is not enough.
    """
    beside_python = Path(sys.executable).parent / "kaggle"
    if beside_python.exists():
        return [str(beside_python)]
    on_path = shutil.which("kaggle")
    if on_path:
        return [on_path]
    return [sys.executable, "-m", "kaggle"]


def main() -> int:
    if not have_credential():
        print(SETUP_HELP, file=sys.stderr)
        return 1

    credential = find_credential()
    if credential is not None:
        print(f"Using credential at {credential}")
        # A world-readable token is a real exposure; fix it rather than warn.
        if credential.stat().st_mode & 0o077:
            credential.chmod(0o600)
            print(f"Tightened permissions on {credential} to 600")
    else:
        print("Using KAGGLE_API_TOKEN from the environment")

    DEST.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET} -> {DEST}  (about 2 GB)")
    cli = kaggle_command()
    result = subprocess.run(
        [*cli, "datasets", "download", "-d", DATASET, "-p", str(DEST), "--unzip"],
        check=False,
    )
    if result.returncode != 0:
        print(
            "\nDownload failed. The usual cause is not having accepted the "
            f"dataset's terms:\n  https://www.kaggle.com/datasets/{DATASET}",
            file=sys.stderr,
        )
        return result.returncode

    print("\nDone. Next:  PYTHONPATH=. ./.venv/bin/python scripts/inspect_dataset.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
