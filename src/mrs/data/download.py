from __future__ import annotations

import zipfile
from pathlib import Path

import requests

ML_SMALL_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
EXTRACT_DIR = "ml-latest-small"


def download_movielens_latest_small(data_dir: str | Path) -> Path:
    """
    Download MovieLens ml-latest-small and extract to data_dir/ml-latest-small.
    Returns the path to the extracted directory (containing movies.csv, ratings.csv, etc.).
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "ml-latest-small.zip"
    out_dir = data_dir / EXTRACT_DIR

    if out_dir.exists() and (out_dir / "movies.csv").exists() and (out_dir / "ratings.csv").exists():
        return out_dir

    r = requests.get(ML_SMALL_URL, timeout=120)
    r.raise_for_status()
    zip_path.write_bytes(r.content)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)

    return out_dir
