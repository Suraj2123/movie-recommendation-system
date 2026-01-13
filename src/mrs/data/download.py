from __future__ import annotations

import io
import zipfile
from pathlib import Path

import requests


MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def download_movielens_latest_small(data_dir: str) -> Path:
    """
    Downloads MovieLens latest-small and extracts it.
    Returns the extracted dataset directory path, e.g. data/ml-latest-small
    """
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)

    zip_path = root / "ml-latest-small.zip"
    extract_dir = root / "ml-latest-small"

    if extract_dir.exists():
        return extract_dir

    resp = requests.get(MOVIELENS_URL, timeout=60)
    resp.raise_for_status()

    zip_path.write_bytes(resp.content)

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(root)

    return extract_dir
