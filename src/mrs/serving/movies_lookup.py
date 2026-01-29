from __future__ import annotations

import csv
from pathlib import Path


def load_movies_lookup(dataset_dir: str | Path) -> dict[int, dict[str, str]]:
    """
    Load MovieLens movies.csv into:
      movie_id -> {"title": str, "genres": str}

    Looks for movies.csv in dataset_dir or dataset_dir/ml-latest-small.
    Expected schema (MovieLens): movieId,title,genres
    """
    dataset_dir = Path(dataset_dir)
    candidates = [dataset_dir / "movies.csv", dataset_dir / "ml-latest-small" / "movies.csv"]
    movies_csv = None
    for c in candidates:
        if c.exists():
            movies_csv = c
            break
    if movies_csv is None:
        return {}

    lookup: dict[int, dict[str, str]] = {}
    with movies_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_id = (row.get("movieId") or row.get("movie_id") or "").strip()
            if not raw_id:
                continue
            try:
                movie_id = int(raw_id)
            except ValueError:
                continue

            title = (row.get("title") or "").strip()
            genres = (row.get("genres") or "").strip()
            lookup[movie_id] = {"title": title, "genres": genres}

    return lookup
