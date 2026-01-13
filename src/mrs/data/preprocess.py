from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class PreprocessedData:
    ratings: pd.DataFrame
    movies: pd.DataFrame


def load_raw_movielens(dataset_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_csv(dataset_dir / "ratings.csv")
    movies = pd.read_csv(dataset_dir / "movies.csv")
    return ratings, movies


def preprocess(ratings: pd.DataFrame, movies: pd.DataFrame) -> PreprocessedData:
    # Basic cleaning
    ratings = ratings.dropna(subset=["userId", "movieId", "rating"]).copy()
    movies = movies.dropna(subset=["movieId", "title"]).copy()

    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)

    movies["movieId"] = movies["movieId"].astype(int)
    movies["genres"] = movies.get("genres", "").fillna("").astype(str)

    return PreprocessedData(ratings=ratings, movies=movies)
