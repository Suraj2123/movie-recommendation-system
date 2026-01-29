from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class PreprocessedData:
    ratings: pd.DataFrame
    movies: pd.DataFrame


def load_raw_movielens(dataset_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw MovieLens CSVs from dataset_dir (e.g. data/ml-latest-small).
    Returns (ratings, movies).
    """
    p = Path(dataset_dir)
    ratings = pd.read_csv(p / "ratings.csv")
    movies = pd.read_csv(p / "movies.csv")
    return ratings, movies


def preprocess(ratings: pd.DataFrame, movies: pd.DataFrame) -> PreprocessedData:
    """
    Normalize dtypes and handle missing values. Ratings keep userId, movieId, rating, timestamp.
    Movies keep movieId, title, genres; genres are filled (no NaN).
    """
    ratings = ratings.astype({"userId": "int64", "movieId": "int64", "rating": "float64", "timestamp": "int64"})
    movies = movies.astype({"movieId": "int64", "title": "str"})
    movies["genres"] = movies["genres"].fillna("").astype(str)
    return PreprocessedData(ratings=ratings, movies=movies)
