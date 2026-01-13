from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from mrs.models.base import Rec


@dataclass
class ContentTfidfModel:
    movie_ids: np.ndarray
    tfidf_matrix: np.ndarray  
    vectorizer: TfidfVectorizer

    @staticmethod
    def _make_text(movies: pd.DataFrame) -> pd.Series:
        title = movies["title"].fillna("").astype(str)
        genres = movies["genres"].fillna("").astype(str).str.replace("|", " ", regex=False)
        return (title + " " + genres).str.lower()

    @classmethod
    def train(cls, movies: pd.DataFrame) -> ContentTfidfModel:
        text = cls._make_text(movies)
        vectorizer = TfidfVectorizer(min_df=2, max_features=30_000, ngram_range=(1, 2))
        x = vectorizer.fit_transform(text)
        x_dense = x.toarray().astype(np.float32)
        movie_ids = movies["movieId"].to_numpy(dtype=np.int64)
        return cls(movie_ids=movie_ids, tfidf_matrix=x_dense, vectorizer=vectorizer)

    def similar_items(self, movie_id: int, k: int) -> list[Rec]:
        idx = np.where(self.movie_ids == movie_id)[0]
        if len(idx) == 0:
            return []
        i = int(idx[0])

        sims = cosine_similarity(self.tfidf_matrix[i : i + 1], self.tfidf_matrix)[0]
        sims[i] = -1.0
        top_idx = np.argsort(-sims)[:k]
        return [Rec(int(self.movie_ids[j]), float(sims[j])) for j in top_idx]

    def recommend(self, user_id: int, k: int) -> list[Rec]:
        sims = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        np.fill_diagonal(sims, 0.0)
        centrality = sims.mean(axis=1)
        top_idx = np.argsort(-centrality)[:k]
        return [Rec(int(self.movie_ids[j]), float(centrality[j])) for j in top_idx]

    def save(self, path: str) -> None:
        dump(
            {
                "movie_ids": self.movie_ids,
                "tfidf_matrix": self.tfidf_matrix,
                "vectorizer": self.vectorizer,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> ContentTfidfModel:
        obj = load(path)
        return cls(
            movie_ids=obj["movie_ids"],
            tfidf_matrix=obj["tfidf_matrix"],
            vectorizer=obj["vectorizer"],
        )
