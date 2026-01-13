from dataclasses import dataclass

import pandas as pd

from mrs.models.base import Rec


@dataclass
class PopularityRecommender:
    ranked: list[Rec]

    @classmethod
    def train(cls, ratings: pd.DataFrame) -> "PopularityRecommender":
        grouped = (
            ratings.groupby("movieId")["rating"]
            .agg(["mean", "count"])
            .reset_index()
        )

        global_mean = float(ratings["rating"].mean())
        m = 50  # shrinkage strength

        grouped["score"] = (
            grouped["count"] * grouped["mean"] + m * global_mean
        ) / (grouped["count"] + m)

        grouped = grouped.sort_values("score", ascending=False)

        ranked = [
            Rec(int(row.movieId), float(row.score))
            for row in grouped.itertuples(index=False)
        ]
        return cls(ranked=ranked)

    def recommend(self, user_id: int, k: int) -> list[Rec]:
        return self.ranked[:k]
