from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mrs.evaluation.metrics import EvalResult, catalog_coverage, precision_recall_at_k
from mrs.models.base import Recommender


@dataclass(frozen=True)
class SplitData:
    train: pd.DataFrame
    test: pd.DataFrame


def chronological_split(ratings: pd.DataFrame, test_ratio: float = 0.2) -> SplitData:
    # Simple per-user chronological split
    ratings = ratings.sort_values(["userId", "timestamp"])
    train_parts = []
    test_parts = []

    for uid, grp in ratings.groupby("userId"):
        n = len(grp)
        if n < 5:
            train_parts.append(grp)
            continue
        cut = max(1, int(n * (1 - test_ratio)))
        train_parts.append(grp.iloc[:cut])
        test_parts.append(grp.iloc[cut:])

    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True) if test_parts else ratings.iloc[0:0].copy()
    return SplitData(train=train, test=test)


def evaluate(model: Recommender, train: pd.DataFrame, test: pd.DataFrame, k: int = 10) -> EvalResult:
    truth: dict[int, set[int]] = (
        test.groupby("userId")["movieId"].apply(lambda s: set(map(int, s.tolist()))).to_dict()
    )

    recs: dict[int, list[int]] = {}
    all_recommended: list[int] = []
    for uid in truth.keys():
        r = model.recommend(int(uid), k)
        mids = [int(x.movie_id) for x in r]
        recs[int(uid)] = mids
        all_recommended.extend(mids)

    p, r = precision_recall_at_k(recs, truth, k)
    coverage = catalog_coverage(all_recommended, set(map(int, train["movieId"].unique().tolist())))
    return EvalResult(precision_at_k=p, recall_at_k=r, coverage=coverage)
