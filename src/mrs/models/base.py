from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Rec:
    movie_id: int
    score: float


class Recommender(Protocol):
    def recommend(self, user_id: int, k: int) -> list[Rec]:
        ...
