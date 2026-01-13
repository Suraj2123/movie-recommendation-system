from __future__ import annotations

from dataclasses import dataclass




@dataclass(frozen=True)
class EvalResult:
    precision_at_k: float
    recall_at_k: float
    coverage: float


def precision_recall_at_k(recs: dict[int, list[int]], truth: dict[int, set[int]], k: int) -> tuple[float, float]:
    precisions = []
    recalls = []

    for user_id, rlist in recs.items():
        topk = rlist[:k]
        tset = truth.get(user_id, set())
        if not tset:
            continue
        hits = sum(1 for mid in topk if mid in tset)
        precisions.append(hits / max(k, 1))
        recalls.append(hits / len(tset))

    if not precisions:
        return 0.0, 0.0
    return float(sum(precisions) / len(precisions)), float(sum(recalls) / len(recalls))


def catalog_coverage(all_recommended: list[int], catalog: set[int]) -> float:
    if not catalog:
        return 0.0
    return len(set(all_recommended)) / len(catalog)
