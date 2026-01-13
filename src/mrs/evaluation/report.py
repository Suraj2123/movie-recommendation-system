from __future__ import annotations

from datetime import datetime

from mrs.evaluation.metrics import EvalResult


def render_report(run_id: str, popularity: EvalResult, content: EvalResult) -> str:
    ts = datetime.utcnow().isoformat() + "Z"
    return f"""# Offline Evaluation Report

Run ID: `{run_id}`
Generated: `{ts}`

## Results (k=10)

| Model | Precision@10 | Recall@10 | Coverage |
|---|---:|---:|---:|
| Popularity | {popularity.precision_at_k:.4f} | {popularity.recall_at_k:.4f} | {popularity.coverage:.4f} |
| Content TF-IDF | {content.precision_at_k:.4f} | {content.recall_at_k:.4f} | {content.coverage:.4f} |

## Notes
- This is a simple offline chronological split per user.
- Real-world performance requires online experiments and richer feedback signals.
"""
