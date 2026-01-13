
Commit message:
- `docs: add README with quickstart`

---

## 1.4 Add `MODEL_CARD.md`
**Path:** `MODEL_CARD.md`
```md
# Model Card — Movie Recommendation System (MRS)

## Model overview
This repository currently supports:
- Popularity baseline model
- Content-based TF-IDF similarity model

## Intended use
- Learning/portfolio demonstration of end-to-end ML-ish pipelines and API serving
- Not intended for real production personalization without further privacy/security work

## Data
Default dataset: MovieLens “latest-small” (downloaded during training).

## Metrics (offline)
Generated during training (`artifacts/<run_id>/metrics.json` and `report.md`), including:
- precision@k, recall@k (simple offline split)
- catalog coverage (basic)

## Limitations
- Cold-start for new users is not personalized (falls back to popularity)
- Content model relies on metadata fields present in dataset (title/genres)
- Offline evaluation is simplistic and does not represent online performance

## Ethical considerations
- Dataset may contain popularity bias
- Recommendations may over-represent mainstream content

## Versioning
Each training run is exported under `artifacts/<run_id>/` with a manifest.
