# Architecture

## Offline pipeline
1) Download MovieLens dataset
2) Preprocess ratings + movies
3) Train models:
   - popularity baseline
   - TF-IDF content similarity
4) Offline evaluation
5) Export artifacts under `artifacts/<run_id>/`

## Online serving
FastAPI loads `artifacts/<run_id>/models/*` and serves:
- recommendations (strategy switch)
- similar-items (content model)
