from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from joblib import load

from mrs.config.logging import configure_logging
from mrs.config.settings import settings
from mrs.models.content_tfidf import ContentTfidfModel
from mrs.models.popularity import PopularityRecommender

Strategy = Literal["popularity", "content"]


def load_models(run_id: str) -> tuple[PopularityRecommender, ContentTfidfModel, dict]:
    base = Path(settings.artifacts_dir) / run_id
    models_dir = base / "models"
    if not models_dir.exists():
        raise FileNotFoundError(f"Artifacts not found for run_id={run_id}. Train first.")

    pop = load(models_dir / "popularity.joblib")
    content = ContentTfidfModel.load(str(models_dir / "content_tfidf.joblib"))

    metrics_path = base / "metrics.json"
    metrics: dict = {}
    if metrics_path.exists():
        metrics = __import__("json").loads(metrics_path.read_text())

    return pop, content, metrics


configure_logging()
app = FastAPI(title="Movie Recommendation System", version="0.1.0")

try:
    POP, CONTENT, METRICS = load_models(settings.run_id)
except Exception:
    POP, CONTENT, METRICS = None, None, {}


@app.get("/")
def root() -> dict:
    return {
        "name": "Movie Recommendation System",
        "docs": "/docs",
        "health": "/health",
        "example_recs": "/v1/recommendations?user_id=1&k=10&strategy=popularity",
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "run_id": settings.run_id, "models_loaded": POP is not None}


@app.get("/v1/model-info")
def model_info() -> dict:
    return {"run_id": settings.run_id, "version": "0.1.0", "metrics": METRICS}


@app.get("/v1/recommendations")
def recommendations(
    user_id: int = Query(..., ge=1),
    k: int = Query(10, ge=1, le=100),
    strategy: Strategy = Query("popularity"),
) -> dict:
    if POP is None or CONTENT is None:
        raise HTTPException(status_code=400, detail="Models not loaded. Train first.")

    model = POP if strategy == "popularity" else CONTENT
    recs = model.recommend(user_id=user_id, k=k)
    return {"user_id": user_id, "k": k, "strategy": strategy, "recommendations": [r.__dict__ for r in recs]}


@app.get("/v1/similar-items")
def similar_items(movie_id: int = Query(..., ge=1), k: int = Query(10, ge=1, le=100)) -> dict:
    if CONTENT is None:
        raise HTTPException(status_code=400, detail="Models not loaded. Train first.")
    recs = CONTENT.similar_items(movie_id=movie_id, k=k)
    return {"movie_id": movie_id, "k": k, "similar": [r.__dict__ for r in recs]}


def main() -> None:
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("mrs.serving.api:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
