from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException, Query

from mrs.config.settings import settings
from mrs.models.content_tfidf import ContentTfidfModel
from mrs.models.popularity import PopularityRecommender
from mrs.serving.loader import load_models


# -----------------------------------------------------------------------------
# App setup
# -----------------------------------------------------------------------------

app = FastAPI(
    title="Movie Recommendation System API",
    description="FastAPI backend for movie recommendations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

Strategy = Literal["popularity", "content"]

# -----------------------------------------------------------------------------
# Load models at startup
# -----------------------------------------------------------------------------

popularity_model: PopularityRecommender | None = None
content_model: ContentTfidfModel | None = None


@app.on_event("startup")
def startup_event():
    global popularity_model, content_model

    try:
        popularity_model, content_model, _ = load_models(settings.run_id)
    except Exception as e:
        raise RuntimeError(f"Failed to load models: {e}") from e


# -----------------------------------------------------------------------------
# Health
# -----------------------------------------------------------------------------

@app.get("/health", tags=["health"])
def health():
    return {
        "status": "ok",
        "run_id": settings.run_id,
        "models_loaded": popularity_model is not None,
    }


# -----------------------------------------------------------------------------
# Recommendations
# -----------------------------------------------------------------------------

@app.get("/v1/recommendations", tags=["recommendations"])
def get_recommendations(
    user_id: int = Query(..., ge=1),
    k: int = Query(10, ge=1, le=100),
    strategy: Strategy = Query("popularity"),
):
    if popularity_model is None or content_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    if strategy == "popularity":
        recs = popularity_model.recommend(user_id=user_id, k=k)
    elif strategy == "content":
        recs = content_model.recommend_for_user(user_id=user_id, k=k)
    else:
        raise HTTPException(status_code=400, detail="Invalid strategy")

    return {
        "user_id": user_id,
        "strategy": strategy,
        "k": k,
        "recommendations": recs,
    }


# -----------------------------------------------------------------------------
# Similar items
# -----------------------------------------------------------------------------

@app.get("/v1/similar-items", tags=["recommendations"])
def get_similar_items(
    movie_id: int = Query(..., ge=1),
    k: int = Query(10, ge=1, le=100),
):
    if content_model is None:
        raise HTTPException(status_code=503, detail="Content model not loaded")

    sims = content_model.similar_items(movie_id=movie_id, k=k)

    return {
        "movie_id": movie_id,
        "k": k,
        "similar_items": sims,
    }
