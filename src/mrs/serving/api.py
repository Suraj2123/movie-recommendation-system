from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from joblib import load

from mrs.config.settings import settings
from mrs.models.content_tfidf import ContentTfidfModel
from mrs.models.popularity import PopularityRecommender
from mrs.serving.movies_lookup import load_movies_lookup

Strategy = Literal["popularity", "content"]

POP_MODEL: PopularityRecommender | None = None
CONTENT_MODEL: ContentTfidfModel | None = None
MOVIES_LOOKUP: dict[int, dict[str, str]] = {}
MODELS_LOADED = False


def _models_dir() -> Path:
    return Path(settings.artifacts_dir) / settings.run_id / "models"


def _enrich(movie_id: int) -> dict[str, str | None]:
    meta = MOVIES_LOOKUP.get(movie_id, {})
    return {"title": meta.get("title"), "genres": meta.get("genres")}


def _ensure_loaded() -> None:
    if not MODELS_LOADED or POP_MODEL is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Train first.")


def _normalize_list(raw: Any, list_key: str) -> list[Any]:
    if isinstance(raw, dict) and list_key in raw and isinstance(raw[list_key], list):
        return raw[list_key]
    if isinstance(raw, list):
        return raw
    return []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global CONTENT_MODEL, MODELS_LOADED, MOVIES_LOOKUP, POP_MODEL

    MODELS_LOADED = False
    POP_MODEL = None
    CONTENT_MODEL = None
    MOVIES_LOOKUP = {}

    # Load movie metadata (optional)
    try:
        MOVIES_LOOKUP = load_movies_lookup(settings.data_dir)
    except Exception:
        MOVIES_LOOKUP = {}

    # Load models from artifacts
    try:
        models_dir = _models_dir()
        pop_path = models_dir / "popularity.joblib"
        content_path = models_dir / "content_tfidf.joblib"

        if not pop_path.exists():
            MODELS_LOADED = False
            yield
            return

        POP_MODEL = load(pop_path)

        if content_path.exists():
            CONTENT_MODEL = ContentTfidfModel.load(str(content_path))
        else:
            CONTENT_MODEL = None

        MODELS_LOADED = True
    except Exception:
        MODELS_LOADED = False

    yield


app = FastAPI(
    title="Movie Recommendation System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "name": "Movie Recommendation System",
        "docs": "/docs",
        "health": "/health",
        "example_recs": "/v1/recommendations?user_id=1&k=10&strategy=popularity",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "run_id": settings.run_id,
        "models_loaded": bool(MODELS_LOADED and POP_MODEL is not None),
    }


@app.get("/v1/recommendations")
def recommendations(
    user_id: int = Query(..., ge=1),
    k: int = Query(10, ge=1, le=50),
    strategy: Strategy = Query("popularity"),
):
    _ensure_loaded()

    if strategy == "popularity":
        raw = (
            POP_MODEL.recommend(user_id=user_id, k=k)
            if hasattr(POP_MODEL, "recommend")
            else POP_MODEL.top_k(k)  # type: ignore[attr-defined]
        )
    else:
        if CONTENT_MODEL is None:
            raise HTTPException(status_code=400, detail="Content model is not available.")
        raw = (
            CONTENT_MODEL.recommend_for_user(user_id=user_id, k=k)
            if hasattr(CONTENT_MODEL, "recommend_for_user")
            else CONTENT_MODEL.recommend(user_id=user_id, k=k)  # type: ignore[attr-defined]
        )

    items = _normalize_list(raw, "recommendations")
    recs: list[dict[str, Any]] = []

    for item in items:
        if isinstance(item, dict):
            movie_id = int(item.get("movie_id") or item.get("movieId"))
            score = item.get("score")
        else:
            movie_id = int(item[0])
            score = item[1] if len(item) > 1 else None

        out: dict[str, Any] = {"movie_id": movie_id, **_enrich(movie_id)}
        if score is not None:
            out["score"] = float(score)
        recs.append(out)

    return {"user_id": user_id, "k": k, "strategy": strategy, "recommendations": recs}


@app.get("/v1/similar-items")
def similar_items(
    movie_id: int = Query(..., ge=1),
    k: int = Query(10, ge=1, le=50),
):
    _ensure_loaded()
    if CONTENT_MODEL is None:
        raise HTTPException(status_code=400, detail="Content model is not available.")

    raw = (
        CONTENT_MODEL.similar_items(movie_id=movie_id, k=k)
        if hasattr(CONTENT_MODEL, "similar_items")
        else CONTENT_MODEL.most_similar(movie_id=movie_id, k=k)  # type: ignore[attr-defined]
    )

    items = _normalize_list(raw, "similar_items")
    out_items: list[dict[str, Any]] = []

    for item in items:
        if isinstance(item, dict):
            mid = int(item.get("movie_id") or item.get("movieId"))
            score = item.get("score")
        else:
            mid = int(item[0])
            score = item[1] if len(item) > 1 else None

        out: dict[str, Any] = {"movie_id": mid, **_enrich(mid)}
        if score is not None:
            out["score"] = float(score)
        out_items.append(out)

    return {"movie_id": movie_id, "k": k, "similar_items": out_items}
