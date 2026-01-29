from __future__ import annotations

import json
import math
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from joblib import load

from mrs.config.settings import settings, Settings
from mrs.models.content_tfidf import ContentTfidfModel
from mrs.models.popularity import PopularityRecommender
from mrs.serving.movies_lookup import load_movies_lookup

Strategy = Literal["popularity", "content"]

POP_MODEL: PopularityRecommender | None = None
CONTENT_MODEL: ContentTfidfModel | None = None
MOVIES_LOOKUP: dict[int, dict[str, str]] = {}
MODELS_LOADED = False

def _models_dir() -> Path:
    return Path(settings.artifacts_dir) / Settings.run_id_from_env() / "models"


def _run_dir() -> Path:
    return Path(settings.artifacts_dir) / Settings.run_id_from_env()

def _clean_text(v: Any) -> str | None:
    if v is None:
        return None
    # handle NaN (works for float nan and numpy/pandas nan)
    try:
        if isinstance(v, float) and math.isnan(v):
            return None
        if v != v:  # NaN is not equal to itself
            return None
    except Exception:
        pass
    # ensure plain string
    s = str(v).strip()
    return s if s else None

def _enrich(movie_id: int) -> dict[str, str | None]:
    meta = MOVIES_LOOKUP.get(movie_id, {})
    return {
        "title": _clean_text(meta.get("title")),
        "genres": _clean_text(meta.get("genres")),
    }

def _movie_record(movie_id: int) -> dict[str, Any]:
    meta = MOVIES_LOOKUP.get(movie_id, {})
    return {
        "movie_id": movie_id,
        "title": _clean_text(meta.get("title")),
        "genres": _clean_text(meta.get("genres")),
    }

def _ensure_loaded() -> None:
    if not MODELS_LOADED or POP_MODEL is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Train first.")

def _normalize_list(raw: Any, list_key: str) -> list[Any]:
    if isinstance(raw, dict) and list_key in raw and isinstance(raw[list_key], list):
        return raw[list_key]
    if isinstance(raw, list):
        return raw
    return []


def _item_to_mid_score(item: Any) -> tuple[int, float | None]:
    """Extract (movie_id, score) from dict, tuple-like, or Rec-style object."""
    if isinstance(item, dict):
        mid = int(
            item.get("movie_id") or item.get("movieId") or item.get("id") or item.get("item") or 0
        )
        score = item.get("score")
        return mid, float(score) if score is not None else None
    if hasattr(item, "movie_id") and hasattr(item, "score"):
        return int(item.movie_id), float(item.score)
    try:
        mid = int(item[0])
        score = float(item[1]) if len(item) > 1 and item[1] is not None else None
        return mid, score
    except (IndexError, TypeError, KeyError):
        return 0, None

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

    # Load popularity model from artifacts (required).
    # Content model is lazy-loaded on first /v1/similar-items call (free-tier safe).
    try:
        models_dir = _models_dir()
        pop_path = models_dir / "popularity.joblib"

        if not pop_path.exists():
            MODELS_LOADED = False
            yield
            return

        POP_MODEL = load(pop_path)
        MODELS_LOADED = True

        CONTENT_MODEL = None
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
        "run_id": Settings.run_id_from_env(),
        "models_loaded": bool(MODELS_LOADED and POP_MODEL is not None),
    }


@app.get("/v1/model-info")
def model_info():
    out: dict[str, Any] = {
        "run_id": Settings.run_id_from_env(),
        "models_loaded": bool(MODELS_LOADED and POP_MODEL is not None),
    }
    run_dir = _run_dir()
    for name, f in [("manifest", "manifest.json"), ("metrics", "metrics.json")]:
        p = run_dir / f
        if p.exists():
            try:
                out[name] = json.loads(p.read_text())
            except Exception:
                out[name] = None
    return out


@app.get("/v1/movies/{movie_id}")
def get_movie(movie_id: int):
    rec = _movie_record(movie_id)
    if rec["title"] is None and rec["genres"] is None:
        raise HTTPException(status_code=404, detail="Movie not found.")
    return rec

@app.get("/v1/movies/search")
def search_movies(
    q: str = Query(..., min_length=1),
    limit: int = Query(20, ge=1, le=50),
):
    query = q.strip().casefold()
    if not query:
        raise HTTPException(status_code=400, detail="Empty query.")

    results: list[dict[str, Any]] = []
    for mid, meta in MOVIES_LOOKUP.items():
        title = (meta.get("title") or "")
        if query in title.casefold():
            results.append(
                {"movie_id": mid, "title": _clean_text(meta.get("title")), "genres": _clean_text(meta.get("genres")),}
            )
            if len(results) >= limit:
                break

    return {"q": q, "limit": limit, "results": results}

@app.get("/v1/recommendations")
def recommendations(
    user_id: int = Query(..., ge=1),
    k: int = Query(10, ge=1, le=50),
    strategy: Strategy = Query("popularity"),
):
    _ensure_loaded()

    try:
        if strategy == "popularity":
            # Popularity models vary: some ignore user_id and only support top_k(k)
            if hasattr(POP_MODEL, "recommend"):
                try:
                    raw = POP_MODEL.recommend(user_id=user_id, k=k)  # type: ignore[union-attr]
                except TypeError:
                    # Signature mismatch: fallback to calling without user_id
                    raw = POP_MODEL.recommend(k=k)  # type: ignore[union-attr,call-arg]
            else:
                raw = POP_MODEL.top_k(k)  # type: ignore[union-attr,attr-defined]

        else:
            # Content strategy requires content model
            if CONTENT_MODEL is None:
                raise HTTPException(
                    status_code=400,
                    detail="Content model is not loaded yet. Use Similar Explorer first to lazy-load it.",
                )

            if hasattr(CONTENT_MODEL, "recommend_for_user"):
                raw = CONTENT_MODEL.recommend_for_user(user_id=user_id, k=k)
            elif hasattr(CONTENT_MODEL, "recommend"):
                try:
                    raw = CONTENT_MODEL.recommend(user_id=user_id, k=k)  # type: ignore[call-arg]
                except TypeError:
                    raw = CONTENT_MODEL.recommend(user_id, k)  # type: ignore[misc]
            else:
                raise HTTPException(status_code=500, detail="Content model has no recommend method.")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"recommendations failed: {type(e).__name__}: {e}",
        ) from e

    items = _normalize_list(raw, "recommendations")
    recs: list[dict[str, Any]] = []

    for item in items:
        movie_id, score = _item_to_mid_score(item)
        if movie_id <= 0:
            continue
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

    global CONTENT_MODEL
    if CONTENT_MODEL is None:
        try:
            models_dir = _models_dir()
            content_path = models_dir / "content_tfidf.joblib"
            if not content_path.exists():
                raise HTTPException(status_code=400, detail="Content model is not available.")
            CONTENT_MODEL = ContentTfidfModel.load(str(content_path))
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Content model could not be loaded: {type(e).__name__}: {e}",
            ) from e

    try:
        if hasattr(CONTENT_MODEL, "similar_items"):
            raw = CONTENT_MODEL.similar_items(movie_id=movie_id, k=k)
        elif hasattr(CONTENT_MODEL, "similar_movies"):
            raw = CONTENT_MODEL.similar_movies(movie_id=movie_id, k=k)
        elif hasattr(CONTENT_MODEL, "most_similar"):
            raw = CONTENT_MODEL.most_similar(movie_id=movie_id, k=k)
        elif hasattr(CONTENT_MODEL, "recommend_similar"):
            raw = CONTENT_MODEL.recommend_similar(movie_id=movie_id, k=k)
        else:
            raise HTTPException(
                status_code=500,
                detail="Content model does not implement a similar-items method.",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"similar-items failed: {type(e).__name__}: {e}",
        ) from e

    items = (
        _normalize_list(raw, "similar_items")
        or _normalize_list(raw, "recommendations")
        or _normalize_list(raw, "items")
        or _normalize_list(raw, "results")
        or (raw if isinstance(raw, list) else [])
    )

    out_items: list[dict[str, Any]] = []

    for item in items:
        mid, score = _item_to_mid_score(item)
        if mid <= 0:
            continue
        out_dict: dict[str, Any] = {"movie_id": mid, **_enrich(mid)}
        if score is not None:
            out_dict["score"] = float(score)
        out_items.append(out_dict)

    return {"movie_id": movie_id, "k": k, "similar_items": out_items}