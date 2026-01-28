from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query

from mrs.config.settings import settings
from mrs.serving.loader import load_models
from mrs.serving.movies_lookup import load_movies_lookup

Strategy = Literal["popularity", "content"]

POP_MODEL: Any | None = None
CONTENT_MODEL: Any | None = None
MOVIES_LOOKUP: dict[int, dict[str, str]] = {}
MODELS_LOADED = False


def _enrich_movie(movie_id: int) -> dict[str, str | None]:
    meta = MOVIES_LOOKUP.get(movie_id, {})
    return {
        "title": meta.get("title"),
        "genres": meta.get("genres"),
    }


def _call_recommender(obj: Any, method_names: list[str], *args: Any, **kwargs: Any):
    """
    Try multiple method names on a model object until one exists.
    """
    for name in method_names:
        fn = getattr(obj, name, None)
        if callable(fn):
            return fn(*args, **kwargs)
    raise AttributeError(f"None of these methods exist on {type(obj).__name__}: {method_names}")


def _normalize_recs(raw: Any) -> list[dict[str, Any]]:
    """
    Normalize recommender output into:
      [{"movie_id": int, "score": float}, ...]
    Accepts:
      - list[dict] with movie_id/score
      - list[tuple(movie_id, score)]
      - list[int] (score omitted)
      - dict with key "recommendations"
    """
    if raw is None:
        return []

    if isinstance(raw, dict) and "recommendations" in raw:
        raw = raw["recommendations"]

    if not isinstance(raw, list):
        return []

    out: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            mid = item.get("movie_id") or item.get("movieId") or item.get("id") or item.get("item")
            score = item.get("score")
            if mid is None:
                continue
            try:
                movie_id = int(mid)
            except Exception:
                continue
            rec: dict[str, Any] = {"movie_id": movie_id}
            if score is not None:
                try:
                    rec["score"] = float(score)
                except Exception:
                    pass
            out.append(rec)
            continue

        if isinstance(item, (tuple, list)) and len(item) >= 1:
            try:
                movie_id = int(item[0])
            except Exception:
                continue
            rec2: dict[str, Any] = {"movie_id": movie_id}
            if len(item) >= 2 and item[1] is not None:
                try:
                    rec2["score"] = float(item[1])
                except Exception:
                    pass
            out.append(rec2)
            continue

        # list[int]/list[str]
        try:
            movie_id = int(item)
        except Exception:
            continue
        out.append({"movie_id": movie_id})

    return out


def _ensure_loaded() -> None:
    if not MODELS_LOADED or POP_MODEL is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Train first.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global CONTENT_MODEL, MODELS_LOADED, MOVIES_LOOKUP, POP_MODEL

    MODELS_LOADED = False
    POP_MODEL = None
    CONTENT_MODEL = None
    MOVIES_LOOKUP = {}

    # Load models (from artifacts)
    try:
        pop, content, _metrics = load_models(settings.run_id)
        POP_MODEL = pop
        CONTENT_MODEL = content
        MODELS_LOADED = True
    except Exception:
        # Keep API up, but mark models not loaded
        MODELS_LOADED = False

    # Load movie titles/genres lookup (from downloaded dataset dir)
    # This is optional; API will still work without it.
    try:
        MOVIES_LOOKUP = load_movies_lookup(settings.data_dir)
    except Exception:
        MOVIES_LOOKUP = {}

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
        "models_loaded": MODELS_LOADED and POP_MODEL is not None,
    }


@app.get("/v1/recommendations")
def recommendations(
    user_id: int = Query(..., ge=1),
    k: int = Query(10, ge=1, le=50),
    strategy: Strategy = Query("popularity"),
):
    _ensure_loaded()

    # Popularity model usually ignores user_id; content may use it.
    if strategy == "popularity":
        raw = _call_recommender(
            POP_MODEL,
            ["recommend", "recommend_for_user", "predict", "top_k"],
            user_id,
            k,
        )
    else:
        if CONTENT_MODEL is None:
            raise HTTPException(status_code=400, detail="Content model is not available.")
        raw = _call_recommender(
            CONTENT_MODEL,
            ["recommend_for_user", "recommend", "predict", "top_k"],
            user_id,
            k,
        )

    recs = _normalize_recs(raw)

    # Enrich with title/genres if we have them
    enriched = []
    for r in recs:
        movie_id = int(r["movie_id"])
        out = {
            "movie_id": movie_id,
            **_enrich_movie(movie_id),
            "score": float(r.get("score")) if r.get("score") is not None else None,
        }
        # drop score if None to keep response clean
        if out["score"] is None:
            out.pop("score")
        enriched.append(out)

    return {
        "user_id": user_id,
        "k": k,
        "strategy": strategy,
        "recommendations": enriched,
    }


@app.get("/v1/similar-items")
def similar_items(
    movie_id: int = Query(..., ge=1),
    k: int = Query(10, ge=1, le=50),
):
    _ensure_loaded()
    if CONTENT_MODEL is None:
        raise HTTPException(status_code=400, detail="Content model is not available.")

    raw = _call_recommender(
        CONTENT_MODEL,
        ["similar_items", "similar_movies", "most_similar", "recommend_similar", "similar"],
        movie_id,
        k,
    )

    recs = _normalize_recs(raw)
    enriched = []
    for r in recs:
        mid = int(r["movie_id"])
        out = {
            "movie_id": mid,
            **_enrich_movie(mid),
            "score": float(r.get("score")) if r.get("score") is not None else None,
        }
        if out["score"] is None:
            out.pop("score")
        enriched.append(out)

    return {
        "movie_id": movie_id,
        "k": k,
        "similar_items": enriched,
    }
