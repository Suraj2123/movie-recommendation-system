# ruff: noqa: I001
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any

import requests
import streamlit as st

APP_TITLE = "Movie Picks"
APP_TAGLINE = "Recommendations powered by popularity & content-based models"
API_BASE_URL = (os.getenv("API_BASE_URL", "").strip().rstrip("/") or "http://localhost:8000").rstrip("/")
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "").strip()
LOCAL_DEFAULT = not os.getenv("API_BASE_URL", "").strip()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sleep_backoff(attempt: int) -> None:
    time.sleep(0.8 + attempt * 0.4)


def call_api(
    path: str,
    params: dict[str, Any] | None = None,
    retries: int = 6,
    timeout: int = 45,
) -> tuple[dict[str, Any] | list[Any] | None, str | None]:
    url = f"{API_BASE_URL}{path}"
    last_err: str | None = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code >= 500 and attempt < retries:
                _sleep_backoff(attempt)
                continue
            if r.status_code >= 400:
                body = (r.text or "").strip()
                return None, f"{r.status_code} {r.reason}: {body[:500]}"
            try:
                return r.json(), None
            except Exception:
                ct = r.headers.get("content-type", "(missing)")
                body = (r.text or "").strip()
                return None, f"Expected JSON; got {ct}. {body[:500]}"
        except Exception as e:
            last_err = str(e)
            if attempt < retries:
                _sleep_backoff(attempt)
                continue
            return None, f"Request failed: {last_err}"
    return None, last_err or "Request failed"


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def parse_year(title: str | None) -> int | None:
    if not title:
        return None
    m = re.search(r"\((\d{4})\)\s*$", title)
    return int(m.group(1)) if m else None


def strip_year(title: str | None) -> str:
    if not title:
        return ""
    return re.sub(r"\s*\(\d{4}\)\s*$", "", title).strip()


@st.cache_data(ttl=24 * 3600)
def tmdb_search_poster(title: str, year: int | None) -> str | None:
    if not TMDB_API_KEY:
        return None
    base = "https://api.themoviedb.org/3/search/movie"
    params: dict[str, Any] = {"api_key": TMDB_API_KEY, "query": strip_year(title)}
    if year is not None:
        params["year"] = year
    try:
        r = requests.get(base, params=params, timeout=20)
        if r.status_code != 200:
            return None
        data = r.json()
        results = data.get("results") or []
        if not results:
            return None
        poster_path = results[0].get("poster_path")
        if not poster_path:
            return None
        return f"https://image.tmdb.org/t/p/w342{poster_path}"
    except Exception:
        return None


@dataclass
class Movie:
    movie_id: int
    title: str | None
    genres: str | None
    score: float | None = None


def poster_or_placeholder(m: Movie) -> str:
    if m.title and TMDB_API_KEY:
        u = tmdb_search_poster(m.title, parse_year(m.title))
        if u:
            return u
    return "https://placehold.co/342x513/1a1a2e/eee?text=🎬"


def to_movies(items: Any, key: str) -> list[Movie]:
    if isinstance(items, dict) and key in items and isinstance(items[key], list):
        items = items[key]
    if not isinstance(items, list):
        return []
    out: list[Movie] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        out.append(
            Movie(
                movie_id=safe_int(it.get("movie_id")),
                title=it.get("title"),
                genres=it.get("genres"),
                score=float(it["score"]) if it.get("score") is not None else None,
            )
        )
    return out


def init_state() -> None:
    if "my_list" not in st.session_state:
        st.session_state["my_list"] = {}
    if "selected_movie_id" not in st.session_state:
        st.session_state["selected_movie_id"] = None
    if "last_search" not in st.session_state:
        st.session_state["last_search"] = ""
    if "open_similar" not in st.session_state:
        st.session_state["open_similar"] = False
    if "similar_results" not in st.session_state:
        st.session_state["similar_results"] = []


def add_to_list(m: Movie) -> None:
    st.session_state["my_list"][m.movie_id] = m


def remove_from_list(movie_id: int) -> None:
    st.session_state["my_list"].pop(movie_id, None)


def in_list(movie_id: int) -> bool:
    return movie_id in st.session_state["my_list"]


# ---------------------------------------------------------------------------
# Page config & theme
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS: dark, cinematic
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .main { background: linear-gradient(180deg, #0f0f14 0%, #1a1a24 100%); }
    section[data-testid="stSidebar"] { background: #12121a; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #1e1e2a; color: #a0a0b0; border-radius: 8px;
        padding: 8px 16px; font-weight: 500;
    }
    .stTabs [aria-selected="true"] { background: #2d2d3d; color: #fff; }
    div[data-testid="stHorizontalBlock"] > div { border-radius: 12px; }
    .movie-card { border-radius: 12px; overflow: hidden; background: #1a1a24; border: 1px solid #2a2a3a; }
    .hero { padding: 2rem 0 1.5rem; }
    .hero h1 { font-size: 2.4rem; font-weight: 700; color: #fff; letter-spacing: -0.02em; }
    .hero p { color: #888; font-size: 1.05rem; margin-top: 0.25rem; }
    .local-banner { background: #1e2a1e; color: #8bc34a; padding: 10px 16px; border-radius: 10px; margin-bottom: 1rem; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

init_state()

# ---------------------------------------------------------------------------
# Sidebar: controls, status, My List
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(f"### 🎬 {APP_TITLE}")
    st.caption(APP_TAGLINE)
    st.divider()

    if LOCAL_DEFAULT:
        st.markdown(
            '<div class="local-banner">Running against local API (localhost:8000). '
            'Start it with: <code>uvicorn mrs.serving.api:app --reload</code></div>',
            unsafe_allow_html=True,
        )

    with st.spinner("Checking API…"):
        health, health_err = call_api("/health", timeout=12, retries=3)

    if health_err:
        st.warning("API unreachable. Start the backend first.")
        st.caption(health_err[:200])
    else:
        ok = isinstance(health, dict) and health.get("models_loaded") is True
        st.success("API ready · Models loaded" if ok else "API ready · Models loading…")

    st.divider()
    st.subheader("Controls")
    user_id = st.number_input("User ID", min_value=1, value=1, step=1, key="user_id")
    rec_k = st.slider("Recommendations count", 5, 30, 12, key="rec_k")
    strategy = st.selectbox("Strategy", ["popularity", "content"], index=0, key="strategy")
    st.caption("Popularity = top-rated; Content = TF‑IDF similarity.")

    if TMDB_API_KEY:
        st.caption("🖼️ TMDB posters enabled")
    else:
        st.caption("🖼️ Set TMDB_API_KEY for posters")

    st.divider()
    st.subheader("My List")
    my_list_items = list(st.session_state["my_list"].values())
    if not my_list_items:
        st.caption("Add movies from the home feed.")
    else:
        for m in my_list_items[:12]:
            c1, c2 = st.columns([5, 1])
            with c1:
                st.caption(m.title or f"Movie {m.movie_id}")
            with c2:
                if st.button("✖", key=f"rm_{m.movie_id}"):
                    remove_from_list(m.movie_id)
                    st.rerun()

# ---------------------------------------------------------------------------
# Main: hero, search, tabs
# ---------------------------------------------------------------------------

st.markdown('<div class="hero"><h1>Discover movies</h1><p>Trending, picks for you, and similar titles.</p></div>', unsafe_allow_html=True)

search_col, _ = st.columns([3, 1])
with search_col:
    q = st.text_input(
        "Search by title",
        value=st.session_state["last_search"],
        placeholder="e.g. Shawshank, Toy Story, Matrix…",
        key="search_q",
    )
    do_search = st.button("Search", type="primary", key="do_search")

search_results: list[Movie] = []
if do_search and (q or "").strip():
    st.session_state["last_search"] = q.strip()
    with st.spinner("Searching…"):
        data, err = call_api("/v1/movies/search", params={"q": q.strip(), "limit": 24}, timeout=20, retries=3)
    if err:
        st.error(err)
    else:
        search_results = to_movies(data, "results")

selected_movie_id = st.session_state.get("selected_movie_id")


def render_movie_card(m: Movie, key_prefix: str) -> None:
    poster = poster_or_placeholder(m)
    title = m.title or f"Movie {m.movie_id}"
    subtitle = m.genres or "—"
    with st.container():
        st.image(poster, use_container_width=True)
        st.markdown(f"**{title}**")
        st.caption(subtitle)
        if m.score is not None:
            st.caption(f"Score: {m.score:.3f}")
        b1, b2, b3 = st.columns(3)
        if b1.button("Details", key=f"{key_prefix}_d_{m.movie_id}", use_container_width=True):
            st.session_state["selected_movie_id"] = m.movie_id
            st.rerun()
        if not in_list(m.movie_id):
            if b2.button("➕ List", key=f"{key_prefix}_a_{m.movie_id}", use_container_width=True):
                add_to_list(m)
                st.rerun()
        else:
            b2.button("✓ Saved", key=f"{key_prefix}_s_{m.movie_id}", use_container_width=True, disabled=True)
        if b3.button("Similar", key=f"{key_prefix}_sim_{m.movie_id}", use_container_width=True):
            st.session_state["selected_movie_id"] = m.movie_id
            st.session_state["open_similar"] = True
            st.rerun()


def render_row(title: str, movies: list[Movie], key_prefix: str) -> None:
    if not movies:
        st.caption("No movies to show.")
        return
    st.subheader(title)
    cols = st.columns(6, gap="medium")
    for i, m in enumerate(movies[:12]):
        with cols[i % 6]:
            render_movie_card(m, key_prefix=f"{key_prefix}_{i}")

tab_home, tab_similar, tab_about = st.tabs(["Home", "Similar movies", "About"])

with tab_home:
    with st.spinner("Loading trending…"):
        rec_data, rec_err = call_api(
            "/v1/recommendations",
            params={"user_id": int(user_id), "k": int(rec_k), "strategy": "popularity"},
            timeout=40,
            retries=5,
        )
    trending = [] if rec_err else to_movies(rec_data, "recommendations")
    if rec_err:
        st.warning("Trending temporarily unavailable.")
        st.caption(rec_err)
    else:
        render_row("Trending", trending, "trending")

    with st.spinner("Loading for you…"):
        fy_data, fy_err = call_api(
            "/v1/recommendations",
            params={"user_id": int(user_id), "k": int(rec_k), "strategy": strategy},
            timeout=50,
            retries=5,
        )
    for_you = [] if fy_err else to_movies(fy_data, "recommendations")
    if fy_err:
        st.warning("For you temporarily unavailable.")
        st.caption(fy_err)
    else:
        render_row(f"For you ({strategy})", for_you, "foryou")

    if search_results:
        render_row(f'Search: "{st.session_state["last_search"]}"', search_results, "search")

    if selected_movie_id is not None:
        st.divider()
        st.subheader("Selected movie")
        details, det_err = call_api(f"/v1/movies/{int(selected_movie_id)}", timeout=15, retries=2)
        if det_err:
            st.warning("Could not load details.")
            st.caption(det_err)
        else:
            m = Movie(
                movie_id=int(details["movie_id"]),
                title=details.get("title"),
                genres=details.get("genres"),
                score=None,
            )
            c1, c2 = st.columns([1, 2])
            with c1:
                st.image(poster_or_placeholder(m), use_container_width=True)
            with c2:
                st.markdown(f"## {m.title or f'Movie {m.movie_id}'}")
                st.caption(m.genres or "—")
                d1, d2, d3 = st.columns(3)
                if not in_list(m.movie_id):
                    if d1.button("Add to list", key="sel_add"):
                        add_to_list(m)
                        st.rerun()
                else:
                    d1.button("In list", key="sel_in", disabled=True)
                if d2.button("Find similar", key="sel_sim"):
                    st.session_state["open_similar"] = True
                    st.rerun()
                if d3.button("Clear", key="sel_clear"):
                    st.session_state["selected_movie_id"] = None
                    st.rerun()

with tab_similar:
    st.subheader("Find similar movies")
    st.caption("Pick a movie (by ID or from search) and get content-based recommendations.")

    seed = st.number_input("Movie ID", min_value=1, value=int(selected_movie_id or 318), step=1, key="seed_id")
    k_sim = st.slider("How many similar?", 5, 30, 12, key="k_sim")

    if st.button("Find similar", type="primary", key="find_sim"):
        with st.spinner("Fetching similar movies…"):
            sim_data, sim_err = call_api(
                "/v1/similar-items",
                params={"movie_id": int(seed), "k": int(k_sim)},
                timeout=90,
                retries=6,
            )
        if sim_err:
            st.error(sim_err)
            st.session_state["similar_results"] = []
        else:
            sims = to_movies(sim_data, "similar_items")
            st.session_state["similar_results"] = sims
            render_row("Similar movies", sims, "sim")
    else:
        prev = st.session_state.get("similar_results") or []
        if prev:
            render_row("Similar movies (last run)", prev, "sim_prev")

with tab_about:
    st.subheader("About")
    st.markdown(
        """
        **Movie Picks** uses a FastAPI backend with:
        - **Popularity** recommender (Bayesian-smoothed ratings)
        - **Content-based** model (TF‑IDF on title + genres, cosine similarity)

        **Run locally**
        1. Train: `python -m mrs.pipelines.train --run-id local`
        2. API: `uvicorn mrs.serving.api:app --reload`
        3. UI: `API_BASE_URL=http://localhost:8000 streamlit run app/streamlit_app.py`

        **Deploy** (e.g. Render): set `API_BASE_URL` to your API URL; optional `TMDB_API_KEY` for posters.
        """
    )
