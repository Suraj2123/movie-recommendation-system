# ruff: noqa: I001
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any

import requests
import streamlit as st


APP_TITLE = "🎬 Movie Recommendation System"
API_BASE_URL = os.getenv("API_BASE_URL", "").strip().rstrip("/")
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "").strip()


def _stop(msg: str) -> None:
    st.error(msg)
    st.stop()


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
                return None, f"{r.status_code} {r.reason}: {body[:600]}"

            try:
                return r.json(), None
            except Exception:
                ct = r.headers.get("content-type", "(missing)")
                body = (r.text or "").strip()
                return None, f"Expected JSON but got {ct}. Body: {body[:600]}"

        except Exception as e:
            last_err = str(e)
            if attempt < retries:
                _sleep_backoff(attempt)
                continue
            return None, f"Network error calling {url}: {last_err}"

    return None, f"Unknown error calling {url}: {last_err}"


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def parse_year(title: str | None) -> int | None:
    if not title:
        return None
    m = re.search(r"\((\d{4})\)\s*$", title)
    if not m:
        return None
    return int(m.group(1))


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
        url = tmdb_search_poster(m.title, parse_year(m.title))
        if url:
            return url
    return "https://dummyimage.com/342x513/111/ffffff.png&text=%F0%9F%8E%AC"


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


def add_to_list(m: Movie) -> None:
    st.session_state["my_list"][m.movie_id] = m


def remove_from_list(movie_id: int) -> None:
    st.session_state["my_list"].pop(movie_id, None)


def in_list(movie_id: int) -> bool:
    return movie_id in st.session_state["my_list"]


st.set_page_config(page_title=APP_TITLE, page_icon="🎬", layout="wide")
init_state()

if not API_BASE_URL:
    _stop("API_BASE_URL is not set. Set it in Render for the UI service.")

st.title(APP_TITLE)
st.caption(
    "Netflix-style interactive demo powered by your FastAPI backend. "
    "On Render free tier, cold starts happen — the UI retries automatically."
)

with st.container():
    left, right = st.columns([2, 1], gap="large")
    with left:
        with st.spinner("Connecting to backend..."):
            health, health_err = call_api("/health", timeout=20, retries=6)

        if health_err:
            st.warning("Backend is waking up or temporarily unavailable.")
            st.caption(health_err)
        else:
            loaded = isinstance(health, dict) and health.get("models_loaded") is True
            if loaded:
                st.success("Backend healthy • Models loaded")
            else:
                st.warning("Backend healthy • Models not loaded yet")
            st.json(health)

    with right:
        st.subheader("🎛️ Demo Controls")
        user_id = st.number_input("User ID", min_value=1, value=1, step=1)
        rec_k = st.slider("How many recommendations?", 5, 30, 12)
        strategy = st.selectbox("Strategy", ["popularity", "content"], index=0)

        st.caption("Optional: add TMDB_API_KEY to show real posters.")
        if TMDB_API_KEY:
            st.success("TMDB posters enabled")
        else:
            st.info("Posters disabled (no TMDB_API_KEY)")

st.divider()

search_col, list_col = st.columns([3, 2], gap="large")
with search_col:
    st.subheader("🔎 Search movies")
    q = st.text_input(
        "Type a movie title",
        value=st.session_state["last_search"],
        placeholder="e.g., Shawshank, Toy Story, Matrix…",
    )
    do_search = st.button("Search", type="primary")
with list_col:
    st.subheader("✅ My List")
    my_list_items = list(st.session_state["my_list"].values())
    if not my_list_items:
        st.caption("Save movies here to build a mini watchlist.")
    else:
        for m in my_list_items[:8]:
            cols = st.columns([5, 1])
            cols[0].write(m.title or f"Movie {m.movie_id}")
            if cols[1].button("✖", key=f"rm_{m.movie_id}"):
                remove_from_list(m.movie_id)
                st.rerun()

search_results: list[Movie] = []
if do_search and q.strip():
    st.session_state["last_search"] = q.strip()
    with st.spinner("Searching..."):
        data, err = call_api("/v1/movies/search", params={"q": q.strip(), "limit": 24}, timeout=25, retries=4)
    if err:
        st.error(err)
    else:
        search_results = to_movies(data, "results")

selected_movie_id = st.session_state.get("selected_movie_id")


def render_movie_card(m: Movie, key_prefix: str) -> None:
    poster = poster_or_placeholder(m)
    title = m.title or f"Movie {m.movie_id}"
    subtitle = m.genres or "Genres unavailable"
    score_txt = f"{m.score:.3f}" if m.score is not None else None

    with st.container(border=True):
        st.image(poster, use_container_width=True)
        st.markdown(f"**{title}**")
        st.caption(subtitle)
        if score_txt:
            st.caption(f"Score: `{score_txt}`")

        b1, b2, b3 = st.columns(3)
        if b1.button("Details", key=f"{key_prefix}_details_{m.movie_id}", use_container_width=True):
            st.session_state["selected_movie_id"] = m.movie_id
            st.rerun()

        if not in_list(m.movie_id):
            if b2.button("➕ My List", key=f"{key_prefix}_add_{m.movie_id}", use_container_width=True):
                add_to_list(m)
                st.rerun()
        else:
            if b2.button("✅ Saved", key=f"{key_prefix}_saved_{m.movie_id}", use_container_width=True):
                pass

        if b3.button("🎞️ Similar", key=f"{key_prefix}_sim_{m.movie_id}", use_container_width=True):
            st.session_state["selected_movie_id"] = m.movie_id
            st.session_state["open_similar"] = True
            st.rerun()


def render_row(title: str, movies: list[Movie], key_prefix: str) -> None:
    st.subheader(title)
    if not movies:
        st.caption("No items to show.")
        return

    cols = st.columns(6, gap="medium")
    for i, m in enumerate(movies[:12]):
        with cols[i % 6]:
            render_movie_card(m, key_prefix=f"{key_prefix}_{i}")


tab_home, tab_similar, tab_about = st.tabs(["🏠 Home", "🎞️ Similar Explorer", "ℹ️ About"])

with tab_home:
    with st.spinner("Loading Trending..."):
        rec_data, rec_err = call_api(
            "/v1/recommendations",
            params={"user_id": int(user_id), "k": int(rec_k), "strategy": "popularity"},
            timeout=45,
            retries=6,
        )
    trending = [] if rec_err else to_movies(rec_data, "recommendations")
    if rec_err:
        st.warning("Trending is temporarily unavailable (backend warming up).")
        st.caption(rec_err)
    else:
        render_row("🔥 Trending (Popularity)", trending, "trending")

    with st.spinner("Loading For You..."):
        fy_data, fy_err = call_api(
            "/v1/recommendations",
            params={"user_id": int(user_id), "k": int(rec_k), "strategy": strategy},
            timeout=60,
            retries=6,
        )
    for_you = [] if fy_err else to_movies(fy_data, "recommendations")
    if fy_err:
        st.warning("For You is temporarily unavailable.")
        st.caption(fy_err)
    else:
        render_row(f"✨ For You (strategy={strategy})", for_you, "for_you")

    if search_results:
        render_row(f"🔎 Search results for “{st.session_state['last_search']}”", search_results, "search")

    if selected_movie_id is not None:
        st.divider()
        st.subheader("🎬 Selected movie")
        details, det_err = call_api(f"/v1/movies/{int(selected_movie_id)}", timeout=25, retries=4)
        if det_err:
            st.warning("Could not load movie details.")
            st.caption(det_err)
        else:
            m = Movie(
                movie_id=int(details["movie_id"]),
                title=details.get("title"),
                genres=details.get("genres"),
                score=None,
            )
            cols = st.columns([1, 2], gap="large")
            with cols[0]:
                st.image(poster_or_placeholder(m), use_container_width=True)
            with cols[1]:
                st.markdown(f"## {m.title or f'Movie {m.movie_id}'}")
                st.caption(m.genres or "Genres unavailable")
                b1, b2, b3 = st.columns(3)
                if not in_list(m.movie_id):
                    if b1.button("➕ Add to My List", use_container_width=True):
                        add_to_list(m)
                        st.rerun()
                else:
                    if b1.button("✅ In My List", use_container_width=True):
                        pass
                if b2.button("🎞️ Explore Similar", use_container_width=True):
                    st.session_state["open_similar"] = True
                    st.rerun()
                if b3.button("❌ Clear selection", use_container_width=True):
                    st.session_state["selected_movie_id"] = None
                    st.rerun()

with tab_similar:
    st.subheader("🎞️ Similar Explorer")
    st.caption(
        "Pick a movie and fetch similar titles. First call can be slower on free tier because the content model lazy-loads."
    )

    seed = st.number_input("Seed Movie ID", min_value=1, value=int(selected_movie_id or 318), step=1)
    k_sim = st.slider("How many similar movies?", 5, 30, 12)

    if st.button("Find Similar", type="primary"):
        with st.spinner("Fetching similar movies (may take 30–60s on first call)..."):
            sim_data, sim_err = call_api(
                "/v1/similar-items",
                params={"movie_id": int(seed), "k": int(k_sim)},
                timeout=90,
                retries=8,
            )
        if sim_err:
            st.error(sim_err)
        else:
            sims = to_movies(sim_data, "similar_items")
            render_row("Because you watched…", sims, "similar_row")

with tab_about:
    st.subheader("ℹ️ About this demo")
    st.markdown(
        """


**Render env vars**
- UI: `API_BASE_URL` = your API service URL  
- UI (optional): `TMDB_API_KEY`  
        """.strip()
    )


