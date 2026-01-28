# ruff: noqa: I001
from __future__ import annotations

import os
import time
from typing import Any

import pandas as pd
import requests
import streamlit as st


APP_TITLE = "🎬 Movie Recommendation System"
API_BASE_URL = os.getenv("API_BASE_URL", "").strip().rstrip("/")


def _stop_with_error(msg: str) -> None:
    st.error(msg)
    st.stop()


def call_api(
    path: str,
    params: dict[str, Any] | None = None,
    retries: int = 6,
    timeout: int = 45,
):
    """
    Render free-tier can cold-start. This function retries with small backoff
    and returns (json, err_string).
    """
    url = f"{API_BASE_URL}{path}"
    last_err: str | None = None

    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)

            # If Render proxy returns HTML error pages on startup (502/503),
            # treat as retryable for a bit.
            if r.status_code in (502, 503, 504) and attempt < retries:
                time.sleep(1.0 + attempt * 0.5)
                continue

            if r.status_code >= 400:
                text = (r.text or "").strip()
                return None, f"{r.status_code} {r.reason}: {text[:500]}"

            try:
                return r.json(), None
            except Exception:
                ct = r.headers.get("content-type", "(missing)")
                text = (r.text or "").strip()
                return None, f"Expected JSON but got {ct}. Body: {text[:500]}"

        except Exception as e:
            last_err = str(e)
            if attempt < retries:
                time.sleep(1.0 + attempt * 0.5)
                continue
            return None, f"Network error calling {url}: {last_err}"

    return None, f"Unknown error calling {url}: {last_err}"


def as_df(payload: Any) -> pd.DataFrame:
    if payload is None:
        return pd.DataFrame()

    if isinstance(payload, dict):
        for k in ("recommendations", "similar_items", "items", "results"):
            if k in payload and isinstance(payload[k], list):
                payload = payload[k]
                break

    if isinstance(payload, list):
        if not payload:
            return pd.DataFrame()
        if all(isinstance(x, dict) for x in payload):
            return pd.DataFrame(payload)
        return pd.DataFrame({"item": payload})

    if isinstance(payload, dict):
        return pd.DataFrame([payload])

    return pd.DataFrame({"value": [payload]})


def ensure_rank(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "rank" not in df.columns:
        df.insert(0, "rank", range(1, len(df) + 1))
    return df


def pick_title(row: dict[str, Any]) -> str:
    for key in ("title", "name", "movie_title", "movie", "item"):
        v = row.get(key)
        if v is not None and str(v).strip():
            return str(v)
    if "movie_id" in row:
        return f"Movie {row['movie_id']}"
    return "Recommendation"


def pick_subtitle(row: dict[str, Any]) -> str | None:
    genres = row.get("genres")
    if genres is not None and str(genres).strip():
        return str(genres)
    return None


def pick_score(row: dict[str, Any]) -> str | None:
    for key in ("score", "similarity", "rating", "pred", "pred_rating"):
        if key in row and row[key] is not None:
            try:
                return f"{float(row[key]):.3f}"
            except Exception:
                return str(row[key])
    return None


# -------------------- UI --------------------
st.set_page_config(page_title=APP_TITLE, page_icon="🎬", layout="wide")

if not API_BASE_URL:
    _stop_with_error("API_BASE_URL is not set. Configure it in Render → movie-recommendation-ui → Environment.")

st.title(APP_TITLE)
st.caption("A live, interactive demo backed by a FastAPI recommender. Render free-tier may cold-start; the app will retry automatically.")

with st.sidebar:
    st.subheader("Backend")
    st.code(API_BASE_URL)
    st.link_button("Open API Docs (/docs)", f"{API_BASE_URL}/docs")

    st.divider()
    st.subheader("Recommendations")
    user_id = st.number_input("User ID", min_value=1, value=1, step=1)
    k = st.slider("Recommendations (k)", min_value=1, max_value=50, value=10, step=1)
    strategy = st.selectbox("Strategy", ["popularity", "content"], index=0)

    st.divider()
    st.subheader("Similar movies")
    movie_id = st.number_input("Movie ID", min_value=1, value=318, step=1)
    k_sim = st.slider("Similar movies (k)", min_value=1, max_value=50, value=10, step=1)

    st.divider()
    st.subheader("Quick examples")
    ex1, ex2, ex3 = st.columns(3)
    if ex1.button("User 1", use_container_width=True):
        st.session_state["demo_user_id"] = 1
    if ex2.button("User 5", use_container_width=True):
        st.session_state["demo_user_id"] = 5
    if ex3.button("User 10", use_container_width=True):
        st.session_state["demo_user_id"] = 10

if "demo_user_id" in st.session_state:
    user_id = int(st.session_state.pop("demo_user_id"))

health_col, info_col = st.columns([1, 2], gap="large")

with health_col:
    st.subheader("✅ Backend health")
    with st.spinner("Checking backend..."):
        health, health_err = call_api("/health", timeout=20)

    if health_err:
        st.warning("Backend may be waking up. Try again in a few seconds.")
        st.caption(health_err)
    else:
        loaded = isinstance(health, dict) and health.get("models_loaded") is True
        if loaded:
            st.success("Healthy • Models loaded")
        else:
            st.warning("Healthy • Models not loaded yet")
        st.json(health)

with info_col:
    st.subheader("What to try")
    st.markdown(
        """
- Start with **Popularity** for a quick “Top movies” style list.
- Try **Content** to get recommendations based on movie similarity features.
- Use **Similar movies** with movie `318` (Shawshank) to see related titles.
        """.strip()
    )

st.divider()

tab_recs, tab_sim = st.tabs(["👤 Recommendations", "🎞️ Similar movies"])

with tab_recs:
    st.subheader("Recommendations")
    run = st.button("Get recommendations", type="primary", use_container_width=True)

    if run:
        with st.spinner("Fetching recommendations (may take ~30–60s on cold start)..."):
            data, err = call_api(
                "/v1/recommendations",
                params={"user_id": int(user_id), "k": int(k), "strategy": strategy},
            )

        if err:
            st.error(err)
        else:
            df = ensure_rank(as_df(data))
            recs = df.to_dict(orient="records") if not df.empty else []

            if not recs:
                st.warning("No recommendations returned.")
            else:
                st.success("Done.")

                st.markdown("### ⭐ Top picks")
                for row in recs[:5]:
                    title = pick_title(row)
                    subtitle = pick_subtitle(row)
                    score = pick_score(row)

                    card_left, card_right = st.columns([5, 1], gap="small")
                    with card_left:
                        st.markdown(f"**#{row.get('rank', '')} — {title}**")
                        if subtitle:
                            st.caption(subtitle)
                    with card_right:
                        if score:
                            st.markdown(f"`{score}`")

                st.markdown("### 📋 Full list")
                st.dataframe(df, use_container_width=True, hide_index=True)

            with st.expander("Raw JSON"):
                st.json(data)

with tab_sim:
    st.subheader("Similar movies")
    run2 = st.button("Find similar", type="primary", use_container_width=True)

    if run2:
        with st.spinner("Fetching similar movies (may take ~30–60s on cold start)..."):
            data, err = call_api(
                "/v1/similar-items",
                params={"movie_id": int(movie_id), "k": int(k_sim)},
            )

        if err:
            st.error(err)
        else:
            df = ensure_rank(as_df(data))
            if df.empty:
                st.warning("No similar movies returned.")
            else:
                st.success("Done.")

                st.markdown("### 🎞️ Results")
                st.dataframe(df, use_container_width=True, hide_index=True)

            with st.expander("Raw JSON"):
                st.json(data)
