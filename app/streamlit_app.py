# ruff: noqa: I001
from __future__ import annotations

import os
import time

import pandas as pd
import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "").strip().rstrip("/")
if not API_BASE_URL:
    st.error("API_BASE_URL is not set. Set it in Render → movie-recommendation-ui → Environment.")
    st.stop()


def call_api(path: str, params: dict | None = None, retries: int = 2, timeout: int = 30):
    url = f"{API_BASE_URL}{path}"
    last_err: str | None = None

    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code >= 400:
                return None, f"{r.status_code} {r.reason}: {(r.text or '').strip()[:500]}"
            try:
                return r.json(), None
            except Exception:
                return None, f"Expected JSON but got: {r.headers.get('content-type')} Body: {(r.text or '')[:500]}"
        except Exception as e:
            last_err = str(e)
            if attempt < retries:
                time.sleep(1)
            else:
                return None, f"Network error: {last_err}"
    return None, f"Unknown error: {last_err}"


def to_df(items):
    # Accept list of primitives, list of dicts, or dict-with-list.
    if items is None:
        return pd.DataFrame()

    if isinstance(items, dict):
        # Try common keys
        for k in ("recommendations", "similar_items", "items", "results"):
            if k in items and isinstance(items[k], list):
                items = items[k]
                break

    if isinstance(items, list):
        if not items:
            return pd.DataFrame()
        if all(isinstance(x, dict) for x in items):
            return pd.DataFrame(items)
        # list of ids/strings
        return pd.DataFrame({"item": items})

    return pd.DataFrame({"value": [items]})


def add_rank(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "rank" not in df.columns:
        df.insert(0, "rank", range(1, len(df) + 1))
    return df


st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬", layout="wide")

st.title("🎬 Movie Recommendation System")
st.caption("A live demo playground backed by FastAPI. Try different strategies and explore similar movies.")

with st.sidebar:
    st.subheader("Backend")
    st.code(API_BASE_URL)

    st.divider()
    st.subheader("Recommendations")
    user_id = st.number_input("User ID", min_value=1, value=1, step=1)
    k = st.slider("How many?", 1, 50, 10)
    strategy = st.selectbox("Strategy", ["popularity", "content"], index=0)

    st.divider()
    st.subheader("Similar movies")
    movie_id = st.number_input("Movie ID", min_value=1, value=1, step=1)
    k_sim = st.slider("How many similar?", 1, 50, 10)

colA, colB = st.columns([1, 1], gap="large")

with colA:
    st.subheader("✅ Health")
    health, err = call_api("/health", timeout=15)
    if err:
        st.error(err)
    else:
        if isinstance(health, dict) and health.get("models_loaded") is True:
            st.success("Backend is healthy and models are loaded.")
        else:
            st.warning("Backend reachable, but model status is unclear.")
        st.json(health)

with colB:
    st.subheader("📚 API docs")
    st.write("Open the backend docs here:")
    st.link_button("Open Swagger (/docs)", f"{API_BASE_URL}/docs")


st.divider()
tab1, tab2 = st.tabs(["👤 Get recommendations", "🎞️ Find similar movies"])

with tab1:
    st.subheader("Recommendations")
    go = st.button("Get recommendations", type="primary", use_container_width=True)

    if go:
        with st.spinner("Fetching recommendations..."):
            data, err = call_api(
                "/v1/recommendations",
                params={"user_id": int(user_id), "k": int(k), "strategy": strategy},
            )

        if err:
            st.error(err)
        else:
            df = add_rank(to_df(data))
            st.success("Done.")

            # Show top picks as simple cards
            if not df.empty:
                st.markdown("### ⭐ Top picks")
                top = df.head(5).to_dict(orient="records")
                for row in top:
                    title = row.get("title") or row.get("name") or row.get("movie_title") or row.get("item")
                    score = row.get("score") or row.get("similarity") or row.get("rating")
                    left, right = st.columns([3, 1])
                    with left:
                        st.markdown(f"**#{row.get('rank', '')} — {title}**")
                    with right:
                        if score is not None:
                            st.markdown(f"`{score}`")

                st.markdown("### 📋 Full list")
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.warning("No recommendations returned.")
            with st.expander("Raw JSON"):
                st.json(data)

with tab2:
    st.subheader("Similar movies")
    go2 = st.button("Find similar", type="primary", use_container_width=True)

    if go2:
        with st.spinner("Fetching similar movies..."):
            data, err = call_api(
                "/v1/similar-items",
                params={"movie_id": int(movie_id), "k": int(k_sim)},
            )

        if err:
            st.error(err)
        else:
            df = add_rank(to_df(data))
            st.success("Done.")
            if not df.empty:
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.warning("No similar items returned.")
            with st.expander("Raw JSON"):
                st.json(data)
