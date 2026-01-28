# ruff: noqa: I001
from __future__ import annotations

import os
import time
from typing import Any

import pandas as pd
import requests
import streamlit as st


API_BASE_URL_RAW = os.getenv("API_BASE_URL", "")


def _error_banner(msg: str) -> None:
    st.error(msg)
    st.stop()


def call_api(path: str, params: dict[str, Any] | None = None, retries: int = 2, timeout: int = 25):
    url = f"{API_BASE_URL}{path}"
    last_err: str | None = None

    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
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
                time.sleep(1)
            else:
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
        val = row.get(key)
        if val is not None and str(val).strip():
            return str(val)
    if "movie_id" in row:
        return f"Movie {row['movie_id']}"
    return "Recommendation"


def pick_score(row: dict[str, Any]) -> str | None:
    for key in ("score", "similarity", "rating", "pred", "pred_rating"):
        if key in row and row[key] is not None:
            return str(row[key])
    return None


st.set_page_config(page_title="🎬 Movie Recommendation System", page_icon="🎬", layout="wide")

# ---- DEBUG: show env var exactly (repr reveals hidden whitespace/newlines) ----
st.write("API_BASE_URL raw repr:", repr(API_BASE_URL_RAW))

API_BASE_URL = API_BASE_URL_RAW.strip().rstrip("/")
st.write("API_BASE_URL cleaned repr:", repr(API_BASE_URL))

if not API_BASE_URL:
    _error_banner("API_BASE_URL is not set. Configure it in Render → movie-recommendation-ui → Environment.")

st.title("🎬 Movie Recommendation System")
st.caption("A live, interactive demo backed by a FastAPI recommender.")

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
    movie_id = st.number_input("Movie ID", min_value=1, value=1, step=1)
    k_sim = st.slider("Similar movies (k)", min_value=1, max_value=50, value=10, step=1)

health_col, info_col = st.columns([1, 2], gap="large")

with health_col:
    st.subheader("✅ Backend health")
    health, health_err = call_api("/health", timeout=15)
    st.write("Health err:", health_err)
    if health is not None:
        st.json(health)
    else:
        st.json({})

    if not health_err and isinstance(health, dict):
        if health.get("models_loaded") is True:
            st.success("Healthy • Models loaded")
        else:
            st.warning("Healthy • Models NOT loaded (or unknown)")

with info_col:
    st.subheader("How to use this demo")
    st.markdown(
        """
- **Recommendations:** choose a **User ID**, strategy, and **k**.
- **Similar movies:** enter a **Movie ID** and **k**.
        """.strip()
    )

st.divider()

tab_recs, tab_sim = st.tabs(["👤 Recommendations", "🎞️ Similar movies"])

with tab_recs:
    st.subheader("Recommendations")
    run = st.button("Get recommendations", type="primary", use_container_width=True)

    if run:
        with st.spinner("Fetching recommendations..."):
            data, err = call_api(
                "/v1/recommendations",
                params={"user_id": int(user_id), "k": int(k), "strategy": strategy},
                timeout=30,
            )

        if err:
            st.error(err)
        else:
            df = ensure_rank(as_df(data))
            if df.empty:
                st.warning("No recommendations returned.")
            else:
                st.success("Done.")

                st.markdown("### ⭐ Top picks")
                top = df.head(5).to_dict(orient="records")
                for row in top:
                    title = pick_title(row)
                    score = pick_score(row)
                    left, right = st.columns([4, 1], gap="small")
                    with left:
                        st.markdown(f"**#{row.get('rank', '')} — {title}**")
                    with right:
                        if score is not None:
                            st.markdown(f"`{score}`")

                st.markdown("### 📋 Full list")
                st.dataframe(df, use_container_width=True, hide_index=True)

            with st.expander("Raw JSON"):
                st.json(data)

with tab_sim:
    st.subheader("Similar movies")
    run2 = st.button("Find similar", type="primary", use_container_width=True)

    if run2:
        with st.spinner("Fetching similar movies..."):
            data, err = call_api(
                "/v1/similar-items",
                params={"movie_id": int(movie_id), "k": int(k_sim)},
                timeout=30,
            )

        if err:
            st.error(err)
        else:
            df = ensure_rank(as_df(data))
            if df.empty:
                st.warning("No similar movies returned.")
            else:
                st.success("Done.")
                st.dataframe(df, use_container_width=True, hide_index=True)

            with st.expander("Raw JSON"):
                st.json(data)
