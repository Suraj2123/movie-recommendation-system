# ruff: noqa: I001
from __future__ import annotations

import os
import time
from typing import Any

import pandas as pd
import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "").strip().rstrip("/")
APP_TITLE = "🎬 Movie Recommendation System"


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
    """
    Accepts:
      - list[dict] / list[int|str]
      - dict with common keys: recommendations / similar_items / items / results
      - dict (single object)
    Returns a DataFrame for display.
    """
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
    # fallback: show something stable
    if "movie_id" in row:
        return f"Movie {row['movie_id']}"
    return "Recommendation"


def pick_score(row: dict[str, Any]) -> str | None:
    for key in ("score", "similarity", "rating", "pred", "pred_rating"):
        if key in row and row[key] is not None:
            return str(row[key])
    return None


st.set_page_config(page_title=APP_TITLE, page_icon="🎬", layout="wide")

if not API_BASE_URL:
    _error_banner("API_BASE_URL is not set. Configure it in Render → movie-recommendation-ui → Environment.")


# ---------- Header ----------
st.title(APP_TITLE)
st.caption("A live, interactive demo backed by a FastAPI recommender. Try different strategies, compare results, and explore similar movies.")

with st.sidebar:
    st.subheader("Backend")
    st.code(API_BASE_URL)

    st.divider()
    st.subheader("Quick actions")
    st.link_button("Open API Docs (/docs)", f"{API_BASE_URL}/docs")
    st.caption("Tip: if the first request is slow, Render free tier may be waking up the service.")

    st.divider()
    st.subheader("Demo inputs")

    user_id = st.number_input("User ID", min_value=1, value=1, step=1)
    k = st.slider("Recommendations (k)", min_value=1, max_value=50, value=10, step=1)
    strategy = st.selectbox("Strategy", ["popularity", "content"], index=0)

    st.divider()

    movie_id = st.number_input("Movie ID (Similar)", min_value=1, value=1, step=1)
    k_sim = st.slider("Similar movies (k)", min_value=1, max_value=50, value=10, step=1)

    st.divider()
    st.subheader("Examples")
    ex_cols = st.columns(3)
    if ex_cols[0].button("Try User 1", use_container_width=True):
        st.session_state["user_id_override"] = 1
    if ex_cols[1].button("Try User 5", use_container_width=True):
        st.session_state["user_id_override"] = 5
    if ex_cols[2].button("Try User 10", use_container_width=True):
        st.session_state["user_id_override"] = 10

if "user_id_override" in st.session_state:
    user_id = int(st.session_state.pop("user_id_override"))

# ---------- Health row ----------
health_col, info_col = st.columns([1, 2], gap="large")

with health_col:
    st.subheader("✅ Backend health")
    health, health_err = call_api("/health", timeout=15)
    if health_err:
        st.error(health_err)
    else:
        ok = isinstance(health, dict) and health.get("status") == "ok"
        loaded = isinstance(health, dict) and health.get("models_loaded") is True
        if ok and loaded:
            st.success("Healthy • Models loaded")
        elif ok:
            st.warning("Healthy • Models not loaded yet")
        else:
            st.warning("Unexpected health response")
        st.json(health)

with info_col:
    st.subheader("How to use this demo")
    st.markdown(
        """
- **Recommendations:** choose a **User ID**, strategy, and **k**.
- **Similar movies:** enter a **Movie ID** and **k**.
- Use the **API Docs** button to explore endpoints directly.
        """.strip()
    )

st.divider()

tab_recs, tab_sim, tab_raw = st.tabs(["👤 Recommendations", "🎞️ Similar movies", "🧾 Raw API playground"])

# ---------- Recommendations ----------
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

                # Top picks as cards
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

# ---------- Similar movies ----------
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

# ---------- Raw playground ----------
with tab_raw:
    st.subheader("Raw API Playground")
    st.caption("Useful if you want to copy/paste a URL for the README or debugging.")
    endpoint = st.selectbox(
        "Endpoint",
        ["/health", "/v1/recommendations", "/v1/similar-items"],
        index=1,
    )

    default_params = {
        "/health": {},
        "/v1/recommendations": {"user_id": int(user_id), "k": int(k), "strategy": strategy},
        "/v1/similar-items": {"movie_id": int(movie_id), "k": int(k_sim)},
    }

    params = default_params.get(endpoint, {})
    st.code(f"{API_BASE_URL}{endpoint}", language="text")
    st.write("Params:", params)

    if st.button("Call endpoint", use_container_width=True):
        data, err = call_api(endpoint, params=params or None)
        if err:
            st.error(err)
        else:
            st.json(data)
