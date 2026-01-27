from __future__ import annotations

import os
import time

import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "").strip().rstrip("/")
if not API_BASE_URL:
    st.error("API_BASE_URL is not set. Configure it in Render → Environment.")
    st.stop()


def call_api(path: str, params: dict | None = None, retries: int = 2):
    url = f"{API_BASE_URL}{path}"

    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=30)
        except Exception as e:
            if attempt == retries:
                st.error(f"Network error calling {url}: {e}")
                return None
            time.sleep(1)
            continue

        st.caption(f"GET {r.url}")
        st.write("Status:", r.status_code)
        st.write("Content-Type:", r.headers.get("content-type", "(missing)"))

        try:
            return r.json()
        except Exception:
            st.code((r.text or "")[:3000])
            return None

    return None


st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="centered",
)

st.title("🎬 Movie Recommendation System")
st.caption("Interactive playground powered by your FastAPI backend.")

with st.sidebar:
    st.header("Backend")
    st.code(API_BASE_URL)

tab1, tab2 = st.tabs(["👤 Recommend for user", "🎞️ Similar movies"])

with tab1:
    st.subheader("Get recommendations")
    user_id = st.number_input("user_id", min_value=1, value=1, step=1)
    k = st.slider("k", min_value=1, max_value=50, value=10)
    strategy = st.selectbox("strategy", ["popularity", "content"])

    if st.button("Get recommendations", type="primary"):
        data = call_api(
            "/v1/recommendations",
            params={"user_id": user_id, "k": k, "strategy": strategy},
        )
        if data is not None:
            st.json(data)

with tab2:
    st.subheader("Find similar movies")
    movie_id = st.number_input("movie_id", min_value=1, value=1, step=1)
    k2 = st.slider("k (similar)", min_value=1, max_value=50, value=10)

    if st.button("Find similar", type="primary"):
        data = call_api(
            "/v1/similar-items",
            params={"movie_id": movie_id, "k": k2},
        )
        if data is not None:
            st.json(data)

st.divider()
st.subheader("Health")

health = call_api("/health")
if health is not None:
    st.json(health)
