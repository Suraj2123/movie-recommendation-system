import os
import time

import requests
import streamlit as st

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "https://movie-recommendation-system-oivj.onrender.com",
).rstrip("/")


def call_api(path: str, params: dict | None = None, timeout: int = 30, retries: int = 2):
    url = f"{API_BASE_URL}{path}"

    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout, allow_redirects=True)
        except Exception as e:
            last_exc = e
            continue

        st.caption(f"GET {r.url}")
        st.write("Status:", r.status_code)
        st.write("Content-Type:", r.headers.get("content-type", "(missing)"))

        # Retry on typical transient Render/proxy errors
        if r.status_code in (502, 503, 504) and attempt < retries:
            time.sleep(1)
            continue

        # Try JSON first
        try:
            return r.json()
        except Exception:
            # Not JSON: show raw body so we can see what's happening
            body = (r.text or "").strip()
            if not body:
                st.warning("Empty response body (not JSON).")
            else:
                st.code(body[:3000])
            return None

    st.error(f"Network error calling {url}: {last_exc}")
    return None


st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬", layout="centered")
st.title("🎬 Movie Recommendation System")
st.caption("Interactive playground powered by your FastAPI backend.")

with st.sidebar:
    st.header("Backend")
    st.write("API Base URL:")
    st.code(API_BASE_URL)

tab1, tab2 = st.tabs(["👤 Recommend for user", "🎞️ Similar movies"])

with tab1:
    st.subheader("Get recommendations")
    user_id = st.number_input("user_id", min_value=1, value=1, step=1)
    k = st.slider("k", min_value=1, max_value=50, value=10, step=1)
    strategy = st.selectbox("strategy", ["popularity", "content"], index=0)

    if st.button("Get recommendations", type="primary"):
        data = call_api(
            "/v1/recommendations",
            params={"user_id": int(user_id), "k": int(k), "strategy": strategy},
            timeout=30,
            retries=2,
        )
        if data is not None:
            st.success("Done!")
            st.json(data)

with tab2:
    st.subheader("Find similar movies")
    movie_id = st.number_input("movie_id", min_value=1, value=1, step=1)
    k2 = st.slider("k (similar)", min_value=1, max_value=50, value=10, step=1)

    if st.button("Find similar", type="primary"):
        data = call_api(
            "/v1/similar-items",
            params={"movie_id": int(movie_id), "k": int(k2)},
            timeout=30,
            retries=2,
        )
        if data is not None:
            st.success("Done!")
            st.json(data)

st.divider()
st.subheader("Health")
health = call_api("/health", timeout=15, retries=2)
if health is not None:
    st.json(health)
