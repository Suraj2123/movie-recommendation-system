import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "https://movie-recommendation-system-oivj.onrender.com").rstrip("/")

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
        try:
            resp = requests.get(
                f"{API_BASE_URL}/v1/recommendations",
                params={"user_id": int(user_id), "k": int(k), "strategy": strategy},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            st.success("Done!")
            st.json(data)
        except Exception as e:
            st.error(f"Request failed: {e}")

with tab2:
    st.subheader("Find similar movies")
    movie_id = st.number_input("movie_id", min_value=1, value=1, step=1)
    k2 = st.slider("k (similar)", min_value=1, max_value=50, value=10, step=1)

    if st.button("Find similar", type="primary"):
        try:
            resp = requests.get(
                f"{API_BASE_URL}/v1/similar-items",
                params={"movie_id": int(movie_id), "k": int(k2)},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            st.success("Done!")
            st.json(data)
        except Exception as e:
            st.error(f"Request failed: {e}")

st.divider()
st.write("Health:")
try:
    health = requests.get(f"{API_BASE_URL}/health", timeout=10).json()
    st.json(health)
except Exception as e:
    st.warning(f"Could not read /health: {e}")
