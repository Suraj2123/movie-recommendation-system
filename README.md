# Movie Recommendation System

A **production-style** movie recommender with offline training, FastAPI serving, and a polished Streamlit UI. Train on MovieLens, serve popularity and content-based (TF-IDF) recommendations, and run locally or deploy to Render.

---

## Quick start (local)

**One command** — installs deps, trains models (downloads MovieLens), starts API + UI:

```bash
./scripts/run_local.sh
```

Then open **http://localhost:8501**.

**Manual run:**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]" && pip install -r app/requirements.txt

# 1. Train (downloads MovieLens, writes artifacts/local)
python -m mrs.pipelines.train --run-id local

# 2. Start API
uvicorn mrs.serving.api:app --reload --host 127.0.0.1 --port 8000

# 3. In another terminal, start UI
API_BASE_URL=http://localhost:8000 streamlit run app/streamlit_app.py
```

---

## Features

- **Models:** Popularity (Bayesian-smoothed ratings), Content-based (TF-IDF + cosine similarity).
- **API:** FastAPI — health, recommendations, similar-items, movie search, movie by ID.
- **UI:** Streamlit app — trending, “for you” picks, search, similar movies, watchlist.

---

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Status and model load |
| GET | `/v1/model-info` | Model version and metrics |
| GET | `/v1/recommendations` | Top‑K for user (`strategy=popularity\|content`) |
| GET | `/v1/similar-items` | Similar movies by `movie_id` |
| GET | `/v1/movies/search` | Search by title `q` |
| GET | `/v1/movies/{id}` | Movie details |

---

## Deploy (Render)

1. **Connect** the repo to [Render](https://render.com) and add a **Blueprint** from `render.yaml`.
2. **Deploy** both services:
   - **movie-recommendation-api:** trains on build, serves the API.
   - **movie-recommendation-ui:** Streamlit app.
3. Set the UI env var **`API_BASE_URL`** to your deployed API URL (e.g. `https://movie-recommendation-api.onrender.com`). Update in Render dashboard if the default subdomain differs.
4. Optional: **`TMDB_API_KEY`** on the UI service for poster images.

---

## Project layout

```
├── app/
│   ├── requirements.txt    # Streamlit UI deps
│   └── streamlit_app.py    # UI
├── scripts/
│   └── run_local.sh        # Local run script
├── src/mrs/
│   ├── config/
│   ├── data/               # Download + preprocess
│   ├── evaluation/
│   ├── models/             # Popularity, Content TF-IDF
│   ├── pipelines/          # Train
│   └── serving/            # API + movies lookup
├── render.yaml
└── pyproject.toml
```

---

## License

See project license file.
