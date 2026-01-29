# Movie Recommendation System

Production-grade movie recommender with **offline training**, a **FastAPI service**, and a **polished Streamlit UI**. The system trains on MovieLens, serves popularity and content-based recommendations, and surfaces rich movie metadata using TMDB and IMDb IDs.

---

## Highlights

- **End-to-end ML system**: data download, preprocessing, offline evaluation, artifact versioning, and online serving.
- **Two strategies**: popularity baseline (Bayesian-smoothed ratings) + content-based TF-IDF.
- **Real movie details**: title, genres, year, average rating, rating count, IMDb link, TMDB overview.
- **Deployable**: Render Blueprint (`render.yaml`) with API + UI services.

---

## System design

**Offline pipeline**
- Download MovieLens (latest-small)
- Preprocess ratings + movies
- Train models (popularity + TF-IDF)
- Compute movie stats (avg rating + count)
- Export versioned artifacts and metadata

**Online serving**
- FastAPI loads the latest artifacts on startup
- Endpoints for recommendations, similar items, search, and details
- UI calls the API and renders results with graceful fallbacks

---

## Quick start (local)

**One command** (installs deps, trains models, starts API + UI):

```bash
./scripts/run_local.sh
```

Then open `http://localhost:8501`.

**Manual run:**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]" && pip install -r app/requirements.txt

# 1) Train (downloads MovieLens, writes artifacts/local)
python -m mrs.pipelines.train --run-id local

# 2) Start API
uvicorn mrs.serving.api:app --reload --host 127.0.0.1 --port 8000

# 3) In another terminal, start UI
API_BASE_URL=http://localhost:8000 streamlit run app/streamlit_app.py
```

---

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Status and model load |
| GET | `/v1/model-info` | Model version and metrics |
| GET | `/v1/recommendations` | Top-K recs (`strategy=popularity|content`) |
| GET | `/v1/similar-items` | Similar movies by `movie_id` |
| GET | `/v1/movies/search` | Search by title `q` |
| GET | `/v1/movies/{id}` | Movie details (incl. stats + links) |

---

## Deployment (Render)

1. Connect repo to [Render](https://render.com) and create a **Blueprint** from `render.yaml`.
2. Deploy both services:
   - **movie-recommendation-api**: trains on build, serves API.
   - **movie-recommendation-ui**: Streamlit UI.
3. Set `API_BASE_URL` for the UI service to your API URL (e.g. `https://movie-recommendation-api.onrender.com`).
4. Optional: add `TMDB_API_KEY` for posters and richer details.

---

## Tech stack

- **Python 3.11**
- **FastAPI** + **Uvicorn**
- **Pandas / NumPy / SciKit-Learn**
- **Streamlit**
- **Render** (deploy)

---

## Project layout

```
├── app/                     # Streamlit UI
├── scripts/                 # Local run
├── src/mrs/
│   ├── data/                # Download + preprocess
│   ├── models/              # Popularity, TF-IDF
│   ├── evaluation/          # Offline metrics
│   ├── pipelines/           # Training pipeline
│   └── serving/             # FastAPI
├── render.yaml
└── pyproject.toml
```

---

## Notes for reviewers

- **Scalable design**: models are serialized artifacts, making deploys reproducible.
- **Clear separation**: data, modeling, evaluation, and serving modules are isolated.
- **Production parity**: API + UI deployed separately, API can scale independently.

---

## License

See project license file.
