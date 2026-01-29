# 🎬 Movie Recommendation System

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Production-grade movie recommender with **offline training**, a **FastAPI service**, and a **polished Streamlit UI**. Trained on MovieLens with popularity and content-based strategies, enriched with TMDB/IMDb metadata.

---

## 🚀 Live Demo

**🌐 Web App:** https://movie-recommendation-ui-production-8767.up.railway.app _(first load may take 30s on free tier)_  
**📡 API Docs:** https://movie-recommendation-api-production-8d3c.up.railway.app/docs  
**🔍 API Health:** https://movie-recommendation-api-production-8d3c.up.railway.app/health

---

## ✨ Highlights

- **End-to-end ML pipeline** — data ingestion, preprocessing, training, evaluation, artifact versioning, and serving
- **Two recommendation strategies** — popularity baseline (Bayesian-smoothed) + content-based TF-IDF with cosine similarity
- **Rich metadata** — average ratings, vote counts, release years, IMDb links, TMDB posters and overviews
- **Production-ready** — separate API/UI services, CI/CD with GitHub Actions, deployed on Railway

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

## 🚢 Deployment

**Currently deployed on Railway** ([railway.app](https://railway.app))

### Railway (recommended)
1. Create Railway account and connect GitHub repo
2. Deploy API service:
   - Build command: `pip install -U pip && pip install -e ".[dev]" && python -m mrs.pipelines.train --run-id prod`
   - Start command: `uvicorn mrs.serving.api:app --host 0.0.0.0 --port $PORT`
   - Set env: `RUN_ID=prod`, `MRS_RUN_ID=prod`
3. Deploy UI service:
   - Build command: `pip install -U pip && pip install -r app/requirements.txt`
   - Start command: `streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true`
   - Set env: `API_BASE_URL=<your-api-url>`, `TMDB_API_KEY=<optional>`

### Render
Alternatively, use the included Blueprint:
```bash
# render.yaml included for one-click deploy
```

---

## 🛠️ Tech stack

- **Python 3.11** — Core language
- **FastAPI + Uvicorn** — High-performance API framework
- **Pandas / NumPy / SciKit-Learn** — Data processing and ML
- **Streamlit** — Interactive web UI
- **Railway** — Cloud deployment
- **GitHub Actions** — CI/CD pipeline

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

## 📊 Architecture highlights

- **Reproducible builds** — Models are serialized artifacts with versioned training runs
- **Clean architecture** — Data, models, evaluation, and serving are fully isolated
- **Microservices** — API and UI deploy separately; API can scale independently
- **Graceful degradation** — UI retries on cold starts, falls back to placeholders when metadata unavailable
- **CI/CD** — Automated linting and testing via GitHub Actions

---

## 📸 Screenshots

_Coming soon_

---

## 📝 License

See [LICENSE](LICENSE) file.
