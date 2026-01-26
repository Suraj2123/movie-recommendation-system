# Movie Recommendation System (MRS)
**Live Demo (Swagger UI):**  
👉 https://movie-recommendation-system-oivj.onrender.com/docs  

**Health Check:**  
👉 https://movie-recommendation-system-oivj.onrender.com/health  

---
Production-style movie recommender with:
- Offline training pipeline (baseline + content-based TF-IDF)
- Offline evaluation + generated report
- FastAPI service for recommendations
- Tests + GitHub Actions CI

## What’s implemented
- **Popularity baseline** recommender
- **Content-based** recommender (TF-IDF over genres + title, cosine similarity)
- FastAPI endpoints:
  - `GET /health`
  - `GET /v1/model-info`
  - `GET /v1/recommendations?user_id=...&k=...&strategy=popularity|content`
  - `GET /v1/similar-items?movie_id=...&k=...` (content model)

## Quickstart (Codespaces recommended)
1) Open **Code → Codespaces → Create codespace on main**
2) In the terminal:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
python -m mrs.pipelines.train --run-id local
python -m mrs.serving.api
