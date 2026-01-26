# 🎬 Movie Recommendation System (MRS)

Production-style movie recommendation system with offline training, evaluation, and a live FastAPI service.

**Live Demo (Swagger UI):**  
👉 https://movie-recommendation-system-oivj.onrender.com/docs  

**Health Check:**  
👉 https://movie-recommendation-system-oivj.onrender.com/health  

---

## Overview

This project demonstrates an end-to-end recommender system workflow:

- Offline data processing and model training
- Baseline and content-based recommendation models
- Offline evaluation and artifact versioning
- Production-style FastAPI service
- Publicly deployed, interactive API demo

The system is designed to resemble a real-world ML service rather than a notebook-only project.

---

## Features

### Models
- **Popularity-based recommender**
  - Ranks movies by Bayesian-smoothed average rating
- **Content-based recommender**
  - TF-IDF over movie titles and genres
  - Cosine similarity for item-to-item recommendations

### Pipelines
- Dataset download (MovieLens latest-small)
- Preprocessing and chronological train/test split
- Offline evaluation
- Versioned artifacts (`artifacts/<run_id>/`)

### API (FastAPI)
- Health and model metadata
- User recommendations
- Similar-item recommendations
- Swagger UI for interactive exploration

### Engineering Practices
- Production-style repo layout
- Clear separation of data, models, pipelines, and serving
- GitHub Actions CI (lint + tests)
- Public cloud deployment (Render)

---

## Live Demo (How to Try)

1. Open the Swagger UI:  
   👉 https://movie-recommendation-system-oivj.onrender.com/docs

2. Click **GET /v1/recommendations → Try it out**
   - `user_id`: `1`
   - `k`: `10`
   - `strategy`: `popularity`
   - Click **Execute**

3. Try **GET /v1/similar-items**
   - `movie_id`: `1`
   - `k`: `10`

No setup required — this runs against the deployed service.

---

## API Endpoints

| Method | Endpoint | Description |
|------|---------|-------------|
| GET | `/health` | Service and model status |
| GET | `/v1/model-info` | Model version and offline metrics |
| GET | `/v1/recommendations` | Top-K recommendations for a user |
| GET | `/v1/similar-items` | Similar movies (content-based) |

---

## Offline Training (Local)

If you want to run everything locally:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
python -m mrs.pipelines.train --run-id local

