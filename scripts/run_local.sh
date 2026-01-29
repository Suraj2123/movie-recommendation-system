#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

echo "==> Movie Recommendation System – local run"
echo ""

# 1. Create venv if missing
if [[ ! -d .venv ]]; then
  echo "==> Creating venv..."
  python3 -m venv .venv
fi
source .venv/bin/activate

# 2. Install deps
echo "==> Installing dependencies..."
pip install -q -U pip
pip install -q -e ".[dev]"
pip install -q -r app/requirements.txt

# 3. Train if artifacts missing
ARTIFACTS="$ROOT/artifacts/local/models"
if [[ ! -f "$ARTIFACTS/popularity.joblib" ]]; then
  echo "==> Training models (downloads MovieLens, may take a few minutes)..."
  python -m mrs.pipelines.train --run-id local
else
  echo "==> Using existing artifacts (artifacts/local)"
fi

# 4. Start API in background
echo "==> Starting API on http://127.0.0.1:8000"
uvicorn mrs.serving.api:app --host 127.0.0.1 --port 8000 &
API_PID=$!
trap "kill $API_PID 2>/dev/null || true" EXIT

# Give API time to load models
sleep 5
echo "==> Starting Streamlit UI on http://localhost:8501"
echo ""
export API_BASE_URL=http://127.0.0.1:8000
streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
