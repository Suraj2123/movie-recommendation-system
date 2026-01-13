import argparse
import json
from pathlib import Path
from typing import Any

from joblib import dump

from mrs.config.logging import configure_logging
from mrs.config.settings import settings
from mrs.data.download import download_movielens_latest_small
from mrs.data.preprocess import load_raw_movielens, preprocess
from mrs.evaluation.offline_eval import chronological_split, evaluate
from mrs.evaluation.report import render_report
from mrs.models.content_tfidf import ContentTfidfModel
from mrs.models.popularity import PopularityRecommender


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def train(run_id: str) -> dict[str, Any]:
    configure_logging()

    dataset_dir = download_movielens_latest_small(settings.data_dir)
    ratings_raw, movies_raw = load_raw_movielens(dataset_dir)
    data = preprocess(ratings_raw, movies_raw)

    split = chronological_split(data.ratings, test_ratio=0.2)

    popularity = PopularityRecommender.train(split.train)
    content = ContentTfidfModel.train(data.movies)

    pop_eval = evaluate(popularity, split.train, split.test, k=10)
    content_eval = evaluate(content, split.train, split.test, k=10)

    out_dir = Path(settings.artifacts_dir) / run_id
    models_dir = out_dir / "models"
    _ensure_dir(models_dir)

    dump(popularity, models_dir / "popularity.joblib")
    content.save(str(models_dir / "content_tfidf.joblib"))

    metrics = {
        "run_id": run_id,
        "popularity": pop_eval.__dict__,
        "content_tfidf": content_eval.__dict__,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    report = render_report(run_id, pop_eval, content_eval)
    (out_dir / "report.md").write_text(report)

    manifest = {
        "run_id": run_id,
        "dataset": "movielens-latest-small",
        "rows": {
            "ratings_train": int(len(split.train)),
            "ratings_test": int(len(split.test)),
            "movies": int(len(data.movies)),
        },
        "models": [
            "popularity.joblib",
            "content_tfidf.joblib",
        ],
    }

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=settings.run_id)
    args = parser.parse_args()

    train(args.run_id)


if __name__ == "__main__":
    main()
