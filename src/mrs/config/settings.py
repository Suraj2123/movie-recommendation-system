from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    run_id: str = os.getenv("RUN_ID", "prod")
    artifacts_dir: Path = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
    tmdb_api_key: str | None = os.getenv("TMDB_API_KEY") or None


settings = Settings()
