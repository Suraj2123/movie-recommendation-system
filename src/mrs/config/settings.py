from __future__ import annotations

import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration.

    Values can be overridden via environment variables using the `MRS_` prefix.
    For example:
      - MRS_RUN_ID=prod
      - MRS_ARTIFACTS_DIR=artifacts
    RUN_ID is also supported (e.g. Render) as a fallback for run_id.
    """

    model_config = SettingsConfigDict(env_prefix="MRS_")

    # Which trained run to load (e.g., local, prod)
    run_id: str = "local"

    @classmethod
    def run_id_from_env(cls) -> str:
        return os.getenv("MRS_RUN_ID") or os.getenv("RUN_ID") or "local"

    # Where artifacts are stored
    artifacts_dir: str = "artifacts"

    # Where raw data is downloaded / cached
    data_dir: str = "data"


settings = Settings()
