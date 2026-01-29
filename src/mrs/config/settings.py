from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration.

    Values can be overridden via environment variables using the `MRS_` prefix.
    For example:
      - MRS_RUN_ID=prod
      - MRS_ARTIFACTS_DIR=artifacts
    """

    model_config = SettingsConfigDict(env_prefix="MRS_")

    # Which trained run to load (e.g., local, prod)
    run_id: str = "local"

    # Where artifacts are stored
    artifacts_dir: str = "artifacts"

    # Where raw data is downloaded / cached
    data_dir: str = "data"


settings = Settings()
