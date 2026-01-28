from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central application settings.

    Environment variables override defaults automatically.
    """

    # Render provides RUN_ID=prod
    run_id: str = "local"

    # Where trained artifacts are stored
    artifacts_dir: str = "artifacts"

    # Where datasets are cached/downloaded during training
    data_dir: str = "data"

    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()
