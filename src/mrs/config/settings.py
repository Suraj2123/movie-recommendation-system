from __future__ import annotations

from pathlib import Path

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

    model_config = SettingsConfigDict(
        env_prefix="",          # allows RUN_ID instead of MRS_RUN_ID
        case_sensitive=False,
        extra="ignore",
    )


# Singleton settings instance
settings = Settings()

# Optional sanity output if run directly
if __name__ == "__main__":
    print("run_id:", settings.run_id)
    print("artifacts_dir:", Path(settings.artifacts_dir).resolve())
