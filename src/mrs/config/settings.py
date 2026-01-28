from __future__ import annotations

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central application settings.

    Values are read from environment variables if present,
    otherwise sensible defaults are used.
    """

    # IMPORTANT: Render will inject RUN_ID=prod
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

if __name__ == "__main__":
    print("RUN_ID:", settings.run_id)
    print("ARTIFACTS_DIR:", settings.artifacts_dir)
    print("CWD:", Path.cwd())

