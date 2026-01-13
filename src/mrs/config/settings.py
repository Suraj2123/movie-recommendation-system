from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MRS_", env_file=".env", extra="ignore")

    # Local storage
    data_dir: str = "data"
    artifacts_dir: str = "artifacts"
    # Serving
    run_id: str = "local"  # default artifact run_id to load in API


settings = Settings()
