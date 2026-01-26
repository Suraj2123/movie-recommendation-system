from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MRS_")

    run_id: str = "local"
    artifacts_dir: str = "artifacts"
    data_dir: str = "data"


settings = Settings()

