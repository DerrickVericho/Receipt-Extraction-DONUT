from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABRICKS_HOST: str
    DATABRICKS_TOKEN: str

    DATABRICKS_RUN_IDS: list[str] = []
    DATABRICKS_DEVICE: str = "auto"

    class Config:
        env_file = ".env"
        extra = "ignore"  # .env dishare dengan fe (API_BASE_URL)


settings = Settings()
