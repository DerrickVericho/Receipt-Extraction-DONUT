from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABRICKS_HOST: str
    DATABRICKS_TOKEN: str

    # 3 models
    DATABRICKS_RUN_ID_1: str
    DATABRICKS_MODEL_NAME_1: str
    DATABRICKS_RUN_ID_2: str
    DATABRICKS_MODEL_NAME_2: str
    DATABRICKS_RUN_ID_3: str
    DATABRICKS_MODEL_NAME_3: str

    class Config:
        env_file = ".env"
        extra = "ignore"  # .env dishare dengan fe (API_BASE_URL)


settings = Settings()
