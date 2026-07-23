from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    DATABRICKS_HOST: str
    DATABRICKS_TOKEN: str
    
    # 3 models
    DATABRICKS_RUN_ID_1: Optional[str] = None
    DATABRICKS_RUN_ID_2: Optional[str] = None
    DATABRICKS_RUN_ID_3: Optional[str] = None

    class Config:
        env_file = ".env"

settings = Settings()
