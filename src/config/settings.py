from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True
    
    # Database
    database_url: str = "postgresql://user:password@localhost:5432/outfit_ai"
    
    # File Storage
    upload_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    max_file_size: int = 10485760
    
    # ML Models
    model_dir: str = "data/models"
    device: str = "cpu"
    
    # Security
    secret_key: str = "change-this-to-a-random-secret-key"

    # Add at the end of Settings class, before class Config
    allowed_extensions: list = [".jpg", ".jpeg", ".png", ".webp"]
    
    class Config:
        env_file = ".env"

settings = Settings()

print("SETTINGS LOADED OK")
