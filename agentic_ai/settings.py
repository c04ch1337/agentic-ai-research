# agentic_ai/settings.py
from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    openai_api_key: str | None = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, env="ANTHROPIC_API_KEY")
    huggingface_token: str | None = Field(default=None, env="HUGGINGFACE_TOKEN")
    chroma_db_path: str = Field(default="/app/chroma_db")
    data_path: str = Field(default="/app/data")
    sessions_path: str = Field(default="/app/sessions")

    class Config:
        env_file = ".env"