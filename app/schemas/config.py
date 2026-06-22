from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # GitHub Config
    github_token: str = ""

    # LLM Config
    llm_provider: str = "openai"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    ollama_model: str = "llama3"
    ollama_base_url: str = "http://localhost:11434"

    # Qdrant Config
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # Models
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
