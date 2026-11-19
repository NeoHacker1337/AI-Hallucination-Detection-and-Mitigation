"""Application-wide settings loaded from environment variables (see .env.example)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global settings. Every value can be overridden with an environment variable of the same name."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Storage -----------------------------------------------------------------------------
    data_dir: Path = Field(default=Path("data"), alias="HG_DATA_DIR")
    database_url: str = Field(default="sqlite:///data/halluguard.db", alias="HG_DATABASE_URL")

    # --- LLM providers -----------------------------------------------------------------------
    default_model: str = Field(default="ollama:llama3", alias="HG_DEFAULT_MODEL")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    openai_base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    llm_timeout_s: float = Field(default=180.0, alias="HG_LLM_TIMEOUT_S")
    llm_max_retries: int = Field(default=3, alias="HG_LLM_MAX_RETRIES")

    # --- Models for retrieval and NLI --------------------------------------------------------
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", alias="HG_EMBEDDING_MODEL")
    nli_model: str = Field(default="microsoft/deberta-large-mnli", alias="HG_NLI_MODEL")
    device: str = Field(default="auto", alias="HG_DEVICE")  # "auto", "cpu", "cuda", "mps"
    nli_batch_size: int = Field(default=16, alias="HG_NLI_BATCH_SIZE")

    # --- Datasets (Hugging Face ids, overridable if mirrors are used) --------------------------
    truthfulqa_hf_id: str = Field(default="truthfulqa/truthful_qa", alias="HG_TRUTHFULQA_ID")
    halueval_hf_id: str = Field(default="pminervini/HaluEval", alias="HG_HALUEVAL_ID")

    # --- Retrieval chunking ------------------------------------------------------------------
    chunk_words: int = Field(default=180, alias="HG_CHUNK_WORDS")
    chunk_overlap_words: int = Field(default=40, alias="HG_CHUNK_OVERLAP_WORDS")

    # --- External retrieval (optional; off in every preset) -----------------------------------
    # Wikimedia's robot policy requires a User-Agent with contact details (a URL or e-mail), e.g.
    # "MyThesisBot/1.0 (https://example.org/contact)". Requests without one are refused (HTTP 403).
    web_user_agent: str = Field(default="", alias="HG_WEB_USER_AGENT")
    web_timeout_s: float = Field(default=10.0, alias="HG_WEB_TIMEOUT_S")

    # --- Experiments -------------------------------------------------------------------------
    max_parallel_experiments: int = Field(default=1, alias="HG_MAX_PARALLEL_EXPERIMENTS")

    @property
    def experiments_dir(self) -> Path:
        """Directory holding per-experiment artefacts (CSV, PNG, HTML, config)."""
        return self.data_dir / "experiments"

    @property
    def index_dir(self) -> Path:
        """Directory holding FAISS indexes, one per collection."""
        return self.data_dir / "index"

    def ensure_dirs(self) -> None:
        """Create the data directories if they do not exist."""
        for path in (self.data_dir, self.experiments_dir, self.index_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings singleton."""
    settings = Settings()
    settings.ensure_dirs()
    return settings
