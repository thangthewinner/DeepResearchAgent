"""
Application settings with startup validation.

Uses pydantic-settings to validate all required env vars at startup.
If any required key is missing, the app will fail immediately with a clear error.
"""

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- API Keys (required, validated at startup) ---
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    tavily_api_key: str = Field(..., alias="TAVILY_API_KEY")

    # --- Research Tuning ---
    max_concurrent_researchers: int = Field(
        default=1, alias="MAX_CONCURRENT_RESEARCHERS"
    )
    max_researcher_iterations: int = Field(default=5, alias="MAX_RESEARCHER_ITERATIONS")
    max_context_length: int = Field(default=250000, alias="MAX_CONTEXT_LENGTH")

    # --- Model Names ---
    base_model: str = Field(default="openai:gpt-4o", alias="BASE_MODEL")
    creative_model: str = Field(default="openai:gpt-4o", alias="CREATIVE_MODEL")
    summarization_model: str = Field(
        default="openai:gpt-4o-mini", alias="SUMMARIZATION_MODEL"
    )
    compress_model: str = Field(default="openai:gpt-4o", alias="COMPRESS_MODEL")
    writer_model: str = Field(default="openai:gpt-5", alias="WRITER_MODEL")
    critic_model: str = Field(default="openai:gpt-4o", alias="CRITIC_MODEL")
    judge_model: str = Field(default="openai:gpt-4o", alias="JUDGE_MODEL")
    compressor_model: str = Field(
        default="openai:gpt-4o-mini", alias="COMPRESSOR_MODEL"
    )

    # --- Token Limits ---
    compress_max_tokens: int = Field(default=16000, alias="COMPRESS_MAX_TOKENS")
    writer_max_tokens: int = Field(default=16000, alias="WRITER_MAX_TOKENS")

    # --- Timeouts ---
    # Total time budget for one full research request (seconds)
    request_timeout_seconds: int = Field(default=600, alias="REQUEST_TIMEOUT_SECONDS")
    # Time budget for parallel researcher sub-agents (seconds)
    researcher_timeout_seconds: int = Field(
        default=120, alias="RESEARCHER_TIMEOUT_SECONDS"
    )

    # --- LangSmith Observability ---
    langchain_tracing_v2: Optional[str] = Field(
        default=None, alias="LANGCHAIN_TRACING_V2"
    )
    langchain_api_key: Optional[str] = Field(default=None, alias="LANGCHAIN_API_KEY")
    langchain_project: str = Field(default="deepresearch", alias="LANGCHAIN_PROJECT")

    # --- Logging ---
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


# Singleton — loaded once at import time.
# Raises ValidationError immediately if required keys are missing.
settings = Settings()  # type: ignore[call-arg]

# Re-export flat constants for backward compatibility with existing imports.
OPENAI_API_KEY = settings.openai_api_key
TAVILY_API_KEY = settings.tavily_api_key

MAX_CONCURRENT_RESEARCHERS = settings.max_concurrent_researchers
MAX_RESEARCHER_ITERATIONS = settings.max_researcher_iterations
MAX_CONTEXT_LENGTH = settings.max_context_length

BASE_MODEL = settings.base_model
CREATIVE_MODEL = settings.creative_model
SUMMARIZATION_MODEL = settings.summarization_model
COMPRESS_MODEL = settings.compress_model
WRITER_MODEL = settings.writer_model
CRITIC_MODEL = settings.critic_model
JUDGE_MODEL = settings.judge_model
COMPRESSOR_MODEL = settings.compressor_model

COMPRESS_MAX_TOKENS = settings.compress_max_tokens
WRITER_MAX_TOKENS = settings.writer_max_tokens

REQUEST_TIMEOUT_SECONDS = settings.request_timeout_seconds
RESEARCHER_TIMEOUT_SECONDS = settings.researcher_timeout_seconds
