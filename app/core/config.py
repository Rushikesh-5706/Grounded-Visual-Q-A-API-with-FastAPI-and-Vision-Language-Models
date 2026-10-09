from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
_OPENAI_BASE_URL = "https://api.openai.com/v1"
_GROQ_DEFAULT_MODEL = "qwen/qwen3.8-27b"
_OPENAI_DEFAULT_MODEL = "gpt-4o-mini"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    vlm_provider: Literal["groq", "openai"] = "groq"
    groq_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None

    request_timeout_seconds: float = 60.0
    max_retries: int = Field(default=3, ge=1)
    max_upload_bytes: int = 10 * 1024 * 1024  # 10 MiB
    max_image_dimension: int = 1024
    max_answer_tokens: int = 256
    max_question_chars: int = 2000

    audit_log_path: Path = _PROJECT_ROOT / "audit.log"
    log_level: str = "INFO"

    @property
    def active_provider(self) -> str:
        groq_key = self.groq_api_key and self.groq_api_key.get_secret_value()
        openai_key = self.openai_api_key and self.openai_api_key.get_secret_value()

        if self.vlm_provider == "groq":
            if groq_key:
                return "groq"
            if openai_key:
                return "openai"
            return "groq"  # no key configured; caller must handle
        else:
            if openai_key:
                return "openai"
            if groq_key:
                return "groq"
            return "openai"

    @property
    def active_api_key(self) -> str | None:
        provider = self.active_provider
        if provider == "groq":
            return self.groq_api_key.get_secret_value() if self.groq_api_key else None
        return self.openai_api_key.get_secret_value() if self.openai_api_key else None

    @property
    def active_base_url(self) -> str:
        return _GROQ_BASE_URL if self.active_provider == "groq" else _OPENAI_BASE_URL

    @property
    def active_model(self) -> str:
        return _GROQ_DEFAULT_MODEL if self.active_provider == "groq" else _OPENAI_DEFAULT_MODEL


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
