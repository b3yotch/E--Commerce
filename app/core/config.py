from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central config. Values are pulled from environment variables / .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_research_model: str = "qwen3.5:4b"
    # Runtime context window - independent of a model's advertised max
    # context. Must comfortably fit system+schema prompt + visible_text +
    # review_text (up to ~28k chars / ~7-8k tokens here) or output gets
    # silently truncated. ~4 chars/token is a rough estimate for English text.
    ollama_num_ctx: int = 16384
    ollama_num_predict: int = 4096

    # Scraper
    scrape_timeout_ms: int = 30000
    review_wait_timeout_ms: int = 4000
    # Below this length, review_text is treated as "not really there" (e.g.
    # just a "(3)" count badge matched by the selector) - triggers escalation
    # to scrolling if a review count hint suggests there's more to find.
    min_review_text_chars: int = 40
    scrape_user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
    max_visible_text_chars: int = 20000
    max_review_text_chars: int = 8000

    # Agent
    max_extraction_retries: int = 2


settings = Settings()