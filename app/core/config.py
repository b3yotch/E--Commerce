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
    ollama_num_predict: int = 2048

    # Groq (Creative Strategy Agent - needs genuine creative/marketing
    # judgment, which the local 4B extraction model isn't suited for).
    # Model choice from a 3-way comparison against sample ProductResearch
    # inputs: gpt-oss-120b gave the strongest, best-grounded creative copy;
    # llama-3.3-70b-versatile was close behind and serves as fallback if the
    # primary errors or rate-limits. Free tier is 1K requests/day per model,
    # which is the binding constraint for bulk batches - not TPM.
    groq_api_key: str = ""
    groq_primary_model: str = "openai/gpt-oss-120b"
    groq_fallback_model: str = "llama-3.3-70b-versatile"
    groq_max_tokens: int = 2048
    groq_temperature: float = 0.7  # creative task, not extraction - allow more variance than Ollama's 0.2

    # Scraper
    # Scraper
    scrape_timeout_ms: int = 30000
    scrape_user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
    max_visible_text_chars: int = 20000
    max_review_text_chars: int = 8000
    min_review_text_chars: int = 200  # below this, review_text counts as "not really found" - triggers scroll escalation
    review_wait_timeout_ms: int = 5000  # how long to wait for a review selector to appear after scrolling

    # Agent
    max_extraction_retries: int = 2
    max_creative_retries: int = 2


settings = Settings()