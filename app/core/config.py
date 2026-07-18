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

    # Groq (Prompt Generation Agent - translating CreativeDirection.visual_themes
    # into ComfyUI-ready image/video prompts). Deliberately its OWN settings
    # block, separate from Creative Strategy's groq_* fields above, even
    # though the placeholder values currently match: writing prompts a
    # diffusion/video model responds well to is a different skill from ad
    # copywriting, so model choice here needs its own 2-3 candidate
    # comparison before these placeholders should be trusted - don't assume
    # they should track Creative Strategy's models just because they started
    # out equal.
    prompt_gen_primary_model: str = "llama-3.3-70b-versatile"  # placeholder - re-run comparison before trusting
    prompt_gen_fallback_model: str = "openai/gpt-oss-120b"
    prompt_gen_max_tokens: int = 2048
    # Lower than Creative Strategy's 0.7 - prompt-writing benefits from more
    # precision/consistency than open-ended ad copy, but still needs some
    # variation across 2-3 distinct visual themes per product.
    prompt_gen_temperature: float = 0.6

    # Scraper
    scrape_timeout_ms: int = 30000
    nav_attempt_timeout_ms: int = 10000
    scrape_user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
    max_visible_text_chars: int = 20000
    max_review_text_chars: int = 8000
    min_review_text_chars: int = 200  # below this, review_text counts as "not really found" - triggers scroll escalation
    review_wait_timeout_ms: int = 4000  # how long to wait for a review selector to appear after scrolling

    # Agent
    max_extraction_retries: int = 2
    max_creative_retries: int = 2
    max_prompt_gen_retries: int = 2


settings = Settings()