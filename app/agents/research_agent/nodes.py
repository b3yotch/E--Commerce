from __future__ import annotations

from app.core.config import settings
from app.core.llm import LLMExtractionError, structured_chat
from app.agents.research_agent.prompts import SYSTEM_PROMPT, build_user_prompt
from app.agents.research_agent.schema import ProductResearch
from app.agents.research_agent.scraper import scrape_product_page
from app.agents.research_agent.state import ResearchState


async def scrape_node(state: ResearchState) -> ResearchState:
    """Fetch and pre-parse the product page. Deterministic, no LLM involved."""
    try:
        scraped = await scrape_product_page(state["url"])
    except Exception as exc:  # noqa: BLE001
        return {**state, "error": f"Scrape failed: {exc}"}

    return {**state, "scraped": scraped, "error": None}


async def extract_node(state: ResearchState) -> ResearchState:
    """
    Run the LLM extraction step against the scraped content.

    On a retry (retries > 0), the previous attempt's error is passed back
    into the prompt so the model can actually correct course - a blind
    retry with an identical prompt tends to reproduce the same mistake.
    """
    scraped = state["scraped"]
    previous_error = state.get("error") if state.get("retries", 0) > 0 else None

    user_prompt = build_user_prompt(
        url=scraped.url,
        structured_data=scraped.json_ld_product,
        visible_text=scraped.visible_text,
        review_text=scraped.review_text,
        previous_error=previous_error,
    )

    try:
        research = await structured_chat(
            model=settings.ollama_research_model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            schema=ProductResearch,
        )
        research.source_url = scraped.url
        return {**state, "research": research, "error": None}
    except LLMExtractionError as exc:
        return {**state, "research": None, "error": str(exc)}


def route_after_scrape(state: ResearchState) -> str:
    """Scrape failures are unrelated to extraction quality - don't waste
    extraction retries on them. Go straight to END if scraping failed."""
    return "failed" if state.get("error") else "ok"


def _is_valid(state: ResearchState) -> bool:
    research = state.get("research")
    return bool(research and research.title and research.title.strip())


def validate_node(state: ResearchState) -> ResearchState:
    """
    Sanity-check the extraction. The bar here is deliberately low - we're
    catching outright failures (empty title, no output at all), not grading
    quality. Deeper quality/consistency checks belong to the Review/Critic
    agent later in the pipeline, not this one. This node only inspects state;
    retry bookkeeping happens in bump_retry_node so the counting logic lives
    in exactly one place.
    """
    if _is_valid(state):
        return {**state, "error": None}

    retries = state.get("retries", 0)
    if retries >= settings.max_extraction_retries:
        return {
            **state,
            "error": (state.get("error") or "Extraction produced no usable title")
            + f" (gave up after {retries} retries)",
        }
    return state


def bump_retry_node(state: ResearchState) -> ResearchState:
    """Increments the retry counter, then loops back to extract_node."""
    return {**state, "retries": state.get("retries", 0) + 1}


def route_after_validation(state: ResearchState) -> str:
    """Conditional edge: retry extraction, or stop."""
    if _is_valid(state):
        return "done"
    if state.get("retries", 0) >= settings.max_extraction_retries:
        return "done"  # give up, error is already set on state by validate_node
    return "retry"
