from __future__ import annotations

from typing import TypedDict

from app.agents.creative_strategy_agent.schema import CreativeDirection
from app.agents.prompt_gen_agent.schema import PromptGenerationOutput


class PromptGenState(TypedDict, total=False):
    """
    State threaded through the prompt generation subgraph. `total=False`
    because most fields only exist after their producing node has run.

    No `scraped`-equivalent field here (same reasoning as
    CreativeStrategyState) - this agent's input is already a validated
    CreativeDirection object from Agent 2, not something that needs fetching.
    """

    creative: CreativeDirection

    prompts: PromptGenerationOutput | None
    model_used: str | None  # which model (primary/fallback) actually produced the output - for observability

    retries: int
    error: str | None