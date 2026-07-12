from __future__ import annotations

from typing import TypedDict

from app.agents.research_agent.schema import ProductResearch, ScrapedProductData


class ResearchState(TypedDict, total=False):
    """
    State threaded through the research subgraph. `total=False` because most
    fields only exist after their producing node has run.
    """

    url: str

    scraped: ScrapedProductData | None
    research: ProductResearch | None

    retries: int
    error: str | None
