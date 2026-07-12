from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.agents.research_agent.nodes import (
    bump_retry_node,
    extract_node,
    route_after_scrape,
    route_after_validation,
    scrape_node,
    validate_node,
)
from app.agents.research_agent.state import ResearchState


def build_research_graph():
    """
    scrape -+-> extract -> validate -+-> END  (valid, or retries exhausted)
            |                ^                |
            | (scrape failed)|                v
            +------> END     +--- bump_retry <-+ (invalid, retries remain)
    """
    graph = StateGraph(ResearchState)

    graph.add_node("scrape", scrape_node)
    graph.add_node("extract", extract_node)
    graph.add_node("validate", validate_node)
    graph.add_node("bump_retry", bump_retry_node)

    graph.set_entry_point("scrape")
    graph.add_conditional_edges(
        "scrape",
        route_after_scrape,
        {"ok": "extract", "failed": END},
    )
    graph.add_edge("extract", "validate")

    graph.add_conditional_edges(
        "validate",
        route_after_validation,
        {"done": END, "retry": "bump_retry"},
    )
    graph.add_edge("bump_retry", "extract")

    return graph.compile()


research_graph = build_research_graph()
