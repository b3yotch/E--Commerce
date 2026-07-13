from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.agents.creative_strategy_agent.nodes import (
    bump_retry_node,
    generate_node,
    route_after_validation,
    validate_node,
)
from app.agents.creative_strategy_agent.state import CreativeStrategyState


def build_creative_strategy_graph():
    """
    generate -> validate -+-> END  (valid, or retries exhausted)
        ^                  |
        |                  v
        +--- bump_retry <--+ (invalid, retries remain)

    No scrape/route_after_scrape stage here (unlike the research agent's
    graph) - this agent's input is already a validated ProductResearch
    object, not a URL that can fail to fetch.
    """
    graph = StateGraph(CreativeStrategyState)

    graph.add_node("generate", generate_node)
    graph.add_node("validate", validate_node)
    graph.add_node("bump_retry", bump_retry_node)

    graph.set_entry_point("generate")
    graph.add_edge("generate", "validate")

    graph.add_conditional_edges(
        "validate",
        route_after_validation,
        {"done": END, "retry": "bump_retry"},
    )
    graph.add_edge("bump_retry", "generate")

    return graph.compile()


creative_strategy_graph = build_creative_strategy_graph()