from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.agents.prompt_gen_agent.nodes import (
    bump_retry_node,
    generate_node,
    route_after_validation,
    validate_node,
)
from app.agents.prompt_gen_agent.state import PromptGenState


def build_prompt_gen_graph():
    """
    generate -> validate -+-> END  (valid, or retries exhausted)
        ^                  |
        |                  v
        +--- bump_retry <--+ (invalid, retries remain)

    Same shape as Creative Strategy's graph - no scrape/route_after_scrape
    stage, since this agent's input (a validated CreativeDirection) can't
    fail to fetch the way a URL can.
    """
    graph = StateGraph(PromptGenState)

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


prompt_gen_graph = build_prompt_gen_graph()