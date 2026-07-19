"""
Graph shape:

    start --> generate --> validate --+--> [more themes?] --+--> advance_theme --> generate (next theme)
                 ^                    |                      |
                 |                    v                      +--> finalize --> END (all themes done)
                 +----- bump_retry <--+ (invalid, retries remain)
                                      |
                                      +--> finalize (retries exhausted - see note below)

Two loops layered on Agents 1-3's single retry loop, since this agent also
has to iterate across a variable number of themes (typically 2-3, per
Agent 3's prompt_sets) - something Agents 1-3 didn't need since they each
produced one output per product, not one output per theme within a product.

On exhausted retries: unlike Agents 1-3 (which return best-effort output
after max retries), a theme that never got a valid image has NOTHING to
hand forward - there's no partial image to include. Current behavior: skip
the theme (it's simply absent from theme_results) and continue to the next
one, rather than failing the whole product over one bad theme. Worth
revisiting once real failure rates are known - if a specific theme fails
often, that's more likely a prompt/checkpoint mismatch worth surfacing
loudly than something to quietly skip forever.
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from app.core.config import settings
from app.agents.image_generation.state import ImageGenerationState
from app.agents.image_generation.nodes import (
    start_node,
    generate_node,
    validate_node,
    bump_retry_node,
    advance_theme_node,
    finalize_node,
)


def _after_validate(state: ImageGenerationState) -> str:
    has_error = bool(state.get("error"))

    if has_error:
        if state.get("retries", 0) < settings.max_image_gen_retries:
            return "bump_retry"
        # Retries exhausted for this theme - skip it (see module docstring).
        return "advance_theme"

    return "advance_theme"


def _after_advance(state: ImageGenerationState) -> str:
    prompt_sets = state["prompts"].prompt_sets
    if state.get("current_theme_index", 0) < len(prompt_sets):
        return "generate"
    return "finalize"


def build_graph():
    graph = StateGraph(ImageGenerationState)

    graph.add_node("start", start_node)
    graph.add_node("generate", generate_node)
    graph.add_node("validate", validate_node)
    graph.add_node("bump_retry", bump_retry_node)
    graph.add_node("advance_theme", advance_theme_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("start")
    graph.add_edge("start", "generate")
    graph.add_edge("generate", "validate")

    graph.add_conditional_edges(
        "validate",
        _after_validate,
        {"bump_retry": "bump_retry", "advance_theme": "advance_theme"},
    )
    graph.add_edge("bump_retry", "generate")

    graph.add_conditional_edges(
        "advance_theme",
        _after_advance,
        {"generate": "generate", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)

    return graph.compile()


# Compiled once at import time, matching the pattern research_graph /
# creative_strategy_graph / prompt_gen_graph already use in
# test_full_pipeline_live.py.
image_generation_graph = build_graph()