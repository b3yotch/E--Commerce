"""
Graph assembly for the Image Selection (Critic) subgraph (Agent 5).

Runs between Agent 4 and Agent 6 (Video Generation) in the pipeline.

Loop shape mirrors Agents 4/6: a single node (`select`) processes one theme
per pass and either loops back into itself (next theme, or a retry of the
same theme - see nodes.py's select_node docstring for why one node covers
both) or proceeds to `finalize` once every theme in the input has been
handled.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.agents.image_selection_agent.nodes import (
    finalize_node,
    select_node,
    should_continue,
    start_node,
)
from app.agents.image_selection_agent.state import ImageSelectionState


def build_image_selection_graph():
    graph = StateGraph(ImageSelectionState)

    graph.add_node("start", start_node)
    graph.add_node("select", select_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "start")
    graph.add_edge("start", "select")
    graph.add_conditional_edges(
        "select",
        should_continue,
        {"continue": "select", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)

    return graph.compile()


# Module-level compiled instance, matching how the other agents' graphs are
# imported (research_graph, creative_strategy_graph, image_generation_graph,
# etc.) rather than requiring every caller to invoke the builder itself.
image_selection_graph = build_image_selection_graph()