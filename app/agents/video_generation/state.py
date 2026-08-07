"""
State schema for the Video Generation Agent's graph.

IMPORTANT - see Image_generation.md Challenge 7 before touching this file:
LangGraph silently drops any key a node function returns that isn't
declared here. That bug cost real debugging time on Agent 4 (generation
succeeded every time; validation always saw the result as None because
_pending_result wasn't declared). Every key generate_node/advance_theme_node
return MUST appear below, including the ones prefixed with underscore -
"scratch" state is not exempt from this.
"""

from __future__ import annotations

from typing import TypedDict

from app.agents.prompt_generation.schema import ThemePromptSet  # Agent 3 output
from app.agents.video_generation.schema import GeneratedVideo


class SourceImageRef(TypedDict):
    """
    Minimal adapter shape for an Agent 4 image, indexed by theme. See the
    ADAPTER NOTE in video_schema.py - field names here should match
    whatever Agent 4's actual GeneratedImage exposes; only
    _index_images_by_theme() in video_nodes.py needs to change if they
    don't.
    """

    source_setting: str
    local_path: str
    status: str


class VideoGenerationState(TypedDict):
    # Inputs, set once in start_node
    source_url: str
    prompt_sets: list[ThemePromptSet]
    images_by_theme: dict[str, SourceImageRef]  # keyed by source_setting, built in start_node
    video_workflow_template: dict  # loaded once in start_node, reused (deep-copied) per theme
    run_id: str

    # Theme loop position
    theme_index: int

    # Retry loop position (reset by advance_theme_node between themes)
    retry_count: int

    # Per-attempt scratch state (reset by advance_theme_node between themes)
    current_seed: int | None
    _pending_result: dict | None  # set by generate_node, read by validate_node - see module docstring

    # Accumulated final output
    videos: list[GeneratedVideo]