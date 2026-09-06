"""
State for the Image Selection (Critic) subgraph.

Mirrors ImageGenerationState/VideoGenerationState's shape: single upstream
input (`images`, Agent 4's ImageGenerationOutput), `current_theme_index`
driving the same per-theme loop pattern as Agents 4 and 5, retry scratch
fields named to match the existing convention (retries/error) so any
pipeline-runner code that reads result.get("retries", 0) keeps working
unmodified.

No `current_seed` field here (unlike Agent 4/5's state) - this agent
doesn't generate anything, it judges what's already been generated, so
there's no seed to track.
"""

from __future__ import annotations

from typing import TypedDict

from app.agents.image_generation.schema import ImageGenerationOutput  # Agent 4's output
from app.agents.image_selection_agent.schema import ImageSelectionOutput, ThemeSelectionResult


class ImageSelectionState(TypedDict, total=False):
    # Input
    images: ImageGenerationOutput

    # Progress through theme_results (driven by images.theme_results length,
    # same convention as Agents 4/5 driving off their own upstream lists)
    current_theme_index: int
    completed_theme_results: list[ThemeSelectionResult]

    # Set once in start_node
    run_id: str

    # Per-attempt scratch state - field names match Agents 4/5's exactly
    retries: int
    error: str | None

    # Final output
    selection: ImageSelectionOutput | None