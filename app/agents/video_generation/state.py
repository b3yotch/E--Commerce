"""
State for the Video Generation subgraph.

Mirrors ImageGenerationState's shape and naming closely. Two things worth
calling out:

1. This agent takes TWO upstream inputs, not one - `prompts` (Agent 3's
   PromptGenerationOutput, for video_prompt/motion_description per theme)
   AND `images` (Agent 4's ImageGenerationOutput, for the source frame per
   theme). Agents 1-4 each only ever consumed the single immediately-prior
   agent's output; this is the first agent in the chain that needs two.
2. `_pending_result` is declared here on purpose - see Image_generation.md
   Challenge 7. That bug (a returned-but-undeclared state key silently
   dropped by LangGraph, making validation always fail despite generation
   succeeding every time) is exactly the kind of mistake worth guarding
   against explicitly when porting a pattern to a new agent, not just
   trusting muscle memory to remember it this time.
"""

from __future__ import annotations

from typing import TypedDict

from app.agents.prompt_gen_agent.schema import PromptGenerationOutput  # Agent 3's output
from app.agents.image_generation.schema import ImageGenerationOutput  # Agent 4's output
from app.agents.video_generation.schema import ThemeVideoResult, VideoGenerationOutput


class VideoGenerationState(TypedDict, total=False):
    # Inputs
    prompts: PromptGenerationOutput
    images: ImageGenerationOutput

    # Progress through prompt_sets (driven by prompts.prompt_sets, same as
    # Agent 4 drives off its own prompt_sets length)
    current_theme_index: int
    completed_theme_results: list[ThemeVideoResult]

    # Set once in start_node
    run_id: str
    video_workflow_template: dict  # loaded once, deep-copied per theme in generate_node

    # Per-attempt scratch state - field names match Agent 4's exactly
    # (retries/error/current_seed) for the same reason Agent 4 matched
    # Agents 1-3's names: existing pipeline-runner code that reads
    # result.get("retries", 0) etc. keeps working unmodified.
    retries: int
    error: str | None
    current_seed: int | None
    _pending_result: ThemeVideoResult | None

    # Final output
    videos: VideoGenerationOutput | None