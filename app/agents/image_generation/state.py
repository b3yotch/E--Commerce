"""
State for the Image Generation subgraph.

Unlike Agents 1-3, this agent has no prompts.py - there's no LLM call here,
generation is a deterministic API call to ComfyUI, so there's no prompt to
construct. Same "discuss which files actually apply" call the README
flagged before writing any code.

Iterates one ThemePromptSet at a time (current_theme_index) rather than
firing all themes in parallel, since ComfyUI generation is slow/resource-
heavy compared to Agents 1-3's LLM calls (see README) - sequential keeps
this agent's first version simple; parallelizing across themes is a
concurrency decision to revisit once this works end-to-end, not something
to build speculatively now.
"""

from __future__ import annotations

from typing import TypedDict

from app.agents.prompt_gen_agent.schema import PromptGenerationOutput  # Agent 3's output
from app.agents.image_generation.schema import ImageGenerationOutput, ThemeGenerationResult


class ImageGenerationState(TypedDict, total=False):
    # Input - key name "prompts" matches the pattern research->creative->prompts:
    # each agent's input key is the previous agent's output variable name.
    prompts: PromptGenerationOutput

    # Progress through prompt_sets
    current_theme_index: int
    completed_theme_results: list[ThemeGenerationResult]

    # Set once in start_node, for the whole run
    run_id: str
    theme_image_counts: list[int]  # distribute_total(settings.total_images_per_product, len(prompt_sets))

    # Per-attempt scratch state for the theme currently being generated.
    # "retries"/"error" match Agents 1-3's field names exactly (not
    # retry_count/previous_error) so test_full_pipeline_live.py's existing
    # result.get("retries", 0) / result.get("error") calls work unmodified.
    retries: int
    error: str | None
    current_seed: int | None
    # Bug fix: this was previously used in nodes.py but never declared here.
    # LangGraph only carries forward state keys that exist in this TypedDict
    # schema - any node return value outside it is silently dropped between
    # steps. Without this line, generate_node's _pending_result never
    # reached validate_node, so validation always saw "no result" regardless
    # of whether generation actually succeeded - retrying and eventually
    # skipping every theme even though images were generated and saved to
    # disk correctly on the very first attempt each time.
    _pending_result: ThemeGenerationResult | None

    # Final output, assembled once every theme is done - "images" continues
    # the research/creative/prompts/images naming pattern.
    images: ImageGenerationOutput | None