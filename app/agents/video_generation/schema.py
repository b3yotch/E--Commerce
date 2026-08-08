"""
Output contract for the Video Generation Agent (Agent 5).

Mirrors Agent 4's schema.py shape deliberately (GeneratedVideo /
ThemeVideoResult / VideoGenerationOutput <-> GeneratedImage /
ThemeGenerationResult / ImageGenerationOutput) rather than inventing a
different structure - same reasoning either way: one wrapper per theme,
the wrapper carries provenance + timing + retry count, the leaf object
carries the actual generated-file details.

Two real differences from Agent 4's shape, not oversights:

1. No list[GeneratedVideo] per theme - Agent 4 keeps every candidate in a
   batch because no critic exists yet to pick a winner (see Image_
   generation.md Challenge 2). Video deliberately generates exactly one
   per theme (see Video_generation.md Challenge 2 discussion - video's
   cost profile doesn't reward batching the way cheap image batches did),
   so `video` is a single optional field, not a list.
2. `status` distinguishes "skipped_no_source_image" from "success" -
   Agent 4 has no equivalent because it always has *something* to attempt
   (a prompt). This agent can hit a theme with no valid Agent 4 image to
   animate (that theme was itself skipped upstream after exhausting ITS
   retries - see Image_generation.md Challenge 3) - that's not a
   generation failure worth retrying, so it's recorded explicitly rather
   than silently absent. A genuine retries-exhausted generation failure,
   by contrast, is NOT recorded here at all - same as Agent 4, the theme
   is simply absent from theme_results. Worth revisiting both choices
   together once real failure rates are known.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class GeneratedVideo(BaseModel):
    """One rendered video, for one theme."""

    filename: str = Field(description="Filename as reported by ComfyUI's save node - kept for traceability.")
    subfolder: str = Field(default="", description="Subfolder under ComfyUI's own output/ dir, if any.")
    local_path: str = Field(
        description="Path under this project's own output directory - the path to actually read from, same convention as GeneratedImage.local_path."
    )
    seed: int = Field(description="Seed actually used, for reproducibility/debugging.")
    source_image_local_path: str = Field(
        description=(
            "Which Agent 4 GeneratedImage.local_path was used as the source frame. "
            "PLACEHOLDER SELECTION: currently always theme_result.images[0] - the "
            "first of Agent 4's candidates, not a chosen 'best' one, since no "
            "critic exists yet to choose (see module docstring)."
        )
    )


class ThemeVideoResult(BaseModel):
    """Everything produced (or explicitly not produced) for one theme."""

    source_setting: str = Field(
        description="Echoes ThemePromptSet.source_setting verbatim, so this traces back to its origin theme."
    )
    video: GeneratedVideo | None = Field(
        default=None,
        description="None only when status is 'skipped_no_source_image'.",
    )
    generation_time_seconds: float = 0.0
    retries_used: int = 0
    status: Literal["success", "skipped_no_source_image"] = "success"


class VideoGenerationOutput(BaseModel):
    """Final output for one product."""

    source_url: str
    theme_results: list[ThemeVideoResult] = Field(default_factory=list)