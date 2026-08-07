"""
Output contract for the Video Generation Agent (Agent 5).

Deliberately its own schema, not a reuse of Agent 4's ImageGenerationOutput
- the fields genuinely differ (a video result needs to reference which
Agent 4 image it was generated from; there's no equivalent "which source"
field on an image result), and forcing them into one shared shape would
mean padding one or the other with fields that don't apply. Same reasoning
Prompt_generation.md Challenge 1 already used for keeping Agent 3 and
Agent 4's concerns separate.

ADAPTER NOTE: this assumes Agent 4's ImageGenerationOutput exposes
source_setting, local_path, and status per generated image (used in
video_nodes.py to find which image a given theme's video should animate).
That assumption comes from Image_generation.md's description, not from
seeing Agent 4's actual schema.py - if the real field names differ, only
video_nodes.py's _index_images_by_theme() needs to change, not this file.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class GeneratedVideo(BaseModel):
    """One video result for one visual theme."""

    source_setting: str = Field(
        description=(
            "Echoes ThemePromptSet.source_setting, so this entry traces "
            "back to which visual theme (and which Agent 4 image) produced it."
        )
    )
    source_image_local_path: str | None = Field(
        default=None,
        description="Local path of the Agent 4 image this video was generated from.",
    )
    local_path: str | None = Field(
        default=None,
        description="Local path of the downloaded video file, or None if this theme was skipped/failed.",
    )
    filename: str | None = None
    subfolder: str = ""
    seed: int | None = Field(
        default=None,
        description="Seed used for the attempt that produced this result (or the last attempted seed, on failure).",
    )
    attempts_used: int = Field(default=0, ge=0)
    status: Literal["success", "failed_retries_exhausted", "skipped_no_source_image"] = "failed_retries_exhausted"
    error: str | None = Field(
        default=None,
        description="Last error message, if status is not 'success'.",
    )


class VideoGenerationOutput(BaseModel):
    """Final, validated output for one product."""

    source_url: str
    run_id: str = Field(description="Same run_id convention as Agent 4 - UTC timestamp, one per pipeline invocation.")

    videos: list[GeneratedVideo] = Field(
        default_factory=list,
        description="One entry per input visual_theme (typically 2-3 total) - one video per theme, no batching.",
    )

    total_videos_generated: int = Field(default=0, ge=0)
    total_themes_attempted: int = Field(default=0, ge=0)