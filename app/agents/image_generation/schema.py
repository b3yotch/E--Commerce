"""
Output contract for the Image Generation Agent (Agent 4, image-only for now
- video deferred).

Deliberately NOT modeled here: which of the N generated images per theme is
"best." No critic exists yet (Agent 5), so this agent's job stops at
"generate N candidates, confirm they're real, hand off the manifest" - not
"pick a winner." Picking a winner without a judge would just be Agent 4
quietly guessing, the same trap Agent 3's schema explicitly avoided with
sampler params.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class GeneratedImage(BaseModel):
    """One rendered candidate image."""

    filename: str = Field(description="Filename as reported by ComfyUI's SaveImage node - kept for traceability.")
    subfolder: str = Field(default="", description="Subfolder under ComfyUI's own output/ dir, if any - kept for traceability, not the read path.")
    local_path: str = Field(
        description=(
            "Path under this project's own output directory where the image was "
            "copied to after generation. This is the path Agent 5 (and anything "
            "else downstream) should actually read from - not ComfyUI's install dir."
        )
    )
    width: int
    height: int
    seed: int = Field(description="Seed actually used for this candidate, for reproducibility/debugging.")


class ThemeGenerationResult(BaseModel):
    """
    All candidate images generated from one ThemePromptSet - the image-side
    counterpart to Agent 3's ThemePromptSet. One of these per input
    prompt_sets entry.
    """

    source_setting: str = Field(
        description="Echoes ThemePromptSet.source_setting verbatim, so this traces back to its origin theme."
    )
    aspect_ratio: str = Field(description="The requested ratio string, e.g. '4:5'.")
    resolved_width: int
    resolved_height: int
    images: list[GeneratedImage] = Field(
        default_factory=list,
        description="Candidate images from this theme's batch (batch_size, currently 5, all kept - no selection performed here).",
    )
    generation_time_seconds: float
    retries_used: int = 0


class ImageGenerationOutput(BaseModel):
    """Final output for one product - ready to hand to Agent 5 (Review/Critic) once it exists."""

    source_url: str
    theme_results: list[ThemeGenerationResult] = Field(default_factory=list)