"""
Output contract for the Prompt Generation Agent.

Downstream consumer is Agent 4 (Image/Video Generation via ComfyUI), so this
schema shape is driven by what a ComfyUI workflow can actually consume - not
by what an LLM finds natural to produce. Two design choices worth calling
out up front, since they diverge from what a first draft would probably do:

1. `suggested_hooks`/`suggested_captions` REFERENCE Agent 2's existing
   hooks/captions rather than inventing new copy. Diffusion models render
   legible text unreliably, so hook/caption text is meant to be composited
   as an overlay in a later step, not baked into the generation prompt -
   and letting this agent write new copy would create a second, drifting
   source of truth alongside Agent 2's CreativeDirection.
2. Sampler/workflow parameters (seed, steps, cfg_scale, checkpoint/LoRA
   selection) are deliberately NOT modeled here. Those are deterministic
   pipeline configuration that belongs to Agent 4's ComfyUI workflow JSON,
   the same way scrape_timeout_ms belongs to Settings rather than
   ProductResearch - an LLM asked to invent them would just be guessing.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ImageGenerationPrompt(BaseModel):
    """One image prompt, ComfyUI-ready (CLIPTextEncode positive/negative)."""

    positive_prompt: str = Field(
        description=(
            "Dense natural-language generation prompt covering subject, setting, "
            "lighting, composition, and style. Written in the target checkpoint's "
            "conventions (e.g. natural sentences for Flux-style models); this agent "
            "does not know which checkpoint will run it, so default to natural "
            "language unless style_notes says otherwise."
        )
    )
    negative_prompt: str = Field(
        default="",
        description=(
            "What to actively exclude (common defects, unwanted elements, style "
            "drift). Empty string if a negative prompt wouldn't add anything - "
            "don't pad this out for its own sake."
        ),
    )
    aspect_ratio: str = Field(
        description=(
            "e.g. '1:1', '4:5', '9:16' - pick based on what the composition "
            "actually calls for (a wide landscape setting suits 16:9; a "
            "product-in-hand close-up suits 4:5/9:16 for feed/story placements)."
        )
    )
    style_notes: str = Field(
        default="",
        description=(
            "Checkpoint/LoRA-relevant hints (e.g. 'photorealistic product "
            "photography, not illustration'), kept separate from positive_prompt "
            "so a human can swap checkpoints without rewriting the whole prompt."
        ),
    )


class VideoGenerationPrompt(BaseModel):
    """One video prompt - motion/camera direction layered on the visual description."""

    base_prompt: str = Field(
        description="Visual description, same conventions as ImageGenerationPrompt.positive_prompt."
    )
    motion_description: str = Field(
        description=(
            "Camera and subject motion, in concrete cinematography language "
            "(e.g. 'slow dolly-in, subject remains still, background softly "
            "drifts') - not vague terms like 'dynamic movement'."
        )
    )
    duration_seconds: float = Field(
        default=4.0,
        description=(
            "Target clip length in seconds. Keep short (3-6s) unless the "
            "motion description genuinely needs more to read as intentional "
            "rather than aimless."
        ),
    )
    negative_prompt: str = Field(default="")


class ThemePromptSet(BaseModel):
    """
    Everything generated from one CreativeDirection.visual_themes entry - the
    actual unit of output. One of these per visual_theme, not per hook: hooks
    and captions are text copy meant to pair with a visual, not each drive
    their own separate generation.
    """

    source_setting: str = Field(
        description=(
            "Echoes the source visual_theme.setting verbatim, so this entry "
            "can be traced back to which VisualTheme produced it."
        )
    )
    image_prompt: ImageGenerationPrompt
    video_prompt: VideoGenerationPrompt
    suggested_hooks: list[str] = Field(
        default_factory=list,
        description=(
            "Hooks copied VERBATIM from the input CreativeDirection.hooks that "
            "pair well with this visual - do not invent new hook text here."
        ),
    )
    suggested_captions: list[str] = Field(
        default_factory=list,
        description=(
            "Captions copied VERBATIM from the input CreativeDirection.captions "
            "that pair well with this visual - do not invent new caption text here."
        ),
    )


class PromptGenerationOutput(BaseModel):
    """Final, validated output for one product - ready to hand to Agent 4."""

    source_url: str

    prompt_sets: list[ThemePromptSet] = Field(
        default_factory=list,
        description="One entry per input visual_theme (typically 2-3 total).",
    )

    extraction_confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Self-reported confidence 0-1 that these prompts are grounded in "
            "the input visual_themes/hooks/captions, not invented from scratch."
        ),
    )