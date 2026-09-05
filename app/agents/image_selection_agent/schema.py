"""
Output contract for the Image Selection (Critic) Agent (Agent 5).

Sits between Agent 4 and Agent 6 (Video Generation) in the pipeline - this
is the number Agent 4's own schema docstring anticipated ("no critic
exists yet (Agent 5)") before Video Generation was built first and
temporarily claimed it. Renumbered so agent number matches graph position.
Its job: replace Agent 6's `theme_result.images[0]` placeholder with an
actual chosen frame, judged by a vision model against the theme's creative
brief.

Deliberately NOT modeled here: automatic regeneration when every candidate
scores poorly. No cross-agent retry loop back to Agent 4 exists (the same
non-decision Agent 4 and Agent 5 both already made deliberately) - a
below-threshold pick is recorded explicitly via `status` instead of being
silently treated as a clean success, same pattern as
video_generation.schema.ThemeVideoResult.status.

Also deliberately NOT mutating Agent 4's ImageGenerationOutput in place -
this agent produces its own output referencing Agent 4's candidates by
local_path, the same way Agent 5 references Agent 4's output rather than
editing it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ImageCandidateScore(BaseModel):
    """The critic's judgment on one candidate image."""

    image_index: int = Field(
        description="0-based index matching the order images were shown to the model in this call."
    )
    score: int = Field(
        ge=0,
        le=100,
        description="0-100 against the local-checkpoint-calibrated rubric (see prompts.py), not an absolute quality scale.",
    )
    issues: list[str] = Field(
        default_factory=list,
        description="Brief notes on real problems found, if any. Empty when the candidate is clean.",
    )


class ImageCritiqueResponse(BaseModel):
    """Raw shape returned by the vision model for one theme's candidate set."""

    candidate_scores: list[ImageCandidateScore]
    best_index: int = Field(
        description="0-based index of the strongest candidate, matching candidate_scores indices."
    )
    rationale: str = Field(description="1-2 sentences on why best_index was chosen over the others.")


class ThemeSelectionResult(BaseModel):
    """Everything decided for one theme - the selection-side counterpart to ThemeGenerationResult."""

    source_setting: str = Field(
        description="Echoes ThemeGenerationResult.source_setting verbatim, so this traces back to its origin theme."
    )
    selected_local_path: str = Field(
        description="local_path of the GeneratedImage chosen as this theme's source frame - what Agent 5 should read instead of images[0]."
    )
    candidate_scores: list[ImageCandidateScore] = Field(default_factory=list)
    rationale: str = ""
    status: Literal["selected", "selected_below_threshold"] = "selected"
    retries_used: int = 0


class ImageSelectionOutput(BaseModel):
    """Final output for one product - ready to hand to Agent 5 in place of raw ImageGenerationOutput."""

    source_url: str
    theme_results: list[ThemeSelectionResult] = Field(default_factory=list)