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

ImageCandidateScore/ImageCritiqueResponse were trimmed after a real
json_validate_failed error from Groq's strict JSON-schema decoding
(qwen/qwen3.6-27b): the original shape had an optional field with a
default (issues: list[str] = Field(default_factory=list)) and numeric
range constraints (Field(ge=0, le=100)) on score - either an unsupported
schema feature or the extra verbosity pushing generation past
image_selection_max_tokens before it could close the JSON could explain a
deterministic (not transient) validation failure. Fixed by making every
field required with no defaults, dropping the ge/le constraints (the 0-100
range is enforced by instruction text in prompts.py instead, not the
schema), and removing image_index entirely - candidates are matched to
their image by array position instead of a redundant explicit field,
which also cuts tokens needed per candidate.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ImageCandidateScore(BaseModel):
    """
    The critic's judgment on one candidate image. Matched to its image by
    position in ImageCritiqueResponse.candidate_scores, not by an explicit
    index field - see module docstring for why that field was removed.
    """

    model_config = ConfigDict(extra="forbid")

    score: int = Field(
        description=(
            "0-100 against the local-checkpoint-calibrated rubric (see "
            "prompts.py), not an absolute quality scale. Stay within 0-100 - "
            "not enforced by the schema itself (see module docstring), so "
            "this is instruction-only."
        )
    )
    issue: str = Field(
        description=(
            "The single biggest real problem found with this candidate, in "
            "one short sentence. Use an empty string if there are no real "
            "problems - always include this field, never omit it."
        )
    )


class ImageCritiqueResponse(BaseModel):
    """Raw shape returned by the vision model for one theme's candidate set."""

    model_config = ConfigDict(extra="forbid")

    candidate_scores: list[ImageCandidateScore] = Field(
        description=(
            "One entry per candidate image, in the SAME ORDER the images "
            "were shown. Position in this list is how a score is matched "
            "back to its image - there is no separate index field."
        )
    )
    best_index: int = Field(
        description="0-based position in candidate_scores (and in the images you were shown) of the strongest candidate."
    )
    rationale: str = Field(description="Why best_index was chosen over the others, in one short sentence.")


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