"""
Nodes for the Image Selection (Critic) subgraph.

Two-step judgment per theme, settled in Image_selection.md:
1. Deterministic pre-filter - drop candidates that are corrupt/unreadable
   before spending a model call on them at all.
2. Vision-LLM comparative ranking of whatever survives the filter, shown
   together against the theme's creative brief - a relative choice among a
   small set, not independent scoring compared after the fact.

Failure handling deliberately never leaves a theme with no selection -
Agent 5 needs exactly one source frame per theme, so every branch below
ends in a ThemeSelectionResult, just sometimes flagged
"selected_below_threshold" instead of "selected". See schema.py's module
docstring for why a flagged pick beats no pick.

Retry mechanics: select_node only advances current_theme_index on success
(via _advance). A raised-but-caught LLMExtractionError returns an
incremented `retries` count and leaves the index untouched, so the graph
naturally retries the same theme on the next pass through select_node
without needing a separate retry branch in the conditional edge.
"""

from __future__ import annotations

import asyncio
import uuid

from PIL import Image, UnidentifiedImageError

from app.agents.image_selection_agent.prompts import CRITIC_SYSTEM_PROMPT, build_user_prompt
from app.agents.image_selection_agent.schema import (
    ImageCritiqueResponse,
    ImageSelectionOutput,
    ThemeSelectionResult,
)
from app.agents.image_selection_agent.state import ImageSelectionState
from app.core.config import settings
from app.core.llm import LLMExtractionError, LLMRateLimitError, structured_chat_groq_vision


def _is_valid_image(path: str) -> bool:
    """
    Cheap corruption check before spending a model call on a candidate.
    Opens and calls verify() rather than just checking the file exists -
    catches truncated/zero-byte-but-present files a copy step could in
    principle leave behind, not just missing ones.
    """
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (FileNotFoundError, UnidentifiedImageError, OSError):
        return False


def start_node(state: ImageSelectionState) -> dict:
    return {
        "run_id": str(uuid.uuid4()),
        "current_theme_index": 0,
        "completed_theme_results": [],
        "retries": 0,
        "error": None,
    }


async def select_node(state: ImageSelectionState) -> dict:
    images_output = state["images"]
    idx = state.get("current_theme_index", 0)
    theme = images_output.theme_results[idx]

    valid_images = [img for img in theme.images if _is_valid_image(img.local_path)]

    if not valid_images:
        # Nothing survived the deterministic filter - fall back to whatever
        # Agent 4 reported rather than losing the whole run over one bad
        # theme; recorded explicitly via status, same philosophy as
        # Agent 5's skipped_no_source_image.
        fallback = theme.images[0] if theme.images else None
        result = ThemeSelectionResult(
            source_setting=theme.source_setting,
            selected_local_path=fallback.local_path if fallback else "",
            candidate_scores=[],
            rationale="No candidate passed the deterministic validity check; fell back to the first reported candidate unjudged.",
            status="selected_below_threshold",
            retries_used=state.get("retries", 0),
        )
        return _advance(state, result)

    # Some Groq vision models cap how many images fit in one request - this
    # has changed before and will likely change again, so it's a config
    # value, not a hardcoded number. Truncating (rather than erroring) keeps
    # a theme with more candidates than the model's limit from blocking the
    # whole run; the untruncated candidates are simply never considered.
    valid_images = valid_images[: settings.image_selection_max_images_per_call]
    image_paths = [img.local_path for img in valid_images]

    try:
        critique = await structured_chat_groq_vision(
            model=settings.image_selection_model,
            system_prompt=CRITIC_SYSTEM_PROMPT,
            user_prompt=build_user_prompt(theme.source_setting, len(image_paths)),
            image_paths=image_paths,
            schema=ImageCritiqueResponse,
            temperature=settings.image_selection_temperature,
            max_tokens=settings.image_selection_max_tokens,
        )
    except LLMExtractionError as exc:
        retries = state.get("retries", 0)
        if retries < settings.max_image_selection_retries:
            if isinstance(exc, LLMRateLimitError):
                # Retrying instantly here would land inside the same
                # per-minute OTPM window and hit the identical 429 again -
                # that's exactly what happened before this fix (all 2
                # retries failed identically, for every theme). Wait past
                # the window instead of retrying on the next graph pass.
                await asyncio.sleep(settings.image_selection_rate_limit_backoff_seconds)
            return {"retries": retries + 1, "error": str(exc)}
        # Retries exhausted - fall back to the first valid candidate rather
        # than losing the theme entirely (same reasoning as the
        # no-valid-candidates branch above).
        result = ThemeSelectionResult(
            source_setting=theme.source_setting,
            selected_local_path=valid_images[0].local_path,
            candidate_scores=[],
            rationale=f"Vision critic call failed after {retries} retries ({exc}); fell back to first valid candidate unjudged.",
            status="selected_below_threshold",
            retries_used=retries,
        )
        return _advance(state, result)

    best_index = critique.best_index
    if best_index < 0 or best_index >= len(valid_images):
        best_index = 0  # malformed index from the model - don't propagate it

    best_score = next(
        (c.score for c in critique.candidate_scores if c.image_index == best_index),
        0,
    )
    status = (
        "selected"
        if best_score >= settings.image_selection_score_threshold
        else "selected_below_threshold"
    )

    result = ThemeSelectionResult(
        source_setting=theme.source_setting,
        selected_local_path=valid_images[best_index].local_path,
        candidate_scores=critique.candidate_scores,
        rationale=critique.rationale,
        status=status,
        retries_used=state.get("retries", 0),
    )
    return _advance(state, result)


def _advance(state: ImageSelectionState, result: ThemeSelectionResult) -> dict:
    completed = state.get("completed_theme_results", []) + [result]
    return {
        "completed_theme_results": completed,
        "current_theme_index": state.get("current_theme_index", 0) + 1,
        "retries": 0,
        "error": None,
    }


def should_continue(state: ImageSelectionState) -> str:
    images_output = state["images"]
    if state.get("current_theme_index", 0) >= len(images_output.theme_results):
        return "finalize"
    return "continue"


def finalize_node(state: ImageSelectionState) -> dict:
    images_output = state["images"]
    selection = ImageSelectionOutput(
        source_url=images_output.source_url,
        theme_results=state.get("completed_theme_results", []),
    )
    return {"selection": selection}