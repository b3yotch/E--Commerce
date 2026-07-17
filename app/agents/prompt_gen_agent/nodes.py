from __future__ import annotations

from app.core.config import settings
from app.core.llm import LLMExtractionError, structured_chat_groq
from app.agents.prompt_gen_agent.prompts import SYSTEM_PROMPT, build_user_prompt
from app.agents.prompt_gen_agent.schema import PromptGenerationOutput
from app.agents.prompt_gen_agent.state import PromptGenState


async def generate_node(state: PromptGenState) -> PromptGenState:
    """
    Generate image/video prompts from the creative direction.

    Same primary/fallback model-chain pattern as Creative Strategy's
    generate_node: a provider failure (schema validation error, API error,
    rate limit) falls through to the fallback model immediately and does NOT
    consume a retry, because there's nothing useful to feed back into the
    prompt for a transient provider issue. The retry loop (bump_retry_node)
    is reserved for "a model produced output, but it didn't pass validation."

    Uses its own prompt_gen_primary_model/prompt_gen_fallback_model settings
    rather than Creative Strategy's groq_primary_model/groq_fallback_model -
    writing prompts a diffusion/video model responds well to is a different
    skill from ad copywriting, so model choice here should be validated on
    its own before assuming it should track Creative Strategy's models.
    """
    creative = state["creative"]
    previous_error = state.get("error")
    user_prompt = build_user_prompt(creative, previous_error=previous_error)

    last_error = "No model was attempted"
    for model in (settings.prompt_gen_primary_model, settings.prompt_gen_fallback_model):
        try:
            prompts = await structured_chat_groq(
                model=model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                schema=PromptGenerationOutput,
                temperature=settings.prompt_gen_temperature,
            )
            prompts.source_url = creative.source_url
            return {**state, "prompts": prompts, "error": None, "model_used": model}
        except LLMExtractionError as exc:
            last_error = str(exc)
            continue

    return {**state, "prompts": None, "error": last_error, "model_used": None}


def _missing_reasons(state: PromptGenState) -> list[str]:
    """
    Collect every reason the current output isn't good enough, rather than
    stopping at the first problem found - so validate_node can write one
    complete failure reason instead of needing several retries just to
    surface issues one at a time.
    """
    prompts = state.get("prompts")
    creative = state["creative"]

    if prompts is None:
        return ["model produced no output"]

    reasons: list[str] = []

    expected = len(creative.visual_themes)
    actual = len(prompts.prompt_sets)
    if actual != expected:
        reasons.append(f"expected {expected} prompt_sets (one per visual_theme), got {actual}")

    for i, ps in enumerate(prompts.prompt_sets):
        if not ps.image_prompt.positive_prompt.strip():
            reasons.append(f"prompt_sets[{i}].image_prompt.positive_prompt is empty")
        if not ps.video_prompt.base_prompt.strip():
            reasons.append(f"prompt_sets[{i}].video_prompt.base_prompt is empty")
        if not ps.video_prompt.motion_description.strip():
            reasons.append(f"prompt_sets[{i}].video_prompt.motion_description is empty")

    return reasons


def _is_valid(state: PromptGenState) -> bool:
    return not _missing_reasons(state)


def validate_node(state: PromptGenState) -> PromptGenState:
    """
    Sanity-check the generation. Same low-bar philosophy as Agents 1 and 2 -
    catching outright failures (missing themes, empty prompt fields), not
    grading prompt quality; that judgment belongs to the Review/Critic agent
    once it can actually look at what Agent 4 generates from these prompts.

    Writes a specific failure reason the moment validation fails (not just
    once retries are exhausted), following Creative Strategy's fixed pattern
    rather than Agent 1's older, weaker one - so previous_error actually has
    something useful to feed into the next attempt's prompt.
    """
    if _is_valid(state):
        return {**state, "error": None}

    reason = "; ".join(_missing_reasons(state))

    retries = state.get("retries", 0)
    if retries >= settings.max_prompt_gen_retries:
        return {**state, "error": f"{reason} (gave up after {retries} retries)"}

    return {**state, "error": reason}


def bump_retry_node(state: PromptGenState) -> PromptGenState:
    """Increments the retry counter, then loops back to generate_node."""
    return {**state, "retries": state.get("retries", 0) + 1}


def route_after_validation(state: PromptGenState) -> str:
    """Conditional edge: retry generation, or stop."""
    if _is_valid(state):
        return "done"
    if state.get("retries", 0) >= settings.max_prompt_gen_retries:
        return "done"  # give up, error is already set on state by validate_node
    return "retry"