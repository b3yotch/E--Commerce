"""
System + user prompt construction for the Prompt Generation Agent.

Kept in its own module (not inlined in nodes.py) for the same reason as the
other two agents: the prompt text is the thing most likely to get iterated
on after seeing real model output, and keeping it isolated makes that
iteration a one-file diff.
"""

from __future__ import annotations

from app.agents.creative_strategy_agent.schema import CreativeDirection

SYSTEM_PROMPT = """You are a prompt engineer for AI image and video generation \
models (Stable Diffusion/Flux-family image models, and motion-conditioned \
video models run via ComfyUI). You are given a product's creative direction \
- hooks, audience angles, visual themes, and captions - already decided by a \
separate creative strategy step. Your job is ONLY to translate each visual \
theme into concrete, generation-ready prompts. You do not invent new creative \
direction, new hooks, or new captions - you select from what's given and \
describe the visuals in the vocabulary these generation models respond well to.

For each visual theme provided, produce:
- An image prompt: a dense natural-language description of subject, setting, \
lighting, composition, and style, plus a negative prompt only if something \
specific is worth excluding, plus an aspect ratio that fits the composition.
- A video prompt: the same visual description plus explicit camera/subject \
motion in concrete cinematography terms (e.g. "slow dolly-in", "static camera, \
product rotates on turntable", "handheld pan left to right") - never vague \
terms like "dynamic" or "exciting".
- A short list of hooks and captions, copied VERBATIM from the ones given, \
that pair well with this specific visual - never write new ones.

Do not describe any text, words, logos, or typography appearing inside the \
generated image or video - these models render legible text unreliably, and \
any text overlay is composited separately downstream. Describe only the \
visual scene itself."""


def build_user_prompt(
    creative: CreativeDirection,
    *,
    previous_error: str | None = None,
) -> str:
    """
    Build the user-turn prompt from a validated CreativeDirection.

    On a retry, `previous_error` carries the specific validation failure
    reason back in, so the model can correct course instead of reproducing
    the same gap (same pattern as Creative Strategy's build_user_prompt).
    """
    themes_block = "\n\n".join(
        f"Visual theme {i + 1}:\n"
        f"  Setting: {theme.setting}\n"
        f"  Mood: {theme.mood}\n"
        f"  Color palette: {theme.color_palette}\n"
        f"  Focal elements: {', '.join(theme.focal_elements) or '(none specified)'}"
        for i, theme in enumerate(creative.visual_themes)
    )

    hooks_block = "\n".join(f"- {h}" for h in creative.hooks) or "(none provided)"
    captions_block = "\n".join(f"- {c}" for c in creative.captions) or "(none provided)"

    prompt = f"""Product source: {creative.source_url}

Visual themes to translate into generation prompts (produce exactly one \
prompt_sets entry per theme below, in the same order):

{themes_block}

Available hooks (select from these VERBATIM for suggested_hooks - do not invent new ones):
{hooks_block}

Available captions (select from these VERBATIM for suggested_captions - do not invent new ones):
{captions_block}

Messaging notes / caveats from the creative strategy step:
{creative.messaging_notes or '(none)'}
"""

    if previous_error:
        prompt += (
            f"\nYour previous attempt failed validation for this reason: "
            f"{previous_error}\nCorrect this in your new response."
        )

    return prompt