"""
Node functions for the Video Generation Agent.

Retry loop and fresh-seed-per-retry decision mirror Agent 4's nodes.py
exactly, same reasoning: a deterministic sampler means a same-seed retry
only helps for pure infrastructure flakiness, not a real generation
failure, so a fresh seed is drawn on every retry.

One new case Agent 4 never had to handle: a theme can arrive here with NO
usable Agent 4 image at all, if that theme exhausted ITS retries upstream
and was skipped (Image_generation.md Challenge 3). That's not a
generation failure - retrying a video job with no source image can't
possibly succeed - so it's handled as an immediate, retry-free "skip" in
generate_node itself, not left to burn through max_video_gen_retries
attempts for no reason.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

from app.core.config import settings
from app.agents.image_generation.utils import slugify_url, new_run_id
from app.agents.image_generation.schema import ImageGenerationOutput, ThemeGenerationResult
from app.agents.video_generation.video_comfyui_client import VideoComfyUIClient, ComfyUIError
from app.agents.video_generation.schema import GeneratedVideo, ThemeVideoResult, VideoGenerationOutput
from app.agents.video_generation.state import VideoGenerationState

_client = VideoComfyUIClient()


def _current_theme(state: VideoGenerationState):
    prompt_sets = state["prompts"].prompt_sets
    return prompt_sets[state.get("current_theme_index", 0)]


def _find_source_image_theme(state: VideoGenerationState, source_setting: str) -> ThemeGenerationResult | None:
    images: ImageGenerationOutput = state["images"]
    for theme_result in images.theme_results:
        if theme_result.source_setting == source_setting:
            return theme_result
    return None  # that theme was skipped entirely by Agent 4 (retries exhausted there)


def _pick_source_image(theme_result: ThemeGenerationResult):
    """
    PLACEHOLDER SELECTION - see schema.py's module docstring. Agent 4
    keeps every candidate because no critic exists yet to pick a winner;
    this just takes the first one. Revisit once Agent 6 (Review/Critic)
    can actually judge which candidate is worth animating.
    """
    return theme_result.images[0] if theme_result.images else None


def start_node(state: VideoGenerationState) -> dict:
    """
    Runs once: loads the video workflow template (fails loudly here if
    it's the wrong JSON format - see VideoComfyUIClient.
    load_video_workflow_template - rather than after wasting a generation
    attempt), and generates a run_id following Agent 4's exact convention
    (own scoped output folder, so repeated runs against the same product
    don't pile into the same directory).
    """
    template = _client.load_video_workflow_template()
    return {
        "current_theme_index": 0,
        "completed_theme_results": [],
        "retries": 0,
        "error": None,
        "run_id": new_run_id(),
        "video_workflow_template": template,
    }


def _save_video_locally(
    *, source_url: str, run_id: str, theme_index: int, file_info: dict
) -> tuple[str, str]:
    """
    Same run-scoped save pattern as Agent 4's _save_images_locally:
    <video_output_dir>/<slugified-product>/<run_id>/theme_<n>/<filename>.
    Returns (local_path, filename). Only ever one file per theme (see
    schema.py - no batching), unlike Agent 4's loop over N images.
    """
    product_dir = Path(settings.video_output_dir) / slugify_url(source_url) / run_id / f"theme_{theme_index}"
    product_dir.mkdir(parents=True, exist_ok=True)

    content = _client.fetch_image_bytes(file_info["filename"], file_info["subfolder"])
    local_path = product_dir / file_info["filename"]
    local_path.write_bytes(content)
    return str(local_path), file_info["filename"]


def generate_node(state: VideoGenerationState) -> dict:
    theme = _current_theme(state)
    theme_index = state.get("current_theme_index", 0)

    source_theme = _find_source_image_theme(state, theme.source_setting)
    source_image = _pick_source_image(source_theme) if source_theme else None

    if source_image is None:
        # No retry-worthy failure here - there's nothing generation could
        # succeed at. Skip immediately, don't touch retries.
        result = ThemeVideoResult(
            source_setting=theme.source_setting,
            video=None,
            generation_time_seconds=0.0,
            retries_used=0,
            status="skipped_no_source_image",
        )
        return {"error": None, "_pending_result": result}

    seed = state.get("current_seed") or random.randint(0, 2**32 - 1)
    # CogVideoX's ComfyUI node takes a single "prompt" string - Agent 3's
    # schema deliberately keeps base_prompt (scene) and motion_description
    # (camera/subject motion) as separate fields for clarity/editability,
    # but they're combined here since there's only one text input to feed.
    combined_prompt = f"{theme.video_prompt.base_prompt} {theme.video_prompt.motion_description}"

    # Defensive, not just reactive: clears any job left running from a
    # prior attempt that timed out client-side (see VideoComfyUIClient.
    # interrupt's docstring) before spending time trying to talk to a
    # server that may still be blocked on it. Cheap and a no-op if
    # ComfyUI is already idle.
    _client.interrupt()

    start = time.monotonic()
    try:
        uploaded_filename = _client.upload_image(source_image.local_path)
        workflow = _client.build_video_workflow(
            state["video_workflow_template"],
            image_filename=uploaded_filename,
            positive_prompt=combined_prompt,
            negative_prompt=theme.video_prompt.negative_prompt,
            seed=seed,
            filename_prefix=f"agent5_{theme_index}",
        )
        prompt_id = _client.queue_prompt(workflow)
        history_entry = _client.wait_for_completion(prompt_id)
        raw_outputs = _client.extract_video_outputs(history_entry)
        if not raw_outputs:
            raise ComfyUIError("Video job finished but no output files were found.")

        local_path, filename = _save_video_locally(
            source_url=state["prompts"].source_url,
            run_id=state["run_id"],
            theme_index=theme_index,
            file_info=raw_outputs[0],  # exactly one video per theme, no batching
        )
    except ComfyUIError as exc:
        return {"error": str(exc), "current_seed": None}
    except OSError as exc:
        return {"error": f"Failed saving video locally: {exc}", "current_seed": None}

    elapsed = time.monotonic() - start

    video = GeneratedVideo(
        filename=filename,
        subfolder=raw_outputs[0].get("subfolder", ""),
        local_path=local_path,
        seed=seed,
        source_image_local_path=source_image.local_path,
    )
    result = ThemeVideoResult(
        source_setting=theme.source_setting,
        video=video,
        generation_time_seconds=elapsed,
        retries_used=state.get("retries", 0),
        status="success",
    )
    return {"error": None, "_pending_result": result}


def validate_node(state: VideoGenerationState) -> dict:
    """
    Low bar, same as every other agent's validate_node: did generate_node
    produce SOMETHING (a success or an explicit skip), not whether the
    video is any good - that's Agent 6's job. Unlike Agent 4, there's no
    "expected count" to check against, since it's always exactly one video
    or an explicit skip - the binary presence of a valid _pending_result
    IS the check.
    """
    if state.get("error"):
        return {}  # already has a failure reason from generate_node

    if not state.get("_pending_result"):
        return {"error": "generate_node produced neither a result nor an error - this shouldn't happen."}

    return {}


def bump_retry_node(state: VideoGenerationState) -> dict:
    return {"retries": state.get("retries", 0) + 1}


def advance_theme_node(state: VideoGenerationState) -> dict:
    """
    On success or explicit skip: files the result, resets per-attempt
    scratch state, moves to the next theme. On retries-exhausted failure:
    pending is None here (generate_node's except-branches never set
    _pending_result), so nothing gets appended - same as Agent 4, the
    theme is simply absent from theme_results rather than recorded as a
    distinct "failed" entry. See schema.py's module docstring for why this
    asymmetry (skips ARE recorded, retry-exhausted failures aren't) is
    intentional, not an inconsistency.
    """
    completed = list(state.get("completed_theme_results", []))
    pending = state.get("_pending_result")
    if pending:
        completed.append(pending)

    return {
        "completed_theme_results": completed,
        "current_theme_index": state.get("current_theme_index", 0) + 1,
        "retries": 0,
        "error": None,
        "current_seed": None,
        "_pending_result": None,
    }


def finalize_node(state: VideoGenerationState) -> dict:
    output = VideoGenerationOutput(
        source_url=state["prompts"].source_url,
        theme_results=state.get("completed_theme_results", []),
    )
    return {"videos": output}