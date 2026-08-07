"""
Thin wrapper around ComfyUI's HTTP API.

Image support (build_image_workflow, extract_images) is the original,
already-tested code from Agent 4 - unchanged here.

Video support (build_video_workflow, extract_video_outputs, upload_image)
is new for Agent 5. It reuses queue_prompt/wait_for_completion/
fetch_image_bytes as-is rather than duplicating them, since the
queue/poll/error-detection shape is generic to ComfyUI's API and doesn't
change based on which workflow graph is running - only the workflow
construction and output-key handling differ between image and video (see
Video_generation.md Challenge 1/2 for why the graphs themselves can't be
shared).

One real difference required a signature change rather than a clean reuse:
wait_for_completion originally hardcoded a check for an "images" key on
each node's output. That's correct for the image workflow's SaveImage
node, but the video workflow's save node's output key is unknown ahead of
time - it depends on the custom node package, not on anything this project
controls (confirmed via hand-testing: it isn't "images"). Rather than
guess a name, wait_for_completion now takes an optional expected_output_key
- image callers keep the specific "images" check (no behavior change),
video callers pass None to fall back to "any outputs at all", the same
generic check the hand-test script used successfully.
"""

from __future__ import annotations

import copy
import json
import time
import uuid
from pathlib import Path

import requests

from app.core.config import settings


class ComfyUIError(Exception):
    """Raised for any ComfyUI-side failure: bad checkpoint, node error, or timeout."""


class ComfyUIClient:
    def __init__(self, server: str | None = None):
        self.server = server or settings.comfyui_server
        self.client_id = str(uuid.uuid4())

    def check_checkpoint(self, checkpoint: str) -> None:
        r = requests.get(f"{self.server}/object_info/CheckpointLoaderSimple", timeout=10)
        r.raise_for_status()
        checkpoints = r.json()["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
        if checkpoint not in checkpoints:
            raise ComfyUIError(
                f"Checkpoint {checkpoint!r} not found on ComfyUI server. "
                f"Available: {checkpoints}"
            )

    # -----------------------------------------------------------------
    # Image (Agent 4 - unchanged)
    # -----------------------------------------------------------------

    def build_image_workflow(
        self,
        *,
        positive_prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        seed: int,
        batch_size: int,
        checkpoint: str | None = None,
        steps: int | None = None,
        cfg: float | None = None,
        sampler: str | None = None,
        scheduler: str | None = None,
        filename_prefix: str = "agent4",
    ) -> dict:
        return {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": checkpoint or settings.comfyui_checkpoint},
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": positive_prompt, "clip": ["1", 1]},
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": negative_prompt, "clip": ["1", 1]},
            },
            "4": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": width, "height": height, "batch_size": batch_size},
            },
            "5": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed,
                    "steps": steps or settings.comfyui_steps,
                    "cfg": cfg or settings.comfyui_cfg,
                    "sampler_name": sampler or settings.comfyui_sampler,
                    "scheduler": scheduler or settings.comfyui_scheduler,
                    "denoise": 1.0,
                    "model": ["1", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "latent_image": ["4", 0],
                },
            },
            "6": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
            },
            "7": {
                "class_type": "SaveImage",
                "inputs": {"filename_prefix": filename_prefix, "images": ["6", 0]},
            },
        }

    @staticmethod
    def extract_images(history_entry: dict) -> list[dict]:
        """Returns a flat list of {filename, subfolder} dicts from a completed history entry."""
        images = []
        for node_output in history_entry.get("outputs", {}).values():
            for image in node_output.get("images", []):
                images.append({"filename": image["filename"], "subfolder": image.get("subfolder", "")})
        return images

    # -----------------------------------------------------------------
    # Video (Agent 5 - new)
    # -----------------------------------------------------------------

    def upload_image(self, image_path: str) -> str:
        """
        POSTs a local file to /upload/image, placing it in ComfyUI's own
        input/ directory and returning the server-side filename to
        reference in a LoadImage-style node. Needed for img2vid because
        the source frame (an Agent 4 output, already on disk locally) has
        to round-trip back through ComfyUI before the video workflow's
        image-loading node can reference it - Agent 4 never needed this
        method since text-to-image has no image input.
        """
        path = Path(image_path)
        with open(path, "rb") as f:
            files = {"image": (path.name, f, "image/png")}
            r = requests.post(f"{self.server}/upload/image", files=files, timeout=30)
        r.raise_for_status()
        return r.json()["name"]

    def load_video_workflow_template(self, path: str | None = None) -> dict:
        """
        Loads the API-format workflow JSON once (call from start_node, not
        per-theme - the template itself doesn't change across themes,
        only the per-call patches in build_video_workflow do).

        Guards against the single most likely setup mistake: pointing this
        at ComfyUI's *UI*-format export (nodes/links graph for the canvas)
        instead of the *API*-format export /prompt actually needs (flat
        {node_id: {class_type, inputs}}). This distinction cost real
        debugging time during hand-testing - worth failing loudly and
        immediately here rather than letting it surface as a confusing
        422 from ComfyUI later.
        """
        workflow_path = Path(path or settings.comfyui_video_workflow_json)
        if not workflow_path.exists():
            raise ComfyUIError(f"Video workflow JSON not found: {workflow_path}")
        with open(workflow_path) as f:
            workflow = json.load(f)
        if "nodes" in workflow and "links" in workflow:
            raise ComfyUIError(
                f"{workflow_path} is a UI-format ComfyUI workflow (has "
                "top-level 'nodes'/'links' keys), not the API-format "
                "export the /prompt endpoint needs. Re-export it via "
                "ComfyUI's 'Save (API Format)' option (requires Dev Mode "
                "enabled in Settings)."
            )
        return workflow

    def build_video_workflow(
        self,
        template: dict,
        *,
        image_filename: str,
        positive_prompt: str,
        negative_prompt: str,
        seed: int,
        filename_prefix: str,
    ) -> dict:
        """
        Patches a copy of the loaded template with per-call values. Unlike
        build_image_workflow (which constructs the whole graph from
        scratch, since it's just core nodes), this only patches specific
        node IDs inside an externally-authored graph - the node IDs
        themselves come from Settings (comfyui_video_*_node_id), pinned to
        the specific workflow export that was hand-tested working. See the
        comment on those Settings fields for what breaks this pinning.

        Always deep-copies the template so repeated calls (one per theme)
        never mutate shared state between themes - a mistake that would be
        easy to make silently, since dict mutation wouldn't raise anything,
        it would just corrupt the next theme's job with this one's values.
        """
        workflow = copy.deepcopy(template)

        img_node = settings.comfyui_video_image_load_node_id
        pos_node = settings.comfyui_video_positive_prompt_node_id
        neg_node = settings.comfyui_video_negative_prompt_node_id
        sampler_node = settings.comfyui_video_sampler_node_id
        save_node = settings.comfyui_video_save_node_id

        for node_id in (img_node, pos_node, neg_node, sampler_node, save_node):
            if node_id not in workflow:
                raise ComfyUIError(
                    f"Configured node id {node_id!r} not found in the video "
                    f"workflow template. Available node ids: {list(workflow.keys())}"
                )

        workflow[img_node]["inputs"]["image"] = image_filename
        workflow[pos_node]["inputs"]["prompt"] = positive_prompt
        workflow[neg_node]["inputs"]["prompt"] = (
            negative_prompt or settings.comfyui_video_default_negative_prompt
        )

        sampler_inputs = workflow[sampler_node]["inputs"]
        sampler_inputs["seed"] = seed
        sampler_inputs["num_frames"] = settings.comfyui_video_num_frames
        sampler_inputs["steps"] = settings.comfyui_video_steps
        sampler_inputs["cfg"] = settings.comfyui_video_cfg
        sampler_inputs["scheduler"] = settings.comfyui_video_scheduler
        sampler_inputs["denoise_strength"] = settings.comfyui_video_denoise_strength

        save_inputs = workflow[save_node]["inputs"]
        save_inputs["filename_prefix"] = filename_prefix
        save_inputs["save_output"] = True

        return workflow

    @staticmethod
    def extract_video_outputs(history_entry: dict) -> list[dict]:
        """
        Deliberately generic, unlike extract_images: the video save node's
        output key isn't "images" (confirmed during hand-testing) and
        isn't guaranteed to be any particular name across custom node
        packages. Scans every node's outputs for anything shaped like a
        file reference ({filename: ..., subfolder: ..., type: ...})
        instead of assuming a key name - the same approach the hand-test
        script used successfully before this became agent code.
        """
        found = []
        for node_id, node_output in history_entry.get("outputs", {}).items():
            for key, value in node_output.items():
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict) and "filename" in item:
                            found.append({
                                "node_id": node_id,
                                "output_key": key,
                                "filename": item["filename"],
                                "subfolder": item.get("subfolder", ""),
                                "type": item.get("type", "output"),
                            })
        return found

    # -----------------------------------------------------------------
    # Shared (queue/poll/fetch) - used by both image and video
    # -----------------------------------------------------------------

    def queue_prompt(self, workflow: dict) -> str:
        payload = {"prompt": workflow, "client_id": self.client_id}
        try:
            r = requests.post(f"{self.server}/prompt", json=payload, timeout=10)
            r.raise_for_status()
        except requests.RequestException as exc:
            raise ComfyUIError(f"Failed to queue prompt: {exc}") from exc

        body = r.json()
        if "error" in body:
            raise ComfyUIError(f"ComfyUI rejected the prompt: {body['error']}")
        return body["prompt_id"]

    def wait_for_completion(
        self,
        prompt_id: str,
        timeout_seconds: float | None = None,
        expected_output_key: str | None = "images",
    ) -> dict:
        """
        Polls /history until the job appears, an error surfaces, or
        timeout_seconds elapses. Raises ComfyUIError on timeout or a
        reported node failure.

        expected_output_key: for image jobs (default "images"), a node's
        output must contain that exact key to count as done - this is the
        original Agent 4 behavior, unchanged. Pass None (as video callers
        do) to instead accept ANY non-empty outputs dict as done, since
        the video save node's output key isn't known/stable ahead of time
        - see extract_video_outputs' docstring for the same reasoning.
        """
        timeout_seconds = timeout_seconds or settings.comfyui_generation_timeout_seconds
        deadline = time.monotonic() + timeout_seconds
        poll_interval = 1.0

        while time.monotonic() < deadline:
            r = requests.get(f"{self.server}/history/{prompt_id}", timeout=10)
            r.raise_for_status()
            history = r.json()

            if prompt_id in history:
                entry = history[prompt_id]
                status = entry.get("status", {})

                if status.get("status_str") == "error":
                    raise ComfyUIError(f"ComfyUI reported an error for prompt {prompt_id}: {status}")

                outputs = entry.get("outputs", {})
                if expected_output_key is not None:
                    has_output = any(expected_output_key in node_out for node_out in outputs.values())
                else:
                    has_output = bool(outputs)

                if has_output:
                    return entry

                raise ComfyUIError(
                    f"Prompt {prompt_id} finished with no matching output. Status: {status}"
                )

            time.sleep(poll_interval)

        raise ComfyUIError(f"Timed out after {timeout_seconds}s waiting for prompt {prompt_id}")

    def fetch_image_bytes(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        """
        Pulls raw bytes back from ComfyUI's /view endpoint. Despite the
        name (kept from Agent 4 for continuity), this works for any file
        ComfyUI serves through /view - video files included, since /view
        is a generic file server keyed on filename/subfolder/type, not an
        image-specific endpoint. Video callers use this exactly as-is.
        """
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        r = requests.get(f"{self.server}/view", params=params, timeout=30)
        r.raise_for_status()
        return r.content