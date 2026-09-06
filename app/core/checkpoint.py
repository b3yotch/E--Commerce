"""
Lightweight per-stage checkpointing for the pipeline test script.

Deliberately just flat JSON files under outputs/checkpoints/<slug>/, not a
database - this project is single-user, single-machine, and the existing
--save output already proved flat JSON is enough to hold a stage's full
Pydantic output. Checkpointing reuses that same {"data": ..., **metadata}
shape rather than inventing a second persistence format alongside it.

One file per stage, keyed by product URL only - no run_id in the path.
Two consequences of that choice, both deliberate:

- Re-running the same URL always resumes against the latest checkpoint for
  each stage, rather than needing to know which run_id to resume - there's
  only ever one "current" checkpoint per stage per product.
- Overwriting is the only way this works, and IS the intended behavior:
  once you deliberately re-run a stage (e.g. after the rate-limit fix and
  rerunning Agent 5), its checkpoint is simply replaced, not versioned -
  the point of checkpointing is "what did we last successfully produce,"
  not a history of every attempt.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Type, TypeVar

from pydantic import BaseModel

from app.agents.image_generation.utils import slugify_url

T = TypeVar("T", bound=BaseModel)

CHECKPOINT_ROOT = Path("outputs/checkpoints")


def _checkpoint_path(url: str, stage: str) -> Path:
    return CHECKPOINT_ROOT / slugify_url(url) / f"{stage}.json"


def save_stage(url: str, stage: str, data: BaseModel, **metadata) -> None:
    """
    Writes a stage's output plus any small bits of metadata worth keeping
    (elapsed_seconds, retries, model_used - whatever the caller has handy)
    in the same {"data": ..., **metadata} shape the existing --save summary
    already uses, so both paths stay readable the same way.
    """
    path = _checkpoint_path(url, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**metadata, "data": json.loads(data.model_dump_json())}
    path.write_text(json.dumps(payload, indent=2))


def load_stage(url: str, stage: str, schema: Type[T]) -> tuple[T, dict] | None:
    """
    Returns (validated_model, metadata_dict) if a checkpoint exists for
    this URL/stage, else None. metadata_dict is whatever extra kwargs were
    passed to save_stage (elapsed_seconds, retries, model_used, ...) - if
    the caller only cares about the model, it can just take [0].
    """
    path = _checkpoint_path(url, stage)
    if not path.exists():
        return None

    payload = json.loads(path.read_text())
    data = payload.pop("data")
    return schema.model_validate(data), payload


def clear_checkpoints(url: str) -> None:
    """Deletes every checkpoint for a URL - used by --clear-checkpoints to force a fully clean slate."""
    product_dir = CHECKPOINT_ROOT / slugify_url(url)
    if product_dir.exists():
        for f in product_dir.glob("*.json"):
            f.unlink()