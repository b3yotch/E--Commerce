"""
Thin wrapper around Ollama's chat API that coerces a model's response into a
Pydantic schema. Keeping this generic (not research-agent-specific) so any
later agent (creative strategy, critic, etc.) can reuse it with its own model
and schema.
"""

from __future__ import annotations

import json
import re
import typing
from typing import Type, TypeVar

import ollama
from pydantic import BaseModel

from app.core.config import settings

T = TypeVar("T", bound=BaseModel)


class LLMExtractionError(Exception):
    """Raised when the model output can't be parsed/validated against the schema."""


def _placeholder_for_annotation(annotation: object) -> object:
    """Pick an illustrative placeholder value for a field's Python type, so
    the example instance is valid-shaped JSON without needing real data."""
    origin = typing.get_origin(annotation)

    if origin is typing.Union:
        non_none = [a for a in typing.get_args(annotation) if a is not type(None)]
        return _placeholder_for_annotation(non_none[0]) if non_none else None
    if origin in (list, set, tuple):
        return []
    if origin is dict:
        return {}
    if annotation is str:
        return "..."
    if annotation is int:
        return 0
    if annotation is float:
        return 0.0
    if annotation is bool:
        return False
    return "..."


def _example_instance(schema: Type[BaseModel]) -> dict:
    """
    Build a placeholder JSON *instance* of `schema` - not the schema
    definition itself. Smaller models given a raw JSON Schema (with its own
    `properties`/`type`/`description` meta-keys) can mistake "match this
    schema" for "return this schema" - it looks structurally like a valid
    JSON object with a similar vocabulary. A filled-shape example removes
    that ambiguity entirely.
    """
    return {name: _placeholder_for_annotation(field.annotation) for name, field in schema.model_fields.items()}


def _field_descriptions(schema: Type[BaseModel]) -> str:
    lines = []
    for name, field in schema.model_fields.items():
        lines.append(f"- {name}: {field.description or '(no extra notes)'}")
    return "\n".join(lines)


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json_object(text: str) -> str:
    """
    Best-effort extraction of a single JSON object from model output that may
    still contain stray commentary or leftover thinking-trace fragments
    around it, even with think disabled. Finds the first '{' and its
    brace-matched closing '}' rather than trusting the whole string is
    clean JSON.
    """
    text = _strip_code_fences(text)
    start = text.find("{")
    if start == -1:
        return text  # nothing to salvage - let json.loads raise a clear error

    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]  # unbalanced - let json.loads raise a clear error


async def structured_chat(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    schema: Type[T],
    temperature: float = 0.2,
) -> T:
    """
    Call an Ollama model and coerce its response into `schema`.

    Why this doesn't just use Ollama's native `format=schema` structured
    output: qwen3.5 (a thinking-capable model) has two stacking bugs in
    Ollama as of mid-2026 (ollama/ollama#10929, #14645):
      - think left unset (defaults to True) + format=schema -> the model can
        burn its entire output budget on the hidden thinking trace and never
        emit the final JSON, returning EMPTY content.
      - think=False + format=schema -> the format constraint is silently
        IGNORED and the model returns free-form text instead of JSON.
    Net effect: native `format` isn't reliable on this model family right
    now. Workaround: disable thinking, drop `format`, and instruct the model
    with a filled-shape example (not the raw JSON Schema - see
    _example_instance's docstring for why that distinction matters), then
    parse the response ourselves with a tolerant extractor.

    Also sets num_ctx explicitly: Ollama's runtime default context window
    (2048-4096 depending on version) is unrelated to a model's advertised
    max context, and research prompts here (up to ~20k chars of page text +
    8k of review text) will silently get truncated without this.
    """
    client = ollama.AsyncClient(host=settings.ollama_host)

    example_json = json.dumps(_example_instance(schema), indent=2)
    field_notes = _field_descriptions(schema)

    schema_hint = (
        "\n\nRespond with ONLY a single JSON object - no markdown code fences, "
        "no commentary before or after it. Your response must be a FILLED-IN "
        "INSTANCE containing real data extracted from the page, in exactly "
        "this shape (the values below are placeholders showing the expected "
        "type, not the schema - do not include keys like 'properties', "
        "'type', or 'description' anywhere in your answer):\n"
        f"{example_json}\n\n"
        f"What each field means:\n{field_notes}"
    )

    response = await client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt + schema_hint},
            {"role": "user", "content": user_prompt},
        ],
        think=False,
        options={
            "temperature": temperature,
            "num_ctx": settings.ollama_num_ctx,
            "num_predict": settings.ollama_num_predict,
        },
    )

    raw_content = response["message"]["content"] or ""

    if not raw_content.strip():
        # Some Ollama versions still leak the answer into the thinking field
        # even with think=False, depending on the model's chat template.
        raw_content = response["message"].get("thinking") or ""

    json_text = _extract_json_object(raw_content)

    try:
        parsed = json.loads(json_text)
        return schema.model_validate(parsed)
    except Exception as exc:  # noqa: BLE001 - we want to wrap any parse/validation error
        raise LLMExtractionError(
            f"Model output failed schema validation: {exc}\nRaw output: {raw_content[:500]!r}"
        ) from exc