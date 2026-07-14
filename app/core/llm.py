"""
Thin wrappers around model APIs that coerce a model's response into a
Pydantic schema. Keeping these generic (not agent-specific) so any agent can
reuse them with its own model and schema.

Two separate entry points, not one function with a provider branch inside:
Ollama and Groq need genuinely different workarounds (see each function's
docstring), and folding both into one function with an if/else would hide
that they're solving different problems rather than sharing one mechanism.
"""

from __future__ import annotations

import json
import re
from typing import Type, TypeVar

import ollama
from groq import AsyncGroq
from pydantic import BaseModel

from app.core.config import settings

T = TypeVar("T", bound=BaseModel)


class LLMExtractionError(Exception):
    """Raised when the model output can't be parsed/validated against the schema."""


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _example_from_json_schema(node: dict, defs: dict) -> object:
    """
    Recursively build one plausible placeholder value for a JSON-schema
    node, resolving $ref/$defs (nested Pydantic sub-models) and
    anyOf/Optional wrapping along the way. Used to build a flat, filled
    *instance* of a schema rather than handing a smaller model the raw
    schema definition - see _build_example_json's docstring for why that
    distinction actually matters here, not just cosmetically.
    """
    if "$ref" in node:
        ref_name = node["$ref"].split("/")[-1]
        return _example_from_json_schema(defs[ref_name], defs)

    if "anyOf" in node:
        for option in node["anyOf"]:
            if option.get("type") != "null":
                return _example_from_json_schema(option, defs)
        return None

    node_type = node.get("type")

    if node_type == "object":
        if "properties" in node:
            return {
                name: _example_from_json_schema(prop, defs)
                for name, prop in node["properties"].items()
            }
        if isinstance(node.get("additionalProperties"), dict):
            # dict[str, X] fields (e.g. `specifications: dict[str, str]`)
            return {"example_key": _example_from_json_schema(node["additionalProperties"], defs)}
        return {}

    if node_type == "array":
        return [_example_from_json_schema(node.get("items", {}), defs)]

    if node_type == "string":
        hint = node.get("description") or node.get("title") or "value"
        return f"<{hint[:60]}>"

    if node_type == "integer":
        return 1

    if node_type == "number":
        return 1.0

    if node_type == "boolean":
        return True

    return None


def _build_example_json(schema: Type[BaseModel]) -> str:
    """
    Build a filled-shape example instance from a Pydantic schema, instead of
    handing the model the raw JSON Schema. qwen3.5 (and likely other small
    models) sometimes echo the schema *definition* back verbatim -
    `properties`/`type`/`description` keys with real values nested inside
    `properties` - rather than returning a flat filled instance, because to a
    smaller model the schema definition and a filled instance look
    structurally similar. Showing a concrete example with placeholder values
    (not the schema metadata) removes that ambiguity entirely: there's no
    `properties` key for the model to imitate.

    Generic on purpose - this is used for every schema across every agent,
    not hand-written per schema, so a new agent gets this fix for free.
    """
    schema_dict = schema.model_json_schema()
    defs = schema_dict.get("$defs", {})
    example = _example_from_json_schema(schema_dict, defs)
    return json.dumps(example, indent=2)


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
        emit the final JSON, returning EMPTY content. This is exactly the
        `Invalid JSON: EOF while parsing a value` / empty raw output failure.
      - think=False + format=schema -> the format constraint is silently
        IGNORED and the model returns free-form text instead of JSON.
    Net effect: native `format` isn't reliable on this model family right
    now. Workaround: disable thinking, drop `format`, describe the schema in
    the prompt instead, and parse the response ourselves with a tolerant
    extractor. Revisit once the upstream bugs are fixed - `format` is the
    more robust mechanism when it actually works.

    Also sets num_ctx explicitly: Ollama's runtime default context window
    (2048-4096 depending on version) is unrelated to a model's advertised
    max context, and research prompts here (up to ~20k chars of page text +
    8k of review text) will silently get truncated without this.

    Shows a filled-shape EXAMPLE instance in the prompt, not the raw JSON
    Schema - qwen3.5 (and likely other small models) will otherwise
    sometimes echo the schema definition back verbatim (`properties`/`type`/
    `description` keys, real values nested inside `properties`) instead of
    returning a flat filled instance, since the two look structurally
    similar to a smaller model. See _build_example_json's docstring.
    """
    client = ollama.AsyncClient(host=settings.ollama_host)

    schema_hint = (
        "\n\nRespond with ONLY a single JSON object - no markdown code fences, "
        "no commentary before or after it. Here is the exact shape to return, "
        "with placeholder values showing what kind of content belongs in each "
        "field - replace every placeholder with real content, and do not "
        "return this structure with the placeholders still in it, and do not "
        "return a JSON Schema definition (no 'properties'/'type' keys):\n"
        f"{_build_example_json(schema)}"
    )

    response = await client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt + schema_hint},
            {"role": "user", "content": user_prompt},
        ],
        think=False,
        keep_alive="30m",
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


async def structured_chat_groq(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    schema: Type[T],
    temperature: float = 0.2,
) -> T:
    """
    Call a Groq-hosted model and coerce its response into `schema`.

    Unlike Ollama+qwen3.5 (see structured_chat's docstring), Groq's
    OpenAI-compatible API supports native JSON-schema-constrained decoding
    without the thinking-trace/format collision bug - so this path uses
    response_format directly: the schema is enforced server-side rather than
    just requested in text and parsed tolerantly afterward.

    NOTE: verify against Groq's current docs which models have "strict"
    json_schema support before relying on this for a new model - schema
    enforcement strength has varied by model even within one provider.
    """
    client = AsyncGroq(api_key=settings.groq_api_key)

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=settings.groq_max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": schema.model_json_schema(),
                },
            },
        )
    except Exception as exc:  # noqa: BLE001 - network/API errors also need to trigger fallback in nodes.py
        raise LLMExtractionError(f"Groq API call failed for model {model!r}: {exc}") from exc

    raw_content = response.choices[0].message.content or ""

    try:
        parsed = json.loads(raw_content)
        return schema.model_validate(parsed)
    except Exception as exc:  # noqa: BLE001
        raise LLMExtractionError(
            f"Groq output failed schema validation for model {model!r}: {exc}\nRaw output: {raw_content[:500]!r}"
        ) from exc