"""
Resolves Agent 3's semantic aspect_ratio strings (e.g. "4:5", "9:16") into
concrete pixel dimensions for ComfyUI's EmptyLatentImage node.

This lives in Agent 4, not Agent 3 - see schema.py's docstring for why.
Agent 3 deliberately doesn't know what backend will render its output, so it
hands off a ratio, not dimensions. Agent 4 owns the checkpoint, so it owns
the mapping from ratio -> actual pixel size for that checkpoint's native
resolution.

Values below target ~1 megapixel total (SDXL-family checkpoints, which
JuggernautXL is, are trained around ~1024x1024 / 1MP and degrade in
composition quality well below or above that band) with each dimension
rounded to a multiple of 64, which SDXL's VAE requires cleanly.
"""

from __future__ import annotations

# Curated rather than computed from scratch for every input: these are the
# ratios Agent 3's prompt actually tells the model to choose from (see
# ImageGenerationPrompt's aspect_ratio field description). Precomputed so
# the output dimensions are known-good multiples of 64, rather than trusting
# a generic round-to-64 formula to always land somewhere sane for every
# possible ratio string.
_KNOWN_RATIOS: dict[str, tuple[int, int]] = {
    "1:1": (1024, 1024),
    "4:5": (896, 1152),
    "5:4": (1152, 896),
    "9:16": (768, 1344),
    "16:9": (1344, 768),
    "3:4": (896, 1216),
    "4:3": (1216, 896),
    "2:3": (832, 1216),
    "3:2": (1216, 832),
}

_DEFAULT_RATIO = "1:1"


def resolve_dimensions(aspect_ratio: str) -> tuple[int, int]:
    """
    Returns (width, height) for a given ratio string. Falls back to 1:1
    (logged by the caller, not here - this function stays pure) if Agent 3
    ever emits a ratio outside the curated set, rather than raising and
    failing the whole generation over a cosmetic mismatch.
    """
    normalized = aspect_ratio.strip()
    return _KNOWN_RATIOS.get(normalized, _KNOWN_RATIOS[_DEFAULT_RATIO])


def is_known_ratio(aspect_ratio: str) -> bool:
    return aspect_ratio.strip() in _KNOWN_RATIOS
