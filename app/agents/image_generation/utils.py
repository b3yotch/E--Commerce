"""Turns a product source_url into a filesystem-safe folder name, splits a
total image count across themes, and generates a per-run identifier."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from urllib.parse import urlparse


def slugify_url(url: str) -> str:
    """
    'https://bananaclub.co.in/products/black-turtle-neck-sweater' ->
    'bananaclub.co.in_black-turtle-neck-sweater'

    Keeps the domain (disambiguates products with the same slug across
    different stores) and the last path segment (the actual product slug),
    rather than hashing the URL - a human should be able to glance at the
    output folder name and know which product it is.
    """
    parsed = urlparse(url)
    domain = parsed.netloc
    last_segment = [p for p in parsed.path.split("/") if p][-1] if parsed.path.strip("/") else "product"
    combined = f"{domain}_{last_segment}"
    return re.sub(r"[^a-zA-Z0-9._-]", "-", combined)


def distribute_total(total: int, num_buckets: int) -> list[int]:
    """
    Splits `total` as evenly as possible across `num_buckets`, with any
    remainder going to the first buckets - distribute_total(5, 2) -> [3, 2],
    distribute_total(5, 3) -> [2, 2, 1]. Used to turn "5 images total for
    this product" into a per-theme batch_size, since the number of themes
    (typically 2-3, per Agent 2's visual_themes) varies per product and
    isn't known until Agent 3's output arrives - can't be a fixed
    per-theme constant.
    """
    if num_buckets <= 0:
        return []
    base, remainder = divmod(total, num_buckets)
    return [base + 1 if i < remainder else base for i in range(num_buckets)]


def new_run_id() -> str:
    """
    One of these generated per pipeline invocation (in start_node), used to
    scope each run's output folder. Without this, re-running the same
    product URL writes into the exact same theme_0/theme_1 folders every
    time - harmless for ComfyUI's own uniquely-numbered filenames, but
    images from unrelated runs pile up together with no way to tell which
    run produced which batch, which is exactly what happened when the old
    script ran twice against the same URL before the new one ran once.
    """
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")