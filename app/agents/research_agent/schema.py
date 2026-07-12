"""
Output contract for the Product Research Agent.

Downstream agents (Creative Strategy, Prompt Generation) consume this schema
directly, so it's kept deliberately explicit rather than a free-text blob.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ScrapedProductData(BaseModel):
    """Raw material handed to the LLM extraction step. Not the final output -
    this is what the scraper produces before the model touches it."""

    url: str
    raw_title: str | None = None
    og_image: str | None = None
    json_ld_product: dict | None = None
    visible_text: str = ""
    review_text: str = ""
    review_extraction_method: str = "static"  # "static" | "scroll" | "none" - which
    # strategy actually produced review_text, for observability across bulk runs


class ProductResearch(BaseModel):
    """Final, validated research output for one product page."""

    source_url: str

    title: str = Field(description="The product's name/title as sold.")
    brand: str | None = Field(default=None, description="Brand or manufacturer name, if identifiable.")
    price: str | None = Field(default=None, description="Display price with currency symbol, e.g. '$49.99'. Null if not found.")
    currency: str | None = Field(default=None, description="ISO currency code if determinable, e.g. 'USD'.")

    features: list[str] = Field(
        default_factory=list,
        description="Concrete product features/selling points stated on the page, in the seller's own words where possible.",
    )
    specifications: dict[str, str] = Field(
        default_factory=dict,
        description="Key-value spec sheet entries (material, dimensions, capacity, etc.) as found on the page.",
    )

    @field_validator("specifications", mode="before")
    @classmethod
    def _coerce_specification_values(cls, value: object) -> object:
        """
        Real spec sheets legitimately have multi-valued entries (available
        sizes, available colors) that a model will naturally represent as a
        JSON list rather than a comma-joined string, plus scalar numbers
        (quantity_per_pack: 1) instead of strings. Rather than fighting that
        instinct with an ever-more-detailed prompt, just accept it here and
        normalize to strings so downstream consumers get a consistent
        dict[str, str] without extraction attempts failing over formatting.
        """
        if not isinstance(value, dict):
            return value

        coerced: dict[str, str] = {}
        for key, raw in value.items():
            if isinstance(raw, str):
                coerced[key] = raw
            elif isinstance(raw, bool):
                coerced[key] = "yes" if raw else "no"
            elif isinstance(raw, (list, tuple)):
                coerced[key] = ", ".join(str(item) for item in raw)
            else:
                coerced[key] = str(raw)
        return coerced

    average_rating: float | None = Field(default=None, description="Average review rating out of 5, if shown on the page.")
    review_count: int | None = Field(default=None, description="Number of reviews, if shown on the page.")
    review_summary: str = Field(
        default="",
        description="2-4 sentence synthesis of what reviewers actually say - recurring praise and recurring complaints. Empty string if no review text was available.",
    )

    brand_positioning: str = Field(
        default="",
        description=(
            "Inferred market positioning in 1-2 sentences: who this is for and how it's "
            "pitched (e.g. budget/value, premium/luxury, eco-conscious, performance-focused). "
            "This is the one genuinely interpretive field in this schema - flag low confidence "
            "by saying so explicitly rather than guessing."
        ),
    )

    extraction_confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Self-reported confidence 0-1 that the extracted fields are accurate and grounded in the page content, not invented.",
    )
    missing_fields: list[str] = Field(
        default_factory=list,
        description="Names of fields above that could not be found on the page and were left empty/null.",
    )