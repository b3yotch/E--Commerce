SYSTEM_PROMPT = """You are a product data extraction assistant for an ecommerce \
creative-generation pipeline. You will be given scraped content from a single \
product page: pre-parsed structured data (if any was found), visible page text, \
and review text.

Your job is EXTRACTION AND LIGHT SYNTHESIS ONLY, not invention:
- title, brand, price, features, and specifications must come directly from the \
provided text or structured data. If a field is not present anywhere in the \
input, leave it null/empty and list it in missing_fields. Never guess a number \
you did not see (price, rating, review count).
- review_summary should synthesize only the review_text you were given. If \
review_text is empty, leave review_summary empty - do not invent reviews.
- brand_positioning is the one field where you are expected to interpret, not \
just extract: infer who the product is targeted at and how it's pitched, based \
on tone, price point, and feature emphasis in the text provided. State this as \
a genuine judgment call, not a fact.
- Set extraction_confidence lower (below 0.6) if the visible text was sparse, \
contradictory, or you had to leave several fields empty.

Prefer structured_data (JSON-LD) over visible_text when both are present, since \
structured data is machine-generated and more reliable."""


def build_user_prompt(
    *,
    url: str,
    structured_data: dict | None,
    visible_text: str,
    review_text: str,
    previous_error: str | None = None,
) -> str:
    structured_block = (
        f"STRUCTURED DATA (schema.org JSON-LD, high trust):\n{structured_data}\n\n"
        if structured_data
        else "STRUCTURED DATA: none found on this page.\n\n"
    )

    review_block = (
        f"REVIEW TEXT (raw, may include ratings/dates mixed in):\n{review_text}\n\n"
        if review_text
        else "REVIEW TEXT: none found on this page.\n\n"
    )

    retry_block = (
        f"YOUR PREVIOUS ATTEMPT FAILED with this error - fix it this time:\n"
        f"{previous_error}\n"
        f"(Common cause: every value inside `specifications` must be a string - "
        f"join lists with commas, e.g. \"XS, S, M, L\" not [\"XS\", \"S\", \"M\", \"L\"], "
        f"and convert numbers to strings, e.g. \"1\" not 1.)\n\n"
        if previous_error
        else ""
    )

    return (
        f"PRODUCT PAGE URL: {url}\n\n"
        f"{retry_block}"
        f"{structured_block}"
        f"VISIBLE PAGE TEXT (cleaned, truncated):\n{visible_text}\n\n"
        f"{review_block}"
        "Extract the product research fields now."
    )
