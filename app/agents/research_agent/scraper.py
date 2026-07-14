"""
Scraping layer for the Product Research Agent.

Design goal: pull out as much *deterministic* signal as possible before the
LLM ever sees the page. Most ecommerce platforms (Shopify, WooCommerce,
Magento, custom storefronts using schema.org markup) embed a JSON-LD
`Product` block with name/price/brand/rating - if it's there, we get those
fields with zero hallucination risk and the LLM's job shrinks to the
qualitative fields (positioning, review synthesis) plus gap-filling.

Playwright (async) is used instead of requests/BeautifulSoup alone because
most modern product pages render key content (price, reviews) via JS.
"""

from __future__ import annotations

import json
import re

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

from app.core.config import settings
from app.agents.research_agent.schema import ScrapedProductData

# Common selectors where review text tends to live. Best-effort, not exhaustive -
# sites without matching selectors just fall back to whatever's in visible_text.
REVIEW_SELECTORS = [
    "[class*='review']",
    "[id*='review']",
    "[data-testid*='review']",
]


def _extract_json_ld_product(soup: BeautifulSoup) -> dict | None:
    """Find a schema.org Product block among any JSON-LD script tags."""
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
        except (json.JSONDecodeError, TypeError):
            continue

        candidates = data if isinstance(data, list) else [data]
        for item in candidates:
            if not isinstance(item, dict):
                continue
            item_type = item.get("@type", "")
            type_str = item_type if isinstance(item_type, str) else " ".join(item_type)
            if "Product" in type_str:
                return item
            # Some sites nest the Product inside an @graph array
            for node in item.get("@graph", []) if isinstance(item.get("@graph"), list) else []:
                if isinstance(node, dict) and "Product" in str(node.get("@type", "")):
                    return node
    return None


def _extract_og_image(soup: BeautifulSoup) -> str | None:
    tag = soup.find("meta", property="og:image")
    return tag.get("content") if tag else None


def _extract_visible_text(soup: BeautifulSoup, max_chars: int) -> str:
    """
    Strip page chrome and return trimmed body text.

    The original noise list only caught semantic tags (nav/footer/script/
    style/svg/noscript). In practice, most real ecommerce sites put their
    actual chrome - cookie consent banners, breadcrumb trails, mega-menus,
    newsletter signup blocks, mobile slide-out menus - in plain <div>s
    identified by class/id/role, not semantic tags at all, so none of that
    was being stripped. This directly bloats the prompt handed to the LLM:
    that noise is pure token cost with zero extraction signal, and prefill
    time scales with input length, so cutting it shrinks Ollama latency
    without touching config or hardware.

    Broadened to catch the common patterns above via attribute-contains
    selectors. Best-effort, not exhaustive - a site with unusual markup will
    still leak some chrome through, same caveat as REVIEW_SELECTORS below.
    """
    noise_selectors = [
        "script", "style", "nav", "footer", "svg", "noscript",
        "header", "aside", "form", "iframe",
        "[class*='cookie']", "[id*='cookie']",
        "[class*='banner']",
        "[class*='breadcrumb']", "[id*='breadcrumb']",
        "[class*='newsletter']",
        "[class*='menu']", "[id*='menu']",
        "[role='navigation']", "[role='banner']", "[role='contentinfo']",
        "[aria-hidden='true']",
    ]
    for selector in noise_selectors:
        for element in soup.select(selector):
            element.decompose()

    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    return text[:max_chars]


def _extract_review_text(soup: BeautifulSoup, max_chars: int) -> str:
    """Best-effort collection of text from review-ish sections of the page."""
    chunks: list[str] = []
    seen_ids = set()

    for selector in REVIEW_SELECTORS:
        for element in soup.select(selector):
            if id(element) in seen_ids:
                continue
            seen_ids.add(id(element))
            text = element.get_text(separator=" ", strip=True)
            if len(text) > 20:  # skip near-empty matches like "Reviews (4)" labels
                chunks.append(text)

    combined = re.sub(r"\s+", " ", " ".join(chunks))
    return combined[:max_chars]


async def _goto_with_retries(page, url: str) -> None:
    """
    Navigate with a couple of fallbacks. `net::ERR_HTTP2_PROTOCOL_ERROR` shows
    up on some sites (Myntra among them) when the server's bot-detection layer
    tears down the connection mid-handshake rather than serving a normal
    response - it's not a timeout, so retrying with the same strategy usually
    just reproduces it immediately. We vary wait_until strategy and back off
    briefly between attempts instead of retrying identically.
    """

    # Set headers once
    await page.set_extra_http_headers({
        "Referer": "https://www.google.com/",
        "Upgrade-Insecure-Requests": "1",
    })

    strategies = [
        ("networkidle", 3000),
        ("domcontentloaded", 2000),
        ("commit", 2000),
    ]

    last_error: Exception | None = None

    for attempt, (wait_until, extra_wait) in enumerate(strategies):
        try:
            await page.goto(
                url,
                timeout=settings.scrape_timeout_ms,
                wait_until=wait_until,
            )

            # Give React/SPA apps time to hydrate
            await page.wait_for_timeout(extra_wait)

            return

        except Exception as exc:  # noqa: BLE001
            last_error = exc

            if attempt < len(strategies) - 1:
                await page.wait_for_timeout(1500)

    raise last_error
    


def _review_count_hint(json_ld_product: dict | None) -> int | None:
    """
    Peek at JSON-LD's aggregateRating for a review count, if present, so we
    can skip scroll escalation entirely when structured data already tells
    us there are zero reviews - no point paying for a scroll pass to find
    text that provably doesn't exist. Returns None if we can't tell either
    way (absent or malformed), in which case escalation stays worth trying.
    """
    if not json_ld_product:
        return None
    agg = json_ld_product.get("aggregateRating")
    if not isinstance(agg, dict):
        return None
    for key in ("reviewCount", "ratingCount"):
        try:
            if agg.get(key) is not None:
                return int(agg[key])
        except (TypeError, ValueError):
            continue
    return None


async def _scroll_to_trigger_lazy_content(page) -> None:
    """
    Many third-party review widgets (Judge.me, Loox, Yotpo - common on
    Shopify stores) don't render on page load at all; they lazy-load via a
    scroll-based IntersectionObserver, only injecting content once their
    container scrolls into view. A headless browser that loads the page and
    sits still will never trigger this - there's no fixed wait that fixes it,
    because the trigger is scroll position, not time. Simulate a real visit
    by scrolling through the page in a few steps.
    """
    try:
        height = await page.evaluate("document.body.scrollHeight")
        steps = 6
        for i in range(1, steps + 1):
            await page.mouse.wheel(0, height // steps)
            await page.wait_for_timeout(700)
        await page.evaluate("window.scrollTo(0, 0)")  # back to top, tidy but not required
    except Exception:
        pass


async def scrape_product_page(url: str) -> ScrapedProductData:
    """
    Render `url` with Playwright and return everything the extraction node
    needs: JSON-LD product data (if any), OG image, cleaned visible text,
    and any review-section text found via heuristic selectors.

    Review extraction is a cost-aware escalation ladder, not a fixed
    strategy: check what's already there for free before paying for a
    scroll pass. There's no single strategy that reliably works across every
    site (static HTML, scroll-triggered widgets, click-to-reveal tabs, and
    iframe-embedded review apps all need different handling), and trying
    every strategy on every page would multiply latency across an entire
    bulk CSV run for no benefit on the majority of pages that don't need it.
    """
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=[
                "--disable-http2",
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
            ],
        )
        context = await browser.new_context(
            user_agent=settings.scrape_user_agent,
            locale="en-US",
            timezone_id="Asia/Kolkata",
            viewport={"width": 1366, "height": 900},
            extra_http_headers={
                "Accept-Language": "en-US,en;q=0.9",
            },
        )


        await context.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', {
            get: () => undefined
        });
        """)
        page = await context.new_page()
        page.set_default_timeout(settings.scrape_timeout_ms)
        page.set_default_navigation_timeout(settings.scrape_timeout_ms)

        try:
            await _goto_with_retries(page, url)
            
        except Exception as exc:
            await context.close()
            await browser.close()
            raise RuntimeError(
                f"Navigation failed after retries: {exc}. If this is a "
                f"'net::ERR_HTTP2_PROTOCOL_ERROR' or similar connection-level "
                f"error, the site's bot detection is likely rejecting the "
                f"request rather than the page failing to load - see README "
                f"'Known gaps' for options."
            ) from exc

        # Stage 1 (free): whatever's already in the DOM right after navigation,
        # no scrolling, no extra waiting beyond the goto itself.
        html = await page.content()
        soup = BeautifulSoup(html, "html.parser")
        json_ld_product = _extract_json_ld_product(soup)
        review_text = _extract_review_text(soup, settings.max_review_text_chars)
        review_extraction_method = "static"

        # Only escalate to scrolling if there's an actual signal it's worth
        # it: either we don't know the review count yet (structured data
        # absent, can't rule it out) or JSON-LD says reviews exist but we
        # didn't capture their text. If JSON-LD says reviewCount is 0, don't
        # bother scrolling - there's nothing to find.
        review_count_hint = _review_count_hint(json_ld_product)
        worth_escalating = len(review_text) < settings.min_review_text_chars and (
            review_count_hint is None or review_count_hint > 0
        )

        if worth_escalating:
            # Stage 2 (costs ~3-5s): scroll to trigger scroll-based lazy
            # loading (Judge.me/Loox/Yotpo-style widgets), then re-check.
            await _scroll_to_trigger_lazy_content(page)
            try:
                await page.wait_for_selector(
                    ", ".join(REVIEW_SELECTORS),
                    timeout=settings.review_wait_timeout_ms,
                )
            except Exception:
                pass

            html = await page.content()
            soup = BeautifulSoup(html, "html.parser")
            review_text = _extract_review_text(soup, settings.max_review_text_chars)
            review_extraction_method = "scroll" if len(review_text) >= settings.min_review_text_chars else "none"

        raw_title = await page.title()

        await context.close()
        await browser.close()

    return ScrapedProductData(
        url=url,
        raw_title=raw_title,
        og_image=_extract_og_image(soup),
        json_ld_product=json_ld_product,
        visible_text=_extract_visible_text(soup, settings.max_visible_text_chars),
        review_text=review_text,
        review_extraction_method=review_extraction_method,
    )