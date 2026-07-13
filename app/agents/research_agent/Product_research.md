# Agent 1: Product Research Agent

## Purpose

Takes a single product page URL and returns structured product data (title,
price, features, specs, review synthesis, brand positioning) that the
Creative Strategy Agent consumes downstream. First stage of the AI Product
Creative Generation Workflow.

## How it works

```
scrape ──┬─(ok)──> extract ──> validate ──┬─(valid / retries exhausted)──> END
         │                        ^        │
         └─(failed)──> END        │        v
                                   └── bump_retry (invalid, retries remain)
```

1. **Scrape** (Playwright, deterministic) - render the page, extract
   schema.org JSON-LD `Product` data if present (zero-hallucination-risk
   source for price/title/rating), OG image, cleaned visible text, and
   review-section text via a cost-aware escalation ladder (see Challenge 4).
2. **Extract** (qwen3.5:4b via Ollama) - structures the scraped content
   against a Pydantic schema. JSON-LD is prioritized over free text since
   it's machine-generated and less error-prone.
3. **Validate** - low bar: non-empty title. Real quality judgment is
   deferred to the Review/Critic agent later in the full pipeline.
4. **Retry** - on failure, the previous error is fed back into the next
   attempt's prompt so the model has an actual chance to self-correct,
   rather than blindly repeating the same mistake.

## Major challenges

These are the ones that actually cost significant debugging time and
changed the design, not minor tweaks.

### 1. Ollama's native structured output was unreliable for this model

Expected `format=schema` (Ollama's JSON-schema-constrained decoding) to Just
Work. In practice, qwen3.5 is a thinking-capable model with two bugs that
stack depending on how `think` is set: left unset (defaults to true) plus
`format`, the model can burn its entire output budget on the hidden
reasoning trace and return **empty content**; `think=False` plus `format`
silently **ignores the format constraint** and returns free text instead.
Neither setting works. Had to drop native `format` entirely, disable
thinking, and build a manual, tolerant JSON extractor (strips markdown
fences, finds the first brace-balanced `{...}` block) - a workaround for an
upstream bug, not a clean solution, but the only one that actually worked.

A second-order issue on top of that: once the model was given the JSON
Schema as a text instruction instead of a native constraint, it sometimes
echoed the schema *definition* back (with `properties`/`type`/`description`
keys) instead of returning a filled instance - the two look structurally
similar to a smaller model. Fixed by showing a filled-shape example instead
of the raw schema.

### 2. A retry loop that silently did nothing

A short-circuit check in the extraction node (`if state.get("error"): return
state`), originally meant to skip extraction after a scrape failure, also
triggered after an extraction failure - because extraction itself sets that
same error field on failure. Net effect: once extraction failed once, every
subsequent "retry" returned instantly without ever calling the model again.
The retry counter kept incrementing; the actual work didn't happen.

This one is worth remembering specifically because unit tests with mocks
initially missed it - they only exercised the "model returns
invalid-but-parseable data" path, never "model raises an exception." It only
surfaced against a real product page with genuinely malformed output. Fixed
by removing the now-redundant short-circuit (scrape failures are handled by
a separate routing edge) and adding a regression test for the exact failure
mode (asserting the mock LLM is actually called twice on a retry, not once).

### 3. Bot detection on real sites

Testing against a real ecommerce target (Myntra) exposed a common challenge with
modern JavaScript applications: requests could fail with
`net::ERR_HTTP2_PROTOCOL_ERROR`, a connection-level rejection that is often
consistent with bot-detection terminating the session before a normal HTTP
response is returned.

To improve reliability, the scraper was hardened with several defensive
measures while preserving the existing extraction pipeline:

- Disabled HTTP/2 (`--disable-http2`) to avoid protocol-level connection
  failures observed on some requests.
- Reduced obvious automation fingerprints using
  `--disable-blink-features=AutomationControlled` and overriding
  `navigator.webdriver` before any page scripts execute.
- Configured a realistic browser context (User-Agent, locale, timezone,
  viewport, and `Accept-Language`) instead of relying on Playwright defaults.
- Added navigation fallbacks (`networkidle`, `domcontentloaded`, and `commit`)
  with short backoff intervals, making navigation more resilient across
  different rendering strategies.
- Allowed additional hydration time after navigation so React-based pages could
  finish rendering before HTML extraction.
- Simulated user scrolling via mouse wheel events to trigger lazy-loaded
  content such as review widgets that rely on `IntersectionObserver`.
- Included common browser request headers (for example, `Referer` and
  `Upgrade-Insecure-Requests`) to better resemble a normal browser session.

These improvements significantly reduce failures caused by basic browser
fingerprinting and timing issues. However, they are not intended to bypass
enterprise-grade anti-bot systems. Sites employing advanced bot protection
(Akamai, PerimeterX/Human, Cloudflare Bot Management, DataDome, etc.) may
still require additional infrastructure such as residential proxies, browser
stealth frameworks, session persistence, or authenticated user sessions. Those
concerns are outside the scope of this scraper and are intentionally not
handled by the extraction layer.

### 4. Lazy-loaded content that no fixed wait can solve

Reviews kept coming back empty on a site that clearly had them (JSON-LD
reported `reviewCount: 3`). Root cause, confirmed by fetching the page
directly: the review widget (typical of Shopify apps like Judge.me/Loox)
loads via a **scroll-triggered IntersectionObserver**, not on page load and
not after any fixed delay - a headless browser that loads the page and sits
still will never trigger it, no matter how long it waits.

Fixed with a scroll-simulation step, but the more important design decision
was *not* making every page pay for it: a cost-aware escalation ladder that
checks static content first (free) and only scrolls when there's an actual
signal it's needed (JSON-LD says reviews exist but weren't captured) -
skipping escalation entirely when JSON-LD says `reviewCount: 0`. This
matters specifically because of bulk CSV processing: an unconditional "always
scroll and wait" strategy multiplies latency across every URL in a batch,
most of which don't need it.

**Known limitation, not yet fixed:** this escalation logic is specific to
reviews. The more general version would tie escalation to what the LLM's
`missing_fields` output flags as absent after a static-only attempt, and
have the *retry* trigger an escalated re-scrape - rather than hand-writing a
heuristic per content type. Worth doing before this pattern needs repeating
in a later agent.

## Deliberately deferred (not gaps, decisions)

- **Bot detection on hardened sites (Myntra)**: partially mitigated (see
  Challenge 3), not fully solved. Root cause is likely sophisticated
  fingerprinting (Akamai/PerimeterX-style) that would need residential
  proxies or stealth plugins to reliably defeat - disproportionate effort
  for a first-pass agent. Accepted as a known limitation: some URLs in a
  bulk batch will fail to scrape, and that's an acceptable outcome rather
  than a blocking one.
- **Generalizing lazy-load escalation beyond reviews** (tying it to the
  LLM's `missing_fields` output instead of a review-specific heuristic, as
  discussed in Challenge 4): explicitly not implemented. The current
  review-specific heuristic works for the cases seen so far; generalizing it
  now would be solving a problem not yet observed in another field.
- **LangGraph HITL / persistence / streaming**: none implemented at this
  agent's level. Persistence belongs at the bulk-processing job-queue layer
  (resume a crashed batch run without re-paying for completed LLM calls),
  not inside a single-URL agent. HITL belongs as a gate before the
  expensive image/video generation stages (surface low-confidence research
  before burning compute on bad creatives), not inside extraction itself.
  Streaming is a live-progress UX feature with no functional pipeline value
  unless a dashboard is built. All three are real capabilities for later
  stages of the pipeline, not missing pieces of this one.

## Design decisions worth remembering for later agents

- **Extraction vs. recall**: this agent's job is structuring text already in
  context, not recalling facts from training - which is why a 4B model is
  viable here at all. Doesn't necessarily hold for agents needing actual
  creative judgment (Creative Strategy, Critic).
- **Normalize in code, don't over-constrain the prompt**: when the model's
  output was reasonable but didn't match the schema's types (lists where
  strings were expected), the fix was a coercion validator, not an
  increasingly desperate prompt insisting on a stricter format.
- **Test the state machine before adding real I/O**: the LangGraph
  retry/routing logic was unit-tested with mocked states before ever
  touching a real URL - caught one bug that way, missed another (the silent
  short-circuit) because the mocks didn't cover the right failure mode. Both
  data points matter: mocked tests lock in fixes once found, but discovery
  mostly came from live runs against real, messy pages.