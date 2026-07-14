# AI Product Creative Generation Workflow

A multi-agent LangGraph pipeline for ecommerce brands: takes a product page
URL and produces ad creative direction, on the way to generating marketing
images/videos automatically.

## Pipeline

```
[URL] --> Agent 1: Product Research --> Agent 2: Creative Strategy --> Agent 3: Prompt Generation --> Agent 4: Image/Video Generation --> Agent 5: Review/Critic
              (done)                          (done)                        (not started)                    (not started)                        (not started)
```

Plus a bulk CSV processing layer (not started) for running the whole
pipeline across many product URLs at once.

Each agent is a self-contained LangGraph subgraph living in its own
`app/agents/<name>/` folder (`schema.py`, `prompts.py`, `state.py`,
`nodes.py`, `graph.py`), sharing common infrastructure in `app/core/`.
Design rationale and hard-won debugging lessons for each agent are written
up in detail in `Product_research.md` and `Creative_strategy.md` - this
README covers setup, running, and configuration; those files cover *why*
things are built the way they are.

## Agent 1: Product Research

Takes a product page URL, returns structured product data (title, price,
features, specs, review synthesis, brand positioning).

```
scrape --> extract --> validate --+--> END (valid, or retries exhausted)
             ^                    |
             |                    v
             +------ bump_retry <-+ (invalid, retries remain)
```

1. **scrape** (`scraper.py`, deterministic, no LLM) - Playwright renders the
   page, then we pull out: schema.org `Product` JSON-LD if present, the OG
   image, cleaned visible text (chrome-stripped - cookie banners,
   mega-menus, breadcrumbs, newsletter blocks, not just semantic
   nav/footer tags), and any review-section text found via heuristic CSS
   selectors, escalating to a scroll pass only when there's an actual
   signal it's worth it (see `Product_research.md` Challenge 4).
2. **extract** (`nodes.py` + `llm.py`) - qwen3.5:4b via Ollama. Does **not**
   use Ollama's native `format=` schema constraint - qwen3.5 has stacking
   bugs there (empty output, or the constraint silently ignored - see
   `Product_research.md` Challenge 1) - so thinking is disabled, the schema
   is described via a **filled example instance** in the prompt (not the
   raw JSON Schema, which a smaller model can echo back verbatim instead of
   returning real data), the response streams with **early-stop once the
   JSON object closes** (cuts wasted trailing-commentary generation), and
   is parsed with a tolerant brace-matching extractor.
3. **validate** - low bar sanity check (non-empty title). Deeper quality
   judgment belongs to the Review/Critic agent later in the pipeline.
4. **bump_retry** - increments the retry counter and loops back to
   `extract` if validation failed and retries remain
   (`max_extraction_retries`, default 2).

A 4B local model is viable here because this agent's job is *extraction*,
not *recall* - it's structuring text already in its context window, not
answering from training knowledge. The one genuinely interpretive field is
`brand_positioning`.

## Agent 2: Creative Strategy

Takes Agent 1's `ProductResearch` object, returns ad creative direction:
hooks, audience targeting angles, visual themes, captions, messaging notes.

```
generate --> validate --+--> END (valid, or retries exhausted)
    ^                    |
    |                    v
    +----- bump_retry <--+ (invalid, retries remain)
```

1. **generate** (`nodes.py` + `llm.py`) - calls `structured_chat_groq`
   against a primary model (**gpt-oss-120b**), falling back to a secondary
   model (**llama-3.3-70b-versatile**) on any provider failure. Model
   choice came from a 3-way comparison (Llama 4 Scout / Llama 3.3 70B /
   gpt-oss-120b) against sample inputs before writing any agent code - see
   `Creative_strategy.md` Challenge 1. This agent needs genuine
   creative/marketing judgment, not extraction, which is why it uses a
   different (cloud, larger) model than Agent 1.
2. **validate** - same low-bar philosophy as Agent 1 (non-empty
   hooks/captions), but writes a *specific* failure reason the moment
   validation fails - not just once retries are exhausted - so the
   `previous_error` fed into the next attempt's prompt actually has
   something useful in it. (Agent 1's `validate_node` still has the older,
   weaker version of this pattern - worth backporting.)
3. **bump_retry** - same as Agent 1, `max_creative_retries` (default 2).

Two fields, `messaging_notes` and `ungrounded_claims_flagged`, are prompted
as a required pair rather than interchangeable alternatives - see
`Creative_strategy.md` Challenge 4 for why that distinction had to be made
explicit.

## Setup

```bash
pip install -r requirements.txt
playwright install chromium

cp .env.example .env
```

`.env` needs:

```bash
# Agent 1 - Ollama (local)
OLLAMA_HOST=http://localhost:11434
OLLAMA_RESEARCH_MODEL=qwen3.5:4b

# Agent 2 - Groq (cloud)
GROQ_API_KEY=your-key-here
```

```bash
ollama pull qwen3.5:4b
```

Groq's free tier is enough to develop against (1K requests/day per model as
of writing - see `Creative_strategy.md` Challenge 1 for why that's not
actually the binding constraint it looks like at first).

## Running

```bash
# Agent 1 only, prints the validated ProductResearch object as JSON
python main.py https://example-store.com/products/some-widget
```

```bash
# Agent 1 with step-by-step debugging output (scrape results, JSON-LD
# presence, review-extraction method used, retry counts)
python scripts/test_live.py https://example-store.com/products/some-widget
python scripts/test_live.py https://example-store.com/products/some-widget --scrape-only
python scripts/test_live.py https://example-store.com/products/some-widget --save
```

```bash
# Both agents chained - tests the actual research-to-creative handoff
python scripts/test_pipeline_live.py https://example-store.com/products/some-widget
python scripts/test_pipeline_live.py https://example-store.com/products/some-widget --research-only
python scripts/test_pipeline_live.py https://example-store.com/products/some-widget --save
```

All scripts exit non-zero with a stderr message on failure (scrape failure,
exhausted extraction retries, or exhausted creative-generation retries).

## Configuration notes

Settings live in `app/core/config.py`, env-driven, shared across agents.
Worth understanding rather than just accepting the defaults:

| Setting | Default | Why it matters |
|---|---|---|
| `ollama_num_ctx` | 16384 | Must comfortably fit system+schema prompt + visible_text + review_text or output gets silently truncated - unrelated to the model's advertised max context, which is a separate, larger number. |
| `ollama_num_predict` | 2048 | Worst-case output token ceiling for Agent 1. Real `ProductResearch` output is a few hundred tokens; this just bounds a rambling response. |
| `max_visible_text_chars` / `max_review_text_chars` | 20000 / 8000 | Hard caps on how much scraped text reaches the LLM - keeps prompts (and Ollama latency) bounded regardless of how bloated a page's HTML is. |
| `min_review_text_chars` | 200 | Below this, review text extracted from the static page counts as "not really found," which is what triggers the scroll-escalation pass in the scraper. |
| `review_wait_timeout_ms` | 5000 | How long to wait for a review selector to appear after the scroll-trigger pass, before giving up and calling it `review_extraction_method: "none"`. |
| `max_extraction_retries` | 2 | Agent 1's retry budget. |
| `groq_primary_model` / `groq_fallback_model` | `openai/gpt-oss-120b` / `llama-3.3-70b-versatile` | Agent 2's model chain - see `Creative_strategy.md` Challenge 1 for the comparison that produced this choice. |
| `groq_temperature` | 0.7 | Higher than Agent 1's 0.2 - Agent 2 is a creative task, not extraction, so more variance is wanted rather than suppressed. |
| `max_creative_retries` | 2 | Agent 2's retry budget. |

## Files

| File | Responsibility |
|---|---|
| `app/core/config.py` | Env-driven settings for both agents |
| `app/core/llm.py` | `structured_chat` (Ollama - manual schema-hint, tolerant extractor, streaming early-stop) and `structured_chat_groq` (native schema-constrained decoding) - both reusable by later agents |
| `app/agents/research_agent/schema.py` | `ProductResearch` output contract + `ScrapedProductData` intermediate shape |
| `app/agents/research_agent/scraper.py` | Playwright fetch + JSON-LD/OG/text extraction, cost-aware review-scroll escalation |
| `app/agents/research_agent/prompts.py` | System + user prompt for the extraction call |
| `app/agents/research_agent/state.py` | LangGraph state TypedDict |
| `app/agents/research_agent/nodes.py` | Node functions + retry/validation routing |
| `app/agents/research_agent/graph.py` | Graph assembly |
| `app/agents/creative_strategy_agent/schema.py` | `CreativeDirection` output contract + `AudienceAngle`/`VisualTheme` sub-models |
| `app/agents/creative_strategy_agent/prompts.py` | System + user prompt for the creative-generation call |
| `app/agents/creative_strategy_agent/state.py` | LangGraph state TypedDict |
| `app/agents/creative_strategy_agent/nodes.py` | Node functions, model-fallback chain, retry/validation routing |
| `app/agents/creative_strategy_agent/graph.py` | Graph assembly |
| `scripts/test_live.py` | Standalone step-by-step debugging runner for Agent 1 |
| `scripts/test_pipeline_live.py` | Chains Agent 1 into Agent 2 for end-to-end testing |
| `main.py` | CLI runner for Agent 1, standalone |
| `Product_research.md` | Agent 1 design narrative - challenges hit, decisions made, why |
| `Creative_strategy.md` | Agent 2 design narrative - same format |

## Known gaps to revisit

- **Review pagination** (Agent 1): the current review selector grabs
  whatever's server-rendered/loaded within the wait window. Sites with
  "load more" review pagination will only yield a partial set.
- **Anti-bot pages** (Agent 1): mitigated for basic fingerprinting (see
  `Product_research.md` Challenge 3) but not hardened enterprise anti-bot
  systems (Akamai, PerimeterX, Cloudflare Bot Management, DataDome).
  `scrape_node` will still "succeed" against a challenge page, just with
  junk `visible_text` - worth a minimum-text-length sanity check before
  extraction.
- **Typographic character normalization** (Agent 2): live output uses smart
  quotes/non-breaking hyphens/emoji (a gpt-oss-120b style tendency).
  Harmless in JSON, could matter later for ad-platform APIs or strict
  character-set contexts. Not fixed - no evidence yet it's caused a real
  problem.
- **Provider-level rate-limit-aware retry/backoff** (Agent 2): the
  fallback chain treats any primary-model failure the same way (fall
  through to secondary immediately) rather than distinguishing a
  rate-limit error (worth a backoff-and-retry-same-model) from a genuine
  failure. Not implemented - no evidence yet this distinction has mattered
  at current request volume.
- **Generalizing lazy-load escalation beyond reviews** (Agent 1): the
  scroll-escalation heuristic is review-specific. The more general version
  would tie escalation to whatever the LLM's `missing_fields` output flags
  as absent, and have the retry trigger a re-scrape - not yet implemented,
  no second content type has needed it yet.
- **LangGraph HITL / persistence / streaming-to-UI**: none implemented at
  either agent's level. Persistence belongs at the bulk-processing
  job-queue layer (resume a crashed batch without re-paying for completed
  LLM calls); HITL belongs as a gate before the expensive image/video
  generation stages, not inside extraction/strategy themselves.
