# AI Product Creative Generation Workflow

A multi-agent LangGraph pipeline for ecommerce brands: takes a product page
URL and produces ad creative direction plus ComfyUI-ready generation
prompts, on the way to generating marketing images/videos automatically.

## Pipeline

```
[URL] --> Agent 1: Product Research --> Agent 2: Creative Strategy --> Agent 3: Prompt Generation --> Agent 4: Image/Video Generation (ComfyUI) --> Agent 5: Review/Critic
              (done)                          (done)                        (done)                              (not started)                              (not started)
```

Plus a bulk CSV processing layer (not started) for running the whole
pipeline across many product URLs at once.

Each agent is a self-contained LangGraph subgraph living in its own
`app/agents/<name>/` folder (`schema.py`, `prompts.py`, `state.py`,
`nodes.py`, `graph.py`), sharing common infrastructure in `app/core/`.
Design rationale and hard-won debugging lessons for each agent are written
up in detail in `Product_research.md`, `Creative_strategy.md`, and
`Prompt_generation.md` - this README covers setup, running, and
configuration; those files cover *why* things are built the way they are.

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
   Navigation itself tries several `wait_until` strategies in order
   (fastest/most-reliable first), each capped by its own
   `nav_attempt_timeout_ms` rather than sharing the full `scrape_timeout_ms`
   ceiling - an earlier version let a single strategy (`networkidle`, which
   many real ecommerce pages never cleanly reach due to persistent
   analytics/chat-widget traffic) burn the entire scrape budget before
   falling through to a faster one.
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
   model (**gpt-oss-20b**) on any provider failure. Model choice came from
   a 3-way comparison (Llama 4 Scout / Llama 3.3 70B / gpt-oss-120b)
   against sample inputs before writing any agent code - see
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

**Note on `groq_fallback_model`:** originally `llama-3.3-70b-versatile`,
which Groq deprecated for free/developer tier on June 17, 2026. Now
`openai/gpt-oss-20b` - see `Prompt_generation.md` Challenge 4 for how this
was caught (it hadn't bitten this agent yet only because its primary had
never needed to fall back).

## Agent 3: Prompt Generation

Takes Agent 2's `CreativeDirection` object, returns ComfyUI-ready image and
video generation prompts - one `ThemePromptSet` per input `visual_theme`
(typically 2-3 per product), each pairing an `image_prompt` and a
`video_prompt` with hooks/captions selected (verbatim, never invented) from
Agent 2's existing output.

```
generate --> validate --+--> END (valid, or retries exhausted)
    ^                    |
    |                    v
    +----- bump_retry <--+ (invalid, retries remain)
```

Same shape as Agent 2 - no scrape-equivalent stage.

1. **generate** (`nodes.py` + `llm.py`) - calls `structured_chat_groq`
   against `prompt_gen_primary_model` (**gpt-oss-120b**, an inherited
   placeholder - not yet validated with its own comparison test, see
   `Prompt_generation.md`), falling back to `prompt_gen_fallback_model`
   (**gpt-oss-20b**). Image and video prompts for *all* of a product's
   visual themes are produced in a single call, not split across multiple
   calls - keeps this agent's request footprint equal to Agent 2's rather
   than doubling it (see `Prompt_generation.md` Challenge 3).
2. **validate** - same low-bar philosophy: checks `prompt_sets` count
   matches the input's `visual_themes` count and that no prompt field came
   back empty, not prompt *quality*.
3. **bump_retry** - same pattern, `max_prompt_gen_retries` (default 2).

Sampler/workflow parameters (seed, steps, cfg_scale, checkpoint/LoRA
selection) are deliberately **not** modeled in this agent's schema - those
are deterministic ComfyUI workflow configuration that belongs to Agent 4,
not something an LLM should be guessing at. See `Prompt_generation.md`
Challenge 1.

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

# Agents 2 & 3 - Groq (cloud)
GROQ_API_KEY=your-key-here
```

```bash
ollama pull qwen3.5:4b
```

Groq's free tier is enough to develop against (1K requests/day per model as
of writing - see `Creative_strategy.md` Challenge 1 for why that's not
actually the binding constraint it looks like at first, and
`Prompt_generation.md` Challenge 3 for how Agent 3 avoids doubling its
request footprint despite generating two prompt types per theme). Worth
periodically checking your configured model strings (`groq_primary_model`,
`groq_fallback_model`, `prompt_gen_primary_model`,
`prompt_gen_fallback_model`) against Groq's current model catalog -
deprecated models fail silently into the fallback chain rather than raising
an obvious error; see `Prompt_generation.md` Challenge 4.

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
# Agents 1 + 2 chained - tests the research-to-creative handoff
python scripts/test_pipeline_live.py https://example-store.com/products/some-widget
python scripts/test_pipeline_live.py https://example-store.com/products/some-widget --research-only
python scripts/test_pipeline_live.py https://example-store.com/products/some-widget --save
```

```bash
# All three agents chained - tests the full research-to-prompts handoff
python scripts/test_full_pipeline_live.py https://example-store.com/products/some-widget
python scripts/test_full_pipeline_live.py https://example-store.com/products/some-widget --research-only
python scripts/test_full_pipeline_live.py https://example-store.com/products/some-widget --creative-only
python scripts/test_full_pipeline_live.py https://example-store.com/products/some-widget --save
```

All scripts exit non-zero with a stderr message on failure (scrape failure,
or exhausted retries at whichever stage failed). Each stage prints its
elapsed time and retry count separately - useful for figuring out which
stage actually owns a slow run rather than guessing (see
`Prompt_generation.md` Challenge 5, where scrape latency was initially
mis-attributed to Agent 1's local model).

## Configuration notes

Settings live in `app/core/config.py`, env-driven, shared across agents.
Worth understanding rather than just accepting the defaults:

| Setting | Default | Why it matters |
|---|---|---|
| `ollama_num_ctx` | 16384 | Must comfortably fit system+schema prompt + visible_text + review_text or output gets silently truncated - unrelated to the model's advertised max context, which is a separate, larger number. |
| `ollama_num_predict` | 2048 | Worst-case output token ceiling for Agent 1. Real `ProductResearch` output is a few hundred tokens; this just bounds a rambling response. |
| `max_visible_text_chars` / `max_review_text_chars` | 20000 / 8000 | Hard caps on how much scraped text reaches the LLM - keeps prompts (and Ollama latency) bounded regardless of how bloated a page's HTML is. |
| `min_review_text_chars` | 200 | Below this, review text extracted from the static page counts as "not really found," which is what triggers the scroll-escalation pass in the scraper. |
| `nav_attempt_timeout_ms` | 10000 | Per-strategy navigation timeout, separate from `scrape_timeout_ms`. Without this, a single slow strategy (`networkidle` on a page with persistent background traffic) could burn the entire scrape budget before falling through to a faster fallback strategy. |
| `review_wait_timeout_ms` | 4000 | How long to wait for a review selector to appear after the scroll-trigger pass, before giving up and calling it `review_extraction_method: "none"`. |
| `max_extraction_retries` | 2 | Agent 1's retry budget. |
| `groq_primary_model` / `groq_fallback_model` | `openai/gpt-oss-120b` / `openai/gpt-oss-20b` | Agent 2's model chain - see `Creative_strategy.md` Challenge 1 for the comparison that produced the primary choice. |
| `groq_temperature` | 0.7 | Higher than Agent 1's 0.2 - Agent 2 is a creative task, not extraction, so more variance is wanted rather than suppressed. |
| `max_creative_retries` | 2 | Agent 2's retry budget. |
| `prompt_gen_primary_model` / `prompt_gen_fallback_model` | `openai/gpt-oss-120b` / `openai/gpt-oss-20b` | Agent 3's model chain - kept as its own settings block, separate from Agent 2's, because prompt-writing for generation models is a different skill from ad copywriting. The primary is an inherited placeholder, not yet validated with its own comparison test - see `Prompt_generation.md`. |
| `prompt_gen_temperature` | 0.6 | Slightly lower than Agent 2's 0.7 - prompt-writing benefits from more precision/consistency than open-ended ad copy, but still needs some variation across 2-3 distinct visual themes. |
| `max_prompt_gen_retries` | 2 | Agent 3's retry budget. |

## Files

| File | Responsibility |
|---|---|
| `app/core/config.py` | Env-driven settings for all three agents |
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
| `app/agents/prompt_gen_agent/schema.py` | `PromptGenerationOutput` output contract + `ImageGenerationPrompt`/`VideoGenerationPrompt`/`ThemePromptSet` sub-models |
| `app/agents/prompt_gen_agent/prompts.py` | System + user prompt for the prompt-generation call |
| `app/agents/prompt_gen_agent/state.py` | LangGraph state TypedDict |
| `app/agents/prompt_gen_agent/nodes.py` | Node functions, model-fallback chain, retry/validation routing |
| `app/agents/prompt_gen_agent/graph.py` | Graph assembly |
| `scripts/test_live.py` | Standalone step-by-step debugging runner for Agent 1 |
| `scripts/test_pipeline_live.py` | Chains Agent 1 into Agent 2 for two-stage testing |
| `scripts/test_full_pipeline_live.py` | Chains all three agents for end-to-end testing, with per-stage early-exit flags |
| `main.py` | CLI runner for Agent 1, standalone |
| `Product_research.md` | Agent 1 design narrative - challenges hit, decisions made, why |
| `Creative_strategy.md` | Agent 2 design narrative - same format |
| `Prompt_generation.md` | Agent 3 design narrative - same format |

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
- **Provider-level rate-limit-aware retry/backoff** (Agents 2 & 3): the
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
- **Prompt Generation model choice unvalidated** (Agent 3):
  `prompt_gen_primary_model` is an inherited placeholder from Agent 2, not
  chosen via a real 2-3 candidate comparison against sample
  `CreativeDirection` inputs. Worth running before trusting it - see
  `Prompt_generation.md`.
- **ComfyUI workflow-specific video conditioning** (Agent 3):
  `motion_description` is written in general cinematography language,
  independent of which actual video workflow (AnimateDiff, SVD, etc.) Agent
  4 will run. May need workflow-specific fields once that's settled.
- **LangGraph HITL / persistence / streaming-to-UI**: none implemented at
  any agent's level. Persistence belongs at the bulk-processing job-queue
  layer (resume a crashed batch without re-paying for completed LLM calls);
  HITL belongs as a gate before the expensive image/video generation
  stages, not inside extraction/strategy/prompt-generation themselves.