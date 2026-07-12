# Product Research Agent

First agent in the AI Product Creative Generation Workflow. Takes a product
page URL and returns structured product data (title, price, features, specs,
review synthesis, brand positioning) for the downstream Creative Strategy
Agent to consume.

## How it works

```
scrape --> extract --> validate --+--> END (valid, or retries exhausted)
             ^                    |
             |                    v
             +------ bump_retry <-+ (invalid, retries remain)
```

1. **scrape** (`scraper.py`, deterministic, no LLM) - Playwright renders the
   page, then we pull out: schema.org `Product` JSON-LD if present, the OG
   image, cleaned visible text, and any review-section text found via
   heuristic CSS selectors.
2. **extract** (`nodes.py` + `llm.py`) - qwen3.5:4b via Ollama, called with
   `format=` set to the `ProductResearch` JSON schema so output is
   structurally constrained, not just prompted-for JSON. Structured data is
   given priority over visible text in the prompt since it's machine-generated
   and less likely to be wrong.
3. **validate** - low bar sanity check (did we get a non-empty title at all).
   Deeper quality judgment belongs to the Review/Critic agent later in the
   full pipeline, not here.
4. **bump_retry** - increments the retry counter and loops back to `extract`
   if validation failed and retries remain (`MAX_EXTRACTION_RETRIES`, default
   2). Retry counting lives in exactly one place to avoid the classic
   off-by-one infinite-loop bug in conditional LangGraph edges.

## Why a 4B model is fine here

This agent's job is *extraction*, not *recall* - the model never has to know
anything about the product from training data, it just has to structure text
that's already in its context window. The one genuinely interpretive field is
`brand_positioning`; everything else (title, price, specs, features) should
come straight from the page. If quality on that field turns out to be weak in
practice, swap `OLLAMA_RESEARCH_MODEL` to something larger - it's a config
change, not a rewrite.

## Setup

```bash
pip install -r requirements.txt
playwright install chromium

cp .env.example .env   # adjust OLLAMA_HOST / model name if needed
ollama pull qwen3.5:4b
```

## Run it

```bash
python main.py https://example-store.com/products/some-widget
```

Prints the validated `ProductResearch` object as JSON. Non-zero exit code and
a stderr message on failure (scrape failure or exhausted extraction retries).

## Files

| File | Responsibility |
|---|---|
| `app/agents/research_agent/schema.py` | `ProductResearch` output contract + `ScrapedProductData` intermediate shape |
| `app/agents/research_agent/scraper.py` | Playwright fetch + JSON-LD/OG/text extraction |
| `app/agents/research_agent/prompts.py` | System + user prompt for the extraction call |
| `app/agents/research_agent/state.py` | LangGraph state TypedDict |
| `app/agents/research_agent/nodes.py` | Node functions + retry/validation routing |
| `app/agents/research_agent/graph.py` | Graph assembly |
| `app/core/config.py` | Env-driven settings |
| `app/core/llm.py` | Generic Ollama structured-output wrapper (reusable by later agents) |
| `main.py` | CLI runner for standalone testing |

## Known gaps to revisit

- **Review pagination**: the current review selector grabs whatever's
  server-rendered/loaded within the wait window. Sites with "load more"
  review pagination will only yield a partial set. Fine for a first pass;
  worth a dedicated pagination-aware scraper if review depth matters later.
- **Anti-bot pages**: no proxy/stealth handling yet. Sites with aggressive
  bot detection may return a challenge page instead of product content -
  scrape_node will still "succeed" but visible_text will be junk. Worth
  adding a basic sanity check (e.g. minimum text length) before extraction.
