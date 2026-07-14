# Agent 2: Creative Strategy Agent

## Purpose

Takes the `ProductResearch` object the Product Research Agent produces and
returns ad creative direction for a paid social campaign: hooks, audience
targeting angles, visual themes, captions, and marketing messaging. Second
stage of the AI Product Creative Generation Workflow - its output is what the
Prompt Generation Agent will consume next.

## How it works

```
generate --> validate --+--> END (valid, or retries exhausted)
    ^                    |
    |                    v
    +----- bump_retry <--+ (invalid, retries remain)
```

1. **generate** (`nodes.py` + `llm.py`) - calls `structured_chat_groq`
   against a primary model, falling back to a secondary model on ANY
   failure (API error, rate limit, schema validation error) from the
   primary. This fallback chain is deliberately separate from the retry loop
   below - a transient provider failure isn't a signal that the *approach*
   needs correcting, so it doesn't consume a retry or get a `previous_error`
   message fed back into the prompt.
2. **validate** - low bar, same philosophy as Agent 1: catch outright
   failures (no hooks, no captions at all), not grade creative quality.
   Deeper quality/consistency judgment belongs to the Review/Critic agent
   later in the pipeline.
3. **bump_retry** - increments the retry counter and loops back to
   `generate` if validation failed and retries remain
   (`max_creative_retries`, default 2).

No scrape-equivalent stage - input is already a validated `ProductResearch`
object, not a URL that can fail to fetch.

## Why this agent needed a different model than Agent 1

Agent 1's job is extraction: structuring text that's already in its context
window, which is why a local 4B model was viable there. This agent's job is
genuine creative/marketing judgment - there's no "correct" hook to extract
from the input, only a better or worse one to generate. That's a different
capability class, and it showed immediately: qwen3.5:4b's own attempt at this
task (informal check, not a full test) produced generic, interchangeable ad
copy rather than the distinctly-angled output the schema asks for.

## Major challenges

### 1. Model choice needed an actual test, not a guess

Ran the same 3 hand-built `ProductResearch` inputs (a spec-driven gadget, a
review-driven beauty product, and a product with a strong pre-existing brand
voice) through three Groq-hosted candidates - Llama 4 Scout, Llama 3.3 70B
versatile, and gpt-oss-120b - with an identical prompt, before writing any
agent code. Scout's hooks were noticeably weaker and, in one case, misused a
statistic (conflated `review_count` with "devices charged" - a real
grounding slip, not just a style difference). Llama 3.3 70B and gpt-oss-120b
were both clearly stronger; gpt-oss-120b's structure and specificity edged
ahead, at the cost of unprompted formatting flourishes (markdown tables,
emoji) that weren't asked for and needed to be reined in once the output was
put under real JSON-schema constraints. Landed on **gpt-oss-120b primary,
llama-3.3-70b-versatile fallback**.

Rate limits factored into the choice less than expected: Groq's free tier
caps almost every model at the same 1K requests/day, which matters for the
eventual bulk-CSV layer - but per-call token usage here (one `ProductResearch`
object in, one creative-direction object out) is nowhere near tight enough
for token-per-minute limits to be the deciding factor between candidates.

### 2. Groq doesn't need Agent 1's Ollama workarounds - a second `llm.py` entry point, not a branch

Groq's OpenAI-compatible API supports native `response_format` /
`json_schema` decoding reliably, without qwen3.5's thinking-trace/format
collision bug (see `Product_research.md` Challenge 1). Rather than bolting a
provider `if/else` onto `structured_chat`, added a sibling function,
`structured_chat_groq`, that uses `response_format` directly - the schema is
enforced server-side, not just requested in text and tolerantly parsed
afterward. Keeping the two functions separate makes it clear they're solving
different problems, not sharing one mechanism with a flag.

### 3. The retry-with-context pattern only works if validation writes the reason immediately

Agent 1's `validate_node` only writes an informative error once retries are
exhausted - so on the retries that actually matter (mid-loop, budget
remaining), `previous_error` has nothing useful in it, and the "retry with
context" idea is a no-op in practice. Fixed for this agent by having
`validate_node` write a specific reason (which required field was missing,
or that the model call failed outright) the moment validation fails,
retries remaining or not. Verified this wasn't just a design intention but
actually works, with a regression test mirroring Agent 1's silent
short-circuit bug: mocked a model that returns invalid output twice then
valid output on the third call, and asserted the model is genuinely
re-invoked with the failure reason present in the prompt each time - not
just that the retry counter increments. Worth backporting this fix to
Agent 1.

### 4. `messaging_notes` and `ungrounded_claims_flagged` need to be a required pair, not alternatives

First prompt version treated the two fields as either/or ("flag it in
messaging_notes/ungrounded_claims_flagged"). Live output showed the model
consistently picking one: it would write a hedge in `messaging_notes`
("reviews are too sparse to support specific comfort claims") while leaving
`ungrounded_claims_flagged` empty, even when audience angles in the same
output were themselves inferences built from a single vague detail. Fixed
by making the prompt explicit that the two fields serve different purposes
and must both be filled whenever either applies - `ungrounded_claims_flagged`
lists the specific claims, `messaging_notes` explains why they're uncertain
- with a direct instruction that writing a hedge in one is itself the signal
to go back and populate the other.

### 5. Captions occasionally ran past the char limit - normalized in code, not the prompt

Consistent with Agent 1's "normalize in code, don't over-constrain the
prompt" lesson (specifications coercion validator): rather than an
increasingly insistent prompt about the 150-character caption limit, added a
`field_validator` that trims any overlong caption to 147 chars + `...`. A
caption a few characters over is still usable creative material for a human
to shorten; failing validation over it would waste a retry on otherwise-good
output.

## Deliberately deferred (not gaps, decisions)

- **Typographic character normalization**: live output uses smart
  quotes/non-breaking hyphens/emoji throughout (gpt-oss-120b's style).
  Harmless in JSON, but could matter later for ad-platform APIs or strict
  character-set contexts. Not fixed - no evidence yet it's caused a real
  problem, and normalizing it in the schema layer would risk stripping
  intentional stylistic choices (emoji in captions, for instance) rather
  than just the risky characters.
- **Provider-level retry/backoff for rate limits**: the fallback chain
  handles "primary model errored," but doesn't distinguish a rate-limit
  error (worth a backoff-and-retry-same-model) from a genuine failure (worth
  falling through to the secondary model immediately). Not implemented -
  no evidence yet this distinction has mattered in practice at current
  request volume.
- **Deeper creative quality grading**: `validate_node`'s bar is deliberately
  low (non-empty hooks/captions), same as Agent 1's non-empty-title bar.
  Real quality judgment belongs to the Review/Critic agent later in the
  pipeline, not duplicated here.

## Design decisions worth remembering for later agents

- **Let observed model output shape the schema, not the other way around**:
  the 3-way model test ran against loosely-structured markdown output
  *before* the Pydantic schema was designed - all three candidates
  naturally separated audience angles into segment/angle/rationale and
  visual themes into setting/mood/palette, unprompted, which is what became
  the sub-model structure (`AudienceAngle`, `VisualTheme`) instead of a
  guessed-upfront shape.
- **Model-fallback chains and retry-with-context loops solve different
  problems - keep them separate**: a provider/API failure has nothing
  useful to feed back into a prompt; a validation failure does. Collapsing
  both into one retry mechanism would mean either wasting retry budget on
  transient provider issues, or wasting a fallback-model call on a problem
  the same model could fix if just told what was wrong.
- **A field two humans would consider "obviously related" (messaging_notes,
  ungrounded_claims_flagged) still needs to be pinned down explicitly for a
  model** - describing them as alternatives to each other, even briefly,
  was enough for the model to reliably use only one.
