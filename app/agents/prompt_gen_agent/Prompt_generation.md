# Agent 3: Prompt Generation Agent

## Purpose

Takes the `CreativeDirection` object the Creative Strategy Agent produces
and translates each `visual_theme` into a concrete, ComfyUI-ready image
prompt and video prompt. Third stage of the AI Product Creative Generation
Workflow - its output is what the Image/Video Generation Agent (ComfyUI)
will consume next.

## How it works

```
generate --> validate --+--> END (valid, or retries exhausted)
    ^                    |
    |                    v
    +----- bump_retry <--+ (invalid, retries remain)
```

Same shape as Agent 2 - no scrape-equivalent stage, since input is already a
validated `CreativeDirection` object, not something that can fail to fetch.

1. **generate** (`nodes.py` + `llm.py`) - calls `structured_chat_groq`
   against `prompt_gen_primary_model`, falling back to
   `prompt_gen_fallback_model` on any provider failure. Same
   fallback-vs-retry separation as Agent 2: a provider error isn't a signal
   the *approach* needs correcting, so it doesn't consume a retry.
2. **validate** - same low bar as Agents 1 and 2: catches outright failures
   (wrong number of `prompt_sets`, empty prompt fields), not prompt
   *quality*. Deeper judgment belongs to the Review/Critic agent, once it
   can actually look at what Agent 4 generates from these prompts.
3. **bump_retry** - increments the retry counter and loops back to
   `generate` if validation failed and retries remain
   (`max_prompt_gen_retries`, default 2).

## Why this agent has its own model settings, not Agent 2's

`prompt_gen_primary_model` / `prompt_gen_fallback_model` are deliberately a
separate settings block from Agent 2's `groq_primary_model` /
`groq_fallback_model`, even though they started out pointing at the same
values. Writing a prompt a diffusion/video model actually responds well to
(checkpoint conventions, cinematography vocabulary for motion) is a
different skill from ad copywriting - "best at hooks and captions" isn't
guaranteed to be "best at generation prompts." **This hasn't been validated
with a real comparison test yet** (see Deliberately Deferred) - the current
primary is an inherited placeholder, not a chosen model.

## Major challenges

### 1. The generation backend had to be settled before schema design

An early draft of this agent assumed a generic "prompt string" output
without knowing what would consume it. That's backwards - the schema shape
depends entirely on the target backend. Settled on ComfyUI, needing both
image *and* video prompts (not just images). That decision shaped the
schema toward checkpoint-agnostic natural-language prompts (positive/negative
text) rather than provider-specific parameters, and explicitly excludes
ComfyUI workflow config (seed, steps, cfg_scale, checkpoint/LoRA selection)
from the schema entirely - those are deterministic pipeline settings that
belong with Agent 4's workflow JSON, the same way `scrape_timeout_ms`
belongs to `Settings` rather than `ProductResearch`. An LLM asked to invent
a seed value would just be guessing.

### 2. Variant count is keyed to `visual_themes`, not `hooks`

Considered generating one prompt per hook instead of per visual theme.
Rejected: hooks and captions are *text copy* meant to pair with a visual,
not each drive their own separate image/video - and diffusion models render
legible text unreliably, so hook text was never going to be baked into the
generated image anyway (the system prompt explicitly tells the model not to
describe text/logos/typography in the visual). Settled on one
`ThemePromptSet` per input `visual_theme` (typically 2-3 per product), with
`suggested_hooks`/`suggested_captions` selected **verbatim** from Agent 2's
existing output rather than invented - avoids creating a second, drifting
source of creative copy alongside `CreativeDirection`.

### 3. Bundling image + video into one call per product, not two, to conserve request budget

Since this agent needs both image and video prompts, generating them as two
separate LLM calls per visual theme would double the Groq request count per
product on top of what Agents 1 and 2 already consume - relevant even at
small scale, since it's the *number of requests* that hits Groq's free-tier
ceiling, not token volume (this agent's input, a compact `CreativeDirection`
JSON object, is nowhere near token-limit territory). `PromptGenerationOutput`
bundles `image_prompt` + `video_prompt` into the same `ThemePromptSet`, and
`generate_node` produces every theme's prompt set in a single call per
product - keeping this agent's request footprint equal to Agent 2's (one
request per attempt), not double it.

### 4. A deprecated fallback model was silently masking itself

Live-tested with `prompt_gen_fallback_model` initially set to
`llama-3.3-70b-versatile` - inherited as a placeholder from Agent 2's
config. Groq deprecated that exact model (and `llama-3.1-8b-instant`) for
free/developer tier on June 17, 2026, recommending `gpt-oss-20b` /
`gpt-oss-120b` / `qwen/qwen3.6-27b` as replacements. The bug was invisible
under normal operation: the primary model was succeeding, so the dead
fallback never got exercised, and nothing in `validate_node` would have
caught it even if it had. It surfaced by accident during a manual config
swap that put the deprecated model in the *primary* slot - every run's
`model_used` field showed the fallback model instead of the configured
primary, which is the tell that the primary call is failing outright on
every single attempt, not intermittently.

The same dead model string was sitting in Agent 2's `groq_fallback_model`
too, undetected for the same reason - its primary had simply never needed
to fall back yet. Fixed by swapping both fallback slots to
`openai/gpt-oss-20b` (Groq's own recommended replacement for the deprecated
8B-instant tier) rather than mirroring the 120b primary as the fallback too
- a fallback that matches the primary's cost/speed profile doesn't buy
anything extra.

**Lesson:** `model_used` (already logged per-stage as an observability
field) is what made this catchable at all. A configured-primary vs.
observed-`model_used` mismatch is worth actively watching for, not just
trusting the fallback chain to represent healthy degradation - a fallback
chain can mask a *permanently* broken primary exactly as quietly as it
handles a transient one, which is the whole point of it existing, and also
its blind spot.

### 5. Apparent latency was mis-attributed until it was actually measured

Early live runs showed a ~74s Stage 1 time and the local model (qwen3.5:4b)
was the first suspect. It wasn't the main one - the scraper's
`_goto_with_retries` applied the same `scrape_timeout_ms` (30s) ceiling to
*each* navigation strategy independently, including `networkidle`, which
many real ecommerce pages never cleanly reach (persistent analytics/chat-
widget/ad-pixel traffic keeps the network "busy" indefinitely). That could
burn the entire 30s budget on the first strategy alone before falling
through to a faster one. Fixed at the scraper level (not in this agent) by
reordering navigation strategies fast-and-reliable-first and adding a
separate `nav_attempt_timeout_ms` distinct from the overall
`scrape_timeout_ms` ceiling - cut Stage 1 time from ~74s to ~43s on the same
URL, no model changes involved. Noted here because it's a useful reminder
for evaluating *this* agent's latency too: don't assume which stage owns a
slow pipeline run without checking the per-stage timing the test scripts
already log.

## Deliberately deferred (not gaps, decisions)

- **Model choice not yet validated**: `prompt_gen_primary_model` is still
  an inherited placeholder (`gpt-oss-120b`), not chosen via Agent 2's 3-way
  comparison methodology. Worth running before trusting it - especially
  since Groq now hosts `qwen/qwen3.6-27b`, a plausible candidate given this
  agent's task sits closer to structured translation than Agent 2's more
  open-ended creative writing.
- **`aspect_ratio` and hook-pairing differentiation across themes**:
  promising in live testing (different ratios and different suggested hooks
  per theme, matching composition rather than defaulting) but only observed
  across two live runs - not yet confirmed to hold at scale.
- **ComfyUI workflow-specific video conditioning**: `motion_description` is
  written in general cinematography language, without knowing which video
  workflow (AnimateDiff, SVD, or otherwise) Agent 4 will actually run. May
  need workflow-specific fields (motion strength, conditioning frames) once
  that's settled - deferred rather than guessed at now.
- **Deeper prompt-quality grading**: same low-bar `validate_node` philosophy
  as Agents 1 and 2 - schema completeness only, not whether a prompt would
  actually generate something good. That's the Review/Critic agent's job,
  once it can look at Agent 4's actual output.

## Design decisions worth remembering for later agents

- **Settle the downstream consumer's actual interface before schema
  design, not after.** The schema shape (checkpoint-agnostic prompt text
  vs. provider-specific parameters) depends entirely on what's on the other
  end - guessing here would have meant redesigning the schema anyway.
- **When a shared constraint (request budget) applies across multiple
  related outputs, bundle them into one call rather than splitting by
  output type.** Image and video prompts are conceptually distinct, but the
  constraint that matters (Groq requests/day) doesn't care about that
  distinction - so the schema doesn't split on it either.
- **A "which path actually executed" field (`model_used`) needs to be
  watched, not just logged.** It's the only thing that surfaced the
  deprecated-model bug - the fallback chain itself was working exactly as
  designed, which is precisely why the failure was invisible without it.
- **Provider model catalogs change without warning in code.** A hardcoded
  model string that works today can become dead weight with no runtime
  error, only a behavior mismatch (primary configured vs. fallback always
  used). Worth periodically checking configured model strings against the
  provider's current catalog, not just when something visibly breaks.
