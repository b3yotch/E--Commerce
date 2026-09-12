# Agent 5: Image Selection Agent (Critic)

## Purpose

Takes Agent 4's `ImageGenerationOutput` (several candidate images per
theme, none selected) and picks exactly one - the source frame Agent 6
(Video Generation) will animate. Fifth stage of the AI Product Creative
Generation Workflow, and the first agent whose entire job is *judging*
rather than *producing* new content.

Agent 4's own `schema.py` docstring anticipated this agent's existence
before it was built ("no critic exists yet (Agent 5)"). Video Generation
got built first and temporarily claimed the number 5, using
`theme_result.images[0]` as an explicitly-flagged placeholder in the
meantime (see `Video_generation.md` Challenge 6, original form). Once this
agent actually existed, the pipeline was renumbered so agent number
matches graph position again - see Challenge 4 below, and
`Video_generation.md` Challenge 8 for the other side of that change.

## Why this is one agent, not two

The original "Review/Critic" concept in the pipeline diagram implied a
single agent judging everything after generation. That doesn't survive
contact with the actual pipeline: judging candidate *images* has to
happen **before** Agent 6 runs (it needs one chosen frame to animate), but
judging finished *video* quality can only happen **after** Agent 6 runs
(there's no video to critique yet). Two distinct judgments at two distinct
pipeline moments, with different failure semantics (see Challenge 3) -
not one agent invoked twice. See Challenge 1 for the full reasoning.

This agent is the *image-selection* half only. A *video-quality* critic,
judging Agent 6's finished output, was discussed at the same time and
deliberately deferred until this agent proved itself - see "Deliberately
deferred" below.

## How it works

```
start --> select --+--> [more themes?] --+--> select (next theme)
                    |                      |
                    v                      +--> finalize --> END (all themes done)
                    (always produces a result - see Challenge 3)
```

Simpler loop shape than Agents 4/6: no separate `validate`/`bump_retry`
nodes, because `select_node` never leaves a theme without a result (see
Challenge 3) - a retry either succeeds and advances, or exhausts and
still advances with a flagged fallback. One node handles both the retry
and the theme-loop decision by whether `current_theme_index` moved.

1. **start** - generates a `run_id`, initializes loop state. No ComfyUI
   checkpoint check (unlike Agent 4) - this agent never talks to ComfyUI.
2. **select** (`nodes.py` + `llm.py`'s `structured_chat_vision`) - for the
   current theme: a deterministic pre-filter drops any candidate image
   that fails a basic corruption check (`PIL.Image.verify()`) before
   spending a model call on it at all. Whatever survives is shown together
   to a vision-capable LLM, alongside the theme's creative brief
   (`source_setting`), which returns a score and a short issue note per
   candidate, plus a single best pick and rationale (see Challenge 2 for
   why comparative ranking, not independent scoring). The result is
   recorded either way - see Challenge 3.
3. **finalize** - assembles `ImageSelectionOutput` once every theme has
   been attempted.

No `bump_retry`/`validate` split like Agents 4/6 - see the loop-shape note
above. `retries`/`error` state fields still exist and follow the same
naming convention as every other agent, so the pipeline runner's
`result.get("retries", 0)` calls work unmodified.

## Major challenges

### 1. Two distinct judgments at two distinct pipeline moments - not one agent invoked twice

The tempting design was "one Review/Critic agent, called twice." Rejected
for the same reason `llm.py`'s Ollama/Groq split is two functions, not one
with a branch: folding two things that solve different problems together
hides that they're solving different problems. Image selection is a
*relative* task (rank N candidates) that **cannot fail open** - Agent 6
needs exactly one frame per non-skipped theme, so there's no valid
"reject everything" outcome. Video quality (deferred - see below) would be
an *absolute* task (pass/fail against a bar) that **can and should fail
open** to "flagged, not regenerated" - already a real decision baked into
`max_video_gen_retries: 0` at the config layer (`Video_generation.md`
Challenge 2). An agent invoked twice under one schema would have to encode
both failure semantics as branches inside one contract; two separate
agents keep each contract honest about what it actually guarantees.

### 2. Mechanism: deterministic pre-filter, then comparative ranking - and leniency lives in the rubric, not in Agent 3's prompts

Two layers, not one: a cheap deterministic check (file validity) before
ever spending a model call, then a vision-LLM judgment on whatever
survives. The vision call is explicitly **comparative** - shown all
surviving candidates together against the brief in one call - rather than
scoring each independently and comparing afterward, since independent
scoring tends to produce ties/noise that don't reflect real preference
between similar candidates.

Calibration question worth naming explicitly: candidates come from a
local SDXL checkpoint (`juggernautXL_ragnarok`), not a flagship commercial
image model, so grading them against an unqualified "is this a good
image" bar would reject almost everything. The fix is **not** softening
what Agent 3 asks for at generation time - that would cap the ceiling of
every generation to avoid the floor looking bad, optimizing the wrong
variable. The system prompt (`prompts.py`) instead carries an explicit
"do not penalize / do penalize" calibration: minor SDXL-typical artifacts
(texture softness, small anatomical imperfections off the visual focus,
slightly generic backgrounds) are excluded from scoring; genuine brief
mismatches, broken anatomy, and off-brand output are not.

### 3. Failure handling: never fail open to "no pick" - a below-threshold pick beats no pick

If every candidate for a theme scores below `image_selection_score_
threshold` (default 55), or the vision call fails outright after
exhausting retries, this agent still emits a selection - the best-ranked
candidate anyway, or the first valid one if scoring never happened -
flagged via `status: "selected_below_threshold"` rather than silently
treated as a clean pass. This mirrors `ThemeVideoResult.status` in
`Video_generation.md`: a degraded outcome is recorded explicitly, not
made indistinguishable from success.

No cross-agent retry loop back to Agent 4 exists (the same non-decision
Agents 4 and 6 both already made deliberately - see `Image_generation.md`
Challenge 3, `Video_generation.md`'s deferred list). If real usage shows
most themes landing below threshold, that's evidence the prompt or
checkpoint needs attention, not that a 6th roll of the dice from the same
inputs would help - regenerating with identical prompts into the same
model is likely to reproduce the same quality band. Worth revisiting once
real failure rates are known, same spirit as every other agent's
deliberately-deferred retry philosophy.

### 4. Renumbered to Agent 5 once this agent was actually built

See `Video_generation.md` Challenge 8 for the full mechanics of the swap.
Short version: this agent claims the number Agent 4's own docstring always
expected the critic to have; Video Generation moved from 5 to 6 to make
room. `Video_generation.md`'s `state.py` and `nodes.py` were updated to
consume this agent's `ImageSelectionOutput` (via `selected_local_path`)
instead of reaching into Agent 4's raw candidate list itself.

### 5. First model choice (Groq's qwen/qwen3.6-27b) hit a real rate limit almost immediately

Given the low call volume (a handful of images per theme, a few themes per
product), judgment *quality* seemed to matter more than cost or
throughput - the opposite profile from Agents 2/3's model comparisons - so
a larger hosted vision model (Groq's `qwen/qwen3.6-27b`, a preview model)
was chosen over a local one, reasoning the RTX 3050 was already committed
to ComfyUI and this was too consequential a judgment to save cost on.

First real run hit `Error code: 429 ... tokens per minute (OTPM): Limit
1000, Requested 1347` - this org's on-demand tier caps this specific
preview model at 1000 *output* tokens/minute, and Groq's own estimate
against the 5-candidate schema already exceeded that on a single request.
Not a transient blip: `select_node`'s retry logic retried immediately,
inside the same one-minute window, so every retry hit the identical limit
for every theme.

Fixed in two parts: `image_selection_max_tokens` lowered to fit under the
cap, and a rate-limit-specific exception (`LLMRateLimitError`, distinct
from other `LLMExtractionError`s) added so a 429 specifically backs off
past the one-minute window before retrying, rather than retrying on the
very next graph pass.

### 6. Fixing the rate limit surfaced a second, different failure: Groq's own strict schema validation rejecting the output

With `max_tokens` lowered enough to clear the OTPM cap, a new failure
appeared: `400 json_validate_failed` - Groq's own server-side strict
JSON-schema decoder rejecting the completion outright, deterministically,
on every retry for every affected theme (not a rate limit; a plain 400).

Two plausible causes, addressed together rather than sequentially isolated:
the lowered token budget may have been too tight to let the model finish
the full JSON before running out (truncation), and the schema itself had
an optional field with a default (`issues: list[str] = Field(default_
factory=list)`) plus numeric range constraints (`Field(ge=0, le=100)`) -
either an unsupported schema feature for this model's strict decoding, or
extra verbosity pushing generation past budget. Fixed by trimming the
schema to need fewer tokens per candidate (dropped the redundant
`image_index` field entirely - candidates are matched by array position
instead - and collapsed `issues: list[str]` to a single required `issue:
str`) and making every field required with no defaults or range
constraints, moving the 0-100 range into instruction text instead.

Even after this fix, roughly 2/3 of calls in a real 3-theme run still hit
the identical `json_validate_failed` - the one theme that succeeded
produced a clean, well-reasoned judgment, ruling out a hard schema
incompatibility (that would fail every time, not two-thirds of the time)
and pointing at a probabilistic truncation issue instead.

### 7. Switched to running locally via Ollama, once vision support was confirmed hands-on

Rather than keep tuning against a hosted preview model with real,
demonstrated reliability problems, switched to the same local Ollama
model already used for Agent 1's extraction (`qwen3.5:4b`) - confirmed
multimodal in its underlying architecture (native early-fusion
vision-language, not vision bolted on separately), but Ollama's support
for a given vision-capable GGUF isn't guaranteed just because the
upstream model is multimodal (some vision-packaged models need a separate
mmproj file Ollama doesn't yet load for every architecture).

Verified hands-on before committing, not assumed: `ollama show qwen3.5:4b`
listed `vision` explicitly under Capabilities, and a real test image (an
actual Agent 4 candidate) produced a specific, accurate description
(correct garment type, embroidery colors, placket and cuff details,
background elements) rather than generic or hallucinated text.

This also resolves Challenge 6 by construction, not just by changing
providers: Ollama's path uses this project's own tolerant JSON parsing
(`_extract_json_object`, the same mechanism `structured_chat` already uses
for Agent 1) rather than a provider's strict server-side validator, so a
truncated or malformed response becomes an `LLMExtractionError` this
project controls and can retry - not an opaque 400 from Groq. There is
also no per-minute output-token quota to hit at all, resolving Challenge 5
the same way. A real end-to-end run after switching completed cleanly in
26.36s for 2 themes, with actual populated `candidate_scores` and
brief-referencing rationale for both.

## Deliberately deferred (not gaps, decisions)

- **A video-quality critic**, judging Agent 6's finished output rather
  than Agent 4's candidate frames, was discussed during this agent's
  design and deliberately deferred until this agent proved itself in a
  real run. It has now (Challenge 7) - a natural next agent, with a
  flag-and-pass-through failure mode rather than auto-regeneration, for
  the same cost reasoning behind `max_video_gen_retries: 0`
  (`Video_generation.md` Challenge 2).
- **Cross-agent regeneration loop back to Agent 4**: see Challenge 3.
  Worth building only if real failure rates justify it.
- **Concurrency across themes**: sequential only, matching Agents 4/6 -
  this agent's calls are cheap enough that concurrency would help least
  here, but it wasn't built speculatively regardless.
- **Quality tradeoff of the local model vs. the original hosted one**: not
  rigorously evaluated - the switch was driven by reliability (Groq's
  hosted preview model failing outright most of the time) rather than a
  head-to-head quality comparison against a working baseline that never
  existed. Worth a real comparison once the video-quality critic exists
  and there's a fuller picture of end-to-end output quality to judge
  against.

## Design decisions worth remembering for later agents

- **Verify a provider's real behavior hands-on before designing around
  advertised limits - and before switching away from one, too.** Groq's
  documented image-count and size limits were checked in advance; the
  1000 OTPM cap and the strict-schema rejection rate were not knowable
  until hit in practice. The same discipline applied in reverse before
  committing to Ollama: `ollama show` plus one real test image, not an
  assumption that "the model family is multimodal" meant "this local
  tag works."
- **A failure that's deterministic (every attempt, same input) points at
  a different fix than one that's probabilistic (some attempts succeed).**
  The first Groq failure (429) was purely deterministic within a time
  window - fixable with token budget and backoff. The second (400) looked
  deterministic at first but resolved to roughly 2/3 across a real run
  once retries were let through - the one success ruled out a hard schema
  incompatibility and pointed at truncation instead. Don't diagnose a
  failure's cause from its error message alone; check whether it's
  consistent across attempts first.
- **Leniency belongs in the judge's calibration, not in loosening what was
  asked for at generation time.** Softening Agent 3's prompts to make
  Agent 4's output easier to pass would cap quality across every
  generation to avoid the critic looking harsh on the worst ones - the
  wrong tradeoff. The rubric knowing its own inputs are SDXL-local-
  checkpoint output, not flagship-model output, is the correct place for
  that adjustment.
- **A below-threshold fallback is not the same decision as a
  cross-agent retry loop, even though both "handle failure."** This
  agent's failure handling (Challenge 3) deliberately stops at "record a
  flagged pick" rather than reaching back into Agent 4 - building the
  retry loop can wait for evidence it's needed; never returning a
  selection at all could not wait, since Agent 6 has nothing to animate
  without one.