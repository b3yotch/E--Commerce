# Agent 4: Image Generation Agent

## Purpose

Takes Agent 3's `PromptGenerationOutput` and turns each `ThemePromptSet`'s
`image_prompt` into actual generated images via a local ComfyUI instance.
Fourth stage of the AI Product Creative Generation Workflow. Scoped to
**image generation only** for now - video generation (Agent 3 already
produces `video_prompt` alongside `image_prompt`, per theme) is
deliberately deferred, not forgotten; see Deliberately Deferred below.

## How it works

```
start --> generate --> validate --+--> [more themes?] --+--> advance_theme --> generate (next theme)
             ^                    |                      |
             |                    v                      +--> finalize --> END (all themes done)
             +----- bump_retry <--+ (invalid, retries remain)
                                  |
                                  +--> finalize (retries exhausted for this theme - see Challenge 3)
```

Two loops layered on top of each other, unlike Agents 1-3's single retry
loop: a **retry loop** (same shape as Agents 1-3) and a **theme loop**,
since this agent has to iterate across a variable number of visual themes
(typically 2-3, per Agent 2's `visual_themes`) within one product - Agents
1-3 each produced exactly one output per product, so they never needed a
second loop layer.

1. **start** - verifies the ComfyUI checkpoint actually exists on the
   server before spending any time on generation, generates a `run_id` for
   this invocation, and splits `total_images_per_product` across however
   many themes this product has (see Challenge 2).
2. **generate** (`nodes.py` + `comfyui_client.py`) - builds a ComfyUI
   workflow graph, queues it, polls `/history` until it completes or times
   out, downloads the resulting images via `/view`, and copies them into
   this project's own output directory (see Challenge 5).
3. **validate** - same low-bar philosophy as Agents 1-3: did generation
   produce the expected number of images for this theme, not whether
   they're any good - that's Agent 5's job once it exists.
4. **bump_retry** - increments the retry counter and loops back to
   `generate` if validation failed and retries remain
   (`max_image_gen_retries`, default 2), drawing a **fresh random seed**
   for the retry (see Challenge 3).
5. **advance_theme** - on success (or exhausted retries), files whatever
   result exists, resets per-attempt scratch state, moves to the next
   theme.
6. **finalize** - assembles the final `ImageGenerationOutput` once every
   theme has been attempted.

## Why this agent has no prompts.py

Unlike Agents 1-3, there's no LLM call in the generation step itself -
`positive_prompt`/`negative_prompt` text is already fully formed by Agent
3. Generation is a deterministic API call to ComfyUI, so there's no prompt
to construct, and `prompts.py` doesn't apply. Flagged explicitly before
writing any code, following the same "discuss which files actually apply
before assuming the template carries over wholesale" approach used before
Agents 2 and 3 - the per-agent file template is a convention, not a
contract every agent must fill out identically.

## Major challenges

### 1. `aspect_ratio` -> pixel dimensions belongs in Agent 4, not Agent 3

Agent 3's schema deliberately kept `aspect_ratio` as a semantic string
(`"4:5"`, `"9:16"`) rather than pixel dimensions, for the same reason it
excludes seed/steps/cfg_scale: it doesn't know or shouldn't need to know
which backend renders it. Resolving that string into checkpoint-appropriate
pixel dimensions (`aspect_ratio.py`, curated per-ratio lookup table rounded
to multiples of 64 for SDXL-family checkpoints) is deterministic pipeline
config that belongs with the agent that owns the checkpoint - same
reasoning as `Prompt_generation.md` Challenge 1, applied one level further
down the pipeline. Considered and rejected: changing Agent 3's schema to
emit dimensions directly - that would have made Agent 3 depend on knowing
Agent 4's checkpoint choice, inverting the dependency the wrong way.

### 2. Oversampling ("generate N, keep the best M") needs a critic that doesn't exist yet

Early question: generate more images than needed per theme (e.g. 8) and
keep the best 5, using ComfyUI's `batch_size` to do this cheaply in one
job rather than multiple queue calls. Rejected for now - without Agent 5
(Review/Critic) built yet, there's no judge to do the keeping, so
"oversample and filter" would just produce extra files and push the
selection decision back onto a human, defeating the point of an agentic
pipeline. Settled on generating exactly `total_images_per_product` (5)
images **total per product**, split across however many themes exist via
`distribute_total()` (e.g. 5 images / 2 themes -> `[3, 2]`) - all of them
kept, no selection performed in this agent. Revisit once Agent 5 exists and
can actually judge which candidates are worth keeping.

### 3. Retry loop needed, but same-seed retry would have been the wrong default

Kept the same auto-retry shape as Agents 1-3 (no persistence exists yet, so
restarting the whole pipeline on a single failed generation would be far
more costly here than for Agents 1-3's LLM calls). One deliberate deviation
from a naive port of that pattern: **each retry draws a new random seed**
rather than reusing the same one. With a deterministic sampler, same seed +
same prompt + same params reproduces the exact same output - including the
exact same failure, if the failure wasn't pure infrastructure flakiness
(timeout, connection reset). A same-seed retry only helps for that narrow
failure class and silently wastes a retry attempt on anything else.

Retries-exhausted behavior also had to be decided fresh: unlike Agents 1-3
(which return best-effort output after max retries, since there's always
*some* extracted/generated data to hand off), a theme that never produced a
valid image has nothing to hand forward. Current behavior: skip the theme
entirely and continue to the next one, rather than failing the whole
product over one bad theme - worth revisiting once real failure rates are
known.

### 4. The original ComfyUI test script had no error detection or wait bound

The hand-tested `python.py` smoke test (`queue_prompt` + poll `/history`
until the `prompt_id` appears) worked, but had two gaps that only matter
once this becomes a retry-driven agent rather than a one-off script:
- `wait_for_completion` was `while True: sleep(1)` with no timeout - fatal
  for a retry loop, since a retry can only fire after something actually
  raises, and a stalled/dropped job would hang forever instead.
- History was only checked for the presence of an `images` key, not for a
  ComfyUI-reported error status. A node failing mid-graph (bad checkpoint
  reference, OOM) still writes a history entry; the original script would
  have silently printed nothing rather than surfacing the failure.

Both fixed in `comfyui_client.py`: `wait_for_completion` now takes a
`timeout_seconds` bound and raises `ComfyUIError` on either timeout or an
explicit `status_str == "error"` in the history response.

### 5. Output needed to live in this project, not inside the ComfyUI install

ComfyUI's `SaveImage` node only writes into its own `output/` directory,
identified by filename/subfolder - the original test script never actually
retrieved the bytes. Agent 5 (and anything else downstream) shouldn't need
to know where ComfyUI is installed just to read a product's generated
images. Fixed by adding `fetch_image_bytes()` (pulls raw bytes from
ComfyUI's `/view` endpoint) and having `generate_node` copy every image
into this project's own `image_output_dir` immediately after generation,
recording the copy's path as `local_path` on `GeneratedImage` -
`filename`/`subfolder` are kept only for traceability back to ComfyUI's own
copy, not as the read path.

### 6. Repeated runs against the same product URL silently accumulated images

`_save_images_locally` originally wrote to
`<image_output_dir>/<slugified-product>/theme_<n>/` with no run
identifier. Running the pipeline twice against the same URL (which happened
in practice, while separately debugging a stale-test-script issue) meant
the second run's images landed in the exact same folder as the first run's
- ComfyUI's own filenames are unique, so nothing overwrote, but images from
unrelated runs piled up together with no way to tell which run produced
which batch. Fixed by generating a `run_id` (UTC timestamp) once per
pipeline invocation in `start_node` and inserting it into the save path:
`<image_output_dir>/<slugified-product>/<run_id>/theme_<n>/`.

### 7. A silently-dropped LangGraph state key made validation always fail, despite generation succeeding every time

The most expensive bug so far in this agent. `generate_node` returned
`_pending_result` as part of its state update dict, but `ImageGenerationState`
(the `TypedDict` passed to `StateGraph`) never declared `_pending_result` as
a field. LangGraph only carries forward state keys that exist in the schema
it was constructed with - any key outside it is silently dropped between
node executions. The practical effect: `generate_node` genuinely queued the
job, ComfyUI genuinely generated the correct number of images, and the
images were genuinely downloaded and written to disk correctly - all of
that happens independently of LangGraph's state mechanism. But
`validate_node` always saw `state.get("_pending_result")` as `None`
regardless, reported "expected N images, got 0" every time, and the retry
loop did exactly what it was designed to do: retry, generate more images
(silently multiplying the orphaned files on disk with each attempt),
fail the same way again, and eventually skip the theme after
`max_image_gen_retries` was exhausted - while the pipeline summary reported
`0 images generated`.

What made it visible: the file explorer showing real generated images
sitting in `theme_0`/`theme_1` folders, directly contradicting the
pipeline's own `Total images generated: 0` summary line. That mismatch -
disk state vs. reported state - was the actual signal; the log output alone
looked like a clean (if repeated) generation failure, not a bookkeeping
bug. Fixed by declaring `_pending_result` in `ImageGenerationState`.

**Lesson, in the same spirit as `Prompt_generation.md` Challenge 4's
`model_used` note:** when a pipeline's *reported* outcome and its
*side-effect* outcome (files on disk, rows written, external API calls
made) disagree, trust the side effect and go looking for a bookkeeping bug
- the reporting layer is usually the thing that's wrong, not the work that
was actually done. Also worth a general TypedDict-state lint going
forward: every key any node function returns needs a matching declaration
in the state schema, or LangGraph will drop it without any error at all.

## Deliberately deferred (not gaps, decisions)

- **Video generation**: Agent 3 already produces `video_prompt` per theme,
  but this pass is scoped to `image_prompt` only. Revisit once image
  generation is solid and the video generation backend/workflow (which
  video model - AnimateDiff, SVD, or otherwise - is still unsettled per
  `Prompt_generation.md`'s Deferred section) is decided.
- **Candidate selection / oversampling**: see Challenge 2. All generated
  images are kept; picking a "best" candidate is explicitly left to Agent 5.
- **Concurrency across themes**: themes are processed sequentially, one
  `generate`/`validate` cycle at a time. ComfyUI generation is far
  slower/more resource-heavy than Agents 1-3's LLM calls, so parallelizing
  isn't free the way it might be for a cheap API call - deferred until the
  agent works correctly end-to-end, not built speculatively now.
- **Cleanup of orphaned images from exhausted-retry attempts**: a theme
  that exhausts its retries still leaves behind every failed attempt's
  generated images on disk (see Challenge 7's bug for a concrete case).
  Nothing currently prunes these - worth a cleanup pass once retry failure
  rates in practice are better understood.
- **Same-seed retry as an option**: currently always draws a fresh seed on
  retry (Challenge 3). A same-seed mode would help specifically for
  debugging a suspected transient/infra failure, but isn't needed as a
  default.
- **Provider-level backoff nuance**: same deferred item as Agents 2 and 3 -
  the retry loop treats every failure the same way rather than
  distinguishing (e.g.) a ComfyUI queue-full condition from a genuine node
  error. No evidence yet this distinction has mattered in practice.

## Design decisions worth remembering for later agents

- **Schema design follows the downstream interface, one hop at a time, not
  just once at the top.** Agent 3 already deferred checkpoint-specific
  detail to Agent 4 (`Prompt_generation.md`); Agent 4 in turn resolves
  exactly the pieces it owns (dimensions, seed, sampler config) and no
  more. Each agent should only model what it's actually positioned to
  decide.
- **A retry loop's "same input, try again" default isn't automatically
  correct - check whether retrying with identical inputs can even change
  the outcome.** For a deterministic sampler, it can't; the seed has to
  change for a retry to mean anything beyond "hope the network behaves this
  time."
- **When a pipeline's summary output and its actual side effects disagree,
  believe the side effects.** The bug is almost always in the reporting
  path, not in the work - see Challenge 7.
- **Every key a node function returns must be declared in the graph's state
  schema.** LangGraph does not error on an undeclared key; it just drops it
  silently, which makes this exact class of bug quiet enough to survive
  multiple test runs before surfacing.