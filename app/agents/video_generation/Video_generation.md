# Agent 5: Video Generation Agent

## Purpose

Takes Agent 3's `PromptGenerationOutput` (for `video_prompt` per theme)
**and** Agent 4's `ImageGenerationOutput` (for the source frame per
theme), and turns each theme's `video_prompt` into a generated video via
image-to-video generation against a local ComfyUI instance. Fifth stage
of the AI Product Creative Generation Workflow - and the first agent in
this pipeline that consumes two upstream outputs at once, since
image-to-video generation genuinely needs both a prompt and a starting
image, not just one or the other.

Runs `NimVideo/cogvideox-2b-img2vid` - a community image-to-video
fine-tune of CogVideoX-2B (there's no official THUDM CogVideoX-2B I2V
release; only 5B-I2V exists upstream), confirmed back when Agent 3's
schema was being finalized (see `Prompt_generation.md` Challenge 6).

This landed as its own Agent 5, not an extension of Agent 4's "Image/Video
Generation" combined agent as originally sketched in the pipeline
diagram - decided explicitly rather than left to drift, since video's
cost profile (minutes per attempt vs. Agent 4's seconds) is different
enough to warrant its own retry/batching decisions. That pushes
Review/Critic from Agent 5 to Agent 6.

## How it works

```
start --> generate --> validate --+--> [more themes?] --+--> advance_theme --> generate (next theme)
             ^                    |                      |
             |                    v                      +--> finalize --> END (all themes done)
             +----- bump_retry <--+ (invalid, retries remain)
                                  |
                                  +--> advance_theme (retries exhausted - skip this theme, same choice Agent 4 made)
```

Same two-loop shape as Agent 4 (a retry loop plus a theme loop), for the
same reason: this agent also iterates a variable number of themes
(typically 2-3, driven by `prompts.prompt_sets`' length). One video per
theme, no batching - Agent 4's "generate N, keep the best M" reasoning for
images doesn't transfer here; a single extra video generation is already
expensive enough that batching would need a critic to justify (see
Challenge 6 below), same as it did for images.

1. **start** - loads the API-format ComfyUI workflow JSON once (not
   per-theme, since the template doesn't change across themes - only the
   per-call patches do) and generates a `run_id`, same convention as
   Agent 4's run-scoped output folders.
2. **generate** (`nodes.py` + `video_comfyui_client.py`) - finds the
   Agent 4 image whose `source_setting` matches the current theme, picks
   a source frame from it, uploads that frame back to ComfyUI, builds and
   queues the video workflow, polls to completion, and copies the result
   into this project's own output directory.
3. **validate** - lowest bar of any agent in this pipeline: did
   `generate_node` produce a result at all (a success or an explicit
   skip) - there's no count to check the way Agent 4 checks image count,
   since it's always exactly one video or nothing.
4. **bump_retry** - same fresh-seed-per-retry logic as Agent 4 (a
   deterministic sampler would just reproduce the same failure on a
   same-seed retry), but with `max_video_gen_retries` defaulting to **0**,
   not 2 - see Challenge 2.
5. **advance_theme** - files whatever result exists, resets scratch
   state, moves to the next theme.
6. **finalize** - assembles `VideoGenerationOutput` once every theme has
   been attempted.

## Why this agent has its own ComfyUI client, not Agent 4's

`video_comfyui_client.py` **subclasses** Agent 4's `ComfyUIClient` rather
than either duplicating it or editing it in place. Two things are added
(`upload_image`, `build_video_workflow`, `load_video_workflow_template`,
`extract_video_outputs`) and one thing is deliberately **overridden**, not
patched into the base class: `wait_for_completion`. Agent 4's version
correctly checks for an `"images"` key on each node's output, since
that's what `SaveImage` produces. The video workflow's save node's output
key isn't `"images"` and isn't stable across custom node packages -
confirmed by hand-testing, not assumed - so the override accepts any
non-empty outputs dict instead. Subclassing keeps Agent 4's
already-tested file completely untouched while still reusing
`queue_prompt`/`fetch_image_bytes`/`check_checkpoint` as-is, since those
really are generic to ComfyUI's API regardless of which workflow is
running.

No `prompts.py` here either, same reasoning as Agent 4: generation is a
deterministic API call, not an LLM call.

## Major challenges

### 1. The video workflow needed hand-testing first, and it was a bigger first step than image's was

Agent 4's image workflow used core ComfyUI nodes
(`CheckpointLoaderSimple`/`CLIPTextEncode`/`KSampler`/`VAEDecode`/
`SaveImage`) - any SDXL checkpoint drops into that same graph shape, so
`comfyui_client.py` could stay a thin, generic wrapper. CogVideoX doesn't
work that way: there's no core-node path to it, it requires a custom node
package installed on top of ComfyUI, with its own graph shape entirely.

The bigger, easy-to-miss gotcha: the workflow JSON downloadable from the
model repo is ComfyUI's *UI*-format export (a `nodes`/`links` graph for
the canvas), not the *API*-format export the `/prompt` endpoint actually
needs (a flat `{node_id: {class_type, inputs}}` dict) - these look similar
enough to load without obvious error but aren't interchangeable. Getting
the real API-format export required enabling Dev Mode in ComfyUI's
settings and using "Save (API Format)" - a different button from plain
"Save." `load_video_workflow_template()` now checks for `nodes`/`links`
keys and fails loudly and immediately if it sees the wrong format, rather
than letting `/prompt` reject it with a confusing error later.

Once that export existed, hand-testing followed the same principle as
Agent 4's original test script: prove the workflow works standalone,
patched by hand, before any agent code touches it. Unlike Agent 4's node
IDs (`"1"` through `"7"`, chosen by `build_image_workflow` itself since it
constructs the graph from scratch), the video workflow's relevant node IDs
(`comfyui_video_*_node_id` in Settings) are pinned to the specific
hand-tested export - there's no portable `class_type` convention across
CogVideoX custom node packages the way there is for core ComfyUI nodes,
so these need re-verifying if the workflow is ever re-exported.

### 2. Real hardware numbers, not guesses, ended up driving `max_video_gen_retries` and the generation timeout

Flagged as a risk before any code was written: 6GB VRAM is right at, or
below, the commonly-cited threshold for CogVideoX-2B in ComfyUI. Once
actually measured, the numbers were stark and changed real config
decisions:

- At 49 frames (the value carried over from the hand-test script), one
  sampling step took **~186 seconds** - 20 steps would be ~3,720 seconds
  (62 minutes) for a single video, against an initial
  `comfyui_video_generation_timeout_seconds` of 1800 (30 minutes).
  Every attempt was mathematically guaranteed to time out before finishing.
- At 41 frames, the same card managed **~24.3 seconds/step** - a roughly
  7-8x speedup from dropping 8 frames, disproportionate to the ~16% frame
  reduction alone, suggesting frame count affects more than just linear
  per-step cost on VRAM-constrained hardware. A full generation completed
  in a measured **~601 seconds (~10 minutes)**.

Two settings changed directly because of these numbers, not as
speculative tuning: the timeout was raised to 4500s (real headroom above
a plausible worst case, not a round number picked in advance), and
`max_video_gen_retries` was dropped to **0**, down from an initial
inherited-from-image default of 1. At minutes-per-attempt cost on
constrained hardware, a "free" retry costs real, non-trivial time for a
theme that already failed once - worth deciding deliberately rather than
assuming image's retry philosophy carries over unchanged.

### 3. `num_frames` turned out to be a real runtime parameter - correcting an assumption made back in Agent 3

`Prompt_generation.md` Challenge 6 removed `duration_seconds` from
`VideoGenerationPrompt` on the reasoning that "this checkpoint's output
length is fixed by the checkpoint itself, not a real parameter." Once the
actual ComfyUI sampler node was inspected while building this agent, that
turned out to be wrong: `num_frames` is a plain editable input
(`comfyui_video_num_frames` in Settings, currently 41). fps is still fixed
at 8 regardless of frame count - that part held up.

The schema decision itself doesn't need reversing - an LLM still
shouldn't be choosing this, same as it shouldn't choose seed/steps/cfg -
just the *justification* needed correcting: it's deterministic pipeline
config that belongs to Settings, not something immovable and therefore
moot. Worth remembering as a pattern: an assumption made before the real
consumer/node graph is inspected needs re-auditing once it is, not just
treated as settled because it was reasonable at the time (`Prompt_
generation.md` Challenge 6 itself makes exactly this point about a
different field - this is a second instance of the same lesson,
discovered a stage later).

One real constraint surfaced alongside this: CogVideoX's temporal VAE
requires frame counts of the form `4n+1` (49, 41, 37, 33, ...), so an
exact round-second duration usually isn't reachable at a fixed 8fps - 41
frames gives 5.125s, not a clean 5.0s. Requested durations get rounded to
the nearest valid frame count, not hit exactly.

### 4. ComfyUI's HTTP server can't service any request while synchronously blocked on GPU work - this broke two unrelated things

First surfaced as `ConnectionResetError` on `/upload/image`, with
sampling actively running in ComfyUI's own logs at the same time. The
initial (correct as far as it went) diagnosis: `wait_for_completion`
timing out client-side does *not* tell ComfyUI to stop the job - it keeps
running on the GPU, and while it's running, ComfyUI's single-threaded
aiohttp server can't accept a new connection for anything, including a
plain image upload for a completely different theme's attempt.

That diagnosis was incomplete. The *same* blocking behavior later broke
this agent's own polling loop: `wait_for_completion`'s `GET
/history/{prompt_id}` call has its own short per-request timeout, and if
that GET happened to land while ComfyUI was mid-step, it failed the same
way - except this failure wasn't caught inside the polling loop, so it
propagated up and got treated as "this attempt failed," even though the
job was still running fine and later completed successfully. Confirmed
directly: a video that ComfyUI's own log reported as
"Prompt executed in 00:10:02" showed up in the pipeline's output as a
skipped, retries-exhausted theme.

Fix, in two parts:
- `upload_image` now wraps `requests.RequestException` explicitly as
  `ComfyUIError` with an accurate message, instead of letting it fall
  through to `generate_node`'s `except OSError` branch (which happens to
  catch it too, since `requests.RequestException` subclasses `OSError`,
  but mislabels it as "failed saving video locally" - a message written
  for actual local disk failures, not upload failures).
- `wait_for_completion`'s polling loop now catches `requests.
  RequestException` around each poll and retries the poll rather than
  raising - a failed poll means "we asked at a bad moment," not "the job
  failed." Only an explicit `status_str == "error"` in a successfully
  fetched history entry, or exceeding the full configured wall-clock
  timeout, ends the loop now.

### 5. A defensive fix for Challenge 4 introduced a worse bug: killing jobs that were about to succeed

Once Challenge 4's blocking behavior was understood, the first fix
attempt added a `ComfyUIClient.interrupt()` call (`POST /interrupt`,
ComfyUI's stop-current-job endpoint) in two places: when
`wait_for_completion` times out, and defensively at the very start of
every `generate_node` call, on the reasoning "clear out any job left
running from an abandoned prior attempt before starting a new one."

That reasoning had a hidden assumption: that a still-running job must be
*abandoned*. Challenge 4 showed this isn't true - a job can still be
running and about to succeed even though *our polling* already gave up on
it (a failed poll ≠ a failed job). The defensive `interrupt()` at the top
of `generate_node` couldn't tell the difference, and killed exactly the
jobs that would have succeeded: ComfyUI's own job history showed two
"Failed" entries immediately followed by one genuine success - the
pattern you'd expect if each theme's defensive interrupt call was killing
the *previous* theme's still-legitimately-running job, and only the last
theme (nothing queued after it to interrupt it) survived to completion.

Fixed by removing the defensive interrupt call entirely, rather than
trying to make it smarter. With Challenge 4's polling fix in place, the
polling loop no longer gives up on jobs prematurely, which removes the
actual reason a "clear stuck jobs" call seemed necessary in the first
place. `interrupt()` is now called in exactly one place: after a real,
full-timeout expiration - the one case where a job genuinely can be
considered abandoned rather than merely hard to observe.

**Lesson:** a defensive cleanup call justified by "this shouldn't happen,
but just in case" needs the same scrutiny as the bug it's defending
against - if the underlying detection (here, "the job seems to have
failed") is unreliable, a cleanup action built on top of that detection
inherits the same unreliability, and can do active damage precisely when
the detection was wrong rather than right.

### 6. Source-frame selection has no critic to ask, same gap Agent 4 already flagged

Agent 4 generates multiple candidate images per theme and explicitly
defers picking a "best" one to a future critic agent (see `Image_
generation.md` Challenge 2 - no critic exists yet, so "oversample and
filter" would just produce extra files with nothing to do the filtering).
This agent needs exactly *one* source frame per theme to animate, and
inherits the same gap one level down: there's still no critic to ask.

Resolved for now with `theme_result.images[0]` - the first of Agent 4's
candidates, arbitrary rather than chosen. Flagged explicitly in both
`schema.py`'s docstring and inline at the selection site
(`_pick_source_image` in `nodes.py`) as a placeholder, not a considered
decision, so it's easy to find and replace once Agent 6 (Review/Critic)
exists and can actually judge which candidate is worth animating.

### 7. Two schema fields, one text input - combining `base_prompt` and `motion_description`

Agent 3's `VideoGenerationPrompt` deliberately keeps `base_prompt` (scene
description) and `motion_description` (camera/subject motion) as separate
fields, for clarity and independent editability. The actual ComfyUI
CogVideoX node this agent talks to takes a single `"prompt"` text input,
not two - so `generate_node` concatenates them
(`f"{base_prompt} {motion_description}"`) before sending.

This isn't a schema/backend mismatch to fix - it's confirmed correct by
Agent 3's own system prompt, which explicitly frames the video prompt as
describing "what that starting frame already shows, not invent a
different scene" (see `Prompt_generation.md`). Scene and motion are
genuinely separate *concerns* worth keeping distinct at the point they're
authored, even though they collapse into one string at the point they're
consumed - the same reasoning that keeps `style_notes` separate from
`positive_prompt` in Agent 3's `ImageGenerationPrompt`.

## Deliberately deferred (not gaps, decisions)

- **Candidate selection / oversampling**: see Challenge 6. Revisit once
  Agent 6 exists.
- **Concurrency across themes**: sequential only, same as Agent 4 - more
  true here than for images, given video's measured per-attempt cost.
- **Provider-level backoff nuance**: same deferred item as every prior
  agent - the retry loop (what little of it remains at
  `max_video_gen_retries: 0`) doesn't distinguish failure causes. No
  evidence yet this distinction matters more than the retry-count decision
  in Challenge 2 already addresses.
- **Aspect ratio / resolution reconciliation**: Agent 3 chooses
  `aspect_ratio` per theme for social framing (`4:5`, `9:16`, etc.), but
  this agent's output is a fixed 720x480 landscape regardless of which
  image was selected as the source frame. Currently fed in as-is, with no
  resize/letterbox/crop step. Still open - tracked in the main README's
  Known Gaps, not resolved here.
- **`max_video_gen_retries: 0` as a permanent position**: a cost-driven
  default given real measured attempt time, not a considered stance on
  whether retries help. Revisit with evidence about how often failures are
  transient (worth retrying) vs. hardware-bound (retrying just pays the
  same cost twice for the same outcome).

## Design decisions worth remembering for later agents

- **An assumption made before the real consumer is inspected needs
  re-auditing once it is - not just extended.** `duration_seconds` being
  "fixed by the checkpoint" was reasonable when Agent 3's schema was
  written, before this agent's actual workflow existed to check it
  against. It was wrong. This is the same lesson `Prompt_generation.md`
  Challenge 6 already drew about a different field - worth treating as a
  pattern to watch for generally, not a one-off correction.
- **A failed request and a failed job are not the same thing, and
  conflating them can cascade into worse failures than the original one.**
  Challenge 4's polling bug caused Challenge 5's interrupt bug - a
  detection problem, left unfixed, turned a defensive safety measure into
  an active source of data loss (killed jobs that would have succeeded).
  Fix the detection before building anything on top of it.
- **Subclassing to extend a working, tested file beats editing it in
  place**, even when the "shared" logic (queue/poll/fetch) really is
  generic. `video_comfyui_client.py` never touches Agent 4's
  `comfyui_client.py` - overriding just the one method that genuinely
  differs (`wait_for_completion`'s output-key assumption) keeps Agent 4's
  already-proven code provably unaffected by anything built afterward.
- **A retry count is a cost decision, not just a resilience knob.**
  Agent 4's retry default doesn't transfer to Agent 5 just because the
  retry *mechanism* does - the right number depends on what a retry
  actually costs, which changed by roughly two orders of magnitude between
  these two agents.
