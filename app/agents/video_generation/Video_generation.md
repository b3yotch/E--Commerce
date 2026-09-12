# Agent 6: Video Generation Agent

## Purpose

Takes Agent 3's `PromptGenerationOutput` (for `video_prompt` per theme)
**and** Agent 5's `ImageSelectionOutput` (for the already-chosen source
frame per theme), and turns each theme's `video_prompt` into a generated
video via image-to-video generation against a local ComfyUI instance.
Sixth stage of the AI Product Creative Generation Workflow - and the first
agent in this pipeline that consumes two upstream outputs at once, since
image-to-video generation genuinely needs both a prompt and a starting
image, not just one or the other.

Runs `NimVideo/cogvideox-2b-img2vid` - a community image-to-video
fine-tune of CogVideoX-2B (there's no official THUDM CogVideoX-2B I2V
release; only 5B-I2V exists upstream), confirmed back when Agent 3's
schema was being finalized (see `Prompt_generation.md` Challenge 6).

This landed as its own agent, not an extension of Agent 4's "Image/Video
Generation" combined agent as originally sketched in the pipeline
diagram - decided explicitly rather than left to drift, since video's
cost profile (minutes per attempt vs. Agent 4's seconds) is different
enough to warrant its own retry/batching decisions.

**Renumbered from Agent 5 to Agent 6** once Agent 5 (Image Selection) was
actually built - see Challenge 8. This agent originally consumed Agent 4's
raw `ImageGenerationOutput` directly, picking `images[0]` as a placeholder
source frame (see Challenge 6, original form). It now consumes Agent 5's
`ImageSelectionOutput` instead, which already carries a judged
`selected_local_path` per theme - no picking logic lives in this agent
anymore.

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
expensive enough that batching would need a critic to justify, same as it
did for images (see Challenge 6).

1. **start** - loads the API-format ComfyUI workflow JSON once (not
   per-theme, since the template doesn't change across themes - only the
   per-call patches do) and generates a `run_id`, same convention as
   Agent 4's run-scoped output folders.
2. **generate** (`nodes.py` + `video_comfyui_client.py`) - finds Agent 5's
   already-judged `ThemeSelectionResult` whose `source_setting` matches the
   current theme and reads its `selected_local_path` directly, uploads
   that frame back to ComfyUI, builds and queues the video workflow, polls
   to completion, and copies the result into this project's own output
   directory. A theme with no usable source frame (empty
   `selected_local_path`, or the theme missing from Agent 5's output
   entirely because Agent 4 skipped it further upstream) is recorded as an
   explicit `skipped_no_source_image` result rather than burning a retry
   on something retrying can't fix.
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

A real full pipeline run later measured this stage at **1963.87s (~33
minutes)** for 2 videos, in line with the per-attempt estimate above.

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

### 6. Source-frame selection had no critic to ask - resolved once Agent 5 (Image Selection) existed

Agent 4 generates multiple candidate images per theme and explicitly
defers picking a "best" one to a future critic agent (see
`Image_generation.md` Challenge 2 - no critic exists yet, so "oversample
and filter" would just produce extra files with nothing to do the
filtering). This agent needed exactly *one* source frame per theme to
animate, and inherited the same gap one level down: there was still no
critic to ask.

Originally resolved with `theme_result.images[0]` - the first of Agent 4's
candidates, arbitrary rather than chosen - flagged explicitly in both
`schema.py`'s docstring and inline at the selection site
(`_pick_source_image` in `nodes.py`) as a placeholder, not a considered
decision.

**Now actually resolved**, once Agent 5 (Image Selection) was built (see
`Image_selection.md`). `_pick_source_image` is gone entirely - it existed
only to hold that placeholder logic. `nodes.py` now looks up Agent 5's
`ThemeSelectionResult` for the current theme (matched by `source_setting`,
same convention used throughout this pipeline) and reads
`selected_local_path` directly. `state.py`'s upstream-images field was
renamed from `images: ImageGenerationOutput` to
`selection: ImageSelectionOutput` - not just a rename, the type changed
too, deliberately, so a stale read against the old shape would fail loudly
at the type level rather than silently doing the wrong thing.

One small, easy-to-miss bug caught during the renumbering that produced
this fix (see Challenge 8): `generate_node`'s ComfyUI `filename_prefix`
was still `f"agent5_{theme_index}"`, a leftover from when this agent was
itself numbered 5. It doesn't error - it just mislabels every saved video
file with the wrong agent number - which is exactly the kind of bug that
survives silently forever if nothing forces a full read-through of the
file. Now `f"agent6_{theme_index}"`.

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

### 8. Renumbered from Agent 5 to Agent 6 once Agent 5 (Image Selection) was actually built

Agent 4's own `schema.py` docstring anticipated the eventual critic as
"Agent 5" before this agent existed. This agent got built first and
temporarily claimed that number, using the `images[0]` placeholder
described in Challenge 6. Once Image Selection was actually designed and
built, the mismatch became worth fixing rather than living with
permanently: the critic is now Agent 5 (matching what Agent 4's docstring
always expected), and this agent moved to Agent 6 - agent number matches
graph position again.

Renumbering touched more than a comment: `schema.py` and `state.py` here
were updated to reference Agent 5 by name and consume
`ImageSelectionOutput` (see Challenge 6's resolution above), and
`nodes.py`'s ComfyUI filename prefix bug (also Challenge 6) was caught
specifically *because* the renumbering forced a full read-through of this
file rather than a targeted edit.

### 9. LangGraph-native checkpointing added, given how expensive a partial loss actually is here

This agent's own measured cost (Challenge 2: up to ~4500s per attempt,
~1964s for a real 2-video run) makes it one of two stages (with Agent 4)
where losing partial progress to a crash is genuinely expensive - unlike
Agents 1-3 and Agent 5, where redoing a whole stage from zero costs little.

Rather than the flat per-stage JSON checkpoint used for the cheaper
stages (write the whole output once a stage fully finishes), this agent's
`graph.py` now accepts an optional `checkpointer` at compile time,
compiled with LangGraph's own `AsyncSqliteSaver` by the pipeline runner.
LangGraph persists state after every node, not just at the end - so a
crash after video 1 of 2 succeeded resumes at video 2, not video 0, via
`graph.aget_state(config)` distinguishing three cases (never started,
partway through, already finished) rather than the flat JSON mechanism's
two (finished or not). The module-level `video_generation_graph` compiled
at import time stays uncheckpointed, for any caller that doesn't need
resumability - the pipeline runner builds its own checkpointed instance
separately, since the checkpointer's connection lifecycle belongs to
whoever's actually running the pipeline.

## Deliberately deferred (not gaps, decisions)

- **Candidate selection / oversampling**: see Challenge 6 - now resolved
  by Agent 5 (Image Selection). Whether Agent 5's judgment itself should
  ever trigger Agent 4 to regenerate (rather than falling back to a
  best-available pick) is Agent 5's own deferred decision, not this
  agent's - see `Image_selection.md`.
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
- **A video-quality critic (post-generation)**: judging the *finished*
  video, not just the source frame that went into it, was discussed
  during Agent 5's design and deliberately deferred until Agent 5 proved
  itself - see `Image_selection.md`. Would need a flag-and-pass-through
  failure mode, not auto-regeneration, for the same reason
  `max_video_gen_retries` sits at 0: a failed video costs another ~60
  minutes to redo.

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
- **A placeholder that's flagged loudly at write time is worth the
  discipline, even months later.** `_pick_source_image`'s docstring
  ("PLACEHOLDER SELECTION... revisit once a critic exists") is exactly why
  Challenge 6's resolution was a clean swap instead of an archaeology
  project - the thing to change and the reason it existed were both
  written down at the point the shortcut was taken, not left to be
  reconstructed later.
- **A renumbering is a good forcing function for a full read-through, not
  just a find-and-replace.** The `agent5_`/`agent6_` filename-prefix bug
  in Challenge 6/8 wasn't caught by design review - it was caught because
  updating this file's agent number required actually reading the whole
  file, not just the parts that obviously referenced "Agent 5" as a
  concept.
- **Match the checkpointing mechanism's granularity to what a partial loss
  actually costs**, not uniformly across every agent. This agent and
  Agent 4 get LangGraph's own node-level checkpointer specifically because
  a partial loss here is expensive (minutes to an hour); Agents 1-3 and 5
  stay on a cheaper flat-JSON, whole-stage-only mechanism because a
  partial loss there is cheap. Neither is a strictly better default - it's
  a cost-matched choice per agent, the same category of reasoning as the
  retry-count lesson above.