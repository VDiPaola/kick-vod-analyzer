# Kick VOD Analyser

Turns a full-length Kick VOD into a timestamped activity timeline: which game was
on screen, when the streamer switched to reacting to videos, when they went AFK.

A 12-hour VOD costs a few cents to classify.

## What it produces

Four files per VOD, in `out/<vod_id>/`:

| File | Purpose |
| --- | --- |
| `timeline.json` | Machine-readable segments plus every individual sample, for downstream tools |
| `chapters.vtt` | WebVTT chapter track. Drop it into VLC or a web player |
| `segments.csv` | Flat table for spreadsheets and dataframes |
| `summary_report.md` | Human-readable overview: duration per game, AFK share, timeline table |

## How it works

```
Kick VOD URL
     |
     +-- resolve metadata and the HLS master playlist
     |
     +-- pick renditions:  160p for detection,  720p for frame extraction
     |
     +-- scene detection (keyframes only) --> candidate timestamps
     |        + heartbeat every 15 min      --> gaps get a checkpoint too
     |
     +-- burst extraction: 4 frames per timestamp (T-6s, T-2s, T+2s, T+6s)
     |        composited into one 2x2 grid image
     |
     +-- chat window (optional): +/-45s, deduplicated, noise-filtered, top 30 lines
     |
     +-- multimodal classification (Gemini or OpenAI, sync or batch)
     |
     +-- temporal smoothing: absorb alt-tabs, confirm real transitions
     |
     +-- timeline.json / chapters.vtt / segments.csv / summary_report.md
```

Four design choices carry most of the cost and runtime savings:

**One grid image instead of four frames.** A 2x2 composite costs roughly a
quarter of the vision tokens of four separate images, and gives the model a
12-second motion window so it can tell a paused menu from live gameplay.

**Keyframe-only scene detection.** `-skip_frame nokey` decodes roughly one frame
per two to four seconds instead of every frame. Real cuts always land on or next
to a keyframe, so nothing meaningful is lost.

**Separate renditions per stage.** Scene detection reads the whole VOD end to
end, so it runs on the smallest variant available (often 160p at 230 kbps).
Frame extraction only fetches segments around each sample point, so it can
afford 720p. On a typical Kick VOD this is a 40x reduction in bytes downloaded.

**Sampling driven by change, not by the clock.** Fixed 30-second sampling on a
10-hour VOD means 1,200 requests. On a measured 12-hour reaction-and-gaming VOD,
scene detection fired 909 times and collapsed to 345 classification points after
the 45-second minimum gap. `--max-samples` caps it further when you want a fixed
budget.

## Install

Requires Python 3.11+ and `ffmpeg`/`ffprobe` on `PATH`.

From the project root:

```bash
pip install -e ".[gemini,kick,dev]"
```

Extras:

- `gemini` - the Google Gemini client
- `openai` - the OpenAI client
- `api` - FastAPI and uvicorn for the REST API and debug UI
- `kick` - `curl_cffi`, for browser TLS impersonation. Kick sits behind
  Cloudflare and rejects the default Python TLS fingerprint, so without this
  extra every Kick request returns 403.
- `dev` - pytest and coverage

Then create `.env.local` in the project root with the key for whichever provider
you use:

```
GEMINI_API_KEY=your-key-here
```

### How to invoke it

Two forms work. Use the module form unless you have confirmed the console script
is on your `PATH`:

```bash
python -m kick_vod_analyser.cli <command> [options]
```

`pip install -e` also creates a `kick-vod-analyser` executable, but pip places it
in a user Scripts directory that is often not on `PATH` (pip prints a warning
when this happens). Check with `where kick-vod-analyser` on Windows or
`which kick-vod-analyser` elsewhere. If it resolves, `kick-vod-analyser` is
interchangeable with `python -m kick_vod_analyser.cli` in every example below.

On Windows, prefix commands with `PYTHONIOENCODING=utf-8` if a stream title
contains emoji, otherwise printing the title can raise a `UnicodeEncodeError`.

## Usage

Every example uses a real Kick VOD URL. Substitute your own; the URL must be a
VOD, in either the `kick.com/<channel>/videos/<uuid>` or `kick.com/video/<uuid>`
form. Run all commands from the project root.

**Inspect a VOD.** No API calls, no cost. Start here to confirm the URL resolves:

```bash
python -m kick_vod_analyser.cli info --url "https://kick.com/xqc/videos/709c0cd8-b2d5-4b9d-b47f-e969a84fcd65"
```

**Estimate cost** for an assumed request count:

```bash
python -m kick_vod_analyser.cli estimate --url "https://kick.com/xqc/videos/709c0cd8-b2d5-4b9d-b47f-e969a84fcd65" --provider gemini --mode batch --samples 100
```

**Plan the run** without calling any classification API. This does run scene
detection, so it takes as long as a real run minus the classification step:

```bash
python -m kick_vod_analyser.cli analyse --url "https://kick.com/xqc/videos/709c0cd8-b2d5-4b9d-b47f-e969a84fcd65" --dry-run
```

**Smoke test the whole pipeline without credentials or cost**, using the mock
provider and a small sample cap:

```bash
python -m kick_vod_analyser.cli analyse --url "https://kick.com/xqc/videos/709c0cd8-b2d5-4b9d-b47f-e969a84fcd65" --provider mock --chat none --max-samples 5
```

**First real run.** Cap the samples so a misconfiguration costs a fraction of a
cent rather than a full VOD:

```bash
python -m kick_vod_analyser.cli analyse --url "https://kick.com/xqc/videos/709c0cd8-b2d5-4b9d-b47f-e969a84fcd65" --provider gemini --chat none --max-samples 5
```

**Full run**, batched at half price, chat enabled:

```bash
python -m kick_vod_analyser.cli analyse --url "https://kick.com/xqc/videos/709c0cd8-b2d5-4b9d-b47f-e969a84fcd65" --provider gemini --mode batch
```

Results land in `out/<vod_id>/`. Scene detection and classifications are cached
in `work/cache.sqlite`, so re-running the same VOD skips work already done. The
first run on a 12-hour VOD spends about 24 minutes in scene detection; every
later run on that VOD starts from the cache.

**Read the results:**

```bash
cat out/709c0cd8-b2d5-4b9d-b47f-e969a84fcd65/summary_report.md
```

### Options

| Flag | Default | Meaning |
| --- | --- | --- |
| `--provider` | `gemini` | `gemini`, `openai`, or `mock` |
| `--model` | provider default | Override the model id |
| `--mode` | `sync` | `sync` returns immediately; `batch` is half price with a 24h window |
| `--chat` | `none` | `none`, `file`, or `kick` (see below) |
| `--chat-file` | | Chat JSON or JSONL, required when `--chat file` |
| `--scene-threshold` | `0.35` | Higher means fewer scene triggers |
| `--heartbeat` | `900` | Seconds between fallback checkpoints |
| `--max-samples` | `0` | Cap on classified points, 0 for unlimited |
| `--dry-run` | off | Plan and cost only |
| `--no-resume` | off | Ignore cached scenes and classifications |
| `--keep-frames` | off | Retain the raw burst frames |
| `--no-wait` | off | Submit the batch job and exit |

Use `mock` as the provider to exercise the whole pipeline without credentials or
cost. It derives a deterministic verdict from each grid image.

## Chat

Kick publishes no documented VOD chat API, but the web player's replay endpoint
is reachable without a login and returns complete history. The pipeline still
treats chat as enrichment, not a dependency: every source degrades to an empty
index rather than failing the run.

### Downloading chat on its own

`chat` downloads the full replay for a VOD to a JSONL file. This works
standalone, without ffmpeg or any model credentials:

```bash
python -m kick_vod_analyser.cli chat --url "https://kick.com/xqc/videos/709c0cd8-b2d5-4b9d-b47f-e969a84fcd65"
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--out` | `out/<vod_id>/chat.jsonl` | Output path |
| `--raw` | off | Write Kick's original records (ids, sender identity, badges) instead of the normalised format |
| `--workers` | `8` | Parallel download threads |
| `--chunk-seconds` | `600` | VOD seconds walked per thread task |

An 8.6 hour VOD with 30,000 messages downloads in about 20 seconds with 16
workers. The command exits non-zero when no messages were collected.

The normalised format is one message per line:

```json
{"offset_seconds": 1234.0, "username": "viewer", "text": "PogU wow", "emotes": ["PogU"]}
```

Kick embeds emotes in message text as `[emote:<id>:<name>]`. The normaliser
replaces each token with its name and lists the names under `emotes`.

### How the download works

`web.kick.com/api/v1/chat/{channel_id}/history` accepts two query forms:

- `start_time=<ISO 8601>` returns one fixed five second bucket, floor aligned,
  **truncated to its earliest 25 messages**. Busy moments lose chat.
- `cursor=<epoch microseconds>` returns the 25 messages before the cursor,
  newest first, plus the cursor for the next older page. Paging is complete.

The downloader uses the cursor form. The VOD window is split into chunks of
`--chunk-seconds`, and each chunk is walked backwards from its end on its own
thread until a message older than the chunk start appears. Request count scales
with message volume, not VOD length, and quiet stretches cost one request per
chunk. Messages are deduplicated by id and sorted by `created_at`.

Measured in August 2026: no authorization header is needed, and 16 concurrent
workers were not throttled. `KVA_KICK_AUTH_TOKEN` is read if set and sent as a
bearer token, for the case where Kick begins requiring one. `curl_cffi` (the
`kick` extra) is still required to pass Cloudflare.

### Chat sources for `analyse`

`--chat kick` runs the same download inline and indexes the result:

```bash
python -m kick_vod_analyser.cli analyse --url "https://kick.com/xqc/videos/709c0cd8-b2d5-4b9d-b47f-e969a84fcd65" --chat kick
```

`--chat none` (the default) skips chat entirely and classifies from the
screenshots alone.

`--chat file` loads an export from `chat` or any external scraper. `--chat-file`
is required with it:

```bash
python -m kick_vod_analyser.cli analyse --url "https://kick.com/xqc/videos/709c0cd8-b2d5-4b9d-b47f-e969a84fcd65" --chat file --chat-file ./out/709c0cd8-b2d5-4b9d-b47f-e969a84fcd65/chat.jsonl
```

The loader accepts a JSON array, a JSON object with a `messages`/`comments`/`data`
key, or JSON Lines, and understands Kick's own payload shape, `chat-downloader`
output, and this package's export format. The minimum a record needs is message
text plus something to place it on the timeline:

```json
{"offset_seconds": 1234.5, "username": "viewer", "text": "what game is this"}
```

Without chat the classifier runs vision-only and lowers its confidence where the
visuals are ambiguous. Chat matters most for reaction content, where the browser
window alone does not say what is being watched.

### Chat window construction

For each sample point, messages within +/-45 seconds are collapsed into at most
30 lines:

- Repeats are deduplicated with a multiplier (`KEKW (x45)`), which is what makes
  a busy chat fit in a few hundred tokens.
- Bot accounts and `!command` messages are dropped.
- Content-free single tokens (`lol`, `W`, `gg`) are dropped unless they carry an
  emote.
- Lines are ranked by how much they say about what is on screen. Phrases like
  "what game is this" outrank spam volume, which is capped so a repeat wave can
  never displace a viewer naming the content.

## Temporal smoothing

Raw per-sample verdicts fragment badly: streamers alt-tab, games show loading
screens, the model occasionally misreads a frame. Smoothing enforces three
rules.

**A single verdict never confirms a transition.** A state earns its own segment
by being observed repeatedly: `confirm_consecutive` samples agreeing, or two
samples spread past `min_segment_seconds`.

**A-B-A sandwiches rejoin A.** A brief desktop click or loading screen between
two stretches of the same game is absorbed. The absorption is bounded, because
rewriting 30 seconds is smoothing and rewriting 700 seconds is fabrication.

**The bound scales with the sampling cadence.** At a 900-second heartbeat, the
gap around a lone verdict is dominated by how often you sampled, not by how long
the interruption lasted, so anything under half a sampling interval is treated
as unresolvable. At a 60-second cadence the same gap is well resolved and the
state is kept.

Low-confidence samples carry the current state forward rather than opening a new
one: a hesitant verdict is weaker evidence than continuity.

## Cost

Rates as of August 2026, per million tokens. Batch endpoints bill at half.

| Model | Input | Output | 100 requests (batch) | 345 requests (batch) |
| --- | ---: | ---: | ---: | ---: |
| `gemini-3.5-flash-lite` (default) | $0.30 | $2.50 | ~$0.024 | ~$0.084 |
| `gpt-4o-mini` | $0.15 | $0.60 | ~$0.009 | ~$0.032 |
| `gemini-3.7-flash` | $0.75 | $3.75 | ~$0.050 | ~$0.172 |

`gemini-2.5-flash-lite` was cheaper still at $0.10/$0.40, but Google has retired
it for new API keys: requests return `404 NOT_FOUND` telling you to move to
`gemini-3.5-flash-lite`. If an older key of yours still has access, pass
`--model gemini-2.5-flash-lite`.

Per request: ~258 vision tokens for the grid, ~300 for the chat window, ~320 for
the system prompt and schema, ~90 output.

A real 12-hour VOD sampled without a cap produced 345 requests, which is $0.084
on the default model batched and $0.032 on `gpt-4o-mini`. Cap it with
`--max-samples` when you want a fixed budget.

Prices live in `classify/pricing.py` as a plain table. Add or override an entry
without touching the estimator.

## Caching and resume

Everything expensive is cached in `work/cache.sqlite`:

- Scene detection results, keyed by VOD id
- Classifications, keyed by VOD id, sample offset, and model

A second run reuses both and classifies only what is new. `--no-resume` forces a
fresh pass. Changing the model invalidates classifications but not scene
detection, so trying a different model does not re-download the VOD.

## Rate limits and transient errors

Synchronous classification retries each request on rate-limit and server
errors (HTTP 408, 429, 5xx, or messages mentioning quota, rate limit, or
overload). Up to 8 attempts per request. The wait honours the provider's
`Retry-After` header or Gemini's `retryDelay` hint when present, otherwise it
backs off exponentially from 2s, capped at 120s. Free-tier quota exhaustion
pauses the run instead of failing it. Errors that are not transient (bad API
key, invalid request) are recorded against the sample and the run continues.

## Batch mode

Batch halves the token price in exchange for an asynchronous turnaround of up to
24 hours.

- **Gemini** uploads each grid through the Files API and references it by URI
  from a JSONL manifest. Inline base64 would breach the request size ceiling
  well before a full-length VOD is covered.
- **OpenAI** embeds each grid as a data URI directly in the JSONL.

Without `--no-wait` the pipeline submits the job and polls until it reaches a
terminal state, then writes the outputs:

```bash
python -m kick_vod_analyser.cli analyse --url "https://kick.com/xqc/videos/709c0cd8-b2d5-4b9d-b47f-e969a84fcd65" --provider gemini --mode batch
```

`--no-wait` submits and exits immediately, writing the job id to
`work/<vod_id>/batch_job.txt`:

```bash
python -m kick_vod_analyser.cli analyse --url "https://kick.com/xqc/videos/709c0cd8-b2d5-4b9d-b47f-e969a84fcd65" --provider gemini --mode batch --no-wait
```

Re-run the same command later without `--no-wait` to collect the results. Scene
detection and frame extraction come from the cache, so only the classification
step repeats.

## Configuration

Settings come from environment variables, `.env`, and `.env.local`, with CLI
flags taking precedence.

| Variable | Default |
| --- | --- |
| `GEMINI_API_KEY` / `OPENAI_API_KEY` | |
| `KVA_GEMINI_MODEL` | `gemini-3.5-flash-lite` |
| `KVA_OPENAI_MODEL` | `gpt-4o-mini` |
| `KVA_WORK_DIR` | `./work` |
| `KVA_OUT_DIR` | `./out` |
| `KVA_SAMPLING_SCENE_THRESHOLD` | `0.35` |
| `KVA_SAMPLING_HEARTBEAT_SECONDS` | `900` |
| `KVA_SAMPLING_MIN_GAP_SECONDS` | `45` |
| `KVA_SAMPLING_PHASH_DISTANCE` | `6` |
| `KVA_KICK_AUTH_TOKEN` | unset |
| `KVA_KICK_CHAT_WORKERS` | `8` |
| `KVA_CHAT_WINDOW_SECONDS` | `45` |
| `KVA_CHAT_MAX_LINES` | `30` |
| `KVA_SMOOTHING_MIN_SEGMENT_SECONDS` | `60` |
| `KVA_SMOOTHING_ALT_TAB_WINDOW_SECONDS` | `90` |
| `KVA_SMOOTHING_CONFIRM_CONSECUTIVE` | `2` |
| `KVA_SMOOTHING_MIN_CONFIDENCE` | `0.35` |

## REST API and debug UI

`serve` starts an HTTP API for queueing analyses plus a browser UI for watching
them run. Install the extra first:

```bash
pip install -e ".[api]"
python -m kick_vod_analyser.cli serve --port 8765
```

Open `http://127.0.0.1:8765/` for the debug UI and `/docs` for the interactive
OpenAPI reference. `--work-dir`, `--out-dir`, `--host`, and `-v` work as on
`analyse`.

Jobs run one at a time on a background worker, in submission order. The queue
is persisted to `work/jobs.sqlite`, so it survives restarts; a job that was
running when the process stopped is marked `failed` on the next start.

### Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/jobs` | Queue a VOD. Returns `202` and the job |
| `GET` | `/jobs?status=&limit=` | List jobs, newest first |
| `GET` | `/jobs/{id}` | Job status, stage, result, error |
| `DELETE` | `/jobs/{id}` | Cancel a queued job. `409` if it already started |
| `POST` | `/jobs/{id}/retry` | Queue a finished job again with the same request |
| `DELETE` | `/jobs/{id}/record` | Delete a finished job's record and events |
| `GET` | `/jobs/{id}/events?after=` | Progress events. Pass the returned `cursor` back as `after` to tail |
| `GET` | `/jobs/{id}/outputs` | Output files with existence and size |
| `GET` | `/jobs/{id}/outputs/{name}` | Download one output (`timeline_json`, `chapters_vtt`, `segments_csv`, `summary_md`) |
| `GET` | `/queue` | Worker liveness, current job, counts per status |
| `GET` | `/logs?limit=` | Recent worker log lines |
| `GET` | `/health` | Liveness and version |

Job statuses: `queued`, `running`, `succeeded`, `failed`, `cancelled`.

### Queueing a job

The request body mirrors the `analyse` flags. Only `url` is required:

```bash
curl -X POST http://127.0.0.1:8765/jobs   -H "content-type: application/json"   -d '{"url": "https://kick.com/xqc/videos/709c0cd8-b2d5-4b9d-b47f-e969a84fcd65", "provider": "gemini", "mode": "batch", "max_samples": 50}'
```

| Field | Default | Meaning |
| --- | --- | --- |
| `url` | required | Kick VOD URL |
| `provider` | `gemini` | `gemini`, `openai`, or `mock` |
| `model` | provider default | Model id override |
| `mode` | `sync` | `sync` or `batch` |
| `chat` | `none` | `none`, `file`, or `kick` |
| `chat_file` | | Required when `chat` is `file` |
| `scene_threshold` | settings | 0 to 1 |
| `heartbeat_seconds` | settings | Seconds between fallback checkpoints |
| `max_samples` | settings | 0 for unlimited |
| `resume` | `true` | Reuse cached scenes and classifications |
| `keep_frames` | `false` | Retain raw burst frames |
| `dry_run` | `false` | Plan and cost only |
| `wait_for_batch` | `true` | Poll a batch job until it finishes |

Poll `/jobs/{id}` until `status` is terminal, then read `result.outputs` or
fetch the files through `/jobs/{id}/outputs/{name}`. A `failed` job carries the
traceback or the pipeline's error list in `error`.

### Debug UI

The page at `/` shows the queue counts, a submission form, the job list with a
status filter, and a detail panel per job with tabs for progress events, the
result summary, output files (with download links), the original request, and
errors. The worker log panel tails the most recent log lines. It refreshes every
two seconds; untick "auto refresh" to freeze it.

### Embedding

`create_app(settings)` returns a FastAPI application, so it can be mounted
inside another ASGI app or run directly:

```bash
uvicorn --factory kick_vod_analyser.api.app:build_default_app --port 8765
```

## Library use

Save as `example.py` in the project root and run `python example.py`:

```python
from kick_vod_analyser import Pipeline, RunOptions, load_settings
from kick_vod_analyser.ingest.chat import build_chat_source

URL = "https://kick.com/xqc/videos/709c0cd8-b2d5-4b9d-b47f-e969a84fcd65"

settings = load_settings()
settings.sampling.max_samples = 5

pipeline = Pipeline(settings, chat_source=build_chat_source("none"))
report = pipeline.run(RunOptions(url=URL, provider="gemini", chat_source_kind="none"))

for segment in report.timeline.segments:
    print(f"{segment.start_seconds:8.0f}s  {segment.label}")

print("cost: $%.4f" % report.cost["total_cost_usd"])
```

## Categories

| Category | Covers |
| --- | --- |
| Gaming | Any game, including menus, queues, and loading screens |
| Just Chatting / Podcast | Talking to chat, guests, no other media on screen |
| IRL / Outdoors | Phone or camera streams away from a desk |
| Reaction / Media Share | Watching someone else's video, browsing sites |
| Coding / Creative | Editors, art tools, music production |
| Gambling / Slots | Casino sites and slot machines |
| Intermission / AFK / BRB | BRB cards, empty chair, static placeholder scenes |
| Technical Difficulties / Offline | The stream itself is broken |

Each classification also carries a specific title, a granular sub-activity, an
on-screen flag, an AFK flag, a confidence score, and a one-sentence justification
naming the visual markers that drove the decision.

## Tests

Run from the project root:

```bash
python -m pytest -q
```

With a coverage report:

```bash
python -m pytest -q --cov=kick_vod_analyser --cov-report=term-missing
```

A single file, or a single test:

```bash
python -m pytest tests/test_smoothing.py -v
python -m pytest tests/test_smoothing.py::TestAltTabAbsorption -v
```

585 tests, 96% line coverage. The suite includes real ffmpeg integration:
`tests/conftest.py` builds a 90-second synthetic clip with three visually
distinct acts, and the sampling, extraction, and end-to-end pipeline tests run
against it. Provider and network paths are exercised through fakes; no test
touches the network.

## Troubleshooting

**`kick-vod-analyser: command not found`** — the console script is not on your
`PATH`. Use `python -m kick_vod_analyser.cli` instead, which always works from
the project root.

**`404 NOT_FOUND ... no longer available to new users`** — the model you asked
for has been retired for new API keys. Pass a current one with `--model`, or
update `KVA_GEMINI_MODEL` in your `.env.local`. List what your key can reach:

```bash
python -c "from google import genai; from kick_vod_analyser.config import load_settings; c=genai.Client(api_key=load_settings().gemini_api_key); print('
'.join(m.name for m in c.models.list() if 'generateContent' in (m.supported_actions or [])))"
```

Note that a value in `.env` or `.env.local` overrides the built-in default, so a
stale `KVA_GEMINI_MODEL` line will keep pinning a retired model even after an
upgrade. Check what is actually in effect:

```bash
python -c "from kick_vod_analyser.config import load_settings; print(load_settings().gemini_model)"
```

**`404 NOT_FOUND` / `Kick video API returned 404` on a VOD that plays fine in a
browser** — handled automatically, but worth understanding. Kick VOD URLs now
carry a version 7 UUID that no read endpoint accepts, and that id appears
nowhere in the API payloads. The resolver decodes the creation timestamp the v7
id embeds, looks the channel's video listing up, and matches on time to recover
the version 4 `video.uuid` the endpoints do accept. You will see this in the log:

```
mapped URL id to video uuid 1e7fa39e-09ad-47c5-9f25-f08eedafa16d (4.0s apart)
```

The listing only covers recent VODs, so an older one may no longer be findable.
The `kick.com/video/<id>` form has no channel to look up and cannot be mapped;
use `kick.com/<channel>/videos/<id>`.

Because the mapping resolves to the canonical video id, `out/<vod_id>/` is named
after the v4 id, not the id in the URL you pasted.

**`GEMINI_API_KEY is not set`** — create `.env.local` in the project root with
`GEMINI_API_KEY=your-key-here`, or export it in your shell.

**Every Kick request returns 403** — `curl_cffi` is missing. Install the extra:

```bash
pip install -e ".[kick]"
```

**`required binaries not found on PATH: ffmpeg`** — install ffmpeg and make sure
both `ffmpeg` and `ffprobe` resolve.

**`UnicodeEncodeError` when printing a stream title** — Windows console encoding.
Prefix the command with `PYTHONIOENCODING=utf-8`.

**A run finishes with an empty timeline and N non-fatal issues** — every
classification failed. The listed errors carry the reason; the model id and the
API key are the usual causes. Nothing is cached on failure, so fixing the cause
and re-running only repeats the classification step.

## Project layout

```
src/kick_vod_analyser/
  cli.py             command line interface
  api/
    app.py           FastAPI routes
    jobs.py          persistent job queue and worker thread
    ui.html          debug interface
  pipeline.py        stage orchestration, caching, batch polling
  config.py          settings
  models.py          pydantic domain models
  store.py           sqlite cache
  ffmpeg.py          ffmpeg and ffprobe wrappers
  ingest/
    vod.py           Kick URL -> metadata and playback URL
    chat.py          chat sources and the offset-indexed store
    http.py          Cloudflare-capable HTTP client
  sampling/
    renditions.py    HLS master playlist parsing and rendition selection
    scene.py         scene detection, heartbeats, sample planning
    burst.py         burst extraction and deduplication
    grid.py          2x2 compositing and perceptual hashing
  chatwindow/
    slicer.py        chat window condensation and ranking
  classify/
    prompts.py       system prompt and JSON schema
    base.py          provider interface and response parsing
    gemini.py        Gemini sync and batch
    openai_provider.py  OpenAI sync and batch
    factory.py       provider construction, mock provider
    pricing.py       price table and cost estimation
  postprocess/
    smoothing.py     debounce and hysteresis state machine
    outputs.py       timeline.json, chapters.vtt, segments.csv, summary_report.md
```

## Known limitations

- **Chat replay is best-effort.** Kick's history endpoint is undocumented and
  unversioned. If it breaks, use `--chat file` with an external scraper's export.
- **Scene boundaries quantise to keyframes**, roughly two to four seconds. This
  is well inside the 60-second segment floor, so it does not affect the timeline.
- **Live VODs report duration 0.** The pipeline falls back to probing the HLS
  playlist, but an in-progress stream's duration is a moving target.
- **Detection is network-bound.** A 12-hour VOD streams the full 160p rendition
  once, roughly 350 MB. On a connection sustaining ~0.8 MB/s from Kick's CDN that
  measured 24 minutes, about 28x realtime. The result is cached, so it is paid
  once per VOD. Frame extraction is unaffected: input seeking fetches only the
  segments around each sample point, measured at 4.2 seconds per point
  regardless of position in the VOD.
