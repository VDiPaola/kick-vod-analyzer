# Kick VOD Analyser

Paste in the link to a Kick stream recording (a VOD) and get back a timeline of
what the streamer was doing, minute by minute: which game was on screen, when
they switched to watching videos, when they stepped away from the desk.

The tool takes screenshots at the moments the picture changes, asks an AI model
what it sees, and stitches the answers into chapters you can open in a video
player or a spreadsheet. An 8-hour VOD costs about five cents to process.

## Contents

**Getting started**

- [Quick start](#quick-start)
- [What you get](#what-you-get)
- [How it works](#how-it-works)
- [Install](#install)
- [Running commands](#running-commands)

**Using the command line**

- [Usage](#usage)
- [Options](#options)
- [Chat](#chat)
- [Cost](#cost)
- [Batch mode](#batch-mode)
- [Configuration](#configuration)

**Other ways to use it**

- [REST API and debug UI](#rest-api-and-debug-ui)
- [Library use](#library-use)

**Reference**

- [Categories](#categories)
- [Tests](#tests)
- [Troubleshooting](#troubleshooting)
- [Project layout](#project-layout)
- [Glossary](#glossary)

Internals (sampling, chat download, smoothing, caching, limitations) are in
[TECHNICAL.md](TECHNICAL.md).

## Quick start

1. Install Python 3.11 or newer and [ffmpeg](https://ffmpeg.org/download.html).
   Check both work: `python --version` and `ffmpeg -version`.
2. From the project folder, install the tool:

   ```bash
   pip install -e ".[gemini,kick,dev]"
   ```

3. Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/)
   and save it in a file named `.env.local` in the project folder:

   ```
   GEMINI_API_KEY=your-key-here
   ```

4. Try a small run first. This classifies only 5 points, so a mistake costs a
   fraction of a cent:

   ```bash
   python -m kick_vod_analyser.cli analyse --url "https://kick.com/xqc/videos/709c0cd8-b2d5-4b9d-b47f-e969a84fcd65" --provider gemini --chat none --max-samples 5
   ```

5. Open the summary:

   ```bash
   cat out/<vod_id>/summary_report.md
   ```

Drop `--max-samples 5` for a full run. Add `--mode batch` to pay half price in
exchange for waiting up to 24 hours. No API key yet? Use `--provider mock` to
run the whole pipeline for free with made-up verdicts.

## What you get

Four files per VOD, in `out/<vod_id>/`:

| File | Purpose |
| --- | --- |
| `summary_report.md` | Human-readable overview: duration per game, AFK share, timeline table. Start here |
| `chapters.vtt` | Chapter track. Drop it into VLC or a web player to jump between activities |
| `segments.csv` | Flat table for spreadsheets and dataframes |
| `timeline.json` | Machine-readable segments plus every individual sample, for downstream tools |

Each segment carries a category (see [Categories](#categories)), a specific title
such as the game name, a sub-activity, an on-screen flag, an AFK flag, and a
confidence score.

## How it works

1. The tool reads the VOD once at very low quality and notes every moment the
   picture changes noticeably. It also drops a checkpoint every 15 minutes so
   long, unchanging stretches are still covered.
2. At each of those moments it grabs four higher-quality screenshots spanning
   12 seconds and tiles them into one image.
3. Optionally, it collects what chat was saying around that moment.
4. The image and chat go to an AI model, which answers with a category, a title,
   and a confidence score.
5. The answers are cleaned up so that a quick alt-tab or loading screen does not
   split a gaming session into pieces.
6. The result is written out as the four files above.

Scene detection and classifications are cached in `work/cache.sqlite`, so
re-running the same VOD skips work already done. Details of every stage are in
[TECHNICAL.md](TECHNICAL.md).

## Install

Requires Python 3.11+ and `ffmpeg`/`ffprobe` on `PATH`.

```bash
pip install -e ".[gemini,kick,dev]"
```

| Extra | Needed for |
| --- | --- |
| `gemini` | Classifying with Gemini (the default) |
| `openai` | Classifying with OpenAI models |
| `kick` | Talking to Kick at all. Without it every Kick request returns 403 |
| `api` | The [REST API and debug UI](#rest-api-and-debug-ui) |
| `dev` | Running the [tests](#tests) |

Then create `.env.local` in the project root with the key for whichever provider
you use:

```
GEMINI_API_KEY=your-key-here
```

## Running commands

Run everything from the project root in this form:

```bash
python -m kick_vod_analyser.cli <command> [options]
```

`pip install -e` also creates a `kick-vod-analyser` executable, but it is often
not on `PATH`. If `kick-vod-analyser --help` works, it is interchangeable with
the module form.

On Windows, prefix commands with `PYTHONIOENCODING=utf-8` if a stream title
contains emoji.

| Command | What it does | Costs money? |
| --- | --- | --- |
| `info` | Show a VOD's title, duration, and available qualities | No |
| `estimate` | Predict the cost of a run | No |
| `chat` | Download the VOD's chat replay to a file | No |
| `analyse` | Build the activity timeline | Yes, unless `--dry-run` or `--provider mock` |
| `serve` | Start the REST API and debug UI | Only when jobs run |

## Usage

The URL must be a VOD, in either the `kick.com/<channel>/videos/<uuid>` or
`kick.com/video/<uuid>` form.

**Inspect a VOD.** No API calls, no cost. Start here to confirm the URL resolves:

```bash
python -m kick_vod_analyser.cli info --url "https://kick.com/xqc/videos/709c0cd8-b2d5-4b9d-b47f-e969a84fcd65"
```

**Estimate cost** for an assumed request count:

```bash
python -m kick_vod_analyser.cli estimate --url "https://kick.com/xqc/videos/709c0cd8-b2d5-4b9d-b47f-e969a84fcd65" --provider gemini --mode batch --samples 100
```

**Plan the run** without calling any classification API. This still runs scene
detection:

```bash
python -m kick_vod_analyser.cli analyse --url "https://kick.com/xqc/videos/709c0cd8-b2d5-4b9d-b47f-e969a84fcd65" --dry-run
```

**Smoke test without credentials or cost:**

```bash
python -m kick_vod_analyser.cli analyse --url "https://kick.com/xqc/videos/709c0cd8-b2d5-4b9d-b47f-e969a84fcd65" --provider mock --chat none --max-samples 5
```

**Full run**, batched at half price, chat enabled:

```bash
python -m kick_vod_analyser.cli analyse --url "https://kick.com/xqc/videos/709c0cd8-b2d5-4b9d-b47f-e969a84fcd65" --provider gemini --mode batch --chat kick
```

Results land in `out/<vod_id>/`. The first run on a 12-hour VOD spends about 24
minutes in scene detection; later runs on the same VOD start from the cache.

## Options

Flags for `analyse`:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--provider` | `gemini` | `gemini`, `openai`, or `mock` |
| `--model` | provider default | Override the model id |
| `--mode` | `sync` | `sync` returns immediately; `batch` is half price with a 24h window |
| `--chat` | `none` | `none`, `file`, or `kick` (see [Chat](#chat)) |
| `--chat-file` | | Chat JSON or JSONL, required when `--chat file` |
| `--scene-threshold` | `0.35` | Higher means fewer scene triggers |
| `--heartbeat` | `900` | Seconds between fallback checkpoints |
| `--max-samples` | `0` | Cap on classified points, 0 for unlimited |
| `--dry-run` | off | Plan and cost only |
| `--no-resume` | off | Ignore cached scenes and classifications |
| `--keep-frames` | off | Retain the raw burst frames |
| `--no-wait` | off | Submit the batch job and exit |

## Chat

Chat is optional context for the AI model. Without it the classifier works from
the screenshots alone and is less sure about reaction content, where the browser
window does not say what is being watched. If chat cannot be fetched the run
continues without it.

`--chat kick` downloads the replay from Kick during the run:

```bash
python -m kick_vod_analyser.cli analyse --url "https://kick.com/xqc/videos/709c0cd8-b2d5-4b9d-b47f-e969a84fcd65" --chat kick
```

`--chat file` loads a previously downloaded file:

```bash
python -m kick_vod_analyser.cli analyse --url "https://kick.com/xqc/videos/709c0cd8-b2d5-4b9d-b47f-e969a84fcd65" --chat file --chat-file ./out/<vod_id>/chat.jsonl
```

### Downloading chat on its own

`chat` saves the full replay to `out/<vod_id>/chat.jsonl`, one message per
line. It needs no ffmpeg or API key:

```bash
python -m kick_vod_analyser.cli chat --url "https://kick.com/xqc/videos/709c0cd8-b2d5-4b9d-b47f-e969a84fcd65"
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--out` | `out/<vod_id>/chat.jsonl` | Output path |
| `--raw` | off | Keep Kick's original records instead of the normalised format |
| `--workers` | `8` | Parallel download threads |
| `--chunk-seconds` | `600` | VOD seconds walked per thread task |

```json
{"offset_seconds": 1234.0, "username": "viewer", "text": "PogU wow", "emotes": ["PogU"]}
```

An 8.6 hour VOD with 30,000 messages downloads in about 20 seconds. Accepted
input formats for `--chat file` are listed in [TECHNICAL.md](TECHNICAL.md#chat).

## Cost

On the default model, an 8-hour VOD runs to roughly 230 classification
requests:

| Mode | Approximate cost |
| --- | --- |
| `--mode sync` | $0.11 |
| `--mode batch` | $0.06 |

Cost scales with the number of sample points, which depends on how often the
picture changes. `--max-samples N` caps it to a fixed budget; `estimate`
predicts the cost before you commit. Every run prints its actual cost at the
end.

## Batch mode

`--mode batch` halves the price in exchange for waiting up to 24 hours. By
default the command submits the job and waits for it. To submit and come back
later:

```bash
python -m kick_vod_analyser.cli analyse --url "..." --provider gemini --mode batch --no-wait
```

Re-run the same command without `--no-wait` to collect the results.

## Configuration

Settings come from environment variables, `.env`, and `.env.local`, with CLI
flags taking precedence. Most users only need an API key.

| Variable | Default | Meaning |
| --- | --- | --- |
| `GEMINI_API_KEY` / `OPENAI_API_KEY` | | Provider credentials |
| `KVA_GEMINI_MODEL` | `gemini-3.5-flash-lite` | Default Gemini model |
| `KVA_OPENAI_MODEL` | `gpt-4o-mini` | Default OpenAI model |
| `KVA_WORK_DIR` | `./work` | Cache and scratch files |
| `KVA_OUT_DIR` | `./out` | Output files |
| `KVA_SAMPLING_SCENE_THRESHOLD` | `0.35` | Scene change sensitivity, higher means fewer triggers |
| `KVA_SAMPLING_HEARTBEAT_SECONDS` | `900` | Seconds between fallback checkpoints |
| `KVA_SAMPLING_MIN_GAP_SECONDS` | `45` | Minimum spacing between sample points |
| `KVA_SAMPLING_PHASH_DISTANCE` | `6` | Perceptual hash distance below which grids count as duplicates |
| `KVA_KICK_AUTH_TOKEN` | unset | Optional bearer token for Kick's chat endpoint |
| `KVA_KICK_CHAT_WORKERS` | `8` | Parallel chat download threads |
| `KVA_CHAT_WINDOW_SECONDS` | `45` | Chat seconds either side of a sample point |
| `KVA_CHAT_MAX_LINES` | `30` | Chat lines sent per sample |
| `KVA_SMOOTHING_MIN_SEGMENT_SECONDS` | `60` | Shortest segment the timeline will contain |
| `KVA_SMOOTHING_ALT_TAB_WINDOW_SECONDS` | `90` | Longest interruption absorbed into the surrounding segment |
| `KVA_SMOOTHING_CONFIRM_CONSECUTIVE` | `2` | Agreeing samples needed to confirm a transition |
| `KVA_SMOOTHING_MIN_CONFIDENCE` | `0.35` | Verdicts below this carry the previous state forward |

## REST API and debug UI

`serve` starts an HTTP API for queueing analyses plus a browser UI for watching
them run:

```bash
pip install -e ".[api]"
python -m kick_vod_analyser.cli serve --port 8765
```

Open `http://127.0.0.1:8765/` for the debug UI and `/docs` for the interactive
OpenAPI reference. Jobs run one at a time in submission order. The queue is
persisted to `work/jobs.sqlite`, so it survives restarts.

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

The request body mirrors the `analyse` flags in snake_case (`max_samples`,
`chat_file`, `wait_for_batch`, and so on). Only `url` is required:

```bash
curl -X POST http://127.0.0.1:8765/jobs -H "content-type: application/json" -d '{"url": "https://kick.com/xqc/videos/709c0cd8-b2d5-4b9d-b47f-e969a84fcd65", "provider": "gemini", "mode": "batch", "max_samples": 50}'
```

Poll `/jobs/{id}` until `status` is terminal, then fetch the files through
`/jobs/{id}/outputs/{name}`. The full request schema is at `/docs`.

### Embedding

`create_app(settings)` returns a FastAPI application:

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

## Tests

```bash
python -m pytest -q
python -m pytest -q --cov=kick_vod_analyser --cov-report=term-missing
python -m pytest tests/test_smoothing.py -v
```

585 tests, 96% line coverage. `tests/conftest.py` builds a 90-second synthetic
clip with ffmpeg for the integration tests. No test touches the network.

## Troubleshooting

### `kick-vod-analyser: command not found`

Use `python -m kick_vod_analyser.cli` instead.

### `404 NOT_FOUND ... no longer available to new users`

The model has been retired for new API keys. Pass a current one with `--model`,
or update `KVA_GEMINI_MODEL` in `.env.local`. List what your key can reach:

```bash
python -c "from google import genai; from kick_vod_analyser.config import load_settings; c=genai.Client(api_key=load_settings().gemini_api_key); print('\n'.join(m.name for m in c.models.list() if 'generateContent' in (m.supported_actions or [])))"
```

A stale `KVA_GEMINI_MODEL` line in `.env` or `.env.local` overrides the
built-in default. Check what is in effect:

```bash
python -c "from kick_vod_analyser.config import load_settings; print(load_settings().gemini_model)"
```

### `Kick video API returned 404` on a VOD that plays in a browser

Handled automatically for `kick.com/<channel>/videos/<id>` URLs. The
`kick.com/video/<id>` form cannot be mapped; use the channel form. Old VODs may
no longer be findable. Note that `out/<vod_id>/` is named after Kick's internal
id, not the id in the URL. Details in
[TECHNICAL.md](TECHNICAL.md#kick-url-id-mapping).

### `GEMINI_API_KEY is not set`

Create `.env.local` in the project root with `GEMINI_API_KEY=your-key-here`.

### Every Kick request returns 403

`curl_cffi` is missing: `pip install -e ".[kick]"`.

### `required binaries not found on PATH: ffmpeg`

Install ffmpeg and make sure both `ffmpeg` and `ffprobe` resolve.

### `UnicodeEncodeError` when printing a stream title

Prefix the command with `PYTHONIOENCODING=utf-8`.

### A run finishes with an empty timeline and N non-fatal issues

Every classification failed. The listed errors carry the reason; the model id
and the API key are the usual causes. Fix the cause and re-run; only the
classification step repeats.

## Project layout

```
src/kick_vod_analyser/
  cli.py             command line interface
  api/               FastAPI routes, job queue, debug UI
  pipeline.py        stage orchestration, caching, batch polling
  config.py          settings
  models.py          pydantic domain models
  store.py           sqlite cache
  ffmpeg.py          ffmpeg and ffprobe wrappers
  ingest/            Kick VOD resolution, chat download, HTTP client
  sampling/          rendition selection, scene detection, burst extraction, grids
  chatwindow/        chat window condensation and ranking
  classify/          prompts, Gemini and OpenAI providers, mock provider, pricing
  postprocess/       temporal smoothing and output writers
```

## Glossary

| Term | Meaning |
| --- | --- |
| VOD | Video on demand. A recording of a past live stream |
| Sample | A moment in the VOD at which screenshots are taken and classified |
| Heartbeat | A fallback sample taken at a fixed interval when nothing has changed |
| Provider | The AI service that classifies screenshots: Gemini, OpenAI, or the built-in mock |
| Batch mode | Sending all requests at once for half price, with results up to 24 hours later |
| Segment | A stretch of the timeline with one activity |
| Confidence | The model's own 0 to 1 estimate of how sure it is about a verdict |
| AFK | Away from keyboard. The streamer is not at the desk |
