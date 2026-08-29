# Technical notes

Internals of the Kick VOD Analyser. The [README](README.md) covers installation
and usage; this file explains why the pipeline is built the way it is.

## Contents

- [Design choices](#design-choices)
- [Sampling](#sampling)
- [Chat](#chat)
- [Classification](#classification)
- [Temporal smoothing](#temporal-smoothing)
- [Caching](#caching)
- [Rate limits and transient errors](#rate-limits-and-transient-errors)
- [Batch mode](#batch-mode)
- [Kick URL id mapping](#kick-url-id-mapping)
- [Known limitations](#known-limitations)

## Design choices

Four decisions carry most of the cost and runtime savings.

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
the 45-second minimum gap. `--max-samples` caps it further for a fixed budget.

## Sampling

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

Scene detection runs ffmpeg's scene filter over keyframes of the lowest
rendition. Triggers closer than `KVA_SAMPLING_MIN_GAP_SECONDS` (45s) are
collapsed, and a heartbeat sample is inserted wherever the gap between triggers
exceeds `KVA_SAMPLING_HEARTBEAT_SECONDS` (900s). Grids whose perceptual hash is
within `KVA_SAMPLING_PHASH_DISTANCE` of the previous grid are deduplicated
before classification.

## Chat

### Download

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

Kick embeds emotes in message text as `[emote:<id>:<name>]`. The normaliser
replaces each token with its name and lists the names under `emotes`.

### Accepted file formats

`--chat file` accepts a JSON array, a JSON object with a
`messages`/`comments`/`data` key, or JSON Lines, and understands Kick's own
payload shape, `chat-downloader` output, and this package's export format. The
minimum a record needs is message text plus something to place it on the
timeline:

```json
{"offset_seconds": 1234.5, "username": "viewer", "text": "what game is this"}
```

### Window construction

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

Every chat source degrades to an empty index rather than failing the run.

## Classification

Each request sends the grid image, the condensed chat window, and a system
prompt with a JSON schema. Per request: ~258 vision tokens for the grid, ~300
for the chat window, ~320 for the system prompt and schema, ~90 output.

The response carries a category, a specific title, a sub-activity, an on-screen
flag, an AFK flag, a confidence score, and a one-sentence justification naming
the visual markers that drove the decision.

Prices live in `classify/pricing.py` as a plain table. Add or override an entry
without touching the estimator.

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

## Caching

Everything expensive is cached in `work/cache.sqlite`:

- Scene detection results, keyed by VOD id
- Classifications, keyed by VOD id, sample offset, and model

A second run reuses both and classifies only what is new. `--no-resume` forces a
fresh pass. Changing the model invalidates classifications but not scene
detection, so trying a different model does not re-download the VOD. Nothing is
cached on failure.

## Rate limits and transient errors

Synchronous classification retries each request on rate-limit and server
errors (HTTP 408, 429, 5xx, or messages mentioning quota, rate limit, or
overload). Up to 8 attempts per request. The wait honours the provider's
`Retry-After` header or Gemini's `retryDelay` hint when present, otherwise it
backs off exponentially from 2s, capped at 120s. Free-tier quota exhaustion
pauses the run instead of failing it. Errors that are not transient (bad API
key, invalid request) are recorded against the sample and the run continues.

## Batch mode

- **Gemini** uploads each grid through the Files API and references it by URI
  from a JSONL manifest. Inline base64 would breach the request size ceiling
  well before a full-length VOD is covered.
- **OpenAI** embeds each grid as a data URI directly in the JSONL.

The job id is written to `work/<vod_id>/batch_job.txt`. Polling continues until
the job reaches a terminal state, then outputs are written.

## Kick URL id mapping

Kick VOD URLs carry a version 7 UUID that no read endpoint accepts, and that id
appears nowhere in the API payloads. The resolver decodes the creation timestamp
the v7 id embeds, looks the channel's video listing up, and matches on time to
recover the version 4 `video.uuid` the endpoints do accept. The log shows:

```
mapped URL id to video uuid 1e7fa39e-09ad-47c5-9f25-f08eedafa16d (4.0s apart)
```

The listing only covers recent VODs, so an older one may no longer be findable.
The `kick.com/video/<id>` form has no channel to look up and cannot be mapped.
Because the mapping resolves to the canonical video id, `out/<vod_id>/` is named
after the v4 id, not the id in the URL.

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
  once per VOD. Frame extraction fetches only the segments around each sample
  point, measured at 4.2 seconds per point regardless of position in the VOD.
