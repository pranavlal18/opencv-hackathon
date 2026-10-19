# SeaRescue Agent — Demo Script (4:30–4:50 minutes)

Purpose: live walkthrough for a 5-minute slot (the demo fits
**4 min 30 s to 4 min 50 s** including transitions).
**This document is the script, not the recording.** No demo video has
been recorded yet; anything marked SHOW is performed live on screen (or
screen-captured at presentation time).

## Before the demo (pre-flight checklist)

1. `.venv` active; `python scripts/smoke_cv2.py` → PASS (exit 0).
2. `python -m pytest tests -q` → 69 passed (takes ~15–30 s; can be shown
   or just cited).
3. AWS session refreshed (`aws login`) if the demo touches the live
   endpoint; endpoint is IAM-authenticated — unsigned probes return 403
   (by design; nice to show in segment 3).
4. Terminal font large; `output/demo_live_evidence/` and
   `docs/evidence-manifest.csv` open in the editor for segment 4.
5. Fallback (no internet / AWS unavailable): run the SAME demo commands
   with `--mode local-policy` and say clearly on screen: "this path is
   the shared-decision-logic local run; the deployed endpoint run with
   the same measurements is in the evidence directory." Do not present a
   local run as a live AWS run.

## Segment 1 — Stakes (0:00 → 0:35, ~35 s)

SAY: "Finding one small swimmer from a drone is the difference between a
miss and a rescue. One fixed look misses small, low-confidence people;
over-processing every frame burns a CPU you don't have. So the system
should look once, know what it is unsure about, and re-examine exactly
that — under a strict budget."

SHOW (5 s): terminal `python scripts/smoke_cv2.py` output footer:
`PASS: OpenCV 5 smoke test OK` (OpenCV 5.0.0, NumPy 2.5.4, encode/decode
round-trip OK).

## Segment 2 — Real detector + OpenCV 5 quality + adaptive loop (0:35 → 1:20, ~45 s)

SHOW: `python scripts/agentic_detect.py --input data\samples\dl\data\images\val\shard_000\1003.jpg --output output\demo_inline --policy v2`
then open `output\demo_inline\1003_annotated.jpg`.

SAY: "OpenCV 5 measures the frame — Laplacian-variance sharpness,
brightness, contrast — and the detector gives candidate boxes with a
candidate floor preserved below the operating point. The policy finds the
uncertain small swimmer, zooms a context-expanded crop to 960 pixels,
maps the crop boxes back to the frame and merges with NMS — all bounded:
at most two extra inference calls and a 25-megapixel hires budget."

## Segment 3 — The LIVE AWS decision (1:20 → 2:35, ~75 s)

SHOW:
```
python scripts/demo_live_agent.py --input data\samples\dl\data\images\val\shard_000\1003.jpg `
    --output output\live_demo_stage --mode live-aws
```
Console shows the real round-trip near-instantly.

SAY: "The observation is measurements only — classes, confidences,
relative areas, quality signals. The client never chooses the decision.
It's signed with SigV4 and posted to our deployed Lambda Function URL —
IAM-authenticated, so unsigned requests are refused with 403 before the
function even runs. Lambda validates, decides — this is the same decision
module the local code imports — and returns one bounded action."

POINT OUT on the returned JSON in the terminal: `"action": "crop_zoom"`,
`"trace_id": "…"`, `"budget_remaining"`, and that the trace file
afterwards records `"decision_source": "aws-live"` and
`"trace_id_parity": true` — the same 16-hex id as the local copy of the
shared policy, because it is the same code on both sides.

## Segment 4 — Evidence, traces, measured results (2:35 → 4:15, ~100 s)

SHOW `demo_traces.jsonl` line for 1003 (fields: request, endpoint truth,
matched box `target_matched_box`, extra call timings) and flip
`output\demo_live_evidence\1003_before.jpg` → `1003_after.jpg`.

SAY (results, all measured):
- Fixed-640 baseline: F1 **0.715**; fixed-960: F1 **0.803** (mean
  0.149 s/image on CPU); adaptive-v2 on the same 32 images: F1 **0.7263**
  with 62% escalation and **0.4 Mpx** hires (v1: 84%, 73.6 Mpx) — bounded
  re-examination, not a magic accuracy gain.
- The 8-image LIVE demo: **8/8** requests hit the real endpoint (HTTP
  200), 7 triggered one extra crop inference each, cloud/local trace-ID
  parity and decision agreement **8/8**.
- THEN THE HONEST PART: show `summary.json` counts and state plainly:
  "Zoom did **not** recover any new confident detections on these eight
  images — zero net additions above the 0.25 threshold after the merge
  NMS. What this demo proves is the agentic loop itself: a real
  measurement changed the deployed service's action and the action
  executed locally with correctly mapped coordinates."

## Segment 5 — Limitations + wrap (4:15 → 4:40, ~25 s)

SAY: "Limitations, stated: this eight-image set is a demo, not a
benchmark; real maritime video validation is still pending (the official
MOT clip requires dataset registration); confidence thresholds are
heuristics, not probabilities; the AGPL stack still needs a licensing
review before distribution. The evidence, traces and numbers are all in
the repository."

END CLOCK: ~4:40 (safe margins 4:30–4:50).

## Timing budget recap

| Segment | Duration |
|---|---|
| 1 Stakes + smoke | 35 s |
| 2 Detector + OpenCV 5 + adaptive loop | 45 s |
| 3 LIVE AWS decision | 75 s |
| 4 Evidence + measured results | 100 s |
| 5 Limitations + wrap | 25 s |
| **Total** | **4 min 40 s** |
