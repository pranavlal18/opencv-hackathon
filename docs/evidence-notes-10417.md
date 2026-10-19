# Task 14 Evidence Notes — 10417.jpg NMS Suppression Chain (wording-only correction)

Scope: **wording-only documentation correction** approved in the NMS audit follow-up.
No code, tests, model settings, NMS behavior, thresholds, policy logic, or AWS
resources were modified. No evidence images were altered. Nothing was committed.

## Correction 1 — suppression chain (replaces the vaguer phrase "kept 0.418")

During class-specific, confidence-descending NMS, the 0.486-confidence swimmer
detection was first suppressed by the 0.635-confidence detection (IoU 0.5303).
The 0.418-confidence detection was retained because its IoU with the kept
0.635 detection was only 0.3305, even though its IoU with the
already-suppressed 0.486 detection was 0.6940. The targeted 0.364-confidence
detection was subsequently suppressed by the retained 0.418 detection
(IoU 0.6020). The crop inference contributed zero retained detections.

## Correction 2 — image resolution (fixes a factual error in the Task 14 report)

10417.jpg is 3840 × 2160, not 5456 × 3632.

Specifically: an earlier Task 14 sentence described the executed crop region
`[3442, 1608, 3580, 1719]` (138 × 111 px) as being "in the 5456 × 3632
original". That resolution belongs to `1003.jpg` (verified: 1003_final.json,
width 5456, height 3632). `10417_final.json` records width 3840, height 2160,
and the crop region fits inside it. The crop coordinates and the 138 × 111 px
size quoted for 10417.jpg were themselves correct.

## What was already correct and is unchanged

- Final detection count after merge/NMS: 12 (from 14 initial operating
  detections; 2 removed, both by merge NMS, unrelated to the crop call).
- The crop inference returned 3 raw detections (1 boat 0.446 excluded by the
  swimmer-only rule; 2 sub-threshold swimmers 0.227 / 0.197), retained 0,
  added 0 to the final set (n_kept = 0, n_new_added = 0).
- The 14 → 12 reduction therefore cannot be attributed to the crop-zoom call.
- No accuracy improvement is claimed for the crop_zoom action on this image.
- Trace IDs and crop region are identical across the Task 11, Task 13, and
  Task 14 live runs; `10417_final.json` is byte-identical (sha256
  `5c1d3ce58d7087e9a80cca391658353f90029a5ac08f5a21a3c8028be47b1739`) between
  the Task 11 evidence and the Task 14 visualization run.

## Artifact basis (all numbers re-derived before writing these notes)

- `output/demo_live_viz/demo_traces.jsonl` (10417 line: initial_operating 14,
  final 12, extra_calls n_kept 0, raw_crop_detections list), sha256
  `a030477f421b7183b5edc746975f660ffa7b1626008cd9260d1e9aa4f1f540e1`
- `output/demo_live_viz/10417_final.json` (3840 × 2160, 12 detections), sha256
  `5c1d3ce58d7087e9a80cca391658353f90029a5ac08f5a21a3c8028be47b1739`
- `output/demo_live_viz/10417_crop_summary.json` (crop_wh [138, 111], all
  counts, claim field)
- Historical cross-check (identical boxes/IoUs/suppression): Task 11 evidence
  `output/demo_live_evidence/` (10417_final.json sha256 identical to today's;
  trace sha256 `de77f50f3bea7307b75d85d35b2e8c1598e9c811f491036e876e837db9fcfd7f`)
- Verification scripts (read-only): `_audit_nms.py` (instrumented NMS replay:
  events `[(0.486, 0.635, 0.5303), (0.364, 0.418, 0.602)]`), `_verify_notes_values.py`
  (re-derived every IoU/counter before this file was written).

Suppression-chain units: IoU threshold 0.5 (merge setting, unchanged); all
three named detections are class swimmer (class_id 0), so the per-class rule
kept boats and jet-skis out of this comparison by construction.
