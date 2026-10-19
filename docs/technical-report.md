# SeaRescue Agent — Technical Report

Submission for the OpenCV AI Competition 2026, powered by AWS.
Working directory of record: project root (all paths in this document are
relative to it). All numbers below are taken from artifacts in `output/`
and from live AWS operations performed on 2026-10-17/18. No metric in this
report is estimated or invented; values that come from small demo samples
are labeled as such.

---

## 1. Problem

Rescue teams and event marshals need to find people in the water from a
drone video feed. The hard cases are small swimmers and low-confidence
detections: a single fixed-threshold pass either misses them or floods the
operator with noise. Ship-board and UAV deployments also have hard compute
and connectivity limits, so the system must decide where to spend its
bounded re-examination capacity instead of uniformly over-processing every
frame.

## 2. Intended users

- Drone pilots and event marshals covering open-water swim events.
- Coast guard / sea-rescue operators triaging UAV footage.
- Edge teams needing a bounded, explainable re-examination loop that runs
  offline (CPU) with a cloud decision service when connectivity allows.

## 3. Architecture (what is actually implemented)

Two cooperating parts share one decision module
(`scripts/decision_service.py`, imported by both the local code and the
deployed AWS Lambda handler — a single source of truth that cannot
silently diverge):

1. **Vision side (local, OpenCV 5 + YOLO):** `scripts/agentic_detect.py`
   (images) and `scripts/analyze_video.py` (motion-gated video). They
   compute OpenCV quality signals (Laplacian-variance sharpness, mean
   brightness, std contrast on a 960x540 proxy), run the pretrained
   detector, and build a compact observation (detections `class/conf/
   rel_area` + quality + motion + state + budget).
2. **Decision service (deployed on AWS):** the observation is POSTed as
   JSON to an IAM-authenticated Lambda Function URL. The handler validates
   the schema and returns one bounded action:
   `keep_sparse_sampling` | `sample_densely` | `crop_zoom` | 
   `request_human_review`, with a reason string, a review flag and budget
   accounting. The decision is deterministic: the same request always
   yields the same `trace_id` (a 16-hex hash of the normalized request).
3. **Action execution (local):** the returned action is executed with the
   same pixel semantics used by the local policy (context-expanded crop,
   960 px zoom inference, swimmer-only filtering, `map_to_frame`
   coordinate mapping back to the original image, per-class NMS merge).
4. **Evidence:** every run writes annotated before/after images, a final
   detection JSON and a JSON-lines trace that records the exact request,
   the endpoint result (reached/HTTP status/latency/`trace_id`), the
   executed action and timings. Endpoint URLs are stored masked; request
   headers and credentials are never recorded.

A Mermaid rendering of this flow is provided in `docs/architecture.mmd`
(source; see that file's header note about rendering).

## 4. OpenCV 5 contribution

OpenCV 5.0.0 (exactly one `opencv-python==5.0.0.93` package, pinned; NumPy
2.5.4) is used for every image operation in the pipeline:

- quality measurement: `cv2.cvtColor`, `cv2.resize`, `cv2.Laplacian`
  (sharpness proxy), frame statistics for brightness/contrast;
- "agentic" crop re-examination: OpenCV bounding-box geometry
  (`expand_crop`, `map_to_frame`), pixel clamping and per-class NMS
  (`nms`) for merging initial and zoom-pass detections;
- annotating the evidence images (boxes, labels, review banner);
- the video path's motion gating (`cv2.absdiff` statistics) decides when
  the sampling stride relaxes or densifies;
- the project's smoke test (`scripts/smoke_cv2.py`) writes and re-reads a
  video (encode/decode round-trip) and exercises grayscale, Laplacian and
  frame-diff operations to prove the OpenCV 5 stack.

## 5. Model

- Detector: YOLOv8n fine-tuned for SeaDronesSee, weights
  `weights/best.pt` (AGPL-3.0, see `docs/licensing-notes.md`),
  `sha256 cadfb4abaae544742352c228be5508e236d41f8823165d5dffd7d7095f7db001`,
  inference stack `ultralytics==8.4.177` on torch 2.14.1+cpu (CPU only).
- Classes: swimmer, boat, jetski, life_saving_appliances, buoy.
- Operating point: conf >= 0.25 (operator-facing detections), candidate
  floor 0.10 (preserved for re-examination), zoom pool floor 0.15.
- A fallback detector (RF-DETR Nano, permissive license claim) was
  evaluated on paper only (worse swimmer AP 64.81 vs 71.85, ~19x larger)
  and is NOT used; no second model exists in this repo.

## 6. AWS integration (deployed, verified live)

Stack `searescue-agent` (ap-south-1, created 2026-10-17T17:39:36Z,
status CREATE_COMPLETE as last verified 2026-10-18 UTC):

- Exactly 4 resources: the Lambda function (runtime `python3.12`,
  256 MB, 10 s timeout, handler `lambda_handler.handler`), the execution
  role (only `AWSLambdaBasicExecutionRole`), the Function URL
  (AuthType **AWS_IAM**, no CORS), and one CloudWatch log group
  (`/aws/lambda/searescue-agent-DecideFunction-REPLACEME123456` —
  resource-name suffix redacted from this document;
  retention 30 days, JSON-format platform logging).
- No databases, no storage buckets in the application stack, no API
  Gateway, no networking infrastructure, no per-function concurrency
  reservation (account-level quota 10 documented in
  `tests/test_aws_template.py` guard tests).
- Template: `aws/template.yaml`; deployment via SAM package + CloudFormation
  change set (Task 10, real change-set inspection and execution).
- Decision requests are SigV4-signed by a stdlib-only client
  (`scripts/aws_sigv4.py`) at demo time; credentials are resolved through
  the AWS CLI credential provider and are never printed, stored, or
  recorded in traces.

## 7. Evaluation methodology

- Ground truth: SeaDronesSee val images with their official paired
  YOLO-format labels (CC0 mirror `dronefreak/SeaDronesSee`). Set splits:
  `dev-32` (32 labelled val images), `final-16` and `held-16` (two
  additional disjoint 16-image labelled val samples that were never used
  for threshold tuning).
- Evaluator: `scripts/evaluate.py` (TP/FP/FN at IoU 0.5, class-matched,
  confidence threshold 0.25; mAP50 computed with an AP floor of 0.05; the
  evaluator records a `map50_note` — mAP50 computed on predictions
  truncated at conf 0.25 is a conservative lower bound, not a full-PR
  curve metric).
- Comparison modes: fixed-640 single pass, fixed-960 single pass, and the
  adaptive policy (v1: permissive fallbacks; v2: retuned — zoom needs
  >= 0.15 conf, swimmer-only zoom outputs, <= 2 extra inference calls per
  image, 25 Mpx hires pixel cap, no boat/detail fallback).
- Live-decision evaluation (Task 11): 8 labelled images were processed
  through the LIVE AWS endpoint with real detector measurements (no
  client-prescribed decisions). See section 8.3. This is a *demo subset*,
  not a benchmark.

## 8. Measured results (from preserved artifacts; nothing re-run or altered)

### 8.1 Detection quality (dev-32, IoU 0.5, conf 0.25; Windows 11, CPU)

| Configuration | P | R | F1 | mAP50* | mean latency s/image |
|---|---|---|---|---|---|
| fixed-640 single pass | 0.7113 | 0.7188 | 0.7150 | 0.6748 | 0.079 |
| fixed-960 single pass | 0.7703 | 0.8385 | 0.8030 | 0.7854 | 0.149 |
| adaptive v2 (decisions v2, re-evaluated) | 0.7340 | 0.7188 | 0.7263 | 0.6630 | (preds re-eval) |

*mAP50 caveat: conservative value because predictions are truncated at
conf 0.25 (see §7). Per-class (640 vs 960): swimmer F1 0.6614 vs 0.7640;
boat F1 0.8958 vs 0.9462.

Additional disjoint labelled samples (preds re-evaluated, same thresholds):

| Sample | Policy | n images | P | R | F1 | mAP50* |
|---|---|---|---|---|---|---|
| final-16 | v2 | 16 | 0.7895 | 0.7229 | 0.7547 | 0.4924 |
| held-16 | v1 | 16 | 0.7527 | 0.8434 | 0.7955 | 0.9169 |

### 8.2 Agentic behavior (bounded re-examination; from run traces)

| Run | Policy | images | escalations | hires pixels | mean total s |
|---|---|---|---|---|---|
| adapt_dev | v1 | 32 | 27 (84%) | 73.6 Mpx | 0.722 |
| adapt_dev_v2 | v2 | 32 | 20 (62%) | 0.4 Mpx | 1.297 |
| adapt_final | v2 | 16 | 9 (56%) | 0.1 Mpx | 0.924 |
| adapt_held | v1 | 16 | 12 (75%) | 25.1 Mpx | 1.210 |

The v2 policy reduced escalation rate (84% -> 62% on dev-32) and cut
hires-pixel spending by ~180x (73.6 -> 0.4 Mpx) while slightly *improving*
F1 on the same set (0.715 -> 0.7263) and reducing false positives
(56 -> 50) at equal recall. mAP50 *decreased* slightly (.6748 -> .6630) —
we make no accuracy claim from the adaptive policy on small sets; the
deliverable claimed is *bounded, policy-controlled re-examination*, which
the numbers above evidence.

### 8.3 Live AWS decision demo (Task 11, 2026-10-18 per CloudWatch metric buckets; trace artifacts in `output/demo_live_evidence/`)

- Eight labelled images processed through the LIVE deployed endpoint with
  real measurements: **9 live HTTPS POSTs total counting the labelled
  smoke run (1003.jpg), all HTTP 200, all decisions `aws-live`; an
  additional 1018.jpg smoke run was used as a log-delivery probe.**
  Cumulative Lambda invocations since deployment: **24, errors 0**
  (CloudWatch `Invocations`/`Errors` metrics, 2026-10-18).
- **Seven of the eight evidence images triggered one additional crop
  inference** each (returned `crop_zoom` with an exact target echo);
  image 1032 returned `keep_sparse_sampling` and executed as a correct
  no-op. Budget (<= 2 extra calls/image) was enforced client- and
  service-side; actual maximum used: 1.
- **No net new detections above the operating confidence threshold were
  recovered**: total zoom-added detections after the merge NMS = 0 across
  all seven crop runs (one image produced 2 raw crop-pass outputs that
  stayed below the 0.25 operating threshold; six produced none). GT
  IoU-0.5 hit counts were identical before/after for every image. The
  merge NMS deduplicated overlapping initial detections on two images
  (10416: 11->10, 10417: 14->12). We therefore make **no claim that the
  agentic loop improved detection accuracy** on this subset.
- What the demo *does* prove: a real OpenCV 5 + detector measurement
  caused the deployed AWS service to return an action, and that action
  executed locally with correctly mapped coordinates — cloud-vs-local
  `trace_id` parity held on **8/8** requests and the cloud decision
  agreed with the locally imported shared policy on **8/8** requests
  (same module, independently executed on both sides).
- Client-observed endpoint round-trips: 94–1244 ms (TLS + network from
  the demo workstation to ap-south-1; AWS-side billed durations for a
  cold-start probe: 76 ms including init, 43/256 MB used).

## 9. Video path status

`analyze_video.py` implements motion-gated adaptive sampling (base stride
3, relax/densify actions) and a service mode that posts per-frame
observations to the shared decision logic. It is **verified on synthetic
fixture video only** (`output/vidtest/`, `output/vidservice/`; e.g., the
service-mode run records a frame flagged `request_human_review` for
low-sharpness frames). **Genuine maritime video validation remains
incomplete**: the official SeaDronesSee MOT video requires manual
registration/download which has not happened; consequently no real-video
detection, tracking or probability claims are made anywhere in this
submission.

## 10. Limitations (all stated, none hidden)

1. The eight-image live demo is not a comprehensive benchmark; detection
   quality and decision-service behaviour are reported separately, and
   the small datasets are not statistically comprehensive.
2. No net new detections were recovered by crop zoom on the demo subset
   (measured, see 8.3); the adaptive policy's value on this subset is
   bounded, explainable re-examination and honest negative evidence, not
   accuracy gain.
3. Real maritime video validation is pending; the MOT video requires
   manual registration (the project does not bypass dataset barriers).
4. The demo workstation clock ran ~+5:30 ahead of true UTC during the
   live runs; all reported times use AWS-side timestamps.
5. A CloudWatch log-delivery gap was observed for 9 warm-environment
   invocations (invocations metric + HTTP 200 responses + trace-id parity
   carry the evidence instead; documented in the Task 11 report; the
   handler was not modified).
6. Confidence scores are heuristic thresholds, NOT calibrated
   probabilities; no emergency-certainty claim is made.
7. Detector/stack licensing (AGPL-3.0) still requires review before
   distribution or hosted access (see `docs/licensing-notes.md`).

## 11. Security

- Function URL is **AWS_IAM only**; unsigned requests get 403 from the
  auth layer without invoking the function (verified live repeatedly,
  most recently 2026-10-19, no credentials involved).
- Least-privilege execution role (`AWSLambdaBasicExecutionRole` only);
  no VPC/public networking surface beyond the IAM URL; no database or
  bucket in the application stack.
- Decision requests are schema-validated with hard limits: <= 32 KiB,
  <= 200 detections, numeric ranges, and a forbidden-key filter
  (images/binaries, credentials, `url/uri/token/secret`-style keys).
- Demo credentials flow through the AWS CLI credential provider into
  process memory only; signing headers are never logged; stored endpoint
  URLs are masked (`https://w6dqcj***.lambda-url.ap-south-1…`).
- Repo hygiene: weights, datasets, generated outputs and `.aws-sam`
  build artifacts are gitignored; nothing is committed or pushed.

## 12. Reproduction (tested commands; Windows PowerShell, project root)

```powershell
# environment (already set up in this repo's .venv; Python 3.12, OpenCV 5.0.0)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt          # pins opencv-python==5.0.0.93, ultralytics==8.4.177

# verify the OpenCV 5 stack + run the full offline test suite (69 tests)
python scripts/smoke_cv2.py
python -m pytest tests -q

# single-image adaptive agent (local decisions, offline)
python scripts/agentic_detect.py --input <image.jpg> --output output\single --policy v2

# LIVE end-to-end demo (real detector observation -> signed Function URL -> executed action)
python scripts/demo_live_agent.py --images data\samples\dl\data\images\val\shard_000 `
    --limit 8 --output output\demo_live --mode live-aws
# offline variant of the same pipeline (no network; clearly labelled local-policy):
python scripts/demo_live_agent.py --input <image.jpg> --output output\local --mode local-policy

# video pipeline (synthetic/owned clips; service mode uses the shared decision logic)
python scripts/analyze_video.py --input <clip.avi> --output output\video --decision service
```

AWS deployment reproduction is documented in `README.md` (SAM package +
change-set flow). The deployed stack does not need to be re-created to
repeat the demo; `--profile searescue` selects the credential profile on
this workstation.

## 13. Evidence pointers

- `output/eval640_v2/results.json`, `output/eval960_v2/results.json`,
  `output/evaladapt_dev_v2/results.json` (and siblings) — detector/agent
  evaluation artifacts quoted in §8.1–8.2.
- `output/demo_live_evidence/` — 8-image live demo: annotated
  before/after JPGs, `*_final.json`, `demo_traces.jsonl`, `summary.json`;
  mapped examples in `docs/evidence-manifest.csv`.
- `output/lambda_fixtures/` — Task-10 live fixture probes (direct invoke
  + signed URL) and `output/live_smoke_1003/`, `output/live_smoke_1018/`
  — labelled live smoke/probe runs.
- `docs/architecture.mmd`, `docs/licensing-notes.md`,
  `docs/demo-script.md`, `docs/devpost-draft.md`,
  `docs/evidence-manifest.csv`.
