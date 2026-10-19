# SeaRescue Agent — Devpost Draft

## Title

**SeaRescue Agent: Look Twice Where It Matters — Bounded Agentic Re-Examination for People in Water (OpenCV 5 × AWS)**

## Tagline

> A drone-vision pipeline that measures what it sees, post the uncertainty
> to a deployed AWS decision service, and re-examines exactly the frame
> regions it is unsure about — under a strict, explainable budget.

*(alt tagline: "One look is never the whole story. SeaRescue decides
where the second look goes.")*

## Problem

People in open water look small, partial and low-confidence on UAV
footage. A single fixed-threshold detection pass either misses them or
drowns the operator in noise — and uniform high-resolution re-scanning of
every frame is infeasible on rescue-budget compute. Rescue teams need a
system that knows **what it is unsure about** and spends its bounded
re-examination capacity only there, with decisions that are auditable.

## Solution

SeaRescue Agent is an agentic vision loop built on **OpenCV 5** and a
pretrained maritime detector, with the "what should I do next" decision
made by a deterministic, schema-validated service deployed on **AWS**:

1. OpenCV 5 measures the frame: Laplacian-variance sharpness, brightness,
   contrast — plus the detector's candidate boxes down to a preserved
   0.15 floor.
2. A **measurement-only** observation (never image bytes, never
   client-chosen decisions) is SigV4-signed and posted to an
   IAM-authenticated Lambda Function URL.
3. The deployed service — running the same decision module as the local
   code — returns one bounded action: `keep_sparse_sampling`,
   `sample_densely`, `crop_zoom` or `request_human_review`, with a reason
   string, deterministic trace id and budget accounting.
4. The client **executes the returned action locally** (context-expanded
   crop at 960 px, swimmer-only filtering, coordinate mapping back into
   the frame, NMS merge) and records a full evidence trace.
5. Every run outputs annotated before/after images, final-detection JSON
   and JSONL traces that prove whether a real AWS execution happened
   (`decision_source: aws-live`, `trace_id_parity: true`) — as opposed to
   a labelled local-policy simulation.

## Technical implementation

- Python 3.12, Windows 11, CPU only (torch 2.14.1+cpu).
- OpenCV exactly one package: `opencv-python==5.0.0.93` (OpenCV 5.0.0),
  NumPy 2.5.4; inference `ultralytics==8.4.177`.
- Detector: YOLOv8n fine-tuned for SeaDronesSee
  (`sha256 cadfb4aba…7db001`; weights are NOT redistributed).
- Modules: `agentic_detect.py` (image agent, policies v1/v2),
  `analyze_video.py` (motion-gated adaptive video loop; service mode),
  `decision_service.py` (the single-source-of-truth policy),
  `lambda_handler.py` (deployed handler), `evaluate.py`, `detect.py`,
  `aws_sigv4.py` (stdlib signing), `demo_live_agent.py` (live demo).
- 69 offline unit tests (stubbed detector/transport; no live AWS inside
  tests) + an OpenCV 5 smoke test (video encode/decode round-trip).
- Deployment: AWS SAM + CloudFormation change set; the deployed stack
  contains exactly 4 resources (Lambda, IAM role, Function URL, log
  group) — no database, no bucket, no API Gateway, no networking infra.

## OpenCV 5 contribution

OpenCV 5.0.0 does all the image-side work: quality measurement
(cvtColor/Laplacian/resize), the agentic crop geometry that makes
re-examination possible (expand_crop / map_to_frame / clamping), per-class
NMS merging of initial and zoom detections, evidence annotation, and the
motion-gated video sampling path. The submission's "agentic" behavior —
deciding *where a second look happens and how much it costs* — is built
on OpenCV operations end to end.

## AWS contribution

- **Decision service on Lambda** (python3.12, 256 MB, 10 s): schema
  validation with hard limits (≤ 32 KiB, ≤ 200 detections, forbidden
  keys), deterministic decisions with trace ids, bounded budget
  accounting, reason strings and review flags.
- **Security**: Function URL is AWS_IAM-only (unsigned = 403 refused
  before invocation — verified live); execution role is limited to log
  writing; JSON platform logging with 30-day retention.
- **Honest deployment evidence**: stack created via real change-set
  inspection (2026-10-17); cumulative 24 invocations, **0 errors**;
  CloudWatch metrics carry invocation truth (incl. a documented
  warm-environment log-delivery gap we did not paper over).
- **Cost design**: idle = $0 (no always-billing infra in either region;
  ~130 KB total storage; IAM auth prevents free-riding invocations).

## Measured evaluation (all from preserved artifacts)

- dev-32 (SeaDronesSee val, IoU 0.5, conf 0.25, CPU):
  fixed-640 → P 0.7113 / R 0.7188 / F1 0.7150 / mAP50 0.6748 (0.079 s);
  fixed-960 → P 0.7703 / R 0.8385 / F1 0.8030 / mAP50 0.7854 (0.149 s);
  adaptive-v2 → F1 0.7263 with escalation 62% and hires pixels 0.4 Mpx
  (v1: 84%, 73.6 Mpx) — bounded re-examination with slightly fewer false
  positives at equal recall; we do **not** claim an accuracy win from
  adaptivity on small sets.
- Two additional untouched 16-image val samples (final-16 / held-16)
  evaluated the same way (F1 0.7547 / 0.7955).
- LIVE cloud demo (8 labelled images through the deployed endpoint):
  8/8 HTTP 200 aws-live decisions, 7 triggered one real 960 px crop
  inference each with mapped coordinates, cloud/local trace-id parity and
  decision agreement 8/8 — and, stated plainly, **no net new detections
  above the operating threshold were recovered by the zoom** (measured,
  artifacts committed in the repo). The demo proves deterministic cloud
  decision + auditable executed action, not an accuracy improvement.
- The eight images are a demo subset, **not** a comprehensive benchmark;
  real maritime video validation is pending (official MOT video requires
  manual dataset registration).

## Limitations

Stated in the submission and the demo: demo subset is not a benchmark;
no real-video validation yet; confidences are heuristics, not calibrated
probabilities; AGPL-3.0 model/stack obligations still under review before
distribution or hosted access; a warm-environment CloudWatch
log-delivery gap was observed and documented (invocation evidence comes
from metrics + response bodies + trace-id parity).

## Acknowledgements

- OpenCV community and the OpenCV AI Competition 2026 organizers.
- AWS for the competition credits used only within the Free plan limits.
- SeaDronesSee authors (University of Tübingen) for the dataset;
  the `dronefreak` Hugging Face mirror (CC0-marked) used for val images.
- Ultralytics for the YOLO training/eval stack used to produce the
  fine-tuned detector locally.
