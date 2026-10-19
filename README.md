# SeaRescue Agent

Agentic maritime vision system for the OpenCV AI Competition 2026
(powered by AWS): detects people in water from UAV imagery, posts its
uncertainty to a deployed AWS decision service, and re-examines bounded
regions of the frame — with auditable evidence at every step.

Status: **competition-ready build, committed to GitHub** — demo video and
the Devpost submission remain manual human steps (not automated here).
Submission deadline referenced in docs: 2026-10-26, 18:00 IST.

---

## What is proven vs. not (read this first)

Proven with artifacts and live AWS evidence:

- Real OpenCV 5 + detector measurements about an image cause the
  deployed Lambda to return a bounded action; the action executes
  locally with correctly mapped coordinates; cloud/local trace-id parity
  and decision agreement are 8/8 on the live demo set.
- Detection-quality results below are measured on labelled val subsets
  (details in `docs/technical-report.md`).

Not claimed anywhere:

- The adaptive loop did **not** recover net new confident detections on
  the 8-image live demo subset (measured: zoom-added detections above
  the operating threshold = 0). We make no adaptive-accuracy claim.
- Real maritime **video** validation is incomplete (the official
  SeaDronesSee MOT video requires manual registration; not bypassed).
- The small datasets are demo/benchmark-subset scale, not comprehensive.

## Setup (Windows PowerShell; tested)

```powershell
cd C:\path\to\opencv-hackathon
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` pins the stack: `opencv-python==5.0.0.93` (OpenCV
**5**) as the single OpenCV package, `ultralytics==8.4.177`,
`torch` (CPU-only wheel), `numpy` 2.x, `pyyaml`. Use the project
`.venv`, never the global interpreter. Python 3.12.0.

Model weights go to `weights/best.pt` (gitignored; AGPL-3.0 — see
`docs/licensing-notes.md`). They are not shipped with the repository.

## Verified environment & commands

```powershell
python scripts/smoke_cv2.py      # OpenCV 5 smoke (encode/decode round-trip, Laplacian, frame-diff) -> PASS
python -m pytest tests -q        # 74 offline tests -> all pass (no live AWS inside unit tests)

# single-image adaptive agent (local decisions)
python scripts/agentic_detect.py --input <image.jpg> --output out\single --policy v2

# full labelled-set evaluation (fixed / adaptive comparisons -> results.json per run)
python scripts/evaluate.py <args per script help>

# video loop (owned clips; service mode uses the shared decision logic)
python scripts/analyze_video.py --input <clip.avi> --output out\video --decision service

# LIVE end-to-end demo: real observation -> signed Function URL -> executed action
# (--function-name: your deployed DecideFunction name; NOT stored in this repo)
# Dataset val images are NOT in the repo — point --images at your own local copy.
python scripts/demo_live_agent.py --images <your local val image dir> `
    --limit 8 --output output\demo_live --mode live-aws `
    --function-name searescue-agent-DecideFunction-<your-suffix>
# same pipeline offline (labelled local-policy, no network):
python scripts/demo_live_agent.py --input <image.jpg> --output out\local --mode local-policy
```

## Repository layout

```
scripts/   agentic_detect.py, analyze_video.py, decision_service.py,
           lambda_handler.py, evaluate.py, detect.py, demo_live_agent.py,
           aws_sigv4.py, smoke_cv2.py
tests/     74 offline unit tests incl. AWS template guard tests
aws/       template.yaml (deployed SAM template)
output/    preserved evidence & evaluation artifacts (gitignored)
docs/      technical report, architecture diagram, evidence manifest,
           licensing notes, demo script, devpost draft
data/      local dataset mirror (gitignored; never shipped)
weights/   detector weights (gitignored; never shipped)
```

## Measured results (summary; full tables in docs/technical-report.md)

Detection quality (dev-32, 32 labelled val images, IoU 0.5, conf 0.25,
CPU; mAP50 is a conservative truncated-value, see evaluator note):

| Config | P | R | F1 | mAP50 | mean s/image |
|---|---|---|---|---|---|
| fixed-640 | 0.7113 | 0.7188 | 0.7150 | 0.6748 | 0.079 |
| fixed-960 | 0.7703 | 0.8385 | 0.8030 | 0.7854 | 0.149 |
| adaptive v2 | 0.7340 | 0.7188 | 0.7263 | 0.6630 | (re-eval) |

Agentic behavior (from run traces): v2 escalation 62% (v1 84%), hires
0.4 Mpx (v1 73.6 Mpx) — bounded re-examination, negative accuracy claim.
Two additional untouched 16-image samples: F1 0.7547 / 0.7955.

Live AWS demo (Task 11): 8/8 signed requests HTTP 200 `aws-live`; 7
crop_zoom each executed with one local 960 px crop inference; parity and
agreement with the local shared policy 8/8; endpoint RTT 94–1244 ms;
cumulative Lambda invocations 24, errors 0.

Crop-zoom visual evidence (Task 14): for every executed crop the demo now
also saves the exact pixels sent, an enlarged inspection view with all raw
crop detections drawn and status-labeled (pass / below-threshold /
class-excluded), and a machine-readable `*_crop_summary.json`. On
`10417.jpg` the crop call returned 3 raw detections, retained 0, and the
final set change is merge-NMS dedup only (see
`docs/evidence-notes-10417.md` for the exact suppression chain). No
accuracy improvement is claimed.

## AWS architecture & deployment (summary)

Deployed stack `searescue-agent` (ap-south-1, created 2026-10-17T17:39:36Z):
Lambda `python3.12` (256 MB / 10 s) exposing an **AWS_IAM** Function URL,
least-privilege role (`AWSLambdaBasicExecutionRole` only), JSON platform
logging with 30-day retention. No database, no buckets in the application
stack, no API Gateway, no VPC infra, no per-function concurrency
reservation (account quota documented in guard tests). Deployment
reproduction: `sam package` + `aws cloudformation create-change-set`
(with `CAPABILITY_IAM` and `CAPABILITY_AUTO_EXPAND`) → inspect → execute
→ wait for `stack-create-complete` (Task 10 flow, executed and verified).

## Security

- IAM-only Function URL: unsigned requests get **403 before invocation**
  (verified live, including a credential-free probe on 2026-10-19).
- Runs use a dedicated credential profile; passwords/keys/session tokens
  are never printed, stored, or written into traces; endpoint URLs are
  masked in outputs; SigV4 signing is a stdlib-only module the project
  audits (`scripts/aws_sigv4.py`).
- Request schema forbids binaries/credentials/URLs by key filter and
  enforces size/numeric limits server-side.
- No real AWS resource identifiers are embedded in this repository:
  endpoint URLs are masked (6-char prefix) in docs, function-name
  defaults are placeholders, and the deployed names are supplied at
  runtime via CLI flags.

## Limitations

See `docs/technical-report.md` §10 (demo subset not a benchmark; no net
new detections recovered by zoom on the live subset; real video
validation pending; heuristic thresholds, not calibrated probabilities;
warm-environment CloudWatch log-delivery gap documented; AGPL review
before distribution/hosting).

## Licensing (`docs/licensing-notes.md`)

Detector weights: AGPL-3.0 (`dronefreak/seadronessee-yolov8n`).
Inference stack `ultralytics==8.4.177`: AGPL-3.0. Torch/OpenCV/NumPy:
permissive. Dataset: SeaDronesSee **val** images from the CC0-marked
`dronefreak/SeaDronesSee` mirror; the original dataset's MOT video
requires manual registration (treated as a user action; never bypassed).
Source-sharing/attribution obligations and the remote-service question
are an **open review checklist**, not legal conclusions.

## Cleanup instructions (documented only — NOT executed)

To remove all resources this project created in AWS when the competition
is over:

1. Delete the stack:
   `aws cloudformation delete-stack --stack-name searescue-agent
   --region ap-south-1 --profile searescue` then
   `aws cloudformation wait stack-delete-complete --stack-name
   searescue-agent --region ap-south-1 --profile searescue`.
   (Equivalent: `sam delete --stack-name searescue-agent`.)
2. Delete the SAM-managed upload bucket only if it is no longer wanted:
   it is the SAM-managed default source bucket
   (`aws-sam-cli-managed-default-samclisourcebucket-…`), empty it first
   (`aws s3 rm s3://<bucket> --recursive`), then
   `aws s3api delete-bucket --bucket <bucket>`.
3. Do **NOT** delete the two pre-existing SageMaker buckets (created
   2026-07-09 / 2026-09-12) — they predate this project.
4. Optionally delete the four pre-existing orphan log groups
   (`/aws/lambda/csv-validator`, `/aws/lambda/irispredictor`,
   `/aws/sagemaker/studio`, `s3-csv-group`) — harmless KB-scale orphans
   from earlier experiments; not touched during this project.
5. Local cleanup: nothing persistent outside the repo; `output/`,
   `data/`, `.aws-sam/` are local-only.

While the stack is idle it costs $0.00 (verified sweep on 2026-10-18:
zero running billable infra in `ap-south-1` + `us-east-1`, ~130 KB total
storage; plan credits snapshot $179.29, expiry 2026-12-25).

## Submission package map

- `docs/technical-report.md` — full report (problem, architecture,
  evaluation, security, reproduction, limitations).
- `docs/architecture.mmd` — implemented-architecture diagram source.
- `docs/evidence-manifest.csv` — 4 representative live-demo images with
  re-verified checksums, decisions, trace ids and artifact paths.
- `docs/licensing-notes.md`, `docs/demo-script.md`, `docs/devpost-draft.md`.
- `docs/evidence-notes-10417.md` — audited NMS suppression-chain notes
  for `10417.jpg` (wording-only correction, verified against artifacts).

The code and docs in this repository are committed and pushed to
GitHub as a reproducibility artifact. The Devpost submission itself and
any demo video recording remain manual human steps (not performed by
automation).
