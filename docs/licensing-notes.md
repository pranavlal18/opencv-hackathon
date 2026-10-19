# SeaRescue Agent — Licensing Notes (factual, no legal conclusions)

Status: **pending review before distribution**. This document records
provenance facts as observed from installed metadata and dataset sources.
It makes no legal determinations; the review items in §4 need a proper
license review before any of the actions in §5 are attempted.

## 1. Detector weights

- Source: Hugging Face model `dronefreak/seadronessee-yolov8n`
  (pretrained weights adapted for SeaDronesSee val usage).
- License as recorded by the model card metadata: **AGPL-3.0**.
- Local file: `weights/best.pt`
  (`sha256 cadfb4abaae544742352c228be5508e236d41f8823165d5dffd7d7095f7db001`).
- Repo hygiene: `weights/` (and `*.pt/*.onnx/*.bin/*.ckpt/*.safetensors`)
  are gitignored; weights are **not** in the repository and are not part
  of any submitted artifact.

## 2. Inference stack

- `ultralytics==8.4.177` — license as recorded in installed METADATA:
  **AGPL-3.0**. The local detector pipeline imports this package
  (`agentic_detect.py` builds its predict/annotate/split logic on it).
- `torch==2.14.1+cpu`, NumPy, OpenCV — permissive licenses (per package
  metadata; OpenCV = Apache-2.0).
- Consequence to review: any distribution that conveys the combined
  inference program must be checked against AGPL-3.0 source-sharing terms
  (see §4).

## 3. Dataset

- Used data: SeaDronesSee **val** split images with their paired official
  YOLO-format labels, obtained from the Hugging Face dataset
  `dronefreak/SeaDronesSee`, which is published under the **CC0** marker
  on that mirror (this is what the project relies on for its local
  evaluation sets: `dev-32`, `final-16`, `held-16` — all val samples).
- Original dataset: SeaDronesSee (University of Tübingen,
  seadronessee.cs.uni-tuebingen.de). The **multi-object-tracking video
  part requires manual registration and download**; the project has not
  downloaded it and explicitly treats that as a user action (no
  bypassing of dataset access barriers was attempted).
- Claim discipline: the project's documents state exactly what was used
  (val images from a CC0-marked mirror) and never claim rights over the
  original dataset's video assets.

## 4. Obligations requiring review before distribution or hosted access

(unresolved — this list is the review checklist, not a conclusion)

1. Whether distributing this repository (source + scripts, with weights
   supplied separately) triggers AGPL-3.0 source-provision duties through
   the Ultralytics inference stack and the AGPL-3.0 model weights.
2. Whether the AWS Lambda decision service counts as "conveying" or as
   network remote interaction under AGPL-3.0 terms — the deployed handler
   itself imports `scripts/decision_service.py` (project-authored source,
   available with the repository), but the interaction model must be
   reviewed rather than assumed.
3. Whether the competition demo (source shipping and/or hosted demo)
   satisfies attribution and license-notice requirements for both the
   weights (AGPL-3.0) and the Ultralytics stack (AGPL-3.0), and any
   dataset-attribution requirements.
4. Whether hosting the demo behind the current **AWS_IAM-authenticated**
   Function URL (credentialed, demonstration-only access) changes the
   review outcome.
5. The fallback option (RF-DETR Nano, permissive license claim) was
   examined earlier on measured numbers (worse swimmer AP 64.81 vs
   71.85; ~19x larger) and remains NOT installed and NOT used; switching
   would itself require a fresh runtime/compatibility measurement and is
   out of scope for the competition deadline.

## 5. What this submission does and does not include

- Includes: project-authored source (this repository), documentation,
  evidence artifacts (annotated images, traces, JSON results), and
  pointers to the CC0-marked mirrored evaluation set.
- Does **not** include: model weights, the raw dataset, any private video
  file, credentials, or the original SeaDronesSee MOT videos. None of
  these may be attached without completing §4.
