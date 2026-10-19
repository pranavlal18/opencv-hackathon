"""Shared decision logic for the SeaRescue cloud decision service.

Single source of truth used by BOTH the local test path and the AWS Lambda
handler (scripts/lambda_handler.py), so the two cannot silently diverge:
Lambda imports decide_observation() from this module.

Contract (heuristic thresholds, NOT calibrated probabilities):
- Input: compact JSON observation (measurements + refs only; never image
  binaries, credentials, personal data, or public URLs).
- Output: one action of keep_sparse_sampling | sample_densely | crop_zoom |
  request_human_review, with args, reason, review flag, budget accounting.
- Deterministic: identical (observation, state) -> identical decision.
- Bounded: budget.remaining_extra <= 0 forbids crop_zoom/sample_densely
  escalation (falls back to keep_sparse_sampling or human review).
- request_human_review for poor quality or unresolvable uncertainty.
  Confidence scores are NEVER treated as proof of an emergency.

Request schema:
  {"request_id": str, "frame_ref": str,
   "observation": {"detections": [{"class": str, "conf": float,
                                   "rel_area": float}], "quality": {...},
                   "motion": {...} | None},
   "state": {"stride": int, "samples_used": int, "prev_action": str | None},
   "budget": {"remaining_extra": int, "max_extra": int}}
Response schema:
  {"action": str, "args": {...}, "reason": str, "review": bool,
   "trace_id": str, "budget_remaining": int}
"""
import hashlib
import json

ACTIONS = ("keep_sparse_sampling", "sample_densely", "crop_zoom",
           "request_human_review")

MAX_REQUEST_BYTES = 32 * 1024
FORBIDDEN_KEYS = ("image_data", "pixels", "image_base64", "password",
                  "token", "secret", "api_key", "url", "uri", "href")
CLASSES = ("swimmer", "boat", "jetski", "life_saving_appliances", "buoy")

# Heuristic thresholds (engineering choices, documented in traces).
SWIMMER_UNCERTAIN_CONF = 0.50
SWIMMER_ZOOM_FLOOR = 0.15
SHARP_FLOOR = 100.0
DIFF_THRESH = 8.0
FLOW_THRESH = 0.4


class DecisionError(ValueError):
    """Malformed or rejected decision request."""


def _forbid_deep(obj, path="root"):
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = str(k).lower()
            if any(f in kl for f in FORBIDDEN_KEYS):
                raise DecisionError(f"forbidden key '{k}' at {path} "
                                    f"(no binaries/credentials/URLs allowed)")
            _forbid_deep(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _forbid_deep(v, f"{path}[{i}]")


def validate_request(req: dict) -> dict:
    """Validate and normalize a decision request. Raises DecisionError."""
    if not isinstance(req, dict):
        raise DecisionError("request must be a JSON object")
    if len(json.dumps(req).encode()) > MAX_REQUEST_BYTES:
        raise DecisionError(f"request exceeds {MAX_REQUEST_BYTES} bytes")
    _forbid_deep(req)
    for field in ("request_id", "frame_ref", "observation", "state", "budget"):
        if field not in req:
            raise DecisionError(f"missing field: {field}")
    obs = req["observation"]
    if not isinstance(obs, dict) or "detections" not in obs or "quality" not in obs:
        raise DecisionError("observation needs {detections, quality}")
    dets = obs["detections"]
    if not isinstance(dets, list) or len(dets) > 200:
        raise DecisionError("detections must be a list of at most 200")
    for i, d in enumerate(dets):
        if not isinstance(d, dict):
            raise DecisionError(f"detections[{i}] must be an object")
        if d.get("class") not in CLASSES:
            raise DecisionError(f"detections[{i}].class unknown: {d.get('class')!r}")
        for num in ("conf", "rel_area"):
            v = d.get(num)
            if not isinstance(v, (int, float)) or not 0.0 <= v <= 1.0:
                raise DecisionError(f"detections[{i}].{num} must be in [0,1]")
    q = obs["quality"]
    for num in ("sharpness", "brightness", "contrast"):
        if num not in q or not isinstance(q[num], (int, float)):
            raise DecisionError(f"quality.{num} must be a number")
    st = req["state"]
    for num in ("stride", "samples_used"):
        if num not in st or not isinstance(st[num], int) or st[num] < 0:
            raise DecisionError(f"state.{num} must be a non-negative int")
    b = req["budget"]
    for num in ("remaining_extra", "max_extra"):
        if num not in b or not isinstance(b[num], int) or b[num] < 0:
            raise DecisionError(f"budget.{num} must be a non-negative int")
    if b["remaining_extra"] > b["max_extra"]:
        raise DecisionError("budget.remaining_extra exceeds budget.max_extra")
    return req


def decide_observation(req: dict) -> dict:
    """Pure deterministic decision. Validates, then selects an action."""
    req = validate_request(req)
    obs = req["observation"]
    dets = obs["detections"]
    quality = obs["quality"]
    motion = obs.get("motion")
    budget_left = req["budget"]["remaining_extra"]

    swimmers = [d for d in dets if d["class"] == "swimmer"]
    zoomable = [d for d in swimmers if d["conf"] >= SWIMMER_ZOOM_FLOOR]
    uncertain = [d for d in zoomable if d["conf"] < SWIMMER_UNCERTAIN_CONF]
    poor_quality = quality["sharpness"] < SHARP_FLOOR
    hot = bool(motion) and (motion.get("diff_mean", 0) >= DIFF_THRESH
                            or motion.get("flow_mean", 0) >= FLOW_THRESH)

    trace_id = hashlib.sha256(
        json.dumps(req, sort_keys=True).encode()).hexdigest()[:16]

    def resp(action, args, reason, review):
        return {"action": action, "args": args, "reason": reason,
                "review": review, "trace_id": trace_id,
                "budget_remaining": budget_left - (1 if action == "crop_zoom" else 0)}

    if poor_quality and not any(d["conf"] >= SWIMMER_UNCERTAIN_CONF for d in swimmers):
        return resp("request_human_review", {},
                    f"sharpness {quality['sharpness']:.1f} < {SHARP_FLOOR} with no "
                    f"confident swimmer; needs human review, no escalation", True)
    if budget_left <= 0:
        if uncertain or hot:
            return resp("request_human_review", {},
                        "budget exhausted with unresolved evidence; human review, "
                        "no further inference", True)
        return resp("keep_sparse_sampling", {},
                    "budget exhausted and scene quiet; hold sparse sampling", False)
    if uncertain:
        top = sorted(uncertain, key=lambda d: (d["conf"], d["rel_area"]))[0]
        return resp("crop_zoom",
                    {"target": {"class": top["class"],
                                "conf": round(top["conf"], 3),
                                "rel_area": round(top["rel_area"], 6)},
                     "imgsz": 960},
                    f"{len(uncertain)} uncertain swimmer(s); zoom top candidate "
                    f"(conf {top['conf']:.2f})", False)
    if hot:
        return resp("sample_densely", {"stride": 1},
                    f"motion diff={motion['diff_mean']:.1f}/flow={motion['flow_mean']:.3f} "
                    f"above thresholds; densify to stride 1", False)
    return resp("keep_sparse_sampling", {},
                "clear or quiet scene; no extra analysis warranted", False)
