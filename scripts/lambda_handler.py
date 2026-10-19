"""AWS Lambda handler for the SeaRescue decision service (NOT DEPLOYED).

Uses scripts/decision_service.decide_observation() as the single source of
truth — the same function local tests exercise — so cloud and local cannot
diverge. Accepts a Lambda Function URL event (or a direct dict for tests).

Event shapes accepted:
- {"body": "<json string>"} (Function URL / API Gateway proxy)
- {"request_id": ..., ...} (direct invocation / local parity test)

Trace persistence: if the TRACE_TABLE env var is set and boto3 is importable
(Lambda Python runtimes include it), the decision record is written to
DynamoDB; otherwise the trace is returned inline in the response
(`persisted: false`) so local runs stay dependency-free. boto3 is an
OPTIONAL import here — requirements.txt is unchanged.

Response: {"statusCode": 200|400, "body": "<decision JSON>"}.
"""
import json
import os
import time

try:
    from scripts.decision_service import decide_observation, DecisionError
except ImportError:  # Lambda root layout fallback (file deployed at top level)
    from decision_service import decide_observation, DecisionError


def _store_trace(record: dict) -> bool:
    table = os.environ.get("TRACE_TABLE")
    if not table:
        return False
    try:
        import boto3
    except ImportError:
        return False
    boto3.resource("dynamodb").Table(table).put_item(Item=record)
    return True


def _extract_request(event):
    if isinstance(event, dict) and "body" in event:
        body = event["body"]
        if event.get("isBase64Encoded"):
            import base64
            body = base64.b64decode(body).decode("utf-8")
        return json.loads(body) if isinstance(body, str) else body
    return event


def handler(event, context=None):
    """Lambda entry point. Returns proxy-style {statusCode, body}."""
    try:
        req = _extract_request(event)
        decision = decide_observation(req)
    except DecisionError as exc:
        return {"statusCode": 400,
                "body": json.dumps({"error": "rejected", "detail": str(exc)})}
    except (TypeError, json.JSONDecodeError) as exc:
        return {"statusCode": 400,
                "body": json.dumps({"error": "malformed", "detail": str(exc)[:200]})}
    record = {"trace_id": decision["trace_id"],
              "frame_ref": req.get("frame_ref", "unknown"),
              "request_id": req.get("request_id", "unknown"),
              "created_at": int(time.time()),
              "action": decision["action"],
              "reason": decision["reason"],
              "review": decision["review"],
              "budget_remaining": decision["budget_remaining"],
              "policy": os.environ.get("POLICY_VERSION", "shared-decide-v1")}
    try:
        persisted = _store_trace(record)
    except Exception as exc:  # storage must not fail the decision
        persisted = False
        record["store_error"] = str(exc)[:200]
    payload = dict(decision)
    payload["persisted"] = persisted
    return {"statusCode": 200, "body": json.dumps(payload)}
