"""Stdlib-only AWS SigV4 client for the SeaRescue live decision endpoint.

No dependencies beyond the Python standard library. Credentials are obtained
through the configured AWS credential provider (AWS CLI
`configure export-credentials --format env-no-export`, run as a subprocess)
and are kept in memory only. NOTHING in this module prints credentials,
tokens, session values, or Authorization headers; failures raise generic
errors without embedding secret material.

Public API:
  AWSAuthError           Auth/config resolution failure (callers must fail
                         LOUDLY; never fall back silently).
  AWSTransportError      Endpoint unreachable (recorded, not secret-related).
  get_session_credentials(profile) -> dict of env key names -> strings.
  get_function_url(profile, region, function_name) -> (url, auth_method).
      Verifies the URL is AWS_IAM (an unsigned endpoint is refused).
  sign_headers(url, body_json, creds, region) -> header dict (never logged).
  signed_post(url, body_json, creds, region) -> (status, text, latency_ms).
  mask_url(url) -> host with the random function-url id truncated.

The SigV4 canonical-request construction was verified live against the real
IAM-authenticated Function URL (Task 10; signed 200 responses, unsigned 403).
"""
import hashlib
import hmac
import json
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone


class AWSAuthError(RuntimeError):
    """Credential resolution or authentication could not be completed."""


class AWSTransportError(RuntimeError):
    """Network-level failure reaching the endpoint (no auth implication)."""


_ENV_KEYS = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN")


def get_session_credentials(profile: str) -> dict:
    """Resolve live credentials via the AWS CLI. In-memory only; never printed."""
    try:
        proc = subprocess.run(
            ["aws", "configure", "export-credentials", "--profile", profile,
             "--format", "env-no-export", "--no-cli-pager"],
            capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        raise AWSAuthError("aws CLI unavailable for credential resolution")
    if proc.returncode != 0:
        raise AWSAuthError("credential resolution failed; refresh the AWS "
                           "session for the profile (details suppressed)")
    out = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            k = k.strip()
            if k in _ENV_KEYS:
                out[k] = v.strip()
    if not out.get("AWS_ACCESS_KEY_ID") or not out.get("AWS_SECRET_ACCESS_KEY"):
        raise AWSAuthError("credential provider returned no usable keys "
                           "(re-authenticate the profile)")
    return out


def get_function_url(profile: str, region: str, function_name: str) -> tuple:
    """Fetch the Function URL via the AWS CLI; requires AuthType AWS_IAM."""
    try:
        proc = subprocess.run(
            ["aws", "lambda", "get-function-url-config", "--function-name",
             function_name, "--profile", profile, "--region", region,
             "--no-cli-pager", "--output", "json"],
            capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        raise AWSAuthError("aws CLI unavailable for URL discovery")
    if proc.returncode != 0:
        raise AWSAuthError("could not fetch the Function URL "
                           "(details suppressed; check function name)")
    cfg = json.loads(proc.stdout)
    if cfg.get("AuthType") != "AWS_IAM":
        raise AWSAuthError(
            f"refusing endpoint with AuthType={cfg.get('AuthType')!r}; "
            "only AWS_IAM-authenticated URLs are used")
    return cfg.get("FunctionUrl"), cfg.get("AuthType")


def _hmac(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode(), hashlib.sha256).digest()


def sign_headers(url: str, body_json: str, creds: dict, region: str,
                 service: str = "lambda") -> dict:
    """SigV4 headers for POST <url> with body_json. Headers are NEVER logged."""
    host = url.split("//", 1)[1].split("/", 1)[0]
    now = datetime.now(timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = now.strftime("%Y%m%d")
    payload_hash = hashlib.sha256(body_json.encode()).hexdigest()

    canonical_headers = ("content-type:application/json\n"
                         f"host:{host}\nx-amz-date:{amz_date}\n")
    signed_headers = "content-type;host;x-amz-date"
    headers = {"Content-Type": "application/json", "x-amz-date": amz_date}
    token = creds.get("AWS_SESSION_TOKEN") or ""
    if token:
        canonical_headers += f"x-amz-security-token:{token}\n"
        signed_headers = "content-type;host;x-amz-date;x-amz-security-token"
        headers["x-amz-security-token"] = token

    canonical = "\n".join(["POST", "/", "", canonical_headers,
                           signed_headers, payload_hash])
    scope = f"{date_stamp}/{region}/{service}/aws4_request"
    string_to_sign = "\n".join([
        "AWS4-HMAC-SHA256", amz_date, scope,
        hashlib.sha256(canonical.encode()).hexdigest()])
    k_date = _hmac(("AWS4" + creds["AWS_SECRET_ACCESS_KEY"]).encode(), date_stamp)
    k_region = _hmac(k_date, region)
    k_service = _hmac(k_region, service)
    k_signing = _hmac(k_service, "aws4_request")
    signature = hmac.new(k_signing, string_to_sign.encode(),
                         hashlib.sha256).hexdigest()
    headers["Authorization"] = (
        f"AWS4-HMAC-SHA256 Credential={creds['AWS_ACCESS_KEY_ID']}/{scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}")
    return headers


def signed_post(url: str, body_json: str, creds: dict, region: str,
                timeout: float = 30.0) -> tuple:
    """Send the signed POST. Returns (http_status, body_text, latency_ms).
    Network failures raise AWSTransportError; header/credential values are
    never stored in the returned text or printed."""
    headers = sign_headers(url, body_json, creds, region)
    req = urllib.request.Request(url, data=body_json.encode(),
                                 headers=headers, method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode()
            return resp.status, text, int((time.time() - t0) * 1000)
    except urllib.error.HTTPError as exc:
        try:
            text = exc.read().decode()
        except Exception:
            text = ""
        return exc.code, text, int((time.time() - t0) * 1000)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise AWSTransportError(
            f"endpoint unreachable: {getattr(exc, 'reason', exc)}") from None


def mask_url(url: str) -> str:
    """Endpoint host with the random function-url id truncated (log-safe)."""
    try:
        scheme, rest = url.split("//", 1)
        parts = rest.split("/", 1)
        host = parts[0]
        main = host.split(".")[0]
        return f"{scheme}//{main[:6]}***.{host.split('.', 1)[1]}"
    except Exception:
        return "<masked-url>"
