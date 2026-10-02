"""Optional access protection with a key (API key).

Configuration via environment variables:

- ``API_KEY``: one or more keys (comma-separated). Empty = no protection.
- ``API_KEY_FILE``: file with keys (one per line or comma-separated), e.g. a Docker
  secret. Read in addition to ``API_KEY``.
- ``SESSION_HOURS``: validity of the browser session after login (default 12).
- ``COOKIE_SECURE``: ``1``/``true`` forces the Secure flag on the session cookie
  (HTTPS behind a reverse proxy). Set automatically for direct HTTPS.

API clients send the key as ``X-API-Key: <key>`` or ``Authorization: Bearer <key>``.
The web UI logs in once with the key and receives a signed, time-limited session
cookie that does not contain the key itself.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import time
from pathlib import Path

from starlette.requests import HTTPConnection

SESSION_COOKIE = "ai_label_checker_session"
_SESSION_PURPOSE = "ai-label-checker-session"


class ConfigurationError(RuntimeError):
    """API_KEY_FILE is set but not readable."""


def api_keys() -> list[str]:
    """Reads the allowed keys on every call (changes take effect without restart)."""
    raw = os.environ.get("API_KEY", "")
    path = os.environ.get("API_KEY_FILE", "").strip()
    if path:
        try:
            raw += "\n" + Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigurationError(f"API_KEY_FILE not readable: {path}: {exc.strerror or exc}") from exc
    return [k.strip() for k in re.split(r"[,\r\n]", raw) if k.strip()]


def protection_enabled() -> bool:
    try:
        return bool(api_keys())
    except ConfigurationError:
        return True  # fail closed: better locked than unprotected


def _equal(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def key_valid(candidate: str | None) -> bool:
    if not candidate:
        return False
    try:
        allowed = api_keys()
    except ConfigurationError:
        return False
    # Compare against all keys (no early exit), constant time per comparison.
    return any([_equal(candidate, k) for k in allowed])


def key_from_request(conn: HTTPConnection) -> str | None:
    header = conn.headers.get("x-api-key")
    if header:
        return header.strip()
    authorization = conn.headers.get("authorization", "")
    scheme, _, value = authorization.partition(" ")
    if scheme.lower() == "bearer" and value.strip():
        return value.strip()
    return None


# ---------------------------------------------------------------------------
# Browser session
# ---------------------------------------------------------------------------


def session_duration_seconds() -> int:
    try:
        hours = float(os.environ.get("SESSION_HOURS", "12"))
    except ValueError:
        hours = 12.0
    return max(60, int(hours * 3600))


def cookie_secure(conn: HTTPConnection) -> bool:
    forced = os.environ.get("COOKIE_SECURE", "").strip().lower() in {"1", "true", "yes"}
    return forced or conn.url.scheme == "https"


def _signature(key: str, timestamp: str) -> str:
    message = f"{_SESSION_PURPOSE}:{timestamp}".encode()
    return hmac.new(key.encode("utf-8"), message, hashlib.sha256).hexdigest()


def create_session(key: str, now: float | None = None) -> str:
    timestamp = str(int(now if now is not None else time.time()))
    return f"{timestamp}.{_signature(key, timestamp)}"


def session_valid(token: str | None, now: float | None = None) -> bool:
    """Checks signature and age. Changing the key invalidates all sessions."""
    if not token:
        return False
    timestamp, _, signature = token.partition(".")
    if not timestamp.isdigit() or not signature:
        return False
    age = (now if now is not None else time.time()) - int(timestamp)
    if age < -60 or age > session_duration_seconds():
        return False
    try:
        allowed = api_keys()
    except ConfigurationError:
        return False
    return any([_equal(signature, _signature(k, timestamp)) for k in allowed])


def access_allowed(conn: HTTPConnection) -> bool:
    """Key in header or valid session cookie. Always allowed when no key is configured."""
    if not protection_enabled():
        return True
    return key_valid(key_from_request(conn)) or session_valid(conn.cookies.get(SESSION_COOKIE))
