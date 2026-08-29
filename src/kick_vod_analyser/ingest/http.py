"""HTTP client factory that prefers a TLS-impersonating session for Kick.

Kick sits behind Cloudflare bot protection which rejects the default Python TLS
fingerprint. curl_cffi impersonates a real browser handshake when installed;
otherwise the plain httpx client is used and callers must tolerate 403s.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

log = logging.getLogger(__name__)

BROWSER_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://kick.com/",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
}


class HttpResponse(Protocol):
    status_code: int
    text: str

    def json(self) -> Any: ...


class HttpClient(Protocol):
    def get(self, url: str, **kwargs: Any) -> HttpResponse: ...
    def close(self) -> None: ...


def impersonation_available() -> bool:
    try:
        import curl_cffi  # noqa: F401
    except ImportError:
        return False
    return True


def build_client(timeout: float = 30.0) -> HttpClient:
    """Return the best available client for Cloudflare-protected endpoints."""
    if impersonation_available():
        from curl_cffi import requests as curl_requests

        return curl_requests.Session(
            headers=BROWSER_HEADERS, timeout=timeout, impersonate="chrome"
        )

    import httpx

    log.warning(
        "curl_cffi not installed; Kick requests may be blocked by Cloudflare. "
        "Install the kick extra to enable browser TLS impersonation."
    )
    return httpx.Client(headers=BROWSER_HEADERS, timeout=timeout, follow_redirects=True)
