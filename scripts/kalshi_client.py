"""Minimal Kalshi trade-api v2 client with RSA request signing.

Reads KALSHI_KEY_ID and KALSHI_PRIVATE_KEY (inline PEM or a path) from the
environment. Market-data GETs also work unauthenticated; signing mainly raises
rate limits. Run under an interactive zsh so the .zshrc env is present:

    zsh -ic 'uv run python scripts/kalshi_client.py'
"""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

BASE = "https://api.elections.kalshi.com/trade-api/v2"


def _load_key():
    raw = os.environ["KALSHI_PRIVATE_KEY"]
    if raw.strip().startswith("-----BEGIN"):
        data = raw.encode()
    else:
        data = Path(raw).expanduser().read_bytes()
    return serialization.load_pem_private_key(data, password=None)


def _headers(method: str, path: str) -> dict[str, str]:
    key_id = os.environ.get("KALSHI_KEY_ID")
    if not key_id:
        return {}
    ts = str(int(time.time() * 1000))
    msg = f"{ts}{method}/trade-api/v2{path}".encode()
    sig = _load_key().sign(
        msg,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
    return {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
    }


def get(path: str, params: dict | None = None, auth: bool = True) -> requests.Response:
    headers = _headers("GET", path) if auth else {}
    return requests.get(BASE + path, params=params, headers=headers, timeout=30)


if __name__ == "__main__":
    r = get("/portfolio/balance")
    print("auth test /portfolio/balance:", r.status_code, r.text[:120])
    r = get("/exchange/status", auth=False)
    print("public /exchange/status:", r.status_code, r.text[:80])
