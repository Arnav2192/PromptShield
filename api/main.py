"""
PromptShield Demo API — FastAPI wrapper around the promptshield library.

Endpoints:
  POST /inspect   — inspect a text prompt (rate-limited per session)
  GET  /health    — liveness check for start.py status polling
"""
from __future__ import annotations

import os
import time
from collections import defaultdict
from threading import Lock
from typing import Dict, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Config (from env vars with sensible demo defaults)
# ---------------------------------------------------------------------------
DEMO_RATE_LIMIT: int = int(os.getenv("DEMO_RATE_LIMIT", "10"))
DEMO_RATE_WINDOW_SECONDS: int = int(os.getenv("DEMO_RATE_WINDOW_SECONDS", "3600"))
CORS_ORIGIN: str = os.getenv("CORS_ORIGIN", "http://localhost:8080")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="PromptShield Demo API",
    description="Rate-limited demo wrapper around the PromptShield library.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[CORS_ORIGIN, "http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory sliding-window rate limiter
# ---------------------------------------------------------------------------
_rate_store: Dict[str, List[float]] = defaultdict(list)
_rate_lock = Lock()


def _check_rate_limit(session_id: str) -> tuple[bool, int, int]:
    """
    Return (allowed, scans_remaining, retry_after_seconds).

    Uses a sliding window: keep only timestamps within the last
    DEMO_RATE_WINDOW_SECONDS seconds and count them.
    """
    now = time.time()
    window_start = now - DEMO_RATE_WINDOW_SECONDS

    with _rate_lock:
        timestamps = _rate_store[session_id]
        # Evict expired timestamps
        timestamps[:] = [t for t in timestamps if t > window_start]

        count = len(timestamps)
        if count >= DEMO_RATE_LIMIT:
            # Oldest timestamp + window length = when a slot frees up
            oldest = timestamps[0]
            retry_after = int(oldest + DEMO_RATE_WINDOW_SECONDS - now) + 1
            return False, 0, retry_after

        # Record this request
        timestamps.append(now)
        scans_remaining = DEMO_RATE_LIMIT - len(timestamps)
        return True, scans_remaining, 0


# ---------------------------------------------------------------------------
# Lazy PromptShield instance (one per session is expensive; reuse one global)
# ---------------------------------------------------------------------------
_shield_lock = Lock()
_shield = None


def _get_shield():
    global _shield
    if _shield is None:
        with _shield_lock:
            if _shield is None:
                from promptshield import PromptShield  # noqa: PLC0415
                _shield = PromptShield(session_id="__api__", tier="free")
    return _shield


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class InspectRequest(BaseModel):
    text: str
    session_id: str = "default"


class InspectResponse(BaseModel):
    blocked: bool
    risk_score: float
    cumulative_score: float
    confidence: float
    reason: str
    patterns_matched: List[str]
    scans_remaining: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    """Liveness check — returns 200 when the server is ready."""
    return {"status": "ok", "version": "0.1.0"}


@app.post("/inspect", response_model=InspectResponse)
def inspect(body: InspectRequest, request: Request):
    """
    Inspect a prompt for injection / jailbreak signals.

    Rate-limited to DEMO_RATE_LIMIT requests per DEMO_RATE_WINDOW_SECONDS
    per session_id.  Exceeding the limit returns HTTP 429.
    """
    allowed, scans_remaining, retry_after = _check_rate_limit(body.session_id)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "rate_limit_exceeded",
                "retry_after_seconds": retry_after,
                "message": (
                    f"Demo limit reached. {DEMO_RATE_LIMIT} scans/hour per session. "
                    "Install the library to run unlimited scans locally."
                ),
            },
        )

    shield = _get_shield()

    # Create a fresh PromptShield instance scoped to this session so state is
    # tracked per caller rather than globally across all demo users.
    from promptshield import PromptShield  # noqa: PLC0415
    session_shield = PromptShield(session_id=body.session_id, tier="free")
    result = session_shield.inspect_input(body.text)

    return InspectResponse(
        blocked=result.blocked,
        risk_score=round(result.risk_score, 4),
        cumulative_score=round(result.cumulative_score, 4),
        confidence=round(result.classifier_result.confidence, 4),
        reason=result.reason,
        patterns_matched=result.rule_result.patterns,
        scans_remaining=scans_remaining,
    )
