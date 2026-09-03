"""FastAPI wrapper around the existing LiDARFailureAnalyzer agent.

This module is READ-ONLY with respect to src/: it imports the agent and
serves it over HTTP. No RAG logic lives here.

Run:  ./venv/bin/python -m uvicorn web.server:app --reload --port 8000
"""

import random
import re
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.agent import LiDARFailureAnalyzer

STATIC_DIR = Path(__file__).parent / "static"
MAX_QUESTION_CHARS = 4000

# src/agent.py catches every exception internally and returns a normal-looking
# dict whose diagnosis is an error string. Without detecting it the API would
# answer 200 OK with that error text as the "answer".
#
# Matched by pattern rather than a fixed prefix because the agent owns this
# wording and may reword it ("Error during diagnosis:", "Error during
# analysis:", ...). The trailing colon keeps legitimate prose such as
# "Error propagation in the BEV feature space" from being flagged.
AGENT_ERROR_RE = re.compile(r"^Error (?:during|diagnosing|analysing|analyzing)\b[^:\n]{0,40}:")

# Markers of a transient upstream fault (NVIDIA endpoint overloaded / slow)
# as opposed to a permanent one (bad API key, unknown model).
TRANSIENT_MARKERS = (
    "503",
    "502",
    "504",
    "429",
    "timed out",
    "timeout",
    "overloaded",
    "service unavailable",
    "temporarily",
    "connection reset",
    "connection aborted",
)

MAX_ATTEMPTS = 2  # the agent retries internally too; this bounds the outer loop
BACKOFF_BASE_SECONDS = 2.0
BACKOFF_JITTER_SECONDS = 0.75

app = FastAPI(title="LiDAR Failure Analyzer UI")

# The agent loads Chroma + the LLM client once; it is expensive to build,
# so it is created lazily on first request and then reused.
_agent = None
_agent_error = None


def get_agent() -> LiDARFailureAnalyzer:
    global _agent, _agent_error
    if _agent is None:
        try:
            _agent = LiDARFailureAnalyzer()
            _agent_error = None
        except Exception as exc:  # surface init failures to the UI, don't hide them
            _agent_error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            raise HTTPException(status_code=503, detail=f"Agent failed to initialise: {_agent_error}")
    return _agent


def agent_error_message(result: dict) -> str | None:
    """Return the agent's internal error text, or None if the run succeeded."""
    diagnosis = str(result.get("diagnosis") or "")
    return diagnosis if AGENT_ERROR_RE.match(diagnosis) else None


def is_transient(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in TRANSIENT_MARKERS)


def run_with_retries(agent: LiDARFailureAnalyzer, question: str) -> dict:
    """Bounded retry with exponential backoff + jitter for transient upstream
    faults. Permanent errors fail immediately - retrying them only burns time."""
    last_error = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        result = agent.run(question)
        error = agent_error_message(result)
        if error is None:
            return result

        last_error = error
        if not is_transient(error) or attempt == MAX_ATTEMPTS:
            break

        delay = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
        delay += random.uniform(0, BACKOFF_JITTER_SECONDS)
        print(f"  [Retry] attempt {attempt}/{MAX_ATTEMPTS} failed ({error}); retrying in {delay:.1f}s")
        time.sleep(delay)

    status = 503 if is_transient(last_error) else 502
    raise HTTPException(status_code=status, detail=upstream_detail(last_error))


def upstream_detail(error: str) -> str:
    if is_transient(error):
        return (
            "The NVIDIA model endpoint is overloaded or timing out. "
            f"Tried {MAX_ATTEMPTS} times without success - please retry in a moment. "
            f"(upstream said: {error})"
        )
    return f"The model call failed. (upstream said: {error})"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


class Source(BaseModel):
    filename: str
    chunk_id: int | str | None = None
    score: float | None = None
    text: str = ""


class AskResponse(BaseModel):
    answer: str
    confidence: float
    citations: list[str]
    sources: list[Source]
    is_validated: bool
    elapsed_seconds: float


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "agent_loaded": _agent is not None,
        "agent_error": _agent_error,
    }


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """Run one question through the agent. Sync on purpose: FastAPI runs
    non-async endpoints in a threadpool, which keeps the blocking
    LangGraph call from stalling the event loop."""
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question is empty.")

    agent = get_agent()
    started = time.perf_counter()
    try:
        result = run_with_retries(agent, question)
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Agent error: {type(exc).__name__}: {exc}")
    elapsed = time.perf_counter() - started

    papers = result.get("retrieved_papers") or []
    sources = [
        Source(
            filename=str(p.get("filename", "unknown")),
            chunk_id=p.get("chunk_id"),
            score=p.get("score"),
            text=str(p.get("text", ""))[:800],
        )
        for p in papers
        if isinstance(p, dict)
    ]

    return AskResponse(
        answer=str(result.get("diagnosis") or "No diagnosis returned."),
        confidence=float(result.get("confidence") or 0.0),
        citations=[str(c) for c in (result.get("citations") or [])],
        sources=sources,
        is_validated=bool(result.get("is_validated")),
        elapsed_seconds=round(elapsed, 2),
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
