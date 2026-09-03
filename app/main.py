"""
Phase 4.1 -- API endpoints.

Run with:  uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import DB_PATH, QUERY_HISTORY_PATH
from app.db import build_database
from app.orchestrator import Orchestrator
from app.schema_extractor import SchemaExtractor
from app.schemas import QueryResponse

build_database()  # no-op if the sample DB already exists

app = FastAPI(
    title="Text-to-SQL Interface with Guardrails and Hallucination Detection",
    version="1.0.0",
    description="Translates natural-language questions into guarded, validated SQL.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = Orchestrator()
FEEDBACK_PATH = str(Path(QUERY_HISTORY_PATH).parent / "feedback.jsonl")


class QueryRequest(BaseModel):
    question: str
    run_cross_check: bool = True
    row_limit: int | None = None


class FeedbackRequest(BaseModel):
    query_id: str
    correct: bool
    note: str | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    kwargs = {"run_cross_check": req.run_cross_check}
    if req.row_limit:
        kwargs["row_limit"] = req.row_limit
    try:
        return orchestrator.answer(req.question, **kwargs)
    except RuntimeError as e:
        # e.g. missing ANTHROPIC_API_KEY
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/schema")
def schema():
    extractor = SchemaExtractor()
    return extractor.as_dict()


@app.get("/v1/history")
def history(limit: int = 50):
    path = Path(QUERY_HISTORY_PATH)
    if not path.exists():
        return []
    lines = path.read_text().strip().splitlines()
    entries = [json.loads(l) for l in lines[-limit:]]
    entries.reverse()
    return entries


@app.post("/v1/feedback")
def feedback(req: FeedbackRequest):
    """
    Phase 4.3 -- the flywheel. Users mark a result correct/incorrect; incorrect
    results become candidate regression cases for eval/golden_queries.json,
    correct ones become candidate few-shot examples in prompt_builder.py.
    """
    Path(FEEDBACK_PATH).parent.mkdir(parents=True, exist_ok=True)
    entry = {"ts": time.time(), **req.model_dump()}
    with open(FEEDBACK_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return {"status": "recorded"}
