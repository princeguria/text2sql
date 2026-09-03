"""Pydantic models shared across the LLM client, API layer, and eval suite."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ClarificationOption(BaseModel):
    interpretation: str
    example_sql: str


class ClarificationNeeded(BaseModel):
    kind: Literal["clarification_needed"] = "clarification_needed"
    reason: str
    options: list[ClarificationOption]


class GeneratedSQL(BaseModel):
    kind: Literal["sql"] = "sql"
    sql: str
    explanation: str
    confidence: float = Field(ge=0.0, le=1.0, description="Model's own self-reported confidence")
    tables_used: list[str]
    columns_used: list[str]


class GuardrailViolation(BaseModel):
    rule: str
    detail: str


class GuardrailReport(BaseModel):
    passed: bool
    violations: list[GuardrailViolation] = Field(default_factory=list)
    normalized_sql: str | None = None
    applied_row_limit: int | None = None


class SanityFlag(BaseModel):
    check: str
    detail: str
    severity: Literal["info", "warning"] = "warning"


class ConfidenceBreakdown(BaseModel):
    syntax_valid: float
    back_translation_alignment: float
    result_sanity: float
    multi_query_agreement: float
    schema_coverage: float
    overall: float


class ExecutionResult(BaseModel):
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    truncated: bool
    execution_time_ms: float


class QueryResponse(BaseModel):
    status: Literal["ok", "clarification_needed", "blocked", "error"]
    question: str
    sql: str | None = None
    explanation: str | None = None
    result: ExecutionResult | None = None
    confidence: ConfidenceBreakdown | None = None
    sanity_flags: list[SanityFlag] = Field(default_factory=list)
    guardrail: GuardrailReport | None = None
    clarification: ClarificationNeeded | None = None
    alternative_result: ExecutionResult | None = None
    alternative_sql: str | None = None
    error: str | None = None
    query_id: str | None = None
