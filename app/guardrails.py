"""
Phase 2.2 -- Guardrail middleware.

Every generated SQL statement passes through here BEFORE execution. This is
layer 1 of defense; layer 2 is the read-only DB connection in
sandbox_executor.py. Each rule is independently toggleable/configurable via
app.config, and every blocked query is logged with its reason for audit.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import sqlparse
from sqlparse.sql import Statement
from sqlparse.tokens import DDL, DML, Keyword

from app.config import (
    DEFAULT_ROW_LIMIT,
    FORBIDDEN_STATEMENT_TYPES,
    GUARDRAIL_LOG_PATH,
    MAX_ESTIMATED_ROWS_SCANNED,
    MAX_SUBQUERY_DEPTH,
    READ_ONLY_STATEMENT_TYPES,
)
from app.schemas import GuardrailReport, GuardrailViolation


class GuardrailError(Exception):
    pass


def _log(question: str, sql: str, report: GuardrailReport) -> None:
    Path(GUARDRAIL_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": time.time(),
        "question": question,
        "sql": sql,
        "passed": report.passed,
        "violations": [v.model_dump() for v in report.violations],
    }
    with open(GUARDRAIL_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _statement_count(parsed: list[Statement]) -> int:
    return len([s for s in parsed if s.token_first(skip_cm=True) is not None])


def _get_statement_type(stmt: Statement) -> str:
    first = stmt.token_first(skip_cm=True)
    if first is None:
        return "UNKNOWN"
    return first.value.upper()


def _contains_forbidden_keyword(sql_upper: str) -> str | None:
    # Defense-in-depth beyond the leading keyword check: catch forbidden verbs
    # hiding inside CTEs or multi-statement payloads.
    for kw in FORBIDDEN_STATEMENT_TYPES:
        if re.search(rf"\b{kw}\b", sql_upper):
            return kw
    return None


def _max_subquery_depth(sql: str) -> int:
    depth = 0
    max_depth = 0
    in_string = False
    quote_char = ""
    for ch in sql:
        if in_string:
            if ch == quote_char:
                in_string = False
            continue
        if ch in ("'", '"'):
            in_string = True
            quote_char = ch
            continue
        if ch == "(":
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == ")":
            depth = max(0, depth - 1)
    return max_depth


def _has_limit(sql_upper: str) -> bool:
    return re.search(r"\bLIMIT\s+\d+\b", sql_upper) is not None


def _inject_limit(sql: str, limit: int) -> str:
    trimmed = sql.rstrip().rstrip(";").rstrip()
    return f"{trimmed}\nLIMIT {limit};"


def apply_guardrails(
    sql: str,
    question: str = "",
    row_limit: int = DEFAULT_ROW_LIMIT,
    max_subquery_depth: int = MAX_SUBQUERY_DEPTH,
) -> GuardrailReport:
    """
    Validates and (where safe) auto-repairs a generated SQL string.
    Returns a GuardrailReport; callers must check `.passed` before executing
    and must execute `.normalized_sql`, not the original, when passed.
    """
    violations: list[GuardrailViolation] = []
    sql_stripped = sql.strip()

    if not sql_stripped:
        violations.append(GuardrailViolation(rule="empty_query", detail="Generated SQL was empty."))
        report = GuardrailReport(passed=False, violations=violations)
        _log(question, sql, report)
        return report

    try:
        parsed = sqlparse.parse(sql_stripped)
    except Exception as e:
        violations.append(GuardrailViolation(rule="parse_error", detail=str(e)))
        report = GuardrailReport(passed=False, violations=violations)
        _log(question, sql, report)
        return report

    if _statement_count(parsed) != 1:
        violations.append(GuardrailViolation(
            rule="multi_statement",
            detail=f"Exactly one SQL statement is allowed; found {_statement_count(parsed)}. "
                   "Statement stacking is blocked to prevent smuggling a second destructive statement.",
        ))

    stmt = parsed[0] if parsed else None
    sql_upper = sql_stripped.upper()

    if stmt is not None:
        stmt_type = _get_statement_type(stmt)
        if stmt_type not in READ_ONLY_STATEMENT_TYPES:
            violations.append(GuardrailViolation(
                rule="non_read_only_statement",
                detail=f"Statement type '{stmt_type}' is not permitted. Only SELECT/WITH/EXPLAIN are allowed.",
            ))

    forbidden_hit = _contains_forbidden_keyword(sql_upper)
    if forbidden_hit:
        violations.append(GuardrailViolation(
            rule="forbidden_keyword",
            detail=f"Query contains forbidden keyword '{forbidden_hit}'. "
                   "DDL/DML/administrative statements are never permitted.",
        ))

    depth = _max_subquery_depth(sql_stripped)
    if depth > max_subquery_depth:
        violations.append(GuardrailViolation(
            rule="subquery_too_deep",
            detail=f"Nested parenthesis depth {depth} exceeds max allowed {max_subquery_depth}.",
        ))

    if violations:
        report = GuardrailReport(passed=False, violations=violations)
        _log(question, sql, report)
        return report

    normalized = sql_stripped
    applied_limit = None
    if not _has_limit(sql_upper) and not sql_upper.startswith("EXPLAIN"):
        normalized = _inject_limit(sql_stripped, row_limit)
        applied_limit = row_limit

    report = GuardrailReport(passed=True, violations=[], normalized_sql=normalized, applied_row_limit=applied_limit)
    _log(question, sql, report)
    return report


def estimate_scanned_rows(con, sql: str) -> int | None:
    """
    Uses EXPLAIN to get a rough estimate of rows scanned, used to reject
    queries that would hammer the database. Best-effort: returns None if the
    engine's EXPLAIN output can't be parsed (fails open on estimation, but the
    row LIMIT guardrail still bounds the returned result set).
    """
    try:
        plan_rows = con.execute(f"EXPLAIN {sql}").fetchall()
        plan_text = "\n".join(str(r) for r in plan_rows)
        numbers = [int(n) for n in re.findall(r"~?(\d{2,})\s*(?:Rows|rows)", plan_text)]
        return max(numbers) if numbers else None
    except Exception:
        return None


def check_scan_budget(con, sql: str) -> GuardrailViolation | None:
    estimate = estimate_scanned_rows(con, sql)
    if estimate is not None and estimate > MAX_ESTIMATED_ROWS_SCANNED:
        return GuardrailViolation(
            rule="scan_budget_exceeded",
            detail=f"Estimated rows scanned ({estimate:,}) exceeds budget "
                   f"({MAX_ESTIMATED_ROWS_SCANNED:,}).",
        )
    return None
