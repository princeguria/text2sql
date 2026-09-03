"""
Phase 2.3 / 2.4 -- Query sandboxing and the execution layer.

Layer 2 of defense: even if guardrails.py missed something, every query here
runs against a READ-ONLY DuckDB connection (`read_only=True`), so writes are
rejected at the engine level regardless of what slipped through the SQL text
filter. Results are capped, timed, and an EXPLAIN plan is captured for
auditability.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import duckdb

from app.config import DB_PATH
from app.schemas import ExecutionResult


class ExecutionError(Exception):
    pass


@dataclass
class RawExecution:
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    truncated: bool
    execution_time_ms: float
    explain_plan: str = ""


def get_readonly_connection(db_path: str = DB_PATH):
    # read_only=True is enforced by DuckDB itself: any DML/DDL attempted over
    # this handle raises, independent of the guardrail text-based checks.
    return duckdb.connect(db_path, read_only=True)


def execute_sql(sql: str, row_limit: int, db_path: str = DB_PATH) -> RawExecution:
    con = get_readonly_connection(db_path)
    try:
        try:
            explain_rows = con.execute(f"EXPLAIN {sql}").fetchall()
            explain_plan = "\n".join(str(r) for r in explain_rows)
        except Exception:
            explain_plan = ""

        start = time.perf_counter()
        try:
            cursor = con.execute(sql)
        except Exception as e:
            raise ExecutionError(f"Database rejected the query: {e}") from e
        columns = [d[0] for d in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        elapsed_ms = (time.perf_counter() - start) * 1000

        truncated = len(rows) >= row_limit
        rows_out = [list(r) for r in rows[:row_limit]]

        return RawExecution(
            columns=columns,
            rows=rows_out,
            row_count=len(rows_out),
            truncated=truncated,
            execution_time_ms=round(elapsed_ms, 2),
            explain_plan=explain_plan,
        )
    finally:
        con.close()


def to_execution_result(raw: RawExecution) -> ExecutionResult:
    return ExecutionResult(
        columns=raw.columns,
        rows=raw.rows,
        row_count=raw.row_count,
        truncated=raw.truncated,
        execution_time_ms=raw.execution_time_ms,
    )
