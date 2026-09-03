"""
Ties every phase together into one pipeline: this is what the API layer and
the eval suite both call. Keeping this separate from FastAPI routing makes it
directly testable and directly reusable from eval/run_evals.py.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from app.config import DEFAULT_ROW_LIMIT, LOW_CONFIDENCE_THRESHOLD, QUERY_HISTORY_PATH
from app.guardrails import apply_guardrails, check_scan_budget
from app.hallucination import (
    compare_results,
    compute_confidence,
    sanity_check,
    schema_coverage_score,
)
from app.llm_client import get_llm_client
from app.prompt_builder import SchemaFilter
from app.sandbox_executor import ExecutionError, execute_sql, get_readonly_connection, to_execution_result
from app.schema_extractor import SchemaExtractor
from app.schemas import (
    ClarificationNeeded,
    ConfidenceBreakdown,
    GeneratedSQL,
    QueryResponse,
    SanityFlag,
)


class Orchestrator:
    def __init__(self, llm_client=None):
        self.extractor = SchemaExtractor()
        self.llm = llm_client  # lazily created if None, so eval/tests can inject a fake

    def _get_llm(self):
        if self.llm is None:
            self.llm = get_llm_client()
        return self.llm

    def answer(self, question: str, run_cross_check: bool = True, row_limit: int = DEFAULT_ROW_LIMIT) -> QueryResponse:
        query_id = str(uuid.uuid4())
        tables = self.extractor.extract()
        schema_filter = SchemaFilter(tables)
        relevant_tables = schema_filter.select(question)

        llm = self._get_llm()

        # ---- Phase 1.4: ambiguity handling --------------------------------
        generation = llm.generate_sql(question, relevant_tables)
        if isinstance(generation, ClarificationNeeded):
            resp = QueryResponse(
                status="clarification_needed",
                question=question,
                clarification=generation,
                query_id=query_id,
            )
            self._log_history(resp)
            return resp

        assert isinstance(generation, GeneratedSQL)

        # ---- Phase 2.2: guardrails ------------------------------------------
        guardrail_report = apply_guardrails(generation.sql, question=question, row_limit=row_limit)
        if not guardrail_report.passed:
            resp = QueryResponse(
                status="blocked",
                question=question,
                sql=generation.sql,
                explanation=generation.explanation,
                guardrail=guardrail_report,
                error="Query blocked by guardrails before execution.",
                query_id=query_id,
            )
            self._log_history(resp)
            return resp

        safe_sql = guardrail_report.normalized_sql

        # Scan-budget check (EXPLAIN-based) against the read-only connection.
        con = get_readonly_connection()
        try:
            scan_violation = check_scan_budget(con, safe_sql)
        finally:
            con.close()
        if scan_violation:
            guardrail_report.passed = False
            guardrail_report.violations.append(scan_violation)
            resp = QueryResponse(
                status="blocked",
                question=question,
                sql=safe_sql,
                explanation=generation.explanation,
                guardrail=guardrail_report,
                error="Query blocked: exceeds scan budget.",
                query_id=query_id,
            )
            self._log_history(resp)
            return resp

        # ---- Phase 2.3/2.4: sandboxed execution -----------------------------
        try:
            raw = execute_sql(safe_sql, row_limit=row_limit)
        except ExecutionError as e:
            resp = QueryResponse(
                status="error",
                question=question,
                sql=safe_sql,
                explanation=generation.explanation,
                guardrail=guardrail_report,
                error=str(e),
                query_id=query_id,
            )
            self._log_history(resp)
            return resp
        result = to_execution_result(raw)

        # ---- Phase 3.1: back-translation alignment --------------------------
        try:
            restated = llm.back_translate(safe_sql)
            alignment = llm.semantic_similarity(question, restated)
        except Exception:
            alignment = 0.5  # neutral if the LLM call itself fails

        # ---- Phase 3.2: result sanity checking -------------------------------
        sanity_flags, sanity_score = sanity_check(result, relevant_tables)

        # ---- Phase 3.3: multi-query agreement (independent second query) ----
        agreement_score = None
        alt_result = None
        alt_sql = None
        if run_cross_check:
            try:
                alt = llm.generate_alternative_sql(question, relevant_tables, safe_sql)
                alt_guardrail = apply_guardrails(alt["sql"], question=question, row_limit=row_limit)
                if alt_guardrail.passed:
                    alt_raw = execute_sql(alt_guardrail.normalized_sql, row_limit=row_limit)
                    alt_result = to_execution_result(alt_raw)
                    alt_sql = alt_guardrail.normalized_sql
                    agreement_score = compare_results(result, alt_result)
                    if agreement_score < 0.99:
                        sanity_flags.append(
                            SanityFlag(
                                check="cross_check_disagreement",
                                detail=f"Independent second query agreement score: {agreement_score:.2f}. "
                                       "Results diverge -- review both queries below.",
                                severity="warning",
                            )
                        )
            except Exception:
                agreement_score = None  # don't fail the whole request if cross-check errors out

        # ---- Phase 3.4: schema coverage ---------------------------------------
        coverage = schema_coverage_score(question, generation.tables_used, relevant_tables)

        confidence = compute_confidence(
            syntax_valid=True,  # we only reach here if guardrails+execution succeeded
            back_translation_alignment=alignment,
            result_sanity=sanity_score,
            multi_query_agreement=agreement_score,
            schema_coverage=coverage,
        )

        if confidence.overall < LOW_CONFIDENCE_THRESHOLD:
            sanity_flags.append(
                SanityFlag(
                    check="low_overall_confidence",
                    detail=f"Overall confidence {confidence.overall:.2f} is below the "
                           f"{LOW_CONFIDENCE_THRESHOLD} threshold -- treat this result as provisional.",
                    severity="warning",
                )
            )

        resp = QueryResponse(
            status="ok",
            question=question,
            sql=safe_sql,
            explanation=generation.explanation,
            result=result,
            confidence=confidence,
            sanity_flags=sanity_flags,
            guardrail=guardrail_report,
            alternative_result=alt_result,
            alternative_sql=alt_sql,
            query_id=query_id,
        )
        self._log_history(resp)
        return resp

    @staticmethod
    def _log_history(resp: QueryResponse) -> None:
        Path(QUERY_HISTORY_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(QUERY_HISTORY_PATH, "a") as f:
            f.write(resp.model_dump_json() + "\n")
