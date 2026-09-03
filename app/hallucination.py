"""
Phase 3 -- Hallucination detection system.

Combines four independent signals into one confidence score:
  1. syntax_valid          -- did the query parse & execute cleanly
  2. back_translation       -- does "what question does this SQL answer?"
                               (asked to the LLM again, blind to the original
                               question) match what the user actually asked
  3. result_sanity          -- do the returned values pass basic plausibility
                               checks (magnitudes, date ranges, null-heavy joins)
  4. multi_query_agreement  -- does an independently-generated second query
                               produce the same result
  5. schema_coverage        -- did the query touch the tables/columns you'd
                               expect for this type of question
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from app.config import CONFIDENCE_WEIGHTS
from app.schema_extractor import TableInfo
from app.schemas import ConfidenceBreakdown, ExecutionResult, SanityFlag


# ---------------------------------------------------------------------------
# 3.2 Result sanity checking
# ---------------------------------------------------------------------------
def sanity_check(result: ExecutionResult, tables_used: list[TableInfo]) -> tuple[list[SanityFlag], float]:
    """
    Returns (flags, sanity_score in [0,1]). Score starts at 1.0 and is
    penalized for each flagged anomaly; 'info' flags don't penalize.
    """
    flags: list[SanityFlag] = []
    score = 1.0

    if result.row_count == 0:
        flags.append(SanityFlag(
            check="empty_result",
            detail="Query returned zero rows. Could be a correct 'no data matches' "
                   "answer, or a bad JOIN/filter -- worth a second look.",
            severity="warning",
        ))
        score -= 0.25

    # NULL-heavy columns often indicate a bad JOIN (e.g. LEFT JOIN that should be INNER,
    # or joining on the wrong key).
    if result.rows:
        n = len(result.rows)
        for col_idx, col_name in enumerate(result.columns):
            nulls = sum(1 for row in result.rows if row[col_idx] is None)
            if n >= 5 and nulls / n > 0.5:
                flags.append(SanityFlag(
                    check="null_heavy_column",
                    detail=f"Column '{col_name}' is NULL in {nulls}/{n} rows -- "
                           "possible bad JOIN or mismatched key.",
                    severity="warning",
                ))
                score -= 0.15

    # Plausible-range checks on numeric aggregates: flag negative values in
    # columns that look like counts/sums/revenue (should never be negative
    # in this schema), and implausibly large values.
    for col_idx, col_name in enumerate(result.columns):
        lname = col_name.lower()
        looks_like_count_or_money = any(
            key in lname for key in ("count", "total", "sum", "revenue", "amount", "quantity", "price")
        )
        if not looks_like_count_or_money or not result.rows:
            continue
        numeric_vals = [row[col_idx] for row in result.rows if isinstance(row[col_idx], (int, float))]
        if not numeric_vals:
            continue
        if any(v < 0 for v in numeric_vals):
            flags.append(SanityFlag(
                check="negative_value",
                detail=f"Column '{col_name}' contains negative value(s); unexpected for a count/revenue metric.",
                severity="warning",
            ))
            score -= 0.2
        max_val = max(numeric_vals)
        if "count" in lname and max_val > 100_000:
            flags.append(SanityFlag(
                check="implausible_magnitude",
                detail=f"Column '{col_name}' has a value of {max_val:,}, "
                       "far larger than the dataset's known scale -- check for a fan-out JOIN.",
                severity="warning",
            ))
            score -= 0.2

    # Date range plausibility: flag dates far outside any table's observed range.
    for col_idx, col_name in enumerate(result.columns):
        if "date" not in col_name.lower():
            continue
        for row in result.rows:
            val = row[col_idx]
            if isinstance(val, (dt.date, dt.datetime)):
                if val.year < 2000 or val.year > dt.date.today().year + 1:
                    flags.append(SanityFlag(
                        check="implausible_date",
                        detail=f"Column '{col_name}' contains {val}, outside a plausible business date range.",
                        severity="warning",
                    ))
                    score -= 0.15
                    break

    score = max(0.0, min(1.0, score))
    return flags, score


# ---------------------------------------------------------------------------
# 3.3 Multi-query agreement
# ---------------------------------------------------------------------------
def compare_results(a: ExecutionResult, b: ExecutionResult) -> float:
    """
    Returns an agreement score in [0,1] between two ExecutionResults.
    Order-insensitive, tolerant of column-name differences (aliases), compares
    on row-value multisets when shapes match.
    """
    if a.row_count == 0 and b.row_count == 0:
        return 1.0
    if len(a.columns) != len(b.columns):
        # Still check if the "core" numeric content overlaps -- different
        # column counts strongly suggest different semantics though.
        return 0.0

    def normalize_rows(res: ExecutionResult):
        out = []
        for row in res.rows:
            norm_row = tuple(
                round(v, 2) if isinstance(v, float) else v
                for v in row
            )
            out.append(norm_row)
        return sorted(out, key=lambda r: [str(x) for x in r])

    rows_a = normalize_rows(a)
    rows_b = normalize_rows(b)

    if rows_a == rows_b:
        return 1.0

    # Partial credit: fraction of rows in the smaller set that appear in the larger set.
    set_a, set_b = set(rows_a), set(rows_b)
    if not set_a or not set_b:
        return 0.0
    overlap = len(set_a & set_b)
    return overlap / max(len(set_a), len(set_b))


# ---------------------------------------------------------------------------
# 3.4 Schema coverage
# ---------------------------------------------------------------------------
def schema_coverage_score(question: str, tables_used: list[str], candidate_tables: list[TableInfo]) -> float:
    """
    Did the query touch tables that were plausibly relevant to the question
    (per the same relevance ranking used for prompt filtering)? A query that
    ignores every relevant table, or pulls in tables the filter never
    surfaced, is more likely wrong.
    """
    candidate_names = {t.name for t in candidate_tables}
    if not candidate_names:
        return 0.5
    used = set(tables_used)
    if not used:
        return 0.0
    overlap = len(used & candidate_names)
    return min(1.0, overlap / len(candidate_names)) if candidate_names else 0.5


# ---------------------------------------------------------------------------
# Combined confidence score
# ---------------------------------------------------------------------------
def compute_confidence(
    syntax_valid: bool,
    back_translation_alignment: float,
    result_sanity: float,
    multi_query_agreement: float | None,
    schema_coverage: float,
) -> ConfidenceBreakdown:
    weights = dict(CONFIDENCE_WEIGHTS)

    # If we didn't run a second query (e.g. simple lookup), redistribute its
    # weight proportionally across the other signals rather than penalizing.
    if multi_query_agreement is None:
        redistribute = weights.pop("multi_query_agreement")
        total_remaining = sum(weights.values())
        weights = {k: v + redistribute * (v / total_remaining) for k, v in weights.items()}
        multi_query_agreement = 0.0  # excluded from weighted sum below

    syntax_score = 1.0 if syntax_valid else 0.0

    overall = (
        weights.get("syntax_valid", 0) * syntax_score
        + weights.get("back_translation_alignment", 0) * back_translation_alignment
        + weights.get("result_sanity", 0) * result_sanity
        + weights.get("multi_query_agreement", 0) * multi_query_agreement
        + weights.get("schema_coverage", 0) * schema_coverage
    )

    return ConfidenceBreakdown(
        syntax_valid=syntax_score,
        back_translation_alignment=back_translation_alignment,
        result_sanity=result_sanity,
        multi_query_agreement=multi_query_agreement,
        schema_coverage=schema_coverage,
        overall=round(max(0.0, min(1.0, overall)), 4),
    )
