"""
Phase 5.2 -- Automated evals against the golden query dataset.

Measures:
  - execution_match      : does the generated SQL's result match the golden SQL's result
                            (order-insensitive, tolerant of column aliasing)?
  - sql_exact_match       : does the generated SQL string match the golden SQL exactly
                            (informational only -- execution_match is the real bar)
  - hallucination_flagged : for known-ambiguous/unanswerable questions, did the system
                            correctly ask for clarification instead of guessing?
  - guardrail_effective   : for the destructive-request probe, was no write executed?

Usage:
    python -m eval.run_evals                 # requires ANTHROPIC_API_KEY
    python -m eval.run_evals --offline        # dry-run wiring check with a stub LLM, no API calls/cost
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.hallucination import compare_results
from app.orchestrator import Orchestrator
from app.sandbox_executor import execute_sql, to_execution_result
from app.schemas import GeneratedSQL

GOLDEN_PATH = Path(__file__).parent / "golden_queries.json"


class StubLLM:
    """
    Deterministic stand-in for the real LLM so the eval harness (and CI) can be
    exercised without API cost/nondeterminism. Maps each golden question
    directly to its golden SQL / expected behavior. NOT used to claim real
    model accuracy -- only for `--offline` wiring verification.
    """
    def __init__(self, cases: list[dict]):
        self.by_question = {c["question"]: c for c in cases}

    def generate_sql(self, question, tables):
        from app.schemas import ClarificationNeeded, ClarificationOption
        case = self.by_question.get(question)
        if case and case.get("expect_clarification"):
            return ClarificationNeeded(
                reason="Ambiguous or unanswerable with the current schema.",
                options=[ClarificationOption(interpretation="stub", example_sql="SELECT 1")],
            )
        if case and case.get("expect_blocked"):
            return GeneratedSQL(
                sql="DELETE FROM orders WHERE status='cancelled'",
                explanation="stub destructive probe",
                confidence=0.5, tables_used=["orders"], columns_used=["status"],
            )
        sql = case["golden_sql"] if case else "SELECT 1"
        return GeneratedSQL(sql=sql, explanation="stub", confidence=0.9,
                             tables_used=["stub"], columns_used=["stub"])

    def generate_alternative_sql(self, question, tables, first_sql):
        return {"sql": self.by_question[question]["golden_sql"], "explanation": "stub alt", "approach_difference": "none (stub)"}

    def back_translate(self, sql):
        return "stub restated question"

    def semantic_similarity(self, a, b):
        return 0.85


def run(offline: bool, row_limit: int = 1000) -> dict:
    cases = json.loads(GOLDEN_PATH.read_text())
    llm = StubLLM(cases) if offline else None
    orch = Orchestrator(llm_client=llm)

    results = []
    for case in cases:
        t0 = time.time()
        try:
            resp = orch.answer(case["question"], run_cross_check=False, row_limit=row_limit)
        except Exception as e:
            results.append({**case, "outcome": "error", "detail": str(e), "elapsed_s": time.time() - t0})
            continue

        outcome = {"id": case["id"], "question": case["question"], "category": case["category"]}
        outcome["elapsed_s"] = round(time.time() - t0, 2)
        outcome["status"] = resp.status

        if case.get("expect_blocked"):
            outcome["guardrail_effective"] = resp.status == "blocked"
            outcome["pass"] = outcome["guardrail_effective"]
        elif case.get("expect_clarification"):
            outcome["hallucination_flagged"] = resp.status == "clarification_needed"
            outcome["pass"] = outcome["hallucination_flagged"]
        else:
            if resp.status != "ok":
                outcome["pass"] = False
                outcome["detail"] = resp.error or f"unexpected status {resp.status}"
            else:
                outcome["generated_sql"] = resp.sql
                outcome["sql_exact_match"] = _normalize_sql(resp.sql) == _normalize_sql(case["golden_sql"])
                try:
                    golden_raw = execute_sql(case["golden_sql"], row_limit=row_limit)
                    golden_result = to_execution_result(golden_raw)
                    agreement = compare_results(resp.result, golden_result)
                    outcome["execution_match"] = agreement >= 0.999
                    outcome["execution_agreement_score"] = round(agreement, 3)
                except Exception as e:
                    outcome["execution_match"] = False
                    outcome["detail"] = f"golden SQL failed to execute: {e}"
                outcome["confidence"] = resp.confidence.overall if resp.confidence else None
                outcome["pass"] = outcome.get("execution_match", False)

        results.append(outcome)

    return _summarize(results)


def _normalize_sql(sql: str | None) -> str:
    if not sql:
        return ""
    return " ".join(sql.strip().rstrip(";").split()).lower()


def _summarize(results: list[dict]) -> dict:
    n = len(results)
    n_pass = sum(1 for r in results if r.get("pass"))

    regular = [r for r in results if "execution_match" in r]
    n_exec_match = sum(1 for r in regular if r["execution_match"])

    halluc_cases = [r for r in results if "hallucination_flagged" in r]
    n_halluc_correct = sum(1 for r in halluc_cases if r["hallucination_flagged"])

    guardrail_cases = [r for r in results if "guardrail_effective" in r]
    n_guardrail_ok = sum(1 for r in guardrail_cases if r["guardrail_effective"])

    summary = {
        "total_cases": n,
        "total_pass": n_pass,
        "overall_pass_rate": round(n_pass / n, 3) if n else 0,
        "execution_accuracy": f"{n_exec_match}/{len(regular)}" if regular else "n/a",
        "execution_accuracy_pct": round(n_exec_match / len(regular), 3) if regular else None,
        "hallucination_detection_rate": f"{n_halluc_correct}/{len(halluc_cases)}" if halluc_cases else "n/a",
        "hallucination_detection_pct": round(n_halluc_correct / len(halluc_cases), 3) if halluc_cases else None,
        "guardrail_effectiveness": f"{n_guardrail_ok}/{len(guardrail_cases)}" if guardrail_cases else "n/a",
        "unsafe_queries_executed": len(guardrail_cases) - n_guardrail_ok,
        "results": results,
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true",
                         help="Use a deterministic stub LLM instead of calling the real API (wiring check only).")
    parser.add_argument("--out", default=str(Path(__file__).parent / "eval_results.json"))
    args = parser.parse_args()

    summary = run(offline=args.offline)

    print(f"\n{'='*60}\nEVAL SUMMARY {'(OFFLINE STUB -- not real model accuracy)' if args.offline else ''}\n{'='*60}")
    print(f"Overall pass rate:            {summary['total_pass']}/{summary['total_cases']} "
          f"({summary['overall_pass_rate']*100:.1f}%)")
    print(f"Execution accuracy:           {summary['execution_accuracy']}")
    print(f"Hallucination detection rate: {summary['hallucination_detection_rate']}")
    print(f"Guardrail effectiveness:      {summary['guardrail_effectiveness']} "
          f"(unsafe queries executed: {summary['unsafe_queries_executed']})")
    print()
    for r in summary["results"]:
        mark = "PASS" if r.get("pass") else "FAIL"
        print(f"  [{mark}] {r.get('id','?'):28s} {r.get('category',''):20s} {r.get('question','')[:50]}")

    Path(args.out).write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nFull results written to {args.out}")


if __name__ == "__main__":
    main()
