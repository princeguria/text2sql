"""
Central configuration. Everything that a compliance / infra reviewer would want
to be able to tune without touching code lives here, and can be overridden with
environment variables.
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    # Load .env automatically so no shell-specific export/source command is
    # needed on any OS (bash, zsh, PowerShell, cmd.exe all just work).
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass  # python-dotenv not installed -- fall back to real environment variables only

BASE_DIR = Path(__file__).resolve().parent.parent

# ---- Database -------------------------------------------------------------
# DuckDB is used as the "real SQL engine" per the project brief: it enforces
# real ANSI SQL semantics (unlike SQLite's loose typing) but needs no server
# process, which keeps the whole stack runnable in one container.
DB_PATH = os.environ.get("T2SQL_DB_PATH", str(BASE_DIR / "data" / "warehouse.duckdb"))

# ---- LLM --------------------------------------------------------------
LLM_PROVIDER = os.environ.get("T2SQL_LLM_PROVIDER", "gemini")  # "gemini" or "anthropic"

# Gemini (default -- free tier via Google AI Studio)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("T2SQL_LLM_MODEL", "gemini-2.5-flash")

# Anthropic (optional, if T2SQL_LLM_PROVIDER=anthropic)
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.environ.get("T2SQL_ANTHROPIC_MODEL", "claude-sonnet-4-5")

LLM_MAX_TOKENS = int(os.environ.get("T2SQL_LLM_MAX_TOKENS", "2048"))

# ---- Schema filtering ----------------------------------------------------
# Only tables whose relevance score to the question clears this bar are sent
# to the model. Keeps prompts small and generation accuracy high on wide
# schemas. Pure keyword/TF-IDF overlap -- no embedding API call required.
SCHEMA_RELEVANCE_THRESHOLD = float(os.environ.get("T2SQL_SCHEMA_THRESHOLD", "0.05"))
MAX_TABLES_IN_PROMPT = int(os.environ.get("T2SQL_MAX_TABLES", "6"))
SAMPLE_VALUES_PER_COLUMN = 5

# ---- Guardrails ------------------------------------------------------------
DEFAULT_ROW_LIMIT = int(os.environ.get("T2SQL_ROW_LIMIT", "1000"))
MAX_SUBQUERY_DEPTH = int(os.environ.get("T2SQL_MAX_SUBQUERY_DEPTH", "3"))
MAX_ESTIMATED_ROWS_SCANNED = int(os.environ.get("T2SQL_MAX_SCAN_ROWS", "5_000_000"))
FORBIDDEN_STATEMENT_TYPES = {
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE",
    "GRANT", "REVOKE", "ATTACH", "DETACH", "COPY", "EXPORT", "IMPORT",
    "PRAGMA", "CALL", "SET",
}
READ_ONLY_STATEMENT_TYPES = {"SELECT", "WITH", "EXPLAIN"}

# ---- Hallucination detection / confidence weighting -----------------------
CONFIDENCE_WEIGHTS = {
    "syntax_valid": 0.15,
    "back_translation_alignment": 0.30,
    "result_sanity": 0.25,
    "multi_query_agreement": 0.20,
    "schema_coverage": 0.10,
}
LOW_CONFIDENCE_THRESHOLD = float(os.environ.get("T2SQL_LOW_CONF_THRESHOLD", "0.55"))

# ---- History / logging ----------------------------------------------------
GUARDRAIL_LOG_PATH = os.environ.get("T2SQL_GUARDRAIL_LOG", str(BASE_DIR / "data" / "guardrail_log.jsonl"))
QUERY_HISTORY_PATH = os.environ.get("T2SQL_HISTORY_PATH", str(BASE_DIR / "data" / "history.jsonl"))
