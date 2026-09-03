"""
Phase 1.2 / 1.3 -- Build the dynamic, schema-aware prompt and filter it down
to only the tables likely relevant to the question.

Relevance filtering uses lightweight TF-IDF-style keyword overlap between the
question and each table's "document" (table name, column names, descriptions,
sample values). This avoids a network round-trip to an embeddings API while
still meaningfully shrinking the prompt on wide schemas.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from app.config import MAX_TABLES_IN_PROMPT, SCHEMA_RELEVANCE_THRESHOLD
from app.schema_extractor import TableInfo

_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "is", "are",
    "what", "how", "many", "much", "which", "was", "were", "by", "with", "me",
    "show", "list", "give", "find", "get", "who", "did", "do", "does", "each",
    "per", "all", "that", "this", "as", "from", "top", "average", "avg",
}


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-zA-Z_]+", text.lower()) if t not in _STOPWORDS and len(t) > 2]


def _stem(token: str) -> str:
    """Minimal English stemmer -- just enough to match singular/plural table
    and column names (category/categories, order/orders) without pulling in
    a full NLP dependency."""
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("es") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def _table_document(table: TableInfo) -> str:
    parts = [table.name, table.description or ""]
    for col in table.columns:
        parts.append(col.name)
        parts.append(col.description or "")
        parts.extend(str(v) for v in col.sample_values)
    return " ".join(parts)


@dataclass
class TableRelevance:
    table: TableInfo
    score: float


class SchemaFilter:
    """Ranks tables by relevance to a question using keyword overlap (TF-IDF-lite)."""

    def __init__(self, tables: dict[str, TableInfo]):
        self.tables = tables
        self._docs = {name: _tokenize(_table_document(t)) for name, t in tables.items()}
        self._doc_freq = self._build_doc_freq()

    def _build_doc_freq(self) -> Counter:
        df = Counter()
        for tokens in self._docs.values():
            for token in set(tokens):
                df[token] += 1
        return df

    def rank(self, question: str) -> list[TableRelevance]:
        q_tokens = _tokenize(question)
        n_docs = max(len(self._docs), 1)
        results = []
        for name, table in self.tables.items():
            doc_tokens = self._docs[name]
            doc_counts = Counter(doc_tokens)
            stemmed_name = _stem(name.replace("_", ""))
            score = 0.0
            for qt in q_tokens:
                if qt in doc_counts:
                    tf = doc_counts[qt] / max(len(doc_tokens), 1)
                    idf = math.log((n_docs + 1) / (self._doc_freq.get(qt, 0) + 1)) + 1
                    score += tf * idf
                # reward stem match against the table name itself (handles category/categories, etc.)
                if _stem(qt) == stemmed_name or _stem(qt) in name or qt in name:
                    score += 0.5
            results.append(TableRelevance(table=table, score=score))
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def select(self, question: str) -> list[TableInfo]:
        ranked = self.rank(question)
        selected = [r.table for r in ranked if r.score >= SCHEMA_RELEVANCE_THRESHOLD]
        if not selected:
            # Fall back to sending everything rather than an empty schema --
            # better to over-inform than to silently fail.
            selected = [r.table for r in ranked]
        return selected[:MAX_TABLES_IN_PROMPT]


FEW_SHOT_EXAMPLES = [
    {
        "question": "How many customers signed up in 2024?",
        "sql": "SELECT COUNT(*) AS customer_count FROM customers "
               "WHERE signup_date >= DATE '2024-01-01' AND signup_date < DATE '2025-01-01';",
    },
    {
        "question": "What is the total net revenue by product category, "
                     "excluding cancelled and refunded orders?",
        "sql": (
            "SELECT c.category_name, "
            "SUM(oi.quantity * oi.unit_price) AS net_revenue "
            "FROM order_items oi "
            "JOIN orders o ON o.order_id = oi.order_id "
            "JOIN products p ON p.product_id = oi.product_id "
            "JOIN categories c ON c.category_id = p.category_id "
            "WHERE o.status NOT IN ('cancelled', 'refunded') "
            "GROUP BY c.category_name "
            "ORDER BY net_revenue DESC;"
        ),
    },
    {
        "question": "List the top 5 customers by number of completed orders.",
        "sql": (
            "SELECT cu.customer_id, cu.first_name, cu.last_name, COUNT(*) AS order_count "
            "FROM orders o "
            "JOIN customers cu ON cu.customer_id = o.customer_id "
            "WHERE o.status = 'completed' "
            "GROUP BY cu.customer_id, cu.first_name, cu.last_name "
            "ORDER BY order_count DESC "
            "LIMIT 5;"
        ),
    },
]


def format_schema_block(tables: list[TableInfo]) -> str:
    lines = []
    for t in tables:
        lines.append(f"TABLE {t.name} ({t.row_count} rows)" + (f" -- {t.description}" if t.description else ""))
        for c in t.columns:
            flags = []
            if c.is_primary_key:
                flags.append("PK")
            if c.foreign_key:
                flags.append(f"FK -> {c.foreign_key}")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            desc_str = f"  -- {c.description}" if c.description else ""
            sample_str = ""
            if c.sample_values:
                sample_str = f"  (sample values: {', '.join(map(str, c.sample_values))})"
            lines.append(f"  - {c.name}: {c.data_type}{flag_str}{sample_str}{desc_str}")
        lines.append("")
    return "\n".join(lines)


def format_few_shot_block(n: int = 3) -> str:
    lines = []
    for ex in FEW_SHOT_EXAMPLES[:n]:
        lines.append(f"Q: {ex['question']}\nSQL:\n{ex['sql']}\n")
    return "\n".join(lines)


SYSTEM_PROMPT_TEMPLATE = """You are a meticulous analytics engineer that translates natural-language \
business questions into a single read-only SQL query for a DuckDB database.

Rules:
- Only ever write SELECT / WITH (CTE) queries. Never write DDL or DML.
- Use only the tables and columns given in the schema below. Never invent columns.
- Always qualify ambiguous column names with table aliases.
- If the question is genuinely ambiguous (e.g. "revenue" could mean gross or \
net), pick the most standard interpretation, but say so explicitly in your \
explanation and mention the alternative.
- Add a LIMIT clause (<= 1000) unless the question clearly asks for an \
aggregate that returns few rows.
- Return your answer using the `generate_sql` tool. Do not include any SQL \
in plain prose.

DATABASE SCHEMA (relevant tables only):
{schema_block}

EXAMPLES:
{few_shot_block}
"""


def build_system_prompt(tables: list[TableInfo]) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        schema_block=format_schema_block(tables),
        few_shot_block=format_few_shot_block(),
    )
