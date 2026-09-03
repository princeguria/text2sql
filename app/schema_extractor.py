"""
Phase 1.1 -- Auto-extract the database schema.

Produces a structured representation (tables -> columns/types, primary/foreign
keys, sample categorical values) that becomes the grounding context for the
LLM's SQL generation. Uses DuckDB's information_schema directly (equivalent in
spirit to SQLAlchemy's Inspector; swap in SQLAlchemy's `inspect()` unchanged if
you point this at Postgres/MySQL/etc instead of DuckDB).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any

import duckdb

from app.config import DB_PATH, SAMPLE_VALUES_PER_COLUMN


@dataclass
class ColumnInfo:
    name: str
    data_type: str
    is_primary_key: bool = False
    foreign_key: str | None = None  # "other_table.other_column"
    sample_values: list[Any] = field(default_factory=list)
    description: str | None = None


@dataclass
class TableInfo:
    name: str
    columns: list[ColumnInfo]
    row_count: int
    description: str | None = None

    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]


class SchemaExtractor:
    """Introspects a DuckDB database and caches the structured schema."""

    # Optional business-glossary hints. In a real deployment these would come
    # from a data catalog; here they demonstrate the "business glossary terms"
    # the brief calls for.
    TABLE_DESCRIPTIONS = {
        "customers": "People who have signed up and may place orders.",
        "products": "Sellable catalog items, each in one category.",
        "orders": "One row per checkout/order placed by a customer.",
        "order_items": "Line items within an order; use this for revenue/quantity math.",
        "categories": "Product categories/departments.",
    }
    COLUMN_DESCRIPTIONS = {
        "orders.status": "One of: completed, cancelled, refunded, pending. "
                          "'Gross revenue' usually includes all statuses; "
                          "'net revenue' usually excludes cancelled and refunded.",
        "order_items.unit_price": "Price per unit AT TIME OF PURCHASE (already gross, pre-tax).",
    }

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._cache: dict[str, TableInfo] | None = None

    def extract(self, force_refresh: bool = False) -> dict[str, TableInfo]:
        if self._cache is not None and not force_refresh:
            return self._cache

        con = duckdb.connect(self.db_path, read_only=True)
        try:
            tables = {}
            table_rows = con.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main' ORDER BY table_name"
            ).fetchall()

            pk_map = self._primary_keys(con)
            fk_map = self._foreign_keys(con)

            for (table_name,) in table_rows:
                cols_raw = con.execute(
                    "SELECT column_name, data_type FROM information_schema.columns "
                    "WHERE table_schema='main' AND table_name=? ORDER BY ordinal_position",
                    [table_name],
                ).fetchall()

                columns = []
                for col_name, data_type in cols_raw:
                    fq = f"{table_name}.{col_name}"
                    col = ColumnInfo(
                        name=col_name,
                        data_type=data_type,
                        is_primary_key=(table_name, col_name) in pk_map,
                        foreign_key=fk_map.get((table_name, col_name)),
                        description=self.COLUMN_DESCRIPTIONS.get(fq),
                    )
                    if self._is_categorical(data_type):
                        col.sample_values = self._sample_values(con, table_name, col_name)
                    columns.append(col)

                row_count = con.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
                tables[table_name] = TableInfo(
                    name=table_name,
                    columns=columns,
                    row_count=row_count,
                    description=self.TABLE_DESCRIPTIONS.get(table_name),
                )
            self._cache = tables
            return tables
        finally:
            con.close()

    @staticmethod
    def _is_categorical(data_type: str) -> bool:
        return data_type.upper() in {"VARCHAR", "BOOLEAN", "CHAR", "ENUM"}

    @staticmethod
    def _sample_values(con, table: str, column: str) -> list[Any]:
        try:
            rows = con.execute(
                f'SELECT DISTINCT "{column}" FROM "{table}" '
                f'WHERE "{column}" IS NOT NULL LIMIT {SAMPLE_VALUES_PER_COLUMN}'
            ).fetchall()
            return [r[0] for r in rows]
        except Exception:
            return []

    @staticmethod
    def _primary_keys(con) -> set[tuple[str, str]]:
        rows = con.execute("""
            SELECT kcu.table_name, kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
            WHERE tc.constraint_type = 'PRIMARY KEY'
        """).fetchall()
        return {(t, c) for t, c in rows}

    @staticmethod
    def _foreign_keys(con) -> dict[tuple[str, str], str]:
        # DuckDB's constraint introspection for FKs is limited via
        # information_schema alone, so fall back to duckdb_constraints().
        try:
            rows = con.execute("""
                SELECT table_name, constraint_column_names, constraint_text
                FROM duckdb_constraints()
                WHERE constraint_type = 'FOREIGN KEY'
            """).fetchall()
        except Exception:
            return {}
        fk_map = {}
        for table_name, cols, text in rows:
            # constraint_text looks like: FOREIGN KEY (category_id) REFERENCES categories(category_id)
            try:
                ref_part = text.split("REFERENCES")[1].strip()
                ref_table = ref_part.split("(")[0].strip()
                ref_col = ref_part.split("(")[1].split(")")[0].strip()
                for col in cols:
                    fk_map[(table_name, col)] = f"{ref_table}.{ref_col}"
            except Exception:
                continue
        return fk_map

    def as_dict(self) -> dict:
        return {name: asdict(info) for name, info in self.extract().items()}

    def as_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, default=str)


if __name__ == "__main__":
    extractor = SchemaExtractor()
    print(extractor.as_json())
