"""
Import your own CSV / Excel files into the DuckDB warehouse this project
queries. Each file becomes one table, named after the filename (sanitized).

Usage:
    # Import every .csv/.xlsx in a folder into a NEW database file
    python -m app.import_data --dir my_data/ --db data/my_warehouse.duckdb

    # Import specific files, replacing the sample warehouse in place
    python -m app.import_data --files sales.csv customers.xlsx

    # Add to an existing database without wiping other tables
    python -m app.import_data --files new_table.csv --no-replace-db

After importing, point the app at your database:
    # in .env
    T2SQL_DB_PATH=data/my_warehouse.duckdb

Then restart uvicorn so it picks up the new path.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import duckdb
import pandas as pd

from app.config import DB_PATH


def sanitize_table_name(filename: str) -> str:
    name = Path(filename).stem.lower()
    name = re.sub(r"[^a-z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name or name[0].isdigit():
        name = f"t_{name}"
    return name


def load_file_as_dataframe(path: Path, sheet_name: str | int | None = 0) -> dict[str, pd.DataFrame]:
    """
    Returns {table_name: dataframe}. A single-sheet CSV returns one entry;
    a multi-sheet Excel file returns one entry per sheet (table name gets a
    _<sheetname> suffix when there's more than one sheet).
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
        return {sanitize_table_name(path.name): df}

    if suffix in (".xlsx", ".xls"):
        sheets = pd.read_excel(path, sheet_name=None)  # dict of all sheets
        if len(sheets) == 1:
            only_name = next(iter(sheets))
            return {sanitize_table_name(path.name): sheets[only_name]}
        base = sanitize_table_name(path.name)
        return {f"{base}_{sanitize_table_name(sheet)}": df for sheet, df in sheets.items()}

    raise ValueError(f"Unsupported file type: {path.suffix} (only .csv, .xlsx, .xls are supported)")


def import_files(
    files: list[Path],
    db_path: str = DB_PATH,
    replace_db: bool = True,
) -> dict[str, int]:
    """
    Loads each file as one or more tables into db_path.
    If replace_db is True, any existing database file at db_path is deleted
    first (a clean slate). If False, tables are added/replaced in place --
    useful for adding one more file to a database you already built.

    Returns {table_name: row_count} for everything that was loaded.
    """
    path_obj = Path(db_path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    if replace_db and path_obj.exists():
        path_obj.unlink()

    con = duckdb.connect(str(path_obj))
    loaded: dict[str, int] = {}
    try:
        for f in files:
            f = Path(f)
            if not f.exists():
                print(f"  SKIP (not found): {f}")
                continue
            tables = load_file_as_dataframe(f)
            for table_name, df in tables.items():
                con.register("_tmp_df", df)
                con.execute(f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM _tmp_df')
                con.unregister("_tmp_df")
                loaded[table_name] = len(df)
                print(f"  {f.name} -> table \"{table_name}\" ({len(df):,} rows, "
                      f"{len(df.columns)} columns: {', '.join(df.columns[:6])}"
                      f"{', ...' if len(df.columns) > 6 else ''})")
    finally:
        con.close()

    return loaded


def collect_files_from_dir(directory: Path) -> list[Path]:
    exts = {".csv", ".xlsx", ".xls"}
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in exts)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--files", nargs="*", help="Specific CSV/Excel files to import.")
    parser.add_argument("--dir", help="Import every .csv/.xlsx/.xls file found in this directory.")
    parser.add_argument("--db", default=DB_PATH, help=f"Target DuckDB file (default: {DB_PATH})")
    parser.add_argument("--no-replace-db", action="store_true",
                         help="Add to the existing database instead of starting from a clean file.")
    args = parser.parse_args()

    files: list[Path] = []
    if args.files:
        files.extend(Path(f) for f in args.files)
    if args.dir:
        files.extend(collect_files_from_dir(Path(args.dir)))

    if not files:
        parser.error("Provide --files one_or_more.csv, --dir a_folder/, or both.")

    print(f"Importing {len(files)} file(s) into {args.db} "
          f"({'replacing' if not args.no_replace_db else 'adding to'} the database)...\n")
    loaded = import_files(files, db_path=args.db, replace_db=not args.no_replace_db)

    print(f"\nDone. {len(loaded)} table(s) loaded into {args.db}:")
    for name, count in loaded.items():
        print(f"  - {name}: {count:,} rows")
    print(f"\nSet this in your .env to use it:\n  T2SQL_DB_PATH={args.db}")


if __name__ == "__main__":
    main()
