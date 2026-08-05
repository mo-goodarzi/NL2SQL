"""Build the SQLite database from the raw Olist CSVs. Run once.

    python scripts/load_data.py
    python scripts/load_data.py --force          # rebuild from scratch
    python scripts/load_data.py --csv-dir /tmp/olist --db-path /tmp/test.db

Dataset: "Brazilian E-Commerce Public Dataset by Olist" (Kaggle,
``olistbr/brazilian-ecommerce``). Download the nine CSVs into ``data/raw/``.

Why this script exists at all, rather than a one-line ``pandas.to_sql`` loop:

* **Declared foreign keys.** ``to_sql`` infers a schema and declares no keys.
  Foreign keys are what let ``schema.py`` tell the model how tables relate --
  without them the agent guesses joins from column names, and accuracy drops
  noticeably on every multi-table question in your test set.
* **Indexes on join columns.** ``order_items`` has ~112k rows. Unindexed joins
  turn into full scans, queries hit the timeout, and those show up in your
  evaluation as accuracy failures when they are really performance failures.
* **Explicit types.** ID columns are 32-character hashes and zip prefixes have
  meaningful leading zeros; both must stay TEXT. Inferred types silently
  mangle them.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# Make ``src`` importable when this is run directly as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import settings  # noqa: E402


@dataclass(frozen=True)
class TableSpec:
    """Everything needed to load one CSV into one table."""

    name: str
    csv: str
    ddl: str
    #: Columns forced to a pandas dtype on read. Used to protect IDs and zip
    #: prefixes from being read as numbers.
    dtypes: dict[str, str] = field(default_factory=dict)
    indexes: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Schema
#
# Dates are stored as TEXT in 'YYYY-MM-DD HH:MM:SS' form, which is how the CSVs
# already supply them. That format sorts lexicographically and works directly
# with SQLite's strftime/date functions, so the agent can answer "in 2017" or
# "by month" without any conversion. Storing them as epoch integers would be
# faster and much harder for a language model to get right.
# ---------------------------------------------------------------------------

TABLES: list[TableSpec] = [
    TableSpec(
        name="customers",
        csv="olist_customers_dataset.csv",
        dtypes={
            "customer_id": "string",
            "customer_unique_id": "string",
            "customer_zip_code_prefix": "string",
        },
        ddl="""
        CREATE TABLE customers (
            customer_id              TEXT PRIMARY KEY,
            -- One person may place several orders; each order gets a fresh
            -- customer_id, but customer_unique_id is stable across them.
            -- Counting distinct customers means counting this column.
            customer_unique_id       TEXT NOT NULL,
            customer_zip_code_prefix TEXT,
            customer_city            TEXT,
            customer_state           TEXT
        )
        """,
        indexes=("customer_unique_id", "customer_state", "customer_zip_code_prefix"),
    ),
    TableSpec(
        name="sellers",
        csv="olist_sellers_dataset.csv",
        dtypes={"seller_id": "string", "seller_zip_code_prefix": "string"},
        ddl="""
        CREATE TABLE sellers (
            seller_id              TEXT PRIMARY KEY,
            seller_zip_code_prefix TEXT,
            seller_city            TEXT,
            seller_state           TEXT
        )
        """,
        indexes=("seller_state", "seller_zip_code_prefix"),
    ),
    TableSpec(
        name="product_category_translation",
        csv="product_category_name_translation.csv",
        dtypes={
            "product_category_name": "string",
            "product_category_name_english": "string",
        },
        ddl="""
        -- Category names in `products` are Portuguese. Keeping the English
        -- translation as its own table (rather than merging it in) is
        -- deliberate: it forces a real join on questions phrased in English,
        -- which is exactly the skill the test set should be measuring.
        CREATE TABLE product_category_translation (
            product_category_name         TEXT PRIMARY KEY,
            product_category_name_english TEXT NOT NULL
        )
        """,
    ),
    TableSpec(
        name="products",
        csv="olist_products_dataset.csv",
        dtypes={"product_id": "string", "product_category_name": "string"},
        ddl="""
        CREATE TABLE products (
            product_id                 TEXT PRIMARY KEY,
            product_category_name      TEXT,
            -- 'lenght' is misspelled in the source data. Kept as-is: renaming
            -- would break anyone cross-referencing the public dataset docs.
            product_name_lenght        INTEGER,
            product_description_lenght INTEGER,
            product_photos_qty         INTEGER,
            product_weight_g           INTEGER,
            product_length_cm          INTEGER,
            product_height_cm          INTEGER,
            product_width_cm           INTEGER,
            FOREIGN KEY (product_category_name)
                REFERENCES product_category_translation (product_category_name)
        )
        """,
        indexes=("product_category_name",),
    ),
    TableSpec(
        name="orders",
        csv="olist_orders_dataset.csv",
        dtypes={"order_id": "string", "customer_id": "string", "order_status": "string"},
        ddl="""
        CREATE TABLE orders (
            order_id                      TEXT PRIMARY KEY,
            customer_id                   TEXT NOT NULL,
            -- One of: delivered, shipped, canceled, unavailable, invoiced,
            -- processing, created, approved. Lower-case in the source data.
            order_status                  TEXT,
            order_purchase_timestamp      TEXT,
            order_approved_at             TEXT,
            order_delivered_carrier_date  TEXT,
            order_delivered_customer_date TEXT,
            order_estimated_delivery_date TEXT,
            FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
        )
        """,
        indexes=("customer_id", "order_status", "order_purchase_timestamp"),
    ),
    TableSpec(
        name="order_items",
        csv="olist_order_items_dataset.csv",
        dtypes={"order_id": "string", "product_id": "string", "seller_id": "string"},
        ddl="""
        -- The bridge table. Any question relating orders to products or
        -- sellers must route through here, which is why schema retrieval has
        -- to expand along foreign keys rather than picking tables by
        -- similarity alone.
        CREATE TABLE order_items (
            order_id            TEXT NOT NULL,
            -- Sequence number within the order (1, 2, 3...), NOT a quantity.
            order_item_id       INTEGER NOT NULL,
            product_id          TEXT NOT NULL,
            seller_id           TEXT NOT NULL,
            shipping_limit_date TEXT,
            price               REAL,
            freight_value       REAL,
            PRIMARY KEY (order_id, order_item_id),
            FOREIGN KEY (order_id)   REFERENCES orders (order_id),
            FOREIGN KEY (product_id) REFERENCES products (product_id),
            FOREIGN KEY (seller_id)  REFERENCES sellers (seller_id)
        )
        """,
        indexes=("product_id", "seller_id"),
    ),
    TableSpec(
        name="order_payments",
        csv="olist_order_payments_dataset.csv",
        dtypes={"order_id": "string", "payment_type": "string"},
        ddl="""
        CREATE TABLE order_payments (
            order_id             TEXT NOT NULL,
            -- An order may be split across several payment methods.
            payment_sequential   INTEGER NOT NULL,
            payment_type         TEXT,
            payment_installments INTEGER,
            payment_value        REAL,
            PRIMARY KEY (order_id, payment_sequential),
            FOREIGN KEY (order_id) REFERENCES orders (order_id)
        )
        """,
        indexes=("payment_type",),
    ),
    TableSpec(
        name="order_reviews",
        csv="olist_order_reviews_dataset.csv",
        dtypes={"review_id": "string", "order_id": "string"},
        ddl="""
        -- No primary key: review_id is not unique in the published data (a
        -- handful of reviews span multiple orders). Declaring one would make
        -- the load fail. Left as an honest reflection of the source, with
        -- indexes instead.
        CREATE TABLE order_reviews (
            review_id               TEXT NOT NULL,
            order_id                TEXT NOT NULL,
            review_score            INTEGER,
            review_comment_title    TEXT,
            review_comment_message  TEXT,
            review_creation_date    TEXT,
            review_answer_timestamp TEXT,
            FOREIGN KEY (order_id) REFERENCES orders (order_id)
        )
        """,
        indexes=("order_id", "review_id", "review_score"),
    ),
    TableSpec(
        name="geolocation",
        csv="olist_geolocation_dataset.csv",
        dtypes={"geolocation_zip_code_prefix": "string"},
        ddl="""
        -- Many rows per zip prefix, so no primary key and no foreign keys
        -- pointing at it. Mostly useful for state/city aggregation questions.
        CREATE TABLE geolocation (
            geolocation_zip_code_prefix TEXT,
            geolocation_lat             REAL,
            geolocation_lng             REAL,
            geolocation_city            TEXT,
            geolocation_state           TEXT
        )
        """,
        indexes=("geolocation_zip_code_prefix", "geolocation_state"),
    ),
]

# Load order matters: parents before children, so foreign_key_check at the end
# reports genuine data problems rather than ordering artifacts.


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--csv-dir", type=Path, default=settings.csv_dir)
    p.add_argument("--db-path", type=Path, default=settings.db_path)
    p.add_argument(
        "--force",
        action="store_true",
        help="Delete an existing database and rebuild it.",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=20_000,
        help="Rows per insert batch (geolocation is ~1M rows).",
    )
    return p.parse_args()


def check_csvs_present(csv_dir: Path) -> None:
    """Fail early, with instructions, if the raw data is not there."""
    missing = [t.csv for t in TABLES if not (csv_dir / t.csv).exists()]
    if missing:
        raise SystemExit(
            f"Missing {len(missing)} CSV file(s) in {csv_dir}:\n"
            + "\n".join(f"  - {m}" for m in missing)
            + "\n\nDownload the dataset from Kaggle "
            "(olistbr/brazilian-ecommerce) and unzip it into that directory."
        )


def read_csv(csv_path: Path, spec: TableSpec) -> pd.DataFrame:
    """Read one CSV with the column types we care about pinned.

    Two things matter here. Zip prefixes and IDs are read as strings so leading
    zeros survive ('01037' must not become 1037). And empty strings become
    NULL rather than '', so ``WHERE review_comment_title IS NULL`` behaves the
    way a model reasonably expects it to.
    """
    df = pd.read_csv(
        csv_path,
        dtype=spec.dtypes or None,
        keep_default_na=True,
        na_values=["", " "],
    )
    # Nullable dtypes (e.g. the "string" dtype used for IDs) coerce a plain
    # None back into pd.NA on assignment, which sqlite3 cannot bind. Casting
    # to object first makes the None actually stick.
    df = df.astype(object).where(pd.notna(df), None)
    return df


def create_schema(conn: sqlite3.Connection) -> None:
    for spec in TABLES:
        conn.execute(spec.ddl)
    conn.commit()


def create_indexes(conn: sqlite3.Connection) -> int:
    """Create indexes after loading -- much faster than maintaining them during."""
    count = 0
    for spec in TABLES:
        for col in spec.indexes:
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{spec.name}_{col} "
                f"ON {spec.name} ({col})"
            )
            count += 1
    conn.commit()
    return count


def load_table(conn: sqlite3.Connection, spec: TableSpec, csv_dir: Path,
               chunk_size: int) -> int:
    """Insert one CSV's rows, returning the row count."""
    df = read_csv(csv_dir / spec.csv, spec)

    # Align to the declared column order; the CSV order is not guaranteed.
    table_cols = [
        row[1] for row in conn.execute(f"PRAGMA table_info({spec.name})").fetchall()
    ]
    unexpected = set(df.columns) - set(table_cols)
    if unexpected:
        raise ValueError(
            f"{spec.csv} has columns not in the {spec.name} schema: "
            f"{sorted(unexpected)}. The dataset version may have changed."
        )
    df = df.reindex(columns=table_cols)

    placeholders = ", ".join("?" * len(table_cols))
    stmt = (
        f"INSERT INTO {spec.name} ({', '.join(table_cols)}) VALUES ({placeholders})"
    )

    rows = list(df.itertuples(index=False, name=None))
    for i in range(0, len(rows), chunk_size):
        conn.executemany(stmt, rows[i : i + chunk_size])
    conn.commit()
    return len(rows)


def report_fk_violations(conn: sqlite3.Connection) -> None:
    """Report orphaned rows instead of crashing on them.

    Foreign keys are enforced OFF during the bulk load, so a small number of
    orphaned rows in the public dataset cannot abort the whole build. But they
    should not pass silently either -- an orphan is a row the agent can never
    reach through a join, and it will look like a wrong answer later.
    """
    violations = conn.execute("PRAGMA foreign_key_check").fetchall()
    if not violations:
        print("  Foreign keys:   all satisfied")
        return
    by_table: dict[str, int] = {}
    for table, _rowid, parent, _fkid in violations:
        by_table[f"{table} -> {parent}"] = by_table.get(f"{table} -> {parent}", 0) + 1
    print(f"  Foreign keys:   {len(violations)} orphaned row(s) found")
    for pair, n in sorted(by_table.items()):
        print(f"                    {n:>6,}  {pair}")
    print("                  (kept -- these exist in the published dataset)")


def main() -> None:
    args = parse_args()
    csv_dir: Path = args.csv_dir
    db_path: Path = args.db_path

    check_csvs_present(csv_dir)

    if db_path.exists():
        if not args.force:
            raise SystemExit(
                f"{db_path} already exists. Pass --force to rebuild it."
            )
        db_path.unlink()

    db_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Building {db_path}")
    print(f"  from {csv_dir}\n")

    conn = sqlite3.connect(db_path)
    try:
        # Off during load so ordering and known orphans do not abort the build;
        # verified explicitly at the end instead.
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("PRAGMA journal_mode = OFF")
        conn.execute("PRAGMA synchronous = OFF")

        create_schema(conn)

        total = 0
        for spec in TABLES:
            n = load_table(conn, spec, csv_dir, args.chunk_size)
            total += n
            print(f"  {spec.name:<30} {n:>9,} rows")

        n_idx = create_indexes(conn)

        # Lets the query planner make sensible choices, which keeps
        # agent-generated joins inside the timeout.
        conn.execute("ANALYZE")
        conn.commit()

        print(f"\n  {'TOTAL':<30} {total:>9,} rows")
        print(f"  Indexes:        {n_idx}")
        report_fk_violations(conn)

        size_mb = db_path.stat().st_size / 1_000_000
        print(f"  Database size:  {size_mb:.1f} MB")
        print(f"\nDone. Next: python scripts/smoke_test.py")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
