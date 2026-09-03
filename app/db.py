"""
Creates and seeds a sample e-commerce warehouse in DuckDB so the whole system
is runnable out of the box with `python -m app.db`. Swap this for a connection
to a real database by changing DB_PATH / the engine in schema_extractor.py.
"""
import random
import datetime as dt
from pathlib import Path

import duckdb

from app.config import DB_PATH

random.seed(7)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS categories (
    category_id INTEGER PRIMARY KEY,
    category_name VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS customers (
    customer_id INTEGER PRIMARY KEY,
    first_name VARCHAR,
    last_name VARCHAR,
    email VARCHAR,
    country VARCHAR,
    signup_date DATE
);

CREATE TABLE IF NOT EXISTS products (
    product_id INTEGER PRIMARY KEY,
    product_name VARCHAR,
    category_id INTEGER REFERENCES categories(category_id),
    unit_price DECIMAL(10,2),
    is_active BOOLEAN
);

CREATE TABLE IF NOT EXISTS orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(customer_id),
    order_date DATE,
    status VARCHAR -- 'completed', 'cancelled', 'refunded', 'pending'
);

CREATE TABLE IF NOT EXISTS order_items (
    order_item_id INTEGER PRIMARY KEY,
    order_id INTEGER REFERENCES orders(order_id),
    product_id INTEGER REFERENCES products(product_id),
    quantity INTEGER,
    unit_price DECIMAL(10,2) -- price at time of purchase (gross)
);
"""

CATEGORY_NAMES = ["Electronics", "Home & Kitchen", "Books", "Sports & Outdoors", "Apparel"]
COUNTRIES = ["USA", "UK", "Germany", "India", "Canada", "Australia", "Brazil"]
STATUSES = ["completed", "completed", "completed", "cancelled", "refunded", "pending"]
PRODUCT_NAMES_BY_CAT = {
    "Electronics": ["Wireless Earbuds", "4K Monitor", "Mechanical Keyboard", "Smart Speaker", "USB-C Hub"],
    "Home & Kitchen": ["Air Fryer", "French Press", "Cast Iron Skillet", "Robot Vacuum", "Blender"],
    "Books": ["Data Engineering 101", "The Silent Ledger", "SQL for Humans", "Atomic Focus", "Deep Work"],
    "Sports & Outdoors": ["Yoga Mat", "Trail Running Shoes", "Camping Tent", "Resistance Bands", "Water Bottle"],
    "Apparel": ["Wool Sweater", "Rain Jacket", "Running Shorts", "Denim Jacket", "Graphic Tee"],
}


def build_database(db_path: str = DB_PATH, force: bool = False) -> None:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        return
    if path.exists():
        path.unlink()

    con = duckdb.connect(str(path))
    con.execute(SCHEMA_SQL)

    # Categories
    for i, name in enumerate(CATEGORY_NAMES, start=1):
        con.execute("INSERT INTO categories VALUES (?, ?)", [i, name])

    # Customers
    n_customers = 200
    for cid in range(1, n_customers + 1):
        signup = dt.date(2023, 1, 1) + dt.timedelta(days=random.randint(0, 900))
        con.execute(
            "INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?)",
            [cid, f"First{cid}", f"Last{cid}", f"customer{cid}@example.com",
             random.choice(COUNTRIES), signup],
        )

    # Products
    pid = 1
    products = []
    for cat_id, cat_name in enumerate(CATEGORY_NAMES, start=1):
        for pname in PRODUCT_NAMES_BY_CAT[cat_name]:
            price = round(random.uniform(9.99, 299.99), 2)
            active = random.random() > 0.1
            con.execute(
                "INSERT INTO products VALUES (?, ?, ?, ?, ?)",
                [pid, pname, cat_id, price, active],
            )
            products.append((pid, price))
            pid += 1

    # Orders + order_items
    n_orders = 1200
    order_item_id = 1
    start_date = dt.date(2023, 6, 1)
    end_date = dt.date(2026, 8, 30)
    span_days = (end_date - start_date).days

    for oid in range(1, n_orders + 1):
        cust_id = random.randint(1, n_customers)
        order_date = start_date + dt.timedelta(days=random.randint(0, span_days))
        status = random.choice(STATUSES)
        con.execute(
            "INSERT INTO orders VALUES (?, ?, ?, ?)",
            [oid, cust_id, order_date, status],
        )
        n_items = random.randint(1, 4)
        chosen = random.sample(products, k=min(n_items, len(products)))
        for prod_id, price in chosen:
            qty = random.randint(1, 3)
            con.execute(
                "INSERT INTO order_items VALUES (?, ?, ?, ?, ?)",
                [order_item_id, oid, prod_id, qty, price],
            )
            order_item_id += 1

    con.close()
    print(f"Seeded warehouse at {path} "
          f"({n_customers} customers, {pid - 1} products, {n_orders} orders, {order_item_id - 1} order_items)")


if __name__ == "__main__":
    build_database(force=True)
