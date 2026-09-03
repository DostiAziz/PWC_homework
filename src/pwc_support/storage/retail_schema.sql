CREATE TABLE IF NOT EXISTS products (
    product_id TEXT PRIMARY KEY, name TEXT NOT NULL, category TEXT NOT NULL,
    price NUMERIC NOT NULL CHECK(price >= 0), currency TEXT NOT NULL,
    attributes_json TEXT NOT NULL DEFAULT '{}', active INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS inventory (
    product_id TEXT NOT NULL REFERENCES products(product_id), location TEXT NOT NULL,
    quantity INTEGER NOT NULL CHECK(quantity >= 0), PRIMARY KEY(product_id, location)
);
CREATE TABLE IF NOT EXISTS offers (
    offer_id TEXT PRIMARY KEY, product_id TEXT NOT NULL REFERENCES products(product_id),
    description TEXT NOT NULL, discount_percent NUMERIC NOT NULL CHECK(discount_percent >= 0),
    active INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS customers (customer_id TEXT PRIMARY KEY, email TEXT NOT NULL UNIQUE);
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY, customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    status TEXT NOT NULL, total NUMERIC NOT NULL CHECK(total >= 0), currency TEXT NOT NULL,
    delivered_at TEXT
);
CREATE TABLE IF NOT EXISTS order_items (
    item_id TEXT PRIMARY KEY, order_id TEXT NOT NULL REFERENCES orders(order_id),
    product_id TEXT NOT NULL REFERENCES products(product_id), quantity INTEGER NOT NULL CHECK(quantity > 0),
    unit_price NUMERIC NOT NULL CHECK(unit_price >= 0)
);
CREATE TABLE IF NOT EXISTS return_requests (
    return_id TEXT PRIMARY KEY, order_id TEXT NOT NULL REFERENCES orders(order_id), item_id TEXT NOT NULL,
    reason TEXT NOT NULL, status TEXT NOT NULL, idempotency_key TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS refund_requests (
    refund_id TEXT PRIMARY KEY, return_id TEXT NOT NULL REFERENCES return_requests(return_id),
    amount NUMERIC NOT NULL CHECK(amount >= 0), status TEXT NOT NULL, payment_reference TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS retail_audit_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT, event_type TEXT NOT NULL, entity_id TEXT NOT NULL,
    details_json TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_products_category ON products(category);
CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_returns_status ON return_requests(status);
