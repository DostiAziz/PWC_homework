from pathlib import Path

from pwc_support.storage.database import Database


def test_initialize_creates_async_tables_and_indexes(tmp_path: Path) -> None:
    database = Database(tmp_path / "operations.sqlite3")

    database.initialize()

    with database.connect() as connection:
        names = {
            row["name"]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        indexes = {
            row["name"]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='index'")
        }

    assert {
        "inbound_messages",
        "review_decisions",
        "outbox_messages",
        "mailbox_messages",
    } <= names
    assert {"idx_inbound_lease", "idx_outbox_status", "idx_mailbox_thread"} <= indexes


def test_initialize_adds_missing_columns_to_legacy_review_table(tmp_path: Path) -> None:
    database = Database(tmp_path / "legacy.sqlite3")
    with database.connect() as connection:
        connection.execute(
            """
            CREATE TABLE review_requests (
                review_id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                categories_json TEXT NOT NULL,
                original_message TEXT NOT NULL,
                proposed_reply_json TEXT,
                proposed_actions_json TEXT NOT NULL,
                checkpoint_id TEXT,
                response_version INTEGER NOT NULL,
                status TEXT NOT NULL
            )
            """
        )

    database.initialize()

    with database.connect() as connection:
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(review_requests)")}

    assert {"evidence_json", "routing_provenance_json"} <= columns


def test_initialize_adds_case_delivery_columns_before_unique_index(tmp_path: Path) -> None:
    database = Database(tmp_path / "legacy-cases.sqlite3")
    with database.connect() as connection:
        connection.execute(
            """
            CREATE TABLE cases (
                case_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                client_id TEXT NOT NULL,
                category TEXT NOT NULL,
                status TEXT NOT NULL,
                summary TEXT NOT NULL,
                version INTEGER NOT NULL
            )
            """
        )

    database.initialize()

    with database.connect() as connection:
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(cases)")}
        indexes = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='cases'"
            )
        }

    assert {"inbound_message_id", "reply_recipient", "delivery_thread_id"} <= columns
    assert "uq_cases_inbound_message" in indexes
