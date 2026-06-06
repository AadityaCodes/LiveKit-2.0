"""SQLite persistence for the banking voice agent.

Two tables:

* ``pending_customers`` — every account application the agent has
  collected. Rows start out with just the verified profile (written by
  ``save_pending_customer`` at the end of Phase 4) and gain
  ``account_number`` / ``routing_number`` once Phase 5 succeeds.

* ``customer_insights`` — one row per customer keyed by
  ``account_number``. Written by the post-call analytics pass; consumed
  by future-call personalization and downstream marketing campaigns.

The database file path defaults to ``banking.db`` in the working directory
and is configurable via the ``BANKING_DB_PATH`` environment variable. The
file is git-ignored.
"""

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# Resolved at import time so all callers share the same location.
DB_PATH = Path(os.getenv("BANKING_DB_PATH", "banking.db"))


@contextmanager
def _connect():
    """Yield a SQLite connection that commits on success and closes always."""
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create all tables if they don't already exist.

    Idempotent — safe to call from every save/update entry point so the
    schema is guaranteed before any read or write.
    """
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pending_customers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                age INTEGER NOT NULL,
                residential_address TEXT NOT NULL,
                identification_number TEXT NOT NULL,
                date_of_birth TEXT NOT NULL,
                phone_number TEXT NOT NULL,
                email TEXT NOT NULL,
                citizenship_status TEXT NOT NULL,
                employment_status TEXT NOT NULL,
                confirmed_goal TEXT NOT NULL,
                account_number TEXT,
                routing_number TEXT,
                provisioned_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        # account_number is the natural primary key for insights because
        # that is what a returning caller will quote to identify themselves.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS customer_insights (
                account_number TEXT PRIMARY KEY,
                customer_id INTEGER,
                summary TEXT,
                sentiment TEXT,
                insights_json TEXT NOT NULL,
                source_transcript TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (customer_id) REFERENCES pending_customers(id)
            )
            """
        )


def save_pending_customer(
    *,
    first_name: str,
    last_name: str,
    age: int,
    residential_address: str,
    identification_number: str,
    date_of_birth: str,
    phone_number: str,
    email: str,
    citizenship_status: str,
    employment_status: str,
    confirmed_goal: str,
) -> int:
    """Insert a verified pending-customer row and return the new ``id``.

    Called from ``collect_customer_information`` once the agent has
    captured and confirmed all ten Phase 3 fields plus the goal flag.
    """
    init_db()
    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO pending_customers (
                first_name, last_name, age, residential_address,
                identification_number, date_of_birth, phone_number, email,
                citizenship_status, employment_status, confirmed_goal
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                first_name,
                last_name,
                age,
                residential_address,
                identification_number,
                date_of_birth,
                phone_number,
                email,
                citizenship_status,
                employment_status,
                confirmed_goal,
            ),
        )
        return cursor.lastrowid


def attach_account_numbers(
    *, customer_id: int, account_number: str, routing_number: str
) -> None:
    """Write the provisioned account/routing numbers back onto an existing row.

    Called from the Core Banking shim after a successful Phase 5 call.
    Raises ``ValueError`` if no row with ``customer_id`` exists, which
    surfaces as a tool ERROR back to the LLM.
    """
    with _connect() as conn:
        result = conn.execute(
            """
            UPDATE pending_customers
            SET account_number = ?,
                routing_number = ?,
                provisioned_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (account_number, routing_number, customer_id),
        )
        if result.rowcount == 0:
            raise ValueError(f"no pending_customers row with id={customer_id}")


def upsert_customer_insights(
    *,
    account_number: str,
    customer_id: int | None,
    summary: str,
    sentiment: str,
    insights: dict[str, Any],
    source_transcript: str | None = None,
) -> None:
    """Insert or replace the insights row for one customer.

    Called from the post-call analytics pass. ``insights`` is the full
    structured payload extracted by the LLM and is stored as JSON; the
    flat ``summary``/``sentiment`` columns are denormalized for easy
    SQL filtering when picking targets for marketing campaigns.
    """
    init_db()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO customer_insights (
                account_number, customer_id, summary, sentiment,
                insights_json, source_transcript, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(account_number) DO UPDATE SET
                customer_id = excluded.customer_id,
                summary = excluded.summary,
                sentiment = excluded.sentiment,
                insights_json = excluded.insights_json,
                source_transcript = excluded.source_transcript,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                account_number,
                customer_id,
                summary,
                sentiment,
                json.dumps(insights, ensure_ascii=False),
                source_transcript,
            ),
        )


def get_customer_insights(account_number: str) -> dict[str, Any] | None:
    """Return the latest insights for a known account number, or ``None``.

    Intended for future calls: when a returning caller quotes their
    account number, the agent (or any downstream marketing job) can use
    this to personalize the conversation or pick a campaign.
    """
    init_db()
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT account_number, customer_id, summary, sentiment,
                   insights_json, source_transcript, updated_at
            FROM customer_insights WHERE account_number = ?
            """,
            (account_number,),
        ).fetchone()
    if not row:
        return None
    return {
        "account_number": row[0],
        "customer_id": row[1],
        "summary": row[2],
        "sentiment": row[3],
        "insights": json.loads(row[4]) if row[4] else {},
        "source_transcript": row[5],
        "updated_at": row[6],
    }
