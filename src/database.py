import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(os.getenv("BANKING_DB_PATH", "banking.db"))


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
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
