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
            CREATE TABLE IF NOT EXISTS customers (
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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def save_customer(
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
) -> int:
    init_db()
    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO customers (
                first_name, last_name, age, residential_address,
                identification_number, date_of_birth, phone_number, email,
                citizenship_status, employment_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            ),
        )
        return cursor.lastrowid
