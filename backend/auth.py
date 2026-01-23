import sqlite3
import bcrypt
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "users.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def create_users_table():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash BLOB NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def register_user(username: str, password: str) -> bool:
    password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt())

    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, password_hash)
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False


def authenticate_user(username: str, password: str) -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT password_hash FROM users WHERE username = ?",
        (username,)
    )
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return False

    return bcrypt.checkpw(password.encode(), row[0])
