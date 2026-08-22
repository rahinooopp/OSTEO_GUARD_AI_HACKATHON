"""Local patient record store.

A single SQLite file beside the application. Nothing is uploaded anywhere: the
database is an ordinary file on this machine, and it is git-ignored so saved
records cannot be committed by accident.

The store holds clinical text that a clinician chose to save, so treat the file
as containing patient data -- back it up and delete it with that in mind.
"""

import json
import os
import sqlite3
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "osteoguard_records.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS assessments (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT    NOT NULL,
    patient_ref   TEXT    NOT NULL,
    age           INTEGER,
    gender        TEXT,
    bmi           REAL,
    joint         TEXT,
    duration      INTEGER,
    risk_level    TEXT,
    summary       TEXT,
    red_flags     TEXT,
    sources       TEXT,
    notes         TEXT
);
"""


def _connect():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    return connection


def init_store():
    with _connect() as connection:
        connection.executescript(SCHEMA)


def save_assessment(patient_ref, age=None, gender=None, bmi=None, joint=None,
                    duration=None, risk_level=None, summary=None,
                    red_flags=None, sources=None, notes=None):
    """Persist one assessment. Returns its new id."""
    init_store()
    with _connect() as connection:
        cursor = connection.execute(
            """INSERT INTO assessments
               (created_at, patient_ref, age, gender, bmi, joint, duration,
                risk_level, summary, red_flags, sources, notes)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (datetime.now().isoformat(timespec="seconds"),
             patient_ref.strip() or "Unnamed",
             age, gender, bmi, joint, duration, risk_level, summary,
             json.dumps(red_flags or []),
             # Store the citation, not the full passage text -- the guideline
             # is already on disk and the row stays small.
             json.dumps([{k: s.get(k) for k in
                          ("doc_name", "page", "section", "confidence", "url")}
                         for s in (sources or [])]),
             notes),
        )
        return cursor.lastrowid


def list_assessments(limit=200):
    """Saved assessments, newest first, without the bulky text fields."""
    init_store()
    with _connect() as connection:
        rows = connection.execute(
            """SELECT id, created_at, patient_ref, age, gender, joint,
                      risk_level, red_flags
               FROM assessments ORDER BY id DESC LIMIT ?""", (limit,)
        ).fetchall()
    return [dict(row) for row in rows]


def get_assessment(record_id):
    init_store()
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM assessments WHERE id = ?", (record_id,)
        ).fetchone()
    if row is None:
        return None
    record = dict(row)
    for field in ("red_flags", "sources"):
        try:
            record[field] = json.loads(record[field] or "[]")
        except (TypeError, ValueError):
            record[field] = []
    return record


def delete_assessment(record_id):
    init_store()
    with _connect() as connection:
        connection.execute("DELETE FROM assessments WHERE id = ?", (record_id,))


def store_location():
    """Where the database lives, and how big it is, for the UI to show."""
    if not os.path.exists(DB_PATH):
        return DB_PATH, 0
    return DB_PATH, os.path.getsize(DB_PATH)
