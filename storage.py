"""Transactional SQLite storage: the original decision is never overwritten."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


class AppealError(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class Store:
    def __init__(self, path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS content (
                    content_id TEXT PRIMARY KEY,
                    creator_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    status TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS appeals (
                    appeal_id TEXT PRIMARY KEY,
                    content_id TEXT NOT NULL UNIQUE REFERENCES content(content_id),
                    creator_id TEXT NOT NULL,
                    creator_reasoning TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event TEXT NOT NULL
                );
            """)

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def append_audit(connection, event):
        connection.execute("INSERT INTO audit (event) VALUES (?)", (json.dumps(event),))

    def save_decision(self, text, creator_id, result):
        decision = {
            "content_id": str(uuid4()),
            "creator_id": creator_id,
            "timestamp": timestamp(),
            **result,
            "status": "classified",
            "appeal_filed": False,
        }
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO content VALUES (?, ?, ?, ?, ?)",
                (decision["content_id"], creator_id, text, json.dumps(decision), "classified"),
            )
            self.append_audit(connection, {"event_type": "decision", **decision})
        return decision

    @staticmethod
    def current_content(connection, content_id, include_text=False):
        row = connection.execute(
            "SELECT * FROM content WHERE content_id = ?", (content_id,)
        ).fetchone()
        if row is None:
            return None
        decision = json.loads(row["decision"])
        decision["status"] = row["status"]
        appeal = connection.execute(
            "SELECT * FROM appeals WHERE content_id = ?", (content_id,)
        ).fetchone()
        decision["appeal_filed"] = appeal is not None
        if appeal:
            decision["appeal"] = dict(appeal)
        if include_text:
            decision["text"] = row["text"]
        return decision

    def get_content(self, content_id):
        with self.connect() as connection:
            return self.current_content(connection, content_id)

    def appeal(self, content_id, creator_id, reasoning):
        with self.connect() as connection:
            # Serialize appeals before reading: simultaneous retries cannot both win.
            connection.execute("BEGIN IMMEDIATE")
            original = self.current_content(connection, content_id)
            if original is None:
                raise AppealError("Content not found.", 404)
            if original["creator_id"] != creator_id:
                raise AppealError("Only the original creator may appeal this content.", 403)
            if original["appeal_filed"]:
                raise AppealError("An appeal already exists for this content.", 409)
            appeal = {
                "appeal_id": str(uuid4()),
                "content_id": content_id,
                "creator_id": creator_id,
                "creator_reasoning": reasoning,
                "timestamp": timestamp(),
            }
            connection.execute(
                "INSERT INTO appeals VALUES (?, ?, ?, ?, ?)", tuple(appeal.values())
            )
            connection.execute(
                "UPDATE content SET status = 'under_review' WHERE content_id = ?",
                (content_id,),
            )
            event = {
                **original,
                "event_type": "appeal",
                "timestamp": appeal["timestamp"],
                "decision_timestamp": original["timestamp"],
                "appeal_id": appeal["appeal_id"],
                "appeal_reasoning": reasoning,
                "status": "under_review",
                "appeal_filed": True,
            }
            self.append_audit(connection, event)
            return self.current_content(connection, content_id)

    def log(self, limit):
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT sequence, event FROM audit ORDER BY sequence DESC LIMIT ?", (limit,)
            ).fetchall()
            return [{"sequence": row["sequence"], **json.loads(row["event"])} for row in rows]

    def review_queue(self):
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT content_id FROM appeals ORDER BY timestamp ASC LIMIT 100"
            ).fetchall()
            return [self.current_content(connection, row["content_id"], True) for row in rows]
