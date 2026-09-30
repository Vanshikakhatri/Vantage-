"""
ui/db.py — SQLite persistence for the Vantage front-end (chat_history.db).

Frontend-only storage. It never touches the RAG document registry
(data/vantage.db), the vector indices, the graph or the audit log.

Tables
------
chats      chat_id, title, timestamp (created), agent_type, conversation_json,
           favorite, last_updated
settings   key/value pairs (theme, font size, streaming, fast mode ...)
doc_meta   per-document upload size in bytes (the backend Document record
           has pages/chunks/date but not the file size)
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

_lock = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    chat_id            TEXT PRIMARY KEY,
    title              TEXT NOT NULL,
    timestamp          REAL NOT NULL,
    agent_type         TEXT DEFAULT 'auto',
    conversation_json  TEXT NOT NULL DEFAULT '[]',
    favorite           INTEGER NOT NULL DEFAULT 0,
    last_updated       REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chats_updated ON chats(last_updated DESC);
CREATE INDEX IF NOT EXISTS idx_chats_fav ON chats(favorite);

CREATE TABLE IF NOT EXISTS settings (
    key    TEXT PRIMARY KEY,
    value  TEXT
);

CREATE TABLE IF NOT EXISTS doc_meta (
    document_id  TEXT PRIMARY KEY,
    filename     TEXT,
    size_bytes   INTEGER DEFAULT 0
);
"""


def _like(term: str) -> str:
    """Escape LIKE wildcards so user input is matched literally."""
    return "%" + term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


class ChatDB:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------ chats

    def create_chat(self, title: str, agent_type: str = "auto",
                    messages: list[dict] | None = None) -> str:
        chat_id = uuid.uuid4().hex
        now = time.time()
        with _lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO chats (chat_id, title, timestamp, agent_type, conversation_json, "
                "favorite, last_updated) VALUES (?, ?, ?, ?, ?, 0, ?)",
                (chat_id, title, now, agent_type,
                 json.dumps(messages or [], ensure_ascii=False), now),
            )
        return chat_id

    def save_messages(self, chat_id: str, messages: list[dict],
                      agent_type: str | None = None) -> None:
        now = time.time()
        with _lock, self._conn() as conn:
            if agent_type:
                conn.execute(
                    "UPDATE chats SET conversation_json=?, agent_type=?, last_updated=? "
                    "WHERE chat_id=?",
                    (json.dumps(messages, ensure_ascii=False), agent_type, now, chat_id))
            else:
                conn.execute(
                    "UPDATE chats SET conversation_json=?, last_updated=? WHERE chat_id=?",
                    (json.dumps(messages, ensure_ascii=False), now, chat_id))

    def get_chat(self, chat_id: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM chats WHERE chat_id=?", (chat_id,)).fetchone()
        return self._row(row, with_messages=True) if row else None

    def list_chats(self, search: str = "") -> list[dict[str, Any]]:
        """Pinned (favorite) chats first, then most recently updated. With a
        search term: SQL LIKE over the title and the stored conversation."""
        search = search.strip()
        with self._conn() as conn:
            if search:
                pat = _like(search)
                rows = conn.execute(
                    "SELECT * FROM chats WHERE title LIKE ? ESCAPE '\\' "
                    "OR conversation_json LIKE ? ESCAPE '\\' "
                    "ORDER BY favorite DESC, last_updated DESC", (pat, pat)).fetchall()
            else:
                rows = conn.execute(
                    "SELECT chat_id, title, timestamp, agent_type, favorite, last_updated, "
                    "'[]' AS conversation_json FROM chats "
                    "ORDER BY favorite DESC, last_updated DESC").fetchall()
        return [self._row(r, with_messages=bool(search)) for r in rows]

    def rename_chat(self, chat_id: str, title: str) -> None:
        title = title.strip()
        if not title:
            return
        with _lock, self._conn() as conn:
            conn.execute("UPDATE chats SET title=? WHERE chat_id=?", (title[:120], chat_id))

    def set_favorite(self, chat_id: str, value: bool) -> None:
        with _lock, self._conn() as conn:
            conn.execute("UPDATE chats SET favorite=? WHERE chat_id=?", (1 if value else 0, chat_id))

    def delete_chat(self, chat_id: str) -> None:
        with _lock, self._conn() as conn:
            conn.execute("DELETE FROM chats WHERE chat_id=?", (chat_id,))

    def delete_all_chats(self) -> None:
        with _lock, self._conn() as conn:
            conn.execute("DELETE FROM chats")

    def export_all(self) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM chats ORDER BY last_updated DESC").fetchall()
        return [self._row(r, with_messages=True) for r in rows]

    def bookmarks(self) -> list[dict[str, Any]]:
        """Every bookmarked assistant answer across all chats (newest first)."""
        out: list[dict[str, Any]] = []
        for chat in self.export_all():
            for idx, msg in enumerate(chat["messages"]):
                if msg.get("role") == "assistant" and msg.get("bookmarked"):
                    out.append({"chat_id": chat["chat_id"], "chat_title": chat["title"],
                                "index": idx, "content": msg.get("content", ""),
                                "last_updated": chat["last_updated"]})
        return out

    @staticmethod
    def _row(row: sqlite3.Row, with_messages: bool) -> dict[str, Any]:
        data = {
            "chat_id": row["chat_id"], "title": row["title"], "timestamp": row["timestamp"],
            "agent_type": row["agent_type"], "favorite": bool(row["favorite"]),
            "last_updated": row["last_updated"],
        }
        if with_messages:
            try:
                data["messages"] = json.loads(row["conversation_json"])
            except (TypeError, ValueError):
                data["messages"] = []
        return data

    # --------------------------------------------------------------- settings

    def get_settings(self) -> dict[str, Any]:
        with self._conn() as conn:
            rows = conn.execute("SELECT key, value FROM settings").fetchall()
        out: dict[str, Any] = {}
        for r in rows:
            try:
                out[r["key"]] = json.loads(r["value"])
            except (TypeError, ValueError):
                out[r["key"]] = r["value"]
        return out

    def set_setting(self, key: str, value: Any) -> None:
        with _lock, self._conn() as conn:
            conn.execute("INSERT INTO settings (key, value) VALUES (?, ?) "
                         "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                         (key, json.dumps(value)))

    # -------------------------------------------------------------- doc meta

    def set_doc_meta(self, document_id: str, filename: str, size_bytes: int) -> None:
        if not document_id:
            return
        with _lock, self._conn() as conn:
            conn.execute("INSERT INTO doc_meta (document_id, filename, size_bytes) VALUES (?,?,?) "
                         "ON CONFLICT(document_id) DO UPDATE SET size_bytes=excluded.size_bytes, "
                         "filename=excluded.filename", (document_id, filename, size_bytes))

    def doc_meta(self) -> dict[str, int]:
        with self._conn() as conn:
            rows = conn.execute("SELECT document_id, size_bytes FROM doc_meta").fetchall()
        return {r["document_id"]: int(r["size_bytes"] or 0) for r in rows}

    def delete_doc_meta(self, document_id: str) -> None:
        with _lock, self._conn() as conn:
            conn.execute("DELETE FROM doc_meta WHERE document_id=?", (document_id,))
