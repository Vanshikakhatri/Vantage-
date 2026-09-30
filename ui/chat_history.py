"""
ui/chat_history.py — ChatGPT-style conversation history on top of ui/db.py.

Pure helpers (no Streamlit imports) so they are easy to test:
  make_title       first user message -> short title
  group_chats      Starred / Today / Yesterday / Older buckets
  highlight        HTML-escaped text with <mark> around search matches
  match_snippet    short context window around the first match in a chat
  chat_to_markdown export of a single conversation
"""

from __future__ import annotations

import html
import re
from datetime import datetime, timedelta
from typing import Any


def make_title(text: str, limit: int = 48) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    text = re.sub(r"^/\w+\s*", "", text) or text          # drop a leading slash-command
    if not text:
        return "New chat"
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0] or text[:limit]
    return cut.rstrip(",.;:") + "…"


def group_chats(chats: list[dict[str, Any]], now: datetime | None = None) -> list[tuple[str, list[dict]]]:
    """Starred first, then Today / Yesterday / Older, each ordered by last update."""
    now = now or datetime.now()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday = today - timedelta(days=1)
    buckets: dict[str, list[dict]] = {"Starred": [], "Today": [], "Yesterday": [], "Older": []}
    for chat in chats:
        if chat.get("favorite"):
            buckets["Starred"].append(chat)
            continue
        when = datetime.fromtimestamp(chat["last_updated"])
        if when >= today:
            buckets["Today"].append(chat)
        elif when >= yesterday:
            buckets["Yesterday"].append(chat)
        else:
            buckets["Older"].append(chat)
    return [(name, items) for name, items in buckets.items() if items]


def highlight(text: str, term: str) -> str:
    """Escape `text` and wrap case-insensitive matches of `term` in <mark>."""
    safe = html.escape(text or "")
    term = (term or "").strip()
    if not term:
        return safe
    pattern = re.compile(re.escape(html.escape(term)), re.IGNORECASE)
    return pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", safe)


def match_snippet(messages: list[dict], term: str, radius: int = 42) -> str:
    """HTML snippet with the first occurrence of `term` inside the conversation."""
    term = (term or "").strip().lower()
    if not term:
        return ""
    for msg in messages:
        content = msg.get("content", "") or ""
        pos = content.lower().find(term)
        if pos == -1:
            continue
        start = max(0, pos - radius)
        end = min(len(content), pos + len(term) + radius)
        chunk = re.sub(r"\s+", " ", content[start:end])
        prefix = "…" if start > 0 else ""
        suffix = "…" if end < len(content) else ""
        return prefix + highlight(chunk, term) + suffix
    return ""


def chat_to_markdown(chat: dict[str, Any]) -> str:
    lines = [f"# {chat.get('title', 'Conversation')}", ""]
    stamp = datetime.fromtimestamp(chat.get("last_updated", 0)).strftime("%d %b %Y, %H:%M")
    lines += [f"_Exported from Vantage · last updated {stamp}_", ""]
    for msg in chat.get("messages", []):
        who = "You" if msg.get("role") == "user" else "Vantage"
        lines += [f"### {who}", "", msg.get("content", ""), ""]
        meta = msg.get("meta") or {}
        chunks = meta.get("chunks") or []
        if msg.get("role") == "assistant" and chunks:
            lines.append("**Sources**")
            for c in chunks:
                lines.append(f"- {c.get('source', '?')} · page {c.get('page', '?')} "
                             f"· {c.get('match', 0)}% match")
            lines.append("")
    return "\n".join(lines)
