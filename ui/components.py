"""ui/components.py — small, focused render functions used by app.py.

Every function here only draws UI; none of them touch backend.py directly
(app.py wires data from ui/engine.py into these).
"""

from __future__ import annotations

import html
from datetime import datetime

import streamlit as st

from ui.chat_history import highlight, match_snippet


def theme_marker(theme: str) -> None:
    st.markdown(f'<div class="vt-marker vt-theme-{theme}"></div>', unsafe_allow_html=True)


def hero(title: str = "Vantage", tagline: str = "Ask • Search • Analyze") -> None:
    st.markdown(
        f'<div class="vt-hero">'
        f'<div class="vt-hero-mark">V</div>'
        f'<div class="vt-hero-title">{html.escape(title)}</div>'
        f'<div class="vt-hero-tag">{html.escape(tagline)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def sidebar_brand() -> None:
    st.markdown(
        '<div class="vt-brand"><div class="vt-brand-mark">V</div>'
        '<div><div class="vt-brand-title">Vantage</div>'
        '<div class="vt-brand-sub">Ask • Search • Analyze</div></div></div>',
        unsafe_allow_html=True,
    )


def section_label(text: str) -> None:
    st.markdown(f'<div class="vt-section-label">{html.escape(text)}</div>', unsafe_allow_html=True)


def doc_card(*, filename: str, uploaded_at: str, pages: int, size_label: str) -> str:
    return (
        '<div class="vt-doc-card"><div class="vt-doc-icon">📄</div><div>'
        f'<div class="vt-doc-name">{html.escape(filename)}</div>'
        f'<div class="vt-doc-meta">{uploaded_at} · {pages} pages · {size_label}</div>'
        '<span class="vt-doc-badge">✓ Indexed</span></div></div>'
    )


def sidebar_footer(text: str = "Local · private · offline") -> None:
    st.markdown(f'<div class="vt-sidebar-footer"><span>{html.escape(text)}</span></div>',
                unsafe_allow_html=True)


def audit_pill(valid: bool, count: int) -> None:
    icon = "✓" if valid else "⚠"
    text = "Hash Verified" if valid else "Chain broken"
    st.markdown(
        f'<span class="vt-audit-pill">{icon} {text} · {count} records</span>',
        unsafe_allow_html=True,
    )


def route_badge(tool: str) -> str:
    icon = {"chat": "🧠", "rag": "📄", "code": "💻", "blocked": "⛔"}.get(tool, "🧠")
    label = {"chat": "Routed to Chat", "rag": "Routed to Agentic RAG", "code": "Routed to Code",
             "blocked": "Blocked"}.get(tool, "Routed")
    return f'<span class="vt-route-badge">{icon} {label}</span>'


def empty_state() -> None:
    st.markdown(
        '<div class="vt-empty-state">'
        '<div class="vt-empty-illustration">🌾</div>'
        '<div class="vt-empty-title">What would you like to explore today?</div>'
        '<div class="vt-empty-sub">Ask a question, or try one of these:</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def bubble_html(role: str, content_html: str) -> str:
    return (f'<div class="vt-msg-row {role}"><div class="vt-bubble {role}">'
            f'{content_html}</div></div>')


def confidence_card(conf: dict) -> str:
    pct, level, label = conf["percent"], conf["level"], conf["label"]
    reasons = "".join(
        f'<div class="vt-conf-reason"><b>{html.escape(k)}:</b> {html.escape(v)}</div>'
        for k, v in conf["reasons"].items()
    )
    verified = ('<span class="vt-pill good">✓ Verified</span>' if conf["verified"]
                else '<span class="vt-pill">Fast mode</span>')
    return (
        '<div class="vt-conf-card">'
        '<div class="vt-conf-top">'
        '<span class="vt-conf-label">CONFIDENCE</span>'
        f'<span class="vt-conf-pct">{pct}%</span></div>'
        f'<div class="vt-conf-bar-track"><div class="vt-conf-bar-fill {level}" '
        f'style="width:{pct}%"></div></div>'
        f'<div class="vt-conf-tag {level}">{html.escape(label)}</div>'
        f'<div style="margin-top:6px;">{verified} '
        f'<span class="vt-pill">{conf["sections"]} sources</span></div>'
        '</div>'
    ), reasons


def source_card(source: str, page, match: int, snippet: str) -> str:
    return (
        '<div class="vt-source-card">'
        '<div class="vt-source-head">'
        f'<span class="vt-source-name">📄 {html.escape(source)}</span>'
        f'<span class="vt-source-match">{match}% Match</span></div>'
        f'<div class="vt-source-page">Page {html.escape(str(page))}</div>'
        f'<div class="vt-source-snippet">{html.escape(snippet)}</div>'
        '</div>'
    )


def chat_list_item(chat: dict, active: bool, search: str = "") -> None:
    """One row in the sidebar history list: open / rename / favorite / delete."""
    title = chat["title"] or "New chat"
    c_open, c_star, c_menu = st.columns([6, 1, 1])
    with c_open:
        label = ("★ " if chat.get("favorite") else "") + title
        if st.button(label, key=f"open_{chat['chat_id']}", use_container_width=True,
                     type="primary" if active else "secondary"):
            st.session_state.active_chat = chat["chat_id"]
            st.session_state.messages = chat.get("messages") or []
            st.rerun()
        if search:
            snip = match_snippet(chat.get("messages", []), search)
            if snip:
                st.markdown(f'<div class="vt-quiet-caption">{snip}</div>', unsafe_allow_html=True)
    with c_star:
        if st.button("☆" if not chat.get("favorite") else "★", key=f"fav_{chat['chat_id']}",
                     help="Favorite"):
            st.session_state.db.set_favorite(chat["chat_id"], not chat.get("favorite"))
            st.rerun()
    with c_menu:
        with st.popover("⋯", use_container_width=True):
            new_title = st.text_input("Rename", value=title, key=f"rn_{chat['chat_id']}",
                                      label_visibility="collapsed")
            r1, r2 = st.columns(2)
            with r1:
                if st.button("Save", key=f"rns_{chat['chat_id']}", use_container_width=True):
                    st.session_state.db.rename_chat(chat["chat_id"], new_title)
                    st.rerun()
            with r2:
                if st.button("Delete", key=f"del_{chat['chat_id']}", use_container_width=True):
                    st.session_state.db.delete_chat(chat["chat_id"])
                    if st.session_state.active_chat == chat["chat_id"]:
                        st.session_state.active_chat = None
                        st.session_state.messages = []
                    st.rerun()


def time_group_caption(name: str) -> None:
    st.markdown(f'<div class="vt-section-label" style="margin-top:10px;">{name}</div>',
                unsafe_allow_html=True)
