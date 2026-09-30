"""
Vantage — Streamlit front-end.

This file owns presentation only. Every RAG/GraphRAG/guardrail/audit/agent
call still goes through backend.py exactly as before — nothing in src/ was
touched. Features: a warm editorial theme, a ChatGPT-style SQLite chat
history (ui/db.py, ui/chat_history.py), an Auto/RAG mode picker (ui/engine.py —
in Auto the agent itself decides Chat / Code / RAG; RAG is docs-only Agentic
RAG), a friendly confidence card, and Notion-style source cards.

Layout:
  Sidebar : brand, document ingestion + registry, agent mode picker,
            chat history (search / rename / favorite / delete), audit pill.
  Main    : 💬 Chat  and  🕸️ Knowledge Graph tabs.
"""

from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

import backend
from src.core.config import ApplicationConfig
from ui import components as C
from ui import engine
from ui.chat_history import chat_to_markdown, group_chats, make_title
from ui.db import ChatDB
from ui.graph_panel import render_graph_panel
from ui.mdrender import markdown_to_html

st.set_page_config(page_title="Vantage", page_icon="🌿", layout="wide",
                    initial_sidebar_state="expanded")

_ASSETS = Path(__file__).parent / "ui" / "assets"
for _css in ("theme.css",):
    _path = _ASSETS / _css
    if _path.exists():
        st.markdown(f"<style>{_path.read_text()}</style>", unsafe_allow_html=True)

config = ApplicationConfig()


@st.cache_resource(show_spinner=False)
def _get_db() -> ChatDB:
    db_path = Path(config.storage["sqlite"]["path"]).parent / "chat_history.db"
    return ChatDB(db_path)


db = _get_db()
_settings = db.get_settings()

# ------------------------------------------------------------- session state

defaults = {
    "messages": [], "active_chat": None, "direct_rag": False,
    "mode": engine.normalize_mode(_settings.get("mode")), "theme": _settings.get("theme", "light"),
    "font_scale": _settings.get("font_scale", 1.0), "last_query": "",
    "last_route": None, "history_search": "", "pending_prompt": None,
    "expand_sources": {}, "db": db,
}
for key, val in defaults.items():
    st.session_state.setdefault(key, val)

C.theme_marker(st.session_state.theme)
if st.session_state.font_scale != 1.0:
    st.markdown(f"<style>.vt-bubble, .vt-conf-card, .vt-source-card "
                f"{{ font-size: calc(1em * {st.session_state.font_scale}); }}</style>",
                unsafe_allow_html=True)


def _friendly_init_error(exc: Exception) -> None:
    st.error(
        "**Vantage couldn't start its local models.** This almost always means "
        "either Ollama isn't running, or the embedding/reranker weights aren't "
        "available yet (locally under `models/`, or in the Hugging Face cache).",
        icon="⚠️",
    )
    with st.expander("Details"):
        st.code(str(exc))
        st.markdown(
            "- Start Ollama: `ollama serve` (and `ollama pull llama3.2:3b`)\n"
            "- Or drop model weights into the folder named by `VANTAGE_MODELS_DIR` "
            "(defaults to `./models`, or `/app/models` in Docker) — one subfolder "
            "per model, e.g. `models/bge-m3`, `models/bge-reranker-v2-m3`.\n"
            "- See `configs/models.yaml` for exact model names."
        )


def _persist_active_chat() -> None:
    if not st.session_state.messages:
        return
    if st.session_state.active_chat is None:
        title = make_title(st.session_state.messages[0]["content"])
        st.session_state.active_chat = db.create_chat(title, st.session_state.mode,
                                                       st.session_state.messages)
    else:
        db.save_messages(st.session_state.active_chat, st.session_state.messages,
                         st.session_state.mode)


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    C.sidebar_brand()

    frameworks = config.compliance["active_frameworks"]
    st.markdown("".join(f'<span class="vt-pill accent">{f.replace("_", " ")}</span>'
                        for f in frameworks), unsafe_allow_html=True)

    C.section_label("📥 Ingest Documents")
    uploaded = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True,
                                label_visibility="collapsed")
    if uploaded:
        for f in uploaded:
            file_bytes = f.getvalue()
            with st.spinner(f"Ingesting {f.name}..."):
                try:
                    result = backend.ingest_file(file_bytes, f.name)
                except Exception as exc:
                    _friendly_init_error(exc)
                    result = None
            if result is None:
                pass
            elif result.duplicate:
                st.info(f"'{f.name}' already indexed — skipped re-ingestion.")
            elif result.status == "ready":
                db.set_doc_meta(result.document_id, f.name, len(file_bytes))
                st.success(f"'{f.name}' → {result.total_chunks} chunks, "
                          f"{result.total_pages} pages.")
            else:
                st.error(f"Ingestion failed for '{f.name}'.")

    C.section_label("📚 Ingested Documents")
    docs, docs_error = [], None
    try:
        with st.spinner("Loading local models (first run can take a minute)..."):
            docs = backend.list_documents()
    except Exception as exc:
        docs_error = exc

    if docs_error is not None:
        _friendly_init_error(docs_error)
    elif not docs:
        st.caption("No documents ingested yet.")
    else:
        doc_search = st.text_input("Search documents", key="doc_search",
                                   placeholder="🔍 Search documents...",
                                   label_visibility="collapsed")
        sizes = db.doc_meta()
        shown = [d for d in docs if doc_search.lower() in d.filename.lower()] if doc_search else docs
        with st.container(height=min(360, 92 * max(len(shown), 1))):
            for doc in shown:
                size_bytes = sizes.get(doc.document_id, 0)
                size_label = f"{size_bytes / 1024:.0f} KB" if size_bytes < 1024 * 1024 else \
                    f"{size_bytes / 1024 / 1024:.1f} MB"
                uploaded_at = (time.strftime("%d %b %Y", time.localtime(doc.ingested_at))
                              if doc.ingested_at else "—")
                col1, col2 = st.columns([6, 1])
                with col1:
                    st.markdown(C.doc_card(filename=doc.filename, uploaded_at=uploaded_at,
                                           pages=doc.total_pages, size_label=size_label),
                               unsafe_allow_html=True)
                with col2:
                    if st.button("🗑️", key=f"del_{doc.document_id}", help="Delete"):
                        backend.delete_document(doc.document_id)
                        db.delete_doc_meta(doc.document_id)
                        st.rerun()

    st.markdown("---")
    C.section_label("🤖 Agent Router")
    st.session_state.mode = st.segmented_control(
        "Router", options=list(engine.MODES), default=st.session_state.mode,
        label_visibility="collapsed",
    ) or "Auto"
    db.set_setting("mode", st.session_state.mode)
    st.session_state.direct_rag = st.toggle(
        "Fast mode (skip validation)", value=st.session_state.direct_rag,
        help="Skips the confidence check and automatic regeneration for lower "
             "latency. Applies to Auto and RAG.",
    )
    st.caption("**Auto** — the agent decides whether to chat, write code, or search "
               "your documents. **RAG** — answers only from your documents "
               "(Agentic RAG); says so if they don't cover the question.")

    st.markdown("---")
    C.section_label("🔐 Audit Trail")
    if docs_error is None:
        try:
            valid, count = backend.audit_chain_status()
            C.audit_pill(valid, count)
        except Exception:
            st.caption("Audit trail unavailable.")
    else:
        st.caption("Audit trail unavailable.")

    st.markdown("---")
    C.section_label("🗂️ Chats")
    hcol1, hcol2 = st.columns([5, 1])
    with hcol1:
        st.session_state.history_search = st.text_input(
            "Search conversations", value=st.session_state.history_search,
            placeholder="🔍 Search conversations...", label_visibility="collapsed")
    with hcol2:
        if st.button("➕", help="New chat", use_container_width=True):
            st.session_state.active_chat = None
            st.session_state.messages = []
            st.rerun()

    chats = db.list_chats(st.session_state.history_search)
    if not chats:
        st.caption("No conversations yet — ask a question to start one.")
    else:
        with st.container(height=320):
            for group_name, items in group_chats(chats):
                C.time_group_caption(group_name)
                for chat in items:
                    C.chat_list_item(chat, active=(chat["chat_id"] == st.session_state.active_chat),
                                     search=st.session_state.history_search)

    with st.expander("⚙️ Settings"):
        theme_choice = st.radio("Theme", ["light", "dark"],
                                index=["light", "dark"].index(st.session_state.theme),
                                horizontal=True)
        if theme_choice != st.session_state.theme:
            st.session_state.theme = theme_choice
            db.set_setting("theme", theme_choice)
            st.rerun()
        font_choice = st.select_slider("Font size", options=["Small", "Medium", "Large"],
                                       value={0.9: "Small", 1.0: "Medium", 1.15: "Large"}
                                       .get(st.session_state.font_scale, "Medium"))
        new_scale = {"Small": 0.9, "Medium": 1.0, "Large": 1.15}[font_choice]
        if new_scale != st.session_state.font_scale:
            st.session_state.font_scale = new_scale
            db.set_setting("font_scale", new_scale)
            st.rerun()
        st.caption("Streaming is always on — answers appear as they're generated.")
        ec1, ec2 = st.columns(2)
        with ec1:
            if st.button("Clear all chats", use_container_width=True):
                db.delete_all_chats()
                st.session_state.active_chat = None
                st.session_state.messages = []
                st.rerun()
        with ec2:
            all_chats = db.export_all()
            export_md = "\n\n---\n\n".join(chat_to_markdown(c) for c in all_chats) or "_No chats yet._"
            st.download_button("Export all", export_md, file_name="vantage_chat_history.md",
                               use_container_width=True)
        st.caption("Vantage · Governed Agentic RAG — local, private, offline.")

    C.sidebar_footer()

    st.markdown("---")
    C.section_label("📊 Evaluation")
    st.caption("Run `python -m src.eval.run_eval` for a full DPDP/NIST-labeled "
              "eval → produces data/reports/vantage_eval_report.xlsx.")


# ============================================================================
# TOP NAVBAR
# ============================================================================

nav_l, nav_c, nav_r = st.columns([1, 2, 1])
with nav_c:
    st.markdown('<div style="text-align:center;font-family:\'Vantage Serif\',serif;'
               'font-size:20px;color:var(--vt-forest);padding-top:4px;">🌿 Vantage</div>',
               unsafe_allow_html=True)
with nav_r:
    r1, r2 = st.columns(2)
    with r1:
        if st.button("🌗", help="Toggle theme", use_container_width=True):
            st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
            db.set_setting("theme", st.session_state.theme)
            st.rerun()
    with r2:
        with st.popover("⚙️", use_container_width=True):
            st.caption("Open the ⚙️ Settings panel in the sidebar for theme, font size, "
                      "and history export.")

chat_tab, graph_tab = st.tabs(["💬 Chat", "🕸️ Knowledge Graph"])


# ---------------------------------------------------------------- rendering

def _render_confidence(conf: dict, key_prefix: str) -> None:
    card_html, reasons_html = C.confidence_card(conf)
    st.markdown(card_html, unsafe_allow_html=True)
    with st.expander("Why this confidence?"):
        st.markdown(reasons_html, unsafe_allow_html=True)
        if conf.get("note"):
            st.caption(conf["note"])


def _render_sources(chunks: list[dict], key_prefix: str) -> None:
    if not chunks:
        return
    with st.expander(f"📎 Sources — {len(chunks)} passage(s)",
                     expanded=st.session_state.expand_sources.get(key_prefix, False)):
        for i, c in enumerate(chunks):
            st.markdown(C.source_card(c["source"], c["page"], c["match"], c["text"][:220]),
                       unsafe_allow_html=True)
            if len(c["text"]) > 220:
                with st.expander("Expand snippet", expanded=False):
                    st.write(c["text"])


def _render_toolbar(msg: dict, idx: int, is_last: bool) -> None:
    st.markdown('<div class="vt-toolbar">', unsafe_allow_html=True)
    cols = st.columns(5)
    content = msg.get("content", "")
    meta = msg.setdefault("meta", {}) or {}

    with cols[0]:
        with st.popover("📋", use_container_width=True):
            st.caption("Copy from here (hover the block for the copy icon):")
            st.code(content, language=None)
    with cols[1]:
        components_speak(content, key=f"speak_{idx}")
    with cols[2]:
        starred = meta.get("bookmarked", False)
        if st.button("⭐" if starred else "☆", key=f"bm_{idx}", help="Bookmark"):
            meta["bookmarked"] = not starred
            msg["meta"] = meta
            _persist_active_chat()
            st.rerun()
    with cols[3]:
        if is_last and st.button("🔁", key=f"regen_{idx}", help="Regenerate"):
            st.session_state.messages = st.session_state.messages[:idx - 1]
            st.session_state.pending_prompt = _last_user_text_before(idx)
            st.rerun()
    with cols[4]:
        if meta.get("chunks") and st.button("📎", key=f"src_{idx}", help="View sources"):
            key_prefix = f"msg_{idx}"
            st.session_state.expand_sources[key_prefix] = not st.session_state.expand_sources.get(key_prefix, False)
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def _last_user_text_before(idx: int) -> str | None:
    for m in reversed(st.session_state.messages[:idx]):
        if m["role"] == "user":
            return m["content"]
    return None


def components_speak(text: str, key: str) -> None:
    """Real, working text-to-speech via the browser's SpeechSynthesis API.
    Runs inside its own iframe (components.html), which — unlike st.markdown —
    does execute <script> tags."""
    import json as _json
    import streamlit.components.v1 as _components
    safe_text = _json.dumps(text[:4000])
    _components.html(
        f"""
        <style>html,body {{ margin:0; padding:0; background:transparent; }}</style>
        <button id="{key}" style="width:100%;padding:2px 0;border:none;background:transparent;
          color:#8a9690;cursor:pointer;font-size:14px;border-radius:6px;">🔊</button>
        <script>
          document.getElementById("{key}").addEventListener("click", function() {{
            try {{
              const u = new SpeechSynthesisUtterance({safe_text});
              window.speechSynthesis.cancel();
              window.speechSynthesis.speak(u);
            }} catch (e) {{}}
          }});
        </script>
        """, height=30,
    )


def _render_message(msg: dict, idx: int, is_last_assistant: bool) -> None:
    role = msg["role"]
    body = markdown_to_html(msg.get("content", ""))
    st.markdown(C.bubble_html(role, body), unsafe_allow_html=True)

    if role != "assistant":
        return
    meta = msg.get("meta") or {}
    if is_last_assistant and meta.get("tool") in ("chat", "rag", "code"):
        st.markdown(f'<div class="vt-meta-wrap">{C.route_badge(meta["tool"])}</div>',
                   unsafe_allow_html=True)
    if meta.get("sources"):
        chips = "".join(f'<span class="vt-source-chip">📄 {s}</span>' for s in meta["sources"])
        st.markdown(f'<div class="vt-meta-wrap">{chips}</div>', unsafe_allow_html=True)
    if meta.get("confidence"):
        _render_confidence(meta["confidence"], f"msg_{idx}")
    if meta.get("chunks"):
        _render_sources(meta["chunks"], f"msg_{idx}")
    if meta.get("agent_steps"):
        with st.expander("🧭 What the agent did"):
            for step in meta["agent_steps"]:
                st.markdown(f"- {step}")
    if meta.get("guardrail_findings"):
        with st.expander("🛡️ Guardrail activity on this turn"):
            for finding in meta["guardrail_findings"]:
                st.code(finding, language=None)
    if meta.get("error"):
        with st.expander("⚠️ Why did this fail?"):
            st.caption(
                "This usually means Ollama isn't running, or a model/embedding weight "
                "hasn't finished loading. See the sidebar for setup notes."
            )
            st.code(meta["error"], language=None)
    _render_toolbar(msg, idx, is_last_assistant)


def _run_turn(query: str) -> None:
    st.session_state.last_query = query
    st.session_state.messages.append({"role": "user", "content": query})
    _render_message(st.session_state.messages[-1], len(st.session_state.messages) - 1, False)

    route_holder = st.empty()
    status_area = st.empty()
    bubble_area = st.empty()
    routed_tool: list[str | None] = [None]

    def on_route(tool: str) -> None:
        routed_tool[0] = tool
        st.session_state.last_route = tool
        route_holder.markdown(f'<div class="vt-meta-wrap">{C.route_badge(tool)}</div>',
                             unsafe_allow_html=True)

    full_text = ""
    final_result = None
    error_detail: str | None = None
    try:
        for event in engine.stream_answer(query, mode=st.session_state.mode,
                                          fast=st.session_state.direct_rag, on_route=on_route):
            if event["type"] == "status":
                status_area.markdown(
                    f'<div class="vt-status-chip"><span class="vt-status-dot"></span>'
                    f'{event["message"]}</div>', unsafe_allow_html=True)
            elif event["type"] == "reset":
                full_text = ""          # a regeneration is starting; drop the earlier draft
            elif event["type"] == "chunk":
                full_text += event["text"]
                bubble_area.markdown(C.bubble_html("assistant", markdown_to_html(full_text)),
                                     unsafe_allow_html=True)
            elif event["type"] == "result":
                final_result = event["data"]
    except Exception as exc:
        # st.rerun() below would wipe a transient st.error(), so the reason is
        # kept on the message itself (meta.error) and shown via an expander —
        # that way it survives the rerun and future reloads of this chat.
        error_detail = str(exc)
        full_text = full_text or "Sorry — I ran into a problem answering that."

    status_area.empty()
    answer_text = final_result.answer if final_result else full_text
    bubble_area.markdown(C.bubble_html("assistant", markdown_to_html(answer_text)),
                        unsafe_allow_html=True)

    meta = None
    if final_result:
        meta = engine.build_meta(final_result, st.session_state.mode,
                                 st.session_state.direct_rag, routed_tool[0])
    elif error_detail:
        meta = {"tool": "blocked", "mode": st.session_state.mode, "error": error_detail}

    st.session_state.messages.append({"role": "assistant", "content": answer_text, "meta": meta})
    _persist_active_chat()
    st.rerun()


# ============================================================================
# CHAT TAB
# ============================================================================

with chat_tab:
    if not st.session_state.messages:
        C.empty_state()
        cols = st.columns(3)
        for col, (title, sub, prompt) in zip(cols, engine.SUGGESTIONS):
            with col:
                st.markdown(f'<div class="vt-quiet-caption" style="text-align:center;'
                           f'margin-bottom:4px;">{sub}</div>', unsafe_allow_html=True)
                if st.button(title, key=f"sugg_{title}", use_container_width=True):
                    st.session_state.pending_prompt = prompt

    n_msgs = len(st.session_state.messages)
    last_assistant_idx = max((i for i, m in enumerate(st.session_state.messages)
                              if m["role"] == "assistant"), default=-1)
    for i, msg in enumerate(st.session_state.messages):
        _render_message(msg, i, is_last_assistant=(i == last_assistant_idx))

    st.markdown('<div class="vt-chip-row">', unsafe_allow_html=True)
    chip_cols = st.columns(len(engine.QUICK_ACTIONS))
    for col, (label, prompt) in zip(chip_cols, engine.QUICK_ACTIONS.items()):
        with col:
            if st.button(label, key=f"qa_{label}"):
                st.session_state.pending_prompt = prompt
    st.markdown("</div>", unsafe_allow_html=True)

    query = st.chat_input("Ask anything or search your documents...")
    if st.session_state.pending_prompt and not query:
        query = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

    if query:
        query = engine.expand_slash(query)
        _run_turn(query)

with graph_tab:
    render_graph_panel(
        graph_path=config.storage["graph"]["path"],
        highlight_query=st.session_state.last_query,
    )
