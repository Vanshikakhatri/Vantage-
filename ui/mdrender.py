"""
ui/mdrender.py — compact Markdown -> HTML renderer for chat bubbles.

Streaming answers get re-rendered on every chunk, and a bubble is one HTML
string handed to a single st.markdown() call (Streamlit renders each
st.markdown as its own DOM fragment, so splitting an open/close <div> across
calls would not nest correctly) — this needs to be self-contained and fast,
not a full CommonMark implementation. Supports what LLM answers actually use:
headings, bold/italic, inline code, fenced code blocks (+ language badge),
bullet/numbered lists, blockquotes, pipe tables, links and images.
"""

from __future__ import annotations

import html
import re

_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_BOLD = re.compile(r"\*\*([^*\n]+)\*\*|__([^_\n]+)__")
_ITALIC = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)|(?<!_)_([^_\n]+)_(?!_)")
_IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_UL = re.compile(r"^\s*[-*+]\s+(.*)$")
_OL = re.compile(r"^\s*\d+[.)]\s+(.*)$")
_QUOTE = re.compile(r"^\s*>\s?(.*)$")
_FENCE = re.compile(r"^```(\w*)\s*$")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")


def _inline(text: str) -> str:
    text = html.escape(text)
    text = _IMAGE.sub(lambda m: f'<img alt="{m.group(1)}" src="{m.group(2)}" '
                                f'style="max-width:100%;border-radius:12px;" />', text)
    text = _LINK.sub(lambda m: f'<a href="{m.group(2)}" target="_blank" rel="noopener">{m.group(1)}</a>',
                     text)
    text = _INLINE_CODE.sub(lambda m: f"<code>{m.group(1)}</code>", text)
    text = _BOLD.sub(lambda m: f"<strong>{m.group(1) or m.group(2)}</strong>", text)
    text = _ITALIC.sub(lambda m: f"<em>{m.group(1) or m.group(2)}</em>", text)
    return text


def _table(lines: list[str]) -> str:
    def cells(row: str) -> list[str]:
        row = row.strip()
        if row.startswith("|"):
            row = row[1:]
        if row.endswith("|"):
            row = row[:-1]
        return [c.strip() for c in row.split("|")]

    head = cells(lines[0])
    body = [cells(r) for r in lines[2:]]
    out = ['<div style="overflow-x:auto;"><table>', "<thead><tr>"]
    out += [f"<th>{_inline(c)}</th>" for c in head]
    out.append("</tr></thead><tbody>")
    for row in body:
        out.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in row) + "</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def markdown_to_html(text: str) -> str:
    if not text:
        return ""
    lines = text.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    i, n = 0, len(lines)
    list_stack: str | None = None  # "ul" | "ol" | None

    def close_list():
        nonlocal list_stack
        if list_stack:
            out.append(f"</{list_stack}>")
            list_stack = None

    while i < n:
        line = lines[i]

        fence = _FENCE.match(line)
        if fence:
            lang = fence.group(1) or "text"
            close_list()
            body = []
            i += 1
            while i < n and not _FENCE.match(lines[i]):
                body.append(lines[i])
                i += 1
            i += 1  # skip closing fence
            code = html.escape("\n".join(body))
            out.append(f'<div style="font-size:10.5px;color:var(--vt-text-dim);'
                       f'margin-bottom:-6px;text-transform:uppercase;'
                       f'letter-spacing:.5px;">{html.escape(lang)}</div>'
                       f"<pre><code>{code}</code></pre>")
            continue

        if (line.strip().startswith("|") and i + 1 < n and _TABLE_SEP.match(lines[i + 1])):
            close_list()
            j = i + 2
            block = [line, lines[i + 1]]
            while j < n and lines[j].strip().startswith("|"):
                block.append(lines[j])
                j += 1
            out.append(_table(block))
            i = j
            continue

        heading = _HEADING.match(line)
        if heading:
            close_list()
            level = min(len(heading.group(1)) + 2, 6)  # keep headings modest inside a bubble
            out.append(f"<h{level}>{_inline(heading.group(2))}</h{level}>")
            i += 1
            continue

        quote = _QUOTE.match(line)
        if quote and line.strip().startswith(">"):
            close_list()
            out.append(f'<blockquote style="border-left:3px solid var(--vt-border);'
                       f'margin:6px 0;padding:2px 12px;color:var(--vt-text-dim);">'
                       f"{_inline(quote.group(1))}</blockquote>")
            i += 1
            continue

        ul = _UL.match(line)
        ol = _OL.match(line)
        if ul or ol:
            kind = "ul" if ul else "ol"
            if list_stack != kind:
                close_list()
                out.append(f"<{kind}>")
                list_stack = kind
            out.append(f"<li>{_inline((ul or ol).group(1))}</li>")
            i += 1
            continue

        close_list()
        if not line.strip():
            i += 1
            continue
        out.append(f"<p>{_inline(line)}</p>")
        i += 1

    close_list()
    return "".join(out)
