"""Markdown -> Notion block conversion.

Scope is the subset of Markdown this repository's docs actually use: ATX
headings, paragraphs, nested bulleted and numbered lists, task list items,
fenced code, blockquotes, horizontal rules, pipe tables, and the inline set
(bold, italic, code spans, links).

The API constraints are load-bearing, not incidental, and are the reason this
lives in a tested module instead of inline in the sync script:

* a single ``rich_text`` object holds at most 2000 characters, so long lines
  must be split across several objects inside one block — ``.omp/AGENTS.md``
  has individual bullets well past 2000 characters
* a ``rich_text`` array holds at most 100 objects
* one request appends at most 100 children

Anything unrecognized degrades to a paragraph rather than being dropped, so
mirroring never silently loses text.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

MAX_RICH_TEXT_CHARS = 2000
MAX_RICH_TEXT_ITEMS = 100
MAX_BLOCK_CHILDREN = 100

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
BULLET_RE = re.compile(r"^(\s*)[-*+]\s+(.*)$")
NUMBERED_RE = re.compile(r"^(\s*)\d+[.)]\s+(.*)$")
TASK_RE = re.compile(r"^(\s*)[-*+]\s+\[([ xX])\]\s+(.*)$")
FENCE_RE = re.compile(r"^\s*```\s*([A-Za-z0-9+#._-]*)\s*$")
QUOTE_RE = re.compile(r"^\s*>\s?(.*)$")
RULE_RE = re.compile(r"^\s*([-*_])\1{2,}\s*$")
TABLE_ROW_RE = re.compile(r"^\s*\|(.+)\|\s*$")
TABLE_SEP_RE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")

# Notion's code block language enum. Anything outside it is rejected by the
# API, so unknown fences fall back to plain text.
NOTION_LANGUAGES = {
    "bash", "c", "c#", "c++", "css", "diff", "docker", "go", "graphql", "html",
    "java", "javascript", "json", "kotlin", "latex", "less", "lua", "makefile",
    "markdown", "matlab", "mermaid", "nix", "objective-c", "perl", "php",
    "plain text", "powershell", "python", "r", "ruby", "rust", "sass", "scala",
    "scss", "shell", "sql", "swift", "toml", "typescript", "vb.net", "xml",
    "yaml",
}
LANGUAGE_ALIASES = {
    "sh": "shell",
    "zsh": "shell",
    "console": "shell",
    "py": "python",
    "ts": "typescript",
    "js": "javascript",
    "yml": "yaml",
    "text": "plain text",
    "txt": "plain text",
    "": "plain text",
}

# Notion only accepts absolute link targets, and relative repository paths and
# in-page anchors are meaningless in the mirror anyway, because the section
# pages do not share the files' boundaries. Anything else renders as plain text.
LINK_SCHEMES = ("http://", "https://", "mailto:")

INLINE_RE = re.compile(
    r"(?P<code>`[^`]+`)"
    r"|(?P<link>\[[^\]]*\]\([^)]+\))"
    r"|(?P<bold>\*\*[^*]+\*\*|__[^_]+__)"
    r"|(?P<italic>\*[^*\s][^*]*\*|(?<![A-Za-z0-9_])_[^_\s][^_]*_(?![A-Za-z0-9_]))"
)


def _split_text(text: str, limit: int = MAX_RICH_TEXT_CHARS) -> list[str]:
    """Split into chunks within Notion's per-object limit, losslessly.

    Concatenating the result MUST reproduce the input exactly, so the split
    keeps the boundary whitespace on the following chunk instead of stripping
    it. Stripping it silently glues the words on either side of every 2000
    character boundary together.
    """

    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        cut = remaining.rfind(" ", 1, limit)
        if cut <= 0:
            cut = limit
        chunks.append(remaining[:cut])
        remaining = remaining[cut:]
    if remaining:
        chunks.append(remaining)
    return chunks


def _text_object(content: str, annotations: dict[str, bool] | None = None,
                 link: str | None = None) -> dict[str, Any]:
    obj: dict[str, Any] = {
        "type": "text",
        "text": {"content": content},
    }
    if link:
        obj["text"]["link"] = {"url": link}
    if annotations:
        obj["annotations"] = annotations
    return obj


def rich_text(markdown: str) -> list[dict[str, Any]]:
    """Inline Markdown -> Notion rich_text array, respecting API limits."""

    items: list[dict[str, Any]] = []

    def emit(content: str, annotations: dict[str, bool] | None = None,
             link: str | None = None) -> None:
        if not content:
            return
        for chunk in _split_text(content):
            items.append(_text_object(chunk, annotations, link))

    position = 0
    for match in INLINE_RE.finditer(markdown):
        emit(markdown[position : match.start()])
        if match.group("code"):
            emit(match.group("code")[1:-1], {"code": True})
        elif match.group("link"):
            label, _, url = match.group("link")[1:-1].partition("](")
            emit(label or url, None, url if url.startswith(LINK_SCHEMES) else None)
        elif match.group("bold"):
            emit(match.group("bold")[2:-2], {"bold": True})
        elif match.group("italic"):
            emit(match.group("italic")[1:-1], {"italic": True})
        position = match.end()
    emit(markdown[position:])

    if not items:
        items = [_text_object("")]
    # No truncation here: an over-long line is split into several blocks by
    # `_split_oversized` instead, which keeps every character.
    return items


def plain_rich_text(text: str) -> list[dict[str, Any]]:
    """Literal text -> rich_text, with no inline Markdown parsing.

    Code bodies must never be inline-parsed: `[pt_idx](hidden[mask])` is valid
    Python but matches the Markdown link pattern, and Notion rejects the
    resulting URL with a 400 that fails the whole block append.
    """

    items = [_text_object(chunk) for chunk in _split_text(text)]
    return items or [_text_object("")]


def _block(kind: str, markdown: str, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"rich_text": rich_text(markdown)}
    payload.update(extra)
    return {"object": "block", "type": kind, kind: payload}


def _literal_block(kind: str, text: str, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"rich_text": plain_rich_text(text)}
    payload.update(extra)
    return {"object": "block", "type": kind, kind: payload}


@dataclass
class _ListNode:
    indent: int
    block: dict[str, Any]
    children: list[_ListNode] = field(default_factory=list)


def _attach_children(node: _ListNode) -> dict[str, Any]:
    block = node.block
    if node.children:
        kind = block["type"]
        block[kind]["children"] = [_attach_children(child) for child in node.children]
    return block


def _flush_list(roots: list[_ListNode]) -> list[dict[str, Any]]:
    return [_attach_children(node) for node in roots]


def _table_block(rows: list[list[str]]) -> dict[str, Any]:
    width = max(len(row) for row in rows)
    return {
        "object": "block",
        "type": "table",
        "table": {
            "table_width": width,
            "has_column_header": True,
            "has_row_header": False,
            "children": [
                {
                    "object": "block",
                    "type": "table_row",
                    "table_row": {
                        "cells": [
                            rich_text(cell) for cell in (row + [""] * (width - len(row)))
                        ]
                    },
                }
                for row in rows
            ],
        },
    }


def _split_table_row(line: str) -> list[str]:
    inner = TABLE_ROW_RE.match(line)
    body = inner.group(1) if inner else line.strip().strip("|")
    return [cell.strip() for cell in body.split("|")]


def markdown_to_blocks(markdown: str) -> list[dict[str, Any]]:
    """Convert a Markdown document into a list of Notion blocks."""

    blocks: list[dict[str, Any]] = []
    list_roots: list[_ListNode] = []
    list_stack: list[_ListNode] = []
    paragraph: list[str] = []
    quote: list[str] = []
    table: list[list[str]] = []

    def close_list() -> None:
        nonlocal list_roots, list_stack
        if list_roots:
            blocks.extend(_flush_list(list_roots))
        list_roots, list_stack = [], []

    def close_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            blocks.append(_block("paragraph", " ".join(paragraph)))
        paragraph = []

    def close_quote() -> None:
        nonlocal quote
        if quote:
            blocks.append(_block("quote", " ".join(quote)))
        quote = []

    def close_table() -> None:
        nonlocal table
        if table:
            blocks.append(_table_block(table))
        table = []

    def close_all() -> None:
        close_paragraph()
        close_list()
        close_quote()
        close_table()

    lines = markdown.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]

        fence = FENCE_RE.match(line)
        if fence:
            close_all()
            language = fence.group(1).lower()
            language = LANGUAGE_ALIASES.get(language, language)
            if language not in NOTION_LANGUAGES:
                language = "plain text"
            body: list[str] = []
            index += 1
            while index < len(lines) and not FENCE_RE.match(lines[index]):
                body.append(lines[index])
                index += 1
            blocks.append(_literal_block("code", "\n".join(body), language=language))
            index += 1
            continue

        if not line.strip():
            close_all()
            index += 1
            continue

        if RULE_RE.match(line):
            close_all()
            blocks.append({"object": "block", "type": "divider", "divider": {}})
            index += 1
            continue

        heading = HEADING_RE.match(line)
        if heading:
            close_all()
            level = min(len(heading.group(1)), 3)
            blocks.append(_block(f"heading_{level}", heading.group(2).strip()))
            index += 1
            continue

        if TABLE_ROW_RE.match(line):
            close_paragraph()
            close_list()
            close_quote()
            if not TABLE_SEP_RE.match(line):
                table.append(_split_table_row(line))
            index += 1
            continue
        close_table()

        quoted = QUOTE_RE.match(line)
        if quoted:
            close_paragraph()
            close_list()
            quote.append(quoted.group(1).strip())
            index += 1
            continue
        close_quote()

        task = TASK_RE.match(line)
        bullet = None if task else BULLET_RE.match(line)
        numbered = NUMBERED_RE.match(line)
        if task or bullet or numbered:
            close_paragraph()
            if task:
                indent = len(task.group(1))
                node = _ListNode(
                    indent,
                    _block(
                        "to_do",
                        task.group(3).strip(),
                        checked=task.group(2).lower() == "x",
                    ),
                )
            elif bullet:
                indent = len(bullet.group(1))
                node = _ListNode(indent, _block("bulleted_list_item", bullet.group(2).strip()))
            else:
                assert numbered is not None
                indent = len(numbered.group(1))
                node = _ListNode(indent, _block("numbered_list_item", numbered.group(2).strip()))

            while list_stack and list_stack[-1].indent >= indent:
                list_stack.pop()
            if list_stack:
                list_stack[-1].children.append(node)
            else:
                list_roots.append(node)
            list_stack.append(node)
            index += 1
            continue

        # A plain line directly under a list item is a continuation of it.
        if list_stack and line.startswith((" ", "\t")):
            node = list_stack[-1]
            kind = node.block["type"]
            existing = "".join(
                obj["text"]["content"] for obj in node.block[kind]["rich_text"]
            )
            node.block[kind]["rich_text"] = rich_text(f"{existing} {line.strip()}")
            index += 1
            continue

        close_list()
        paragraph.append(line.strip())
        index += 1

    close_all()
    return _split_oversized(blocks)


def _split_oversized(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split any block whose rich_text exceeds the item limit into siblings.

    A rich_text array holds at most 100 objects. Rather than truncate a
    pathologically long line, emit consecutive blocks of the same type so the
    text survives intact. Tables and dividers have no rich_text of their own.
    """

    result: list[dict[str, Any]] = []
    for block in blocks:
        kind = block["type"]
        payload = block.get(kind, {})
        if kind == "table":
            result.append(block)
            continue
        children = payload.get("children")
        if children:
            payload["children"] = _split_oversized(children)
        items = payload.get("rich_text")
        if not items or len(items) <= MAX_RICH_TEXT_ITEMS:
            result.append(block)
            continue
        for offset in range(0, len(items), MAX_RICH_TEXT_ITEMS):
            part = dict(payload)
            part["rich_text"] = items[offset : offset + MAX_RICH_TEXT_ITEMS]
            if offset > 0:
                part.pop("children", None)
            result.append({"object": "block", "type": kind, kind: part})
    return result


@dataclass(frozen=True)
class Task:
    text: str
    done: bool
    source: str
    section: str
    line: int


def extract_tasks(markdown: str, source: str) -> list[Task]:
    """Pull task-list items out of a document with their heading context."""

    tasks: list[Task] = []
    section = ""
    in_fence = False
    for number, line in enumerate(markdown.splitlines(), start=1):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        heading = HEADING_RE.match(line)
        if heading:
            section = heading.group(2).strip()
            continue
        task = TASK_RE.match(line)
        if task:
            tasks.append(
                Task(
                    text=task.group(3).strip(),
                    done=task.group(2).lower() == "x",
                    source=source,
                    section=section,
                    line=number,
                )
            )
    return tasks


def chunked(blocks: list[dict[str, Any]], size: int = MAX_BLOCK_CHILDREN) -> Iterator[list[dict]]:
    """Yield block batches within the per-request child limit."""

    for start in range(0, len(blocks), size):
        yield blocks[start : start + size]
