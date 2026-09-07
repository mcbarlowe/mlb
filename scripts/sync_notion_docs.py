"""Mirror repository documentation into Notion, one way: repo -> Notion.

Git stays the source of truth. Every mirrored page is regenerated from the
markdown on each run and carries a "generated, do not edit" callout, because
edits made in Notion WILL be overwritten. Nothing is ever read back into the
repository.

Layout created under the parent page. The parent is shared across repositories,
so this repo's mirror lives under its own child page (``--repo-page``, default
this directory's name)::

    <parent>
      <repo>
        Overview            README.md, mlb/data/README.md
        Documentation       docs/PIPELINE.md, model_documentation.md, ...
        Plans               docs/*_plan.md, REPO_CLEANUP_AUDIT.md
        Agent context       .omp/AGENTS.md, AGENTS.md, CLAUDE.md, .claude/agents/*
        Tasks (database)    every "- [ ]" item found in the docs above

Notion cannot re-parent an existing page, so changing ``--repo-page`` archives
the old section pages and task database and rebuilds them under the new page.

Task items are extracted into a real Notion database rather than left as
checkboxes, so the 45 items currently buried in two plan files become one
filterable backlog. Each row keeps a stable ``Key`` (hash of source path +
task text) so re-runs update rows in place instead of duplicating them, and
rows whose source line has disappeared are archived.

Idempotency: ``data/notion/sync_state.json`` records a content hash and page id
per file, so unchanged documents are skipped entirely. Changed documents have
their existing blocks deleted and rewritten.

Setup, one time:

1. Create an internal integration at https://www.notion.so/my-integrations
   and copy its token (starts ``ntn_``).
2. Export it as ``NOTION_API_KEY`` or ``NOTION_TOKEN``, and optionally export
   the destination page as ``NOTION_PARENT_PAGE_ID``. Both are also read from
   ``~/.zshrc``, which is the only copy launchd and non-interactive shells see.
3. Open the Notion page that should hold the mirror, and use "..." ->
   "Connections" -> add your integration. The API cannot see pages that have
   not been explicitly connected.

Usage::

    uv run python scripts/sync_notion_docs.py --dry-run
    uv run python scripts/sync_notion_docs.py [--parent-page <notion-url-or-id>]
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import sys
import time
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlb.notion.markdown import (
    MAX_BLOCK_CHILDREN,
    Task,
    chunked,
    extract_tasks,
    markdown_to_blocks,
    rich_text,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = REPO_ROOT / "data" / "notion" / "sync_state.json"
LOCK_PATH = REPO_ROOT / "data" / "notion" / "sync.lock"
# The parent page is shared across repositories, so each repo owns a child page
# named after its directory and every section page hangs off that.
REPO_PAGE_DEFAULT = REPO_ROOT.name

NOTION_API = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"
# Notion allows roughly three requests per second per integration.
REQUEST_INTERVAL = 0.34

SECTIONS: tuple[tuple[str, tuple[str, ...], bool], ...] = (
    ("Overview", ("README.md", "mlb/data/README.md"), False),
    (
        "Documentation",
        ("docs/PIPELINE.md", "docs/model_documentation.md", "docs/evaluation_metrics.md"),
        False,
    ),
    (
        "Plans",
        (
            "docs/training_plan.md",
            "docs/pitcher_strikeout_model_plan.md",
            "docs/pitch_outcome_model_plan.md",
            "docs/REPO_CLEANUP_AUDIT.md",
        ),
        False,
    ),
    (
        "Agent context",
        (".omp/AGENTS.md", "AGENTS.md", "CLAUDE.md", ".claude/agents/code-reviewer.md"),
        True,
    ),
)

# Files that contribute rows to the task database.
TASK_SOURCES = ("docs/",)

MIRROR_NOTICE = (
    "Generated from the repository by scripts/sync_notion_docs.py. "
    "Git is the source of truth — edits made here are overwritten on the next sync."
)
AGENT_NOTICE = (
    "Read-only mirror of an agent context file. This file is loaded automatically "
    "by tooling at session start; it must only ever be edited in the repository."
)


def _from_env_or_zshrc(*names: str) -> str | None:
    """Return the first name that is set, falling back to ``~/.zshrc``.

    Names are tried in order regardless of where they are found, so the
    preferred spelling always wins over a legacy one.
    """
    for name in names:
        value = os.getenv(name)
        if value:
            return value.strip()
    zshrc = Path.home() / ".zshrc"
    if not zshrc.exists():
        return None
    text = zshrc.read_text(errors="replace")
    for name in names:
        match = re.search(
            rf"^\s*export\s+{re.escape(name)}=[\"']?([^\"'\s]+)", text, re.MULTILINE
        )
        if match:
            return match.group(1)
    return None


def load_token() -> str | None:
    return _from_env_or_zshrc("NOTION_API_KEY", "NOTION_TOKEN")


def load_parent_page() -> str | None:
    return _from_env_or_zshrc("NOTION_PARENT_PAGE_ID", "NOTION_PARENT_PAGE")


# A Notion page id is 32 hex characters, optionally dashed. Both patterns are
# anchored against neighbouring hex characters: stripping dashes from a whole
# URL first lets the match slide into the slug, because page titles routinely
# end in a hex letter ("...Page-<id>" matched the "e" of "Page" and returned a
# page id shifted by one character).
_UUID_RE = re.compile(
    r"(?<![0-9a-fA-F])"
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    r"(?![0-9a-fA-F])"
)
_BARE_RE = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{32}(?![0-9a-fA-F])")


def normalize_page_id(value: str) -> str:
    """Accept a Notion URL or a raw id and return the canonical UUID."""

    dashed = _UUID_RE.findall(value)
    if dashed:
        raw = dashed[-1].replace("-", "").lower()
    else:
        bare = _BARE_RE.findall(value)
        if not bare:
            raise SystemExit(f"could not find a Notion page id in {value!r}")
        raw = bare[-1].lower()
    return f"{raw[0:8]}-{raw[8:12]}-{raw[12:16]}-{raw[16:20]}-{raw[20:32]}"


@dataclass
class Document:
    path: str
    section: str
    read_only: bool
    text: str

    @property
    def title(self) -> str:
        return self.path

    @property
    def content_hash(self) -> str:
        return hashlib.blake2b(self.text.encode(), digest_size=8).hexdigest()


def collect_documents() -> list[Document]:
    documents: list[Document] = []
    for section, paths, read_only in SECTIONS:
        for rel in paths:
            path = REPO_ROOT / rel
            if not path.exists():
                print(f"  skip (missing): {rel}")
                continue
            documents.append(
                Document(rel, section, read_only, path.read_text(errors="replace"))
            )
    return documents


def collect_tasks(documents: list[Document]) -> list[Task]:
    tasks: list[Task] = []
    for document in documents:
        if not document.path.startswith(TASK_SOURCES):
            continue
        tasks.extend(extract_tasks(document.text, document.path))
    return tasks


def task_key(task: Task) -> str:
    return hashlib.blake2b(
        f"{task.source}::{task.text}".encode(), digest_size=8
    ).hexdigest()


class NotionClient:
    def __init__(
        self,
        token: str,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
        request_interval: float = REQUEST_INTERVAL,
    ) -> None:
        self._client = httpx.Client(
            base_url=NOTION_API,
            timeout=timeout,
            transport=transport,
            headers={
                "Authorization": f"Bearer {token}",
                "Notion-Version": NOTION_VERSION,
                "Content-Type": "application/json",
            },
        )
        self._interval = request_interval
        self._last_call = 0.0
        self.request_count = 0

    def close(self) -> None:
        self._client.close()

    def _request(self, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        for attempt in range(6):
            wait = self._interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            response = self._client.request(method, url, **kwargs)
            self._last_call = time.monotonic()
            self.request_count += 1
            if response.status_code == 429:
                retry_after = float(response.headers.get("Retry-After", 2 ** attempt))
                print(f"    rate limited, waiting {retry_after:.1f}s")
                time.sleep(retry_after)
                continue
            if response.status_code >= 500:
                time.sleep(2**attempt)
                continue
            if response.status_code >= 400:
                raise SystemExit(
                    f"Notion API {response.status_code} on {method} {url}: {response.text}"
                )
            return response.json()
        raise SystemExit(f"Notion API kept failing on {method} {url}")

    def create_page(self, parent_id: str, title: str) -> str:
        """Create an empty page and return its id.

        Blocks are appended separately and deliberately: the caller must be able
        to record the new page id BEFORE any append, or a failed append leaks a
        half-filled page that the next run cannot find and duplicates instead.
        """
        payload = {
            "parent": {"type": "page_id", "page_id": parent_id},
            "properties": {"title": {"title": rich_text(title)}},
        }
        return self._request("POST", "/pages", json=payload)["id"]

    def append_blocks(self, page_id: str, blocks: list[dict]) -> None:
        self._request("PATCH", f"/blocks/{page_id}/children", json={"children": blocks})

    def clear_page(self, page_id: str) -> None:
        cursor: str | None = None
        ids: list[str] = []
        while True:
            params: dict[str, Any] = {"page_size": 100}
            if cursor:
                params["start_cursor"] = cursor
            data = self._request("GET", f"/blocks/{page_id}/children", params=params)
            ids.extend(block["id"] for block in data.get("results", []))
            if not data.get("has_more"):
                break
            cursor = data.get("next_cursor")
        for block_id in ids:
            self._request("DELETE", f"/blocks/{block_id}")

    def archive_block(self, block_id: str) -> None:
        """Trash any block, including a child_database, which has no page API."""
        self._request("DELETE", f"/blocks/{block_id}")

    def page_exists(self, page_id: str) -> bool:
        try:
            page = self._request("GET", f"/pages/{page_id}")
        except SystemExit:
            return False
        return not page.get("archived", False)

    def create_database(self, parent_id: str, title: str) -> str:
        payload = {
            "parent": {"type": "page_id", "page_id": parent_id},
            "title": rich_text(title),
            "properties": {
                "Task": {"title": {}},
                "Status": {
                    "select": {
                        "options": [
                            {"name": "Open", "color": "yellow"},
                            {"name": "Done", "color": "green"},
                        ]
                    }
                },
                "Source": {"rich_text": {}},
                "Section": {"rich_text": {}},
                "Line": {"number": {}},
                "Key": {"rich_text": {}},
            },
        }
        return self._request("POST", "/databases", json=payload)["id"]

    def query_database(self, database_id: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            payload: dict[str, Any] = {"page_size": 100}
            if cursor:
                payload["start_cursor"] = cursor
            data = self._request("POST", f"/databases/{database_id}/query", json=payload)
            rows.extend(data.get("results", []))
            if not data.get("has_more"):
                break
            cursor = data.get("next_cursor")
        return rows

    def upsert_task(self, database_id: str, task: Task, existing: dict[str, str]) -> str:
        key = task_key(task)
        properties = {
            "Task": {"title": rich_text(task.text)},
            "Status": {"select": {"name": "Done" if task.done else "Open"}},
            "Source": {"rich_text": rich_text(task.source)},
            "Section": {"rich_text": rich_text(task.section or "—")},
            "Line": {"number": task.line},
            "Key": {"rich_text": rich_text(key)},
        }
        page_id = existing.get(key)
        if page_id:
            self._request("PATCH", f"/pages/{page_id}", json={"properties": properties})
            return page_id
        created = self._request(
            "POST",
            "/pages",
            json={"parent": {"database_id": database_id}, "properties": properties},
        )
        return created["id"]

    def archive_page(self, page_id: str) -> None:
        self._request("PATCH", f"/pages/{page_id}", json={"archived": True})


def notice_blocks(document: Document) -> list[dict[str, Any]]:
    text = AGENT_NOTICE if document.read_only else MIRROR_NOTICE
    return [
        {
            "object": "block",
            "type": "callout",
            "callout": {
                "rich_text": rich_text(text),
                "icon": {"type": "emoji", "emoji": "🔒" if document.read_only else "🔁"},
                "color": "gray_background",
            },
        },
        {
            "object": "block",
            "type": "paragraph",
            "paragraph": {"rich_text": rich_text(f"Source: `{document.path}`")},
        },
        {"object": "block", "type": "divider", "divider": {}},
    ]


def empty_state() -> dict[str, Any]:
    return {
        "pages": {},
        "sections": {},
        "database_id": None,
        "parent_id": None,
        "repo_page": None,
        "repo_page_id": None,
    }


def load_state() -> dict[str, Any]:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return empty_state()


def is_stale(
    state: dict[str, Any], documents: list[Document], parent_id: str, repo_page: str
) -> bool:
    """Whether a real sync is needed, decided with NO network calls.

    The post-commit hook gates on this: an already-current sync still costs
    ~60 requests against Notion's ~3/s limit, because every page is verified
    and the task database is re-queried. Only content hashes and the recorded
    layout are consulted, so a commit that touched no mirrored file is free.

    This deliberately cannot detect a page deleted inside Notion; repairing
    that needs a normal (or `--force`) run.
    """
    if state.get("parent_id") != parent_id or state.get("repo_page") != repo_page:
        return True
    if not state.get("repo_page_id") or not state.get("database_id"):
        return True
    sections = state.get("sections", {})
    if any(document.section not in sections for document in documents):
        return True
    pages = state.get("pages", {})
    if len(pages) != len(documents):
        return True
    return any(
        pages.get(document.path, {}).get("hash") != document.content_hash
        for document in documents
    )


@contextlib.contextmanager
def sync_lock() -> Generator[bool, None, None]:
    """Hold an exclusive lock for the duration of a sync.

    Yields False when another sync already holds it. Two concurrent runs would
    interleave reads and writes of the state file and could create duplicate
    pages, which is the same class of defect as the orphan page the id-before-
    fill ordering fixes. flock is released by the kernel if a run is killed, so
    there is no stale lock to clean up.
    """
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True))


def _notice_paragraph() -> dict[str, Any]:
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {"rich_text": rich_text(MIRROR_NOTICE)},
    }


def archive_tree(client: NotionClient, state: dict[str, Any]) -> None:
    """Trash the mirror's section pages and task database, then forget them.

    Notion CANNOT re-parent a page: `PATCH /pages` accepts a `parent` and
    silently ignores it, returning 200 with the old parent intact (verified on
    API versions 2022-06-28 and 2025-09-03; the reference says "a page's parent
    cannot be changed"). So any layout change means archiving the old tree and
    rebuilding it. Archived pages land in the Notion trash and stay restorable.
    """
    for section, page_id in sorted(state.get("sections", {}).items()):
        if client.page_exists(page_id):
            client.archive_page(page_id)
            print(f"  archived old section: {section}")
    database_id = state.get("database_id")
    if database_id:
        client.archive_block(database_id)
        print("  archived old task database")
    state["sections"] = {}
    state["pages"] = {}
    state["database_id"] = None
    state["repo_page_id"] = None


def resolve_repo_page(
    client: NotionClient, state: dict[str, Any], parent_id: str, repo_page: str
) -> str:
    """Return the id of the per-repo page holding this repo's mirror.

    The parent page is shared across repositories, so every repo gets its own
    child page and the sections live under that.
    """
    page_id = state.get("repo_page_id")
    if page_id and state.get("repo_page") == repo_page and client.page_exists(page_id):
        return page_id
    if page_id or state.get("sections") or state.get("database_id"):
        print(f"  layout changed; rebuilding under a '{repo_page}' page")
        archive_tree(client, state)
    page_id = client.create_page(parent_id, repo_page)
    client.append_blocks(page_id, [_notice_paragraph()])
    state["repo_page_id"] = page_id
    state["repo_page"] = repo_page
    save_state(state)
    print(f"  created repo page: {repo_page}")
    return page_id


def dry_run(documents: list[Document], tasks: list[Task]) -> None:
    print("\n=== conversion plan (no network calls) ===")
    total_blocks = 0
    total_requests = 0
    for section, _, _ in SECTIONS:
        in_section = [d for d in documents if d.section == section]
        if not in_section:
            continue
        print(f"\n  {section}")
        for document in in_section:
            blocks = notice_blocks(document) + markdown_to_blocks(document.text)
            requests = 1 + max(0, -(-(len(blocks) - MAX_BLOCK_CHILDREN) // MAX_BLOCK_CHILDREN))
            total_blocks += len(blocks)
            total_requests += requests
            kinds = {}
            for block in blocks:
                kinds[block["type"]] = kinds.get(block["type"], 0) + 1
            summary = " ".join(f"{k}={v}" for k, v in sorted(kinds.items()))
            flag = " [read-only]" if document.read_only else ""
            print(f"    {document.path:<42} {len(blocks):>4} blocks{flag}")
            print(f"      {summary}")
    print(f"\n  totals: {len(documents)} pages, {total_blocks} blocks, "
          f"~{total_requests + len(tasks)} API requests")
    print(f"\n=== task database rows: {len(tasks)} ===")
    by_source: dict[str, list[Task]] = {}
    for task in tasks:
        by_source.setdefault(task.source, []).append(task)
    for source, items in sorted(by_source.items()):
        done = sum(1 for t in items if t.done)
        print(f"  {source:<42} {len(items):>3} tasks ({done} done, {len(items) - done} open)")
        for task in items[:3]:
            mark = "x" if task.done else " "
            section = task.section or "—"
            print(f"    [{mark}] {section[:28]:<28} {task.text[:60]}")
        if len(items) > 3:
            print(f"    ... {len(items) - 3} more")
    keys = [task_key(t) for t in tasks]
    duplicates = len(keys) - len(set(keys))
    if duplicates:
        print(
            f"\n  WARNING: {duplicates} tasks share a (source, text) key and would "
            "collapse into one row. Reword them or they cannot be tracked separately."
        )
    print("\nDry run only. Re-run without --dry-run to sync.")


def sync(
    client: NotionClient,
    parent_id: str,
    documents: list[Document],
    tasks: list[Task],
    force: bool,
    repo_page: str = REPO_PAGE_DEFAULT,
) -> None:
    state = load_state()
    if state.get("parent_id") and state["parent_id"] != parent_id:
        print("  parent page changed; rebuilding all pages")
        archive_tree(client, state)
    state["parent_id"] = parent_id
    repo_page_id = resolve_repo_page(client, state, parent_id, repo_page)

    section_ids: dict[str, str] = state.setdefault("sections", {})
    for section, _, _ in SECTIONS:
        if not any(d.section == section for d in documents):
            continue
        page_id = section_ids.get(section)
        if not page_id or not client.page_exists(page_id):
            page_id = client.create_page(repo_page_id, section)
            client.append_blocks(page_id, [_notice_paragraph()])
            section_ids[section] = page_id
            print(f"  created section: {section}")

    pages: dict[str, Any] = state.setdefault("pages", {})
    for document in documents:
        record = pages.get(document.path, {})
        blocks = notice_blocks(document) + markdown_to_blocks(document.text)
        page_id = record.get("page_id")
        unchanged = record.get("hash") == document.content_hash
        if page_id and unchanged and not force and client.page_exists(page_id):
            print(f"  unchanged: {document.path}")
            continue
        if page_id and client.page_exists(page_id):
            client.clear_page(page_id)
            verb = "updated"
        else:
            page_id = client.create_page(section_ids[document.section], document.title)
            # Record the id with no hash before filling the page, so a failure
            # part-way through the appends is retried against THIS page.
            pages[document.path] = {"page_id": page_id, "hash": None}
            save_state(state)
            verb = "created"
        for batch in chunked(blocks):
            client.append_blocks(page_id, batch)
        print(f"  {verb}: {document.path} ({len(blocks)} blocks)")
        pages[document.path] = {"page_id": page_id, "hash": document.content_hash}
        save_state(state)

    database_id = state.get("database_id")
    if not database_id:
        database_id = client.create_database(repo_page_id, "Tasks")
        state["database_id"] = database_id
        print("  created task database")
    rows = client.query_database(database_id)
    existing: dict[str, str] = {}
    for row in rows:
        key_property = row["properties"].get("Key", {}).get("rich_text", [])
        if key_property:
            existing[key_property[0]["text"]["content"]] = row["id"]

    seen: set[str] = set()
    for task in tasks:
        key = task_key(task)
        if key in seen:
            continue
        seen.add(key)
        client.upsert_task(database_id, task, existing)
    for key, page_id in existing.items():
        if key not in seen:
            client.archive_page(page_id)
            print(f"  archived stale task row {key}")
    print(f"  task database: {len(seen)} rows current, {len(existing) - len(seen & set(existing))} archived")

    save_state(state)
    print(f"\n  {client.request_count} API requests; state at {STATE_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-page", help="Notion page URL or id to mirror into")
    parser.add_argument("--dry-run", action="store_true", help="convert only, no network")
    parser.add_argument("--force", action="store_true", help="rewrite unchanged pages")
    parser.add_argument(
        "--repo-page",
        default=REPO_PAGE_DEFAULT,
        help=f"child page of the parent that holds this repo (default {REPO_PAGE_DEFAULT})",
    )
    parser.add_argument(
        "--if-stale",
        action="store_true",
        help="exit without network calls when no mirrored file changed (for hooks)",
    )
    args = parser.parse_args()

    print("=== repository documents ===")
    documents = collect_documents()
    tasks = collect_tasks(documents)
    for section, _, _ in SECTIONS:
        count = sum(1 for d in documents if d.section == section)
        print(f"  {section:<16} {count} files")

    token = load_token()
    parent_page: str | None = args.parent_page or load_parent_page()
    if args.dry_run:
        dry_run(documents, tasks)
        return

    missing: list[str] = []
    if token is None:
        missing.append("a token (NOTION_API_KEY or NOTION_TOKEN, env or ~/.zshrc)")
    if parent_page is None:
        missing.append("a parent page (--parent-page or NOTION_PARENT_PAGE_ID)")
    if token is None or parent_page is None:
        print("\nrefusing to sync without " + " and ".join(missing))
        print("pass --dry-run to convert without network calls")
        raise SystemExit(2)

    parent_id = normalize_page_id(parent_page)
    skip_when_current = args.if_stale and not args.force
    if skip_when_current and not is_stale(
        load_state(), documents, parent_id, args.repo_page
    ):
        print("\nno mirrored file changed; nothing to sync")
        return

    with sync_lock() as acquired:
        if not acquired:
            print(f"\nanother sync holds {LOCK_PATH}; skipping this run")
            return
        print(f"\n=== syncing into {parent_id} / {args.repo_page} ===")
        client = NotionClient(token)
        try:
            sync(client, parent_id, documents, tasks, args.force, args.repo_page)
        finally:
            client.close()


if __name__ == "__main__":
    main()
