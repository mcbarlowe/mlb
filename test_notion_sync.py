"""Tests for the Notion sync client against a fake Notion API.

The real API is not reachable from tests, but the parts most likely to be wrong
are local: batching pages past the 100-child limit, paginating and clearing
existing blocks, skipping unchanged documents via the state file, upserting task
rows by key instead of duplicating them, archiving stale rows, and retrying a
429. All of that is asserted here against an httpx MockTransport that records
the request sequence.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

_spec = importlib.util.spec_from_file_location(
    "sync_notion_docs", REPO_ROOT / "scripts" / "sync_notion_docs.py"
)
assert _spec and _spec.loader
sync_notion_docs = importlib.util.module_from_spec(_spec)
# Must be in sys.modules before exec: @dataclass resolves its own module.
sys.modules["sync_notion_docs"] = sync_notion_docs
_spec.loader.exec_module(sync_notion_docs)

Document = sync_notion_docs.Document
NotionClient = sync_notion_docs.NotionClient
Task = sync_notion_docs.Task


class FakeNotion:
    """Minimal in-memory Notion: records calls, serves plausible responses."""

    def __init__(
        self,
        fail_once_with: int | None = None,
        fail_append_after: int | None = None,
    ) -> None:
        self.calls: list[tuple[str, str, dict]] = []
        self.appended: list[int] = []
        self.pages: dict[str, dict] = {}
        self.blocks: dict[str, list[str]] = {}
        self.rows: list[dict] = []
        self.archived: list[str] = []
        self.deleted: list[str] = []
        self._fail_once_with = fail_once_with
        self._fail_append_after = fail_append_after
        self._appends = 0
        self._counter = 0

    def _id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}-{self._counter:04d}"

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        path = request.url.path
        self.calls.append((request.method, path, body))

        if self._fail_once_with is not None:
            status = self._fail_once_with
            self._fail_once_with = None
            return httpx.Response(status, json={"message": "slow down"},
                                  headers={"Retry-After": "0"})

        if request.method == "POST" and path == "/v1/pages":
            if "database_id" in body.get("parent", {}):
                row_id = self._id("row")
                self.rows.append({"id": row_id, "properties": body["properties"]})
                return httpx.Response(200, json={"id": row_id})
            page_id = self._id("page")
            self.pages[page_id] = {"archived": False}
            self.blocks[page_id] = []
            return httpx.Response(200, json={"id": page_id})

        if request.method == "PATCH" and path.endswith("/children"):
            page_id = path.split("/")[-2]
            children = body.get("children", [])
            self._appends += 1
            if (
                self._fail_append_after is not None
                and self._appends > self._fail_append_after
            ):
                return httpx.Response(400, json={"message": "Invalid URL for link."})
            self.appended.append(len(children))
            self.blocks.setdefault(page_id, []).extend(self._id("blk") for _ in children)
            return httpx.Response(200, json={"results": []})

        if request.method == "GET" and path.endswith("/children"):
            page_id = path.split("/")[-2]
            ids = self.blocks.get(page_id, [])
            return httpx.Response(
                200,
                json={
                    "results": [{"id": i} for i in ids],
                    "has_more": False,
                    "next_cursor": None,
                },
            )

        if request.method == "DELETE" and "/blocks/" in path:
            self.deleted.append(path.split("/")[-1])
            return httpx.Response(200, json={})

        if request.method == "GET" and "/pages/" in path:
            page_id = path.split("/")[-1]
            if page_id not in self.pages:
                return httpx.Response(404, json={"message": "not found"})
            return httpx.Response(200, json={"id": page_id, **self.pages[page_id]})

        if request.method == "PATCH" and "/pages/" in path:
            page_id = path.split("/")[-1]
            if body.get("archived"):
                self.archived.append(page_id)
            for row in self.rows:
                if row["id"] == page_id and "properties" in body:
                    row["properties"] = body["properties"]
            return httpx.Response(200, json={"id": page_id})

        if request.method == "POST" and path == "/v1/databases":
            return httpx.Response(200, json={"id": self._id("db")})

        if request.method == "POST" and path.endswith("/query"):
            return httpx.Response(
                200, json={"results": self.rows, "has_more": False, "next_cursor": None}
            )

        return httpx.Response(200, json={})

    def client(self) -> NotionClient:
        return NotionClient(
            "ntn_test",
            transport=httpx.MockTransport(self.handler),
            request_interval=0.0,
        )


def _blocks(count: int) -> list[dict]:
    return [
        {"object": "block", "type": "paragraph",
         "paragraph": {"rich_text": [{"type": "text", "text": {"content": f"line {i}"}}]}}
        for i in range(count)
    ]


def test_create_page_makes_an_empty_page():
    fake = FakeNotion()
    client = fake.client()
    page_id = client.create_page("parent-id", "Title")
    assert page_id.startswith("page-")
    create = next(c for c in fake.calls if c[0] == "POST" and c[1] == "/v1/pages")
    assert "children" not in create[2]
    assert fake.appended == []


def test_sync_batches_document_blocks_past_the_hundred_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(sync_notion_docs, "STATE_PATH", tmp_path / "state.json")
    text = "# Title\n\n" + "\n".join(f"- item {i}" for i in range(250)) + "\n"
    document = Document("README.md", "Overview", False, text)
    fake = FakeNotion()
    sync_notion_docs.sync(fake.client(), "parent-id", [document], [], force=False)
    expected = len(
        sync_notion_docs.notice_blocks(document)
        + sync_notion_docs.markdown_to_blocks(document.text)
    )
    assert expected > 100
    assert all(size <= 100 for size in fake.appended)
    # Plus two single-block appends: the repo page's and the section's notice.
    assert sum(fake.appended) == expected + 2


def test_clear_page_deletes_every_existing_block():
    fake = FakeNotion()
    client = fake.client()
    page_id = client.create_page("parent-id", "Title")
    client.append_blocks(page_id, _blocks(5))
    client.clear_page(page_id)
    assert len(fake.deleted) == 5


def test_rate_limit_is_retried_then_succeeds():
    fake = FakeNotion(fail_once_with=429)
    client = fake.client()
    page_id = client.create_page("parent-id", "Title")
    assert page_id.startswith("page-")
    assert client.request_count >= 2


def test_server_error_is_retried():
    fake = FakeNotion(fail_once_with=502)
    client = fake.client()
    assert client.create_page("parent-id", "Title").startswith("page-")


def test_client_error_aborts_loudly():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": "unauthorized"})

    client = NotionClient(
        "bad", transport=httpx.MockTransport(handler), request_interval=0.0
    )
    with pytest.raises(SystemExit, match="401"):
        client.create_page("parent", "Title")


def _task(text: str, done: bool = False, line: int = 1) -> Task:
    return Task(text=text, done=done, source="docs/plan.md", section="Phase", line=line)


def test_task_upsert_updates_existing_row_instead_of_duplicating():
    fake = FakeNotion()
    client = fake.client()
    database_id = "db-1"
    task = _task("do the thing")
    first = client.upsert_task(database_id, task, {})
    assert len(fake.rows) == 1

    # Second run: the row already exists under the same key.
    key = sync_notion_docs.task_key(task)
    second = client.upsert_task(database_id, task, {key: first})
    assert second == first
    assert len(fake.rows) == 1, "re-syncing a task must not create a second row"


def test_task_key_is_stable_across_line_moves_but_splits_on_text_change():
    a = _task("same text", line=10)
    b = _task("same text", line=99)
    c = _task("different text", line=10)
    assert sync_notion_docs.task_key(a) == sync_notion_docs.task_key(b)
    assert sync_notion_docs.task_key(a) != sync_notion_docs.task_key(c)


def test_task_status_reflects_checkbox_state():
    fake = FakeNotion()
    client = fake.client()
    client.upsert_task("db-1", _task("open one", done=False), {})
    client.upsert_task("db-1", _task("closed one", done=True), {})
    statuses = [r["properties"]["Status"]["select"]["name"] for r in fake.rows]
    assert statuses == ["Open", "Done"]


def test_full_sync_skips_unchanged_documents_on_second_run(tmp_path, monkeypatch):
    state_path = tmp_path / "state.json"
    monkeypatch.setattr(sync_notion_docs, "STATE_PATH", state_path)
    fake = FakeNotion()
    client = fake.client()
    documents = [Document("README.md", "Overview", False, "# Title\n\nbody text\n")]
    tasks = [_task("only task")]

    sync_notion_docs.sync(client, "parent-id", documents, tasks, force=False)
    assert state_path.exists()
    first_state = json.loads(state_path.read_text())
    assert "README.md" in first_state["pages"]
    creates_first = sum(1 for c in fake.calls if c[0] == "POST" and c[1] == "/v1/pages")

    calls_before = len(fake.calls)
    sync_notion_docs.sync(client, "parent-id", documents, tasks, force=False)
    creates_second = sum(
        1 for c in fake.calls[calls_before:] if c[0] == "POST" and c[1] == "/v1/pages"
    )
    # The document is unchanged, so no new document page is created; the only
    # POST /pages allowed is the task row upsert path.
    assert creates_second < creates_first
    assert json.loads(state_path.read_text())["pages"]["README.md"]["hash"] == (
        documents[0].content_hash
    )


def test_full_sync_archives_task_rows_that_disappear(tmp_path, monkeypatch):
    monkeypatch.setattr(sync_notion_docs, "STATE_PATH", tmp_path / "state.json")
    fake = FakeNotion()
    client = fake.client()
    documents = [Document("README.md", "Overview", False, "# Title\n")]

    sync_notion_docs.sync(client, "parent-id", documents, [_task("goes away")], force=False)
    assert not fake.archived
    sync_notion_docs.sync(client, "parent-id", documents, [_task("survives")], force=False)
    assert fake.archived, "a task removed from the markdown must be archived in Notion"


def test_changed_document_is_rewritten_not_duplicated(tmp_path, monkeypatch):
    monkeypatch.setattr(sync_notion_docs, "STATE_PATH", tmp_path / "state.json")
    fake = FakeNotion()
    client = fake.client()
    original = [Document("README.md", "Overview", False, "# One\n")]
    sync_notion_docs.sync(client, "parent-id", original, [], force=False)
    page_id = json.loads((tmp_path / "state.json").read_text())["pages"]["README.md"]["page_id"]

    edited = [Document("README.md", "Overview", False, "# One\n\nnew paragraph\n")]
    sync_notion_docs.sync(client, "parent-id", edited, [], force=False)
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["pages"]["README.md"]["page_id"] == page_id, "must reuse the page"
    assert fake.deleted, "old blocks must be cleared before rewriting"


def _page_creates(fake: FakeNotion) -> list[tuple[str, str]]:
    """(parent_id, title) for every page creation, task rows excluded."""
    out = []
    for method, path, body in fake.calls:
        if method == "POST" and path == "/v1/pages" and "page_id" in body.get("parent", {}):
            title = body["properties"]["title"]["title"][0]["text"]["content"]
            out.append((body["parent"]["page_id"], title))
    return out


def test_sections_are_created_under_a_per_repo_page(tmp_path, monkeypatch):
    state_path = tmp_path / "state.json"
    monkeypatch.setattr(sync_notion_docs, "STATE_PATH", state_path)
    fake = FakeNotion()
    documents = [Document("README.md", "Overview", False, "# Title\n")]

    sync_notion_docs.sync(
        fake.client(), "parent-id", documents, [], force=False, repo_page="mlb"
    )
    state = json.loads(state_path.read_text())
    repo_id = state["repo_page_id"]
    assert state["repo_page"] == "mlb"

    creates = {title: parent for parent, title in _page_creates(fake)}
    assert creates["mlb"] == "parent-id", "the repo page hangs off the shared parent"
    assert creates["Overview"] == repo_id, "sections hang off the repo page"
    assert creates["README.md"] == state["sections"]["Overview"]
    database = next(c for c in fake.calls if c[1] == "/v1/databases")
    assert database[2]["parent"]["page_id"] == repo_id, "Tasks belongs to the repo page"


def test_repo_page_is_reused_on_the_next_run(tmp_path, monkeypatch):
    monkeypatch.setattr(sync_notion_docs, "STATE_PATH", tmp_path / "state.json")
    fake = FakeNotion()
    client = fake.client()
    documents = [Document("README.md", "Overview", False, "# Title\n")]

    sync_notion_docs.sync(client, "parent-id", documents, [], force=False, repo_page="mlb")
    first = json.loads((tmp_path / "state.json").read_text())["repo_page_id"]
    sync_notion_docs.sync(client, "parent-id", documents, [], force=False, repo_page="mlb")
    second = json.loads((tmp_path / "state.json").read_text())["repo_page_id"]
    assert first == second
    assert [t for _, t in _page_creates(fake)].count("mlb") == 1


def test_renaming_the_repo_page_archives_the_old_tree(tmp_path, monkeypatch):
    """Notion cannot re-parent a page, so a layout change must rebuild."""
    monkeypatch.setattr(sync_notion_docs, "STATE_PATH", tmp_path / "state.json")
    fake = FakeNotion()
    client = fake.client()
    documents = [Document("README.md", "Overview", False, "# Title\n")]

    sync_notion_docs.sync(client, "parent-id", documents, [], force=False, repo_page="old")
    before = json.loads((tmp_path / "state.json").read_text())

    sync_notion_docs.sync(client, "parent-id", documents, [], force=False, repo_page="new")
    after = json.loads((tmp_path / "state.json").read_text())

    assert before["sections"]["Overview"] in fake.archived, "old section must be trashed"
    assert before["database_id"] in fake.deleted, "old task database must be trashed"
    assert after["repo_page_id"] != before["repo_page_id"]
    assert after["sections"]["Overview"] != before["sections"]["Overview"]
    assert after["pages"]["README.md"]["hash"] == documents[0].content_hash


def test_failed_append_records_the_page_id_instead_of_orphaning_it(tmp_path, monkeypatch):
    """A page created but not filled must be retried, never duplicated.

    This is the real 2026-09-07 failure: a 400 on the second block batch left a
    100-block page in Notion that state never recorded, so the next run built a
    second page with the same title.
    """
    state_path = tmp_path / "state.json"
    monkeypatch.setattr(sync_notion_docs, "STATE_PATH", state_path)
    text = "# Title\n\n" + "\n".join(f"- item {i}" for i in range(150)) + "\n"
    documents = [Document("README.md", "Overview", False, text)]

    # Appends: repo-page notice, section notice, one good document batch, boom.
    broken = FakeNotion(fail_append_after=3)
    with pytest.raises(SystemExit, match="400"):
        sync_notion_docs.sync(broken.client(), "parent-id", documents, [], force=False)
    state = json.loads(state_path.read_text())
    record = state["pages"]["README.md"]
    assert record["hash"] is None, "a half-filled page must not look complete"
    orphan = record["page_id"]

    # A fresh fake stands in for a later run against the same workspace: the
    # repo and section pages survive, so only the document page is retried.
    healthy = FakeNotion()
    for known in (state["repo_page_id"], state["sections"]["Overview"], orphan):
        healthy.pages[known] = {"archived": False}
    healthy.blocks[orphan] = ["blk-existing"]
    sync_notion_docs.sync(healthy.client(), "parent-id", documents, [], force=False)
    retried = json.loads(state_path.read_text())["pages"]["README.md"]
    assert retried["page_id"] == orphan, "the retry must reuse the half-filled page"
    assert retried["hash"] == documents[0].content_hash
    assert healthy.deleted == ["blk-existing"], "the partial content must be cleared"


def _state_after_sync(tmp_path, monkeypatch, documents, repo_page="mlb") -> dict:
    monkeypatch.setattr(sync_notion_docs, "STATE_PATH", tmp_path / "state.json")
    sync_notion_docs.sync(
        FakeNotion().client(), "parent-id", documents, [], force=False,
        repo_page=repo_page,
    )
    return json.loads((tmp_path / "state.json").read_text())


def test_is_stale_is_false_for_an_untouched_mirror(tmp_path, monkeypatch):
    documents = [Document("README.md", "Overview", False, "# Title\n")]
    state = _state_after_sync(tmp_path, monkeypatch, documents)
    assert not sync_notion_docs.is_stale(state, documents, "parent-id", "mlb")


def test_is_stale_detects_every_kind_of_drift(tmp_path, monkeypatch):
    documents = [Document("README.md", "Overview", False, "# Title\n")]
    state = _state_after_sync(tmp_path, monkeypatch, documents)

    edited = [Document("README.md", "Overview", False, "# Title\n\nmore\n")]
    assert sync_notion_docs.is_stale(state, edited, "parent-id", "mlb"), "content change"

    added = documents + [Document("CLAUDE.md", "Agent context", True, "# x\n")]
    assert sync_notion_docs.is_stale(state, added, "parent-id", "mlb"), "new file"

    assert sync_notion_docs.is_stale(state, [], "parent-id", "mlb"), "removed file"
    assert sync_notion_docs.is_stale(state, documents, "other-id", "mlb"), "parent moved"
    assert sync_notion_docs.is_stale(state, documents, "parent-id", "nfl"), "repo renamed"

    for key in ("repo_page_id", "database_id"):
        broken = dict(state, **{key: None})
        assert sync_notion_docs.is_stale(broken, documents, "parent-id", "mlb"), key


def test_is_stale_needs_no_state_file(tmp_path, monkeypatch):
    monkeypatch.setattr(sync_notion_docs, "STATE_PATH", tmp_path / "missing.json")
    documents = [Document("README.md", "Overview", False, "# Title\n")]
    assert sync_notion_docs.is_stale(
        sync_notion_docs.load_state(), documents, "parent-id", "mlb"
    )


def test_sync_lock_is_exclusive(tmp_path, monkeypatch):
    monkeypatch.setattr(sync_notion_docs, "LOCK_PATH", tmp_path / "sync.lock")
    with sync_notion_docs.sync_lock() as outer:
        assert outer is True
        with sync_notion_docs.sync_lock() as inner:
            assert inner is False, "a second holder must be told to back off"
    with sync_notion_docs.sync_lock() as again:
        assert again is True, "the lock must be released on exit"


def test_normalize_page_id_accepts_urls_and_raw_ids():
    expected = "1234abcd-5678-90ef-1234-567890abcdef"
    raw = "1234abcd567890ef1234567890abcdef"
    assert sync_notion_docs.normalize_page_id(raw) == expected
    assert sync_notion_docs.normalize_page_id(f"https://www.notion.so/Page-{raw}") == expected
    assert sync_notion_docs.normalize_page_id(expected) == expected
    with pytest.raises(SystemExit):
        sync_notion_docs.normalize_page_id("https://www.notion.so/no-id-here")


def test_read_only_documents_get_the_locked_notice():
    agent = Document(".omp/AGENTS.md", "Agent context", True, "# x\n")
    normal = Document("README.md", "Overview", False, "# x\n")
    agent_text = json.dumps(sync_notion_docs.notice_blocks(agent))
    normal_text = json.dumps(sync_notion_docs.notice_blocks(normal))
    assert "loaded automatically" in agent_text
    assert "source of truth" in normal_text
    assert "loaded automatically" not in normal_text
