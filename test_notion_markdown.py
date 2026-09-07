"""Tests for Markdown -> Notion block conversion.

These defend the constraints that make a sync succeed or fail against the real
API: the 2000-character rich_text limit, the code-language enum, table shape,
list nesting, and that no text is ever silently dropped.
"""

from __future__ import annotations

from mlb.notion.markdown import (
    MAX_RICH_TEXT_CHARS,
    extract_tasks,
    markdown_to_blocks,
    rich_text,
)


def _plain(block: dict) -> str:
    kind = block["type"]
    return "".join(obj["text"]["content"] for obj in block[kind]["rich_text"])


def test_headings_collapse_beyond_level_three():
    blocks = markdown_to_blocks("# a\n\n## b\n\n### c\n\n#### d\n\n##### e")
    assert [b["type"] for b in blocks] == [
        "heading_1",
        "heading_2",
        "heading_3",
        "heading_3",
        "heading_3",
    ]
    assert _plain(blocks[3]) == "d"


def test_long_line_splits_across_rich_text_objects_without_loss():
    # .omp/AGENTS.md has single bullets far past the per-object limit.
    word = "verylongtoken"
    line = " ".join([word] * 600)
    blocks = markdown_to_blocks(f"- {line}")
    assert len(blocks) == 1
    items = blocks[0]["bulleted_list_item"]["rich_text"]
    assert len(items) > 1
    assert all(len(obj["text"]["content"]) <= MAX_RICH_TEXT_CHARS for obj in items)
    assert "".join(obj["text"]["content"] for obj in items).split() == line.split()


def test_oversized_line_splits_into_sibling_blocks_losslessly():
    # Past 100 rich_text objects a single block is illegal; the text must be
    # spread across sibling blocks rather than truncated.
    line = "x" * (2000 * 130)
    blocks = markdown_to_blocks(line)
    assert len(blocks) > 1
    assert all(b["type"] == "paragraph" for b in blocks)
    assert all(len(b["paragraph"]["rich_text"]) <= 100 for b in blocks)
    assert "".join(_plain(b) for b in blocks) == line


def test_nested_lists_become_children():
    blocks = markdown_to_blocks("- parent\n  - child\n    - grandchild\n- sibling")
    assert len(blocks) == 2
    parent = blocks[0]["bulleted_list_item"]
    child = parent["children"][0]
    assert _plain(child) == "child"
    grandchild = child["bulleted_list_item"]["children"][0]
    assert _plain(grandchild) == "grandchild"
    assert _plain(blocks[1]) == "sibling"


def test_task_items_become_to_do_with_checked_state():
    blocks = markdown_to_blocks("- [ ] open item\n- [x] done item")
    assert [b["type"] for b in blocks] == ["to_do", "to_do"]
    assert blocks[0]["to_do"]["checked"] is False
    assert blocks[1]["to_do"]["checked"] is True
    assert _plain(blocks[1]) == "done item"


def test_unknown_code_language_falls_back_to_plain_text():
    blocks = markdown_to_blocks("```\nplain\n```")
    assert blocks[0]["code"]["language"] == "plain text"
    aliased = markdown_to_blocks("```sh\necho hi\n```")
    assert aliased[0]["code"]["language"] == "shell"
    known = markdown_to_blocks("```python\nx = 1\n```")
    assert known[0]["code"]["language"] == "python"


def test_fenced_code_preserves_markdown_syntax_verbatim():
    source = "```markdown\n# not a heading\n- [ ] not a task\n```"
    blocks = markdown_to_blocks(source)
    assert len(blocks) == 1
    assert blocks[0]["type"] == "code"
    assert _plain(blocks[0]) == "# not a heading\n- [ ] not a task"


def test_fenced_code_never_parses_inline_markdown():
    source = "```python\noutput = heads[pt_idx](hidden[mask])\n```"
    blocks = markdown_to_blocks(source)
    assert _plain(blocks[0]) == "output = heads[pt_idx](hidden[mask])"
    assert all(obj["text"].get("link") is None for obj in blocks[0]["code"]["rich_text"])


def test_table_becomes_table_block_with_header():
    source = "| a | b |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |"
    blocks = markdown_to_blocks(source)
    assert len(blocks) == 1
    table = blocks[0]["table"]
    assert table["table_width"] == 2
    assert table["has_column_header"] is True
    rows = table["children"]
    assert len(rows) == 3
    assert [c[0]["text"]["content"] for c in rows[0]["table_row"]["cells"]] == ["a", "b"]
    assert [c[0]["text"]["content"] for c in rows[2]["table_row"]["cells"]] == ["3", "4"]


def test_ragged_table_rows_are_padded_to_width():
    blocks = markdown_to_blocks("| a | b | c |\n| --- | --- | --- |\n| 1 |")
    row = blocks[0]["table"]["children"][1]["table_row"]["cells"]
    assert len(row) == 3


def test_inline_formatting_and_links():
    items = rich_text("plain **bold** `code` [label](https://example.com) *it*")
    rendered = [(obj["text"]["content"], obj.get("annotations"), obj["text"].get("link"))
                for obj in items]
    assert ("bold", {"bold": True}, None) in rendered
    assert ("code", {"code": True}, None) in rendered
    assert ("label", None, {"url": "https://example.com"}) in rendered
    assert ("it", {"italic": True}, None) in rendered


def test_non_absolute_link_targets_render_as_plain_text():
    for target in ("docs/PIPELINE.md", "#1-test-set-overview", "../README.md"):
        items = rich_text(f"see [the plan]({target})")
        assert all(obj["text"].get("link") is None for obj in items), target
        assert "".join(obj["text"]["content"] for obj in items) == "see the plan"


def test_absolute_link_targets_are_kept():
    items = rich_text("[a](https://example.com) [b](mailto:x@y.z)")
    urls = [obj["text"]["link"]["url"] for obj in items if obj["text"].get("link")]
    assert urls == ["https://example.com", "mailto:x@y.z"]


def test_underscores_inside_identifiers_are_not_italics():
    items = rich_text("call load_to_database and postgres_backfill now")
    assert len(items) == 1
    assert items[0].get("annotations") is None
    assert "load_to_database" in items[0]["text"]["content"]


def test_divider_and_quote():
    blocks = markdown_to_blocks("> quoted line\n\n---\n\nafter")
    kinds = [b["type"] for b in blocks]
    assert kinds == ["quote", "divider", "paragraph"]
    assert _plain(blocks[0]) == "quoted line"


def test_paragraph_lines_join_and_blank_lines_separate():
    blocks = markdown_to_blocks("one\ntwo\n\nthree")
    assert [b["type"] for b in blocks] == ["paragraph", "paragraph"]
    assert _plain(blocks[0]) == "one two"
    assert _plain(blocks[1]) == "three"


def test_extract_tasks_captures_heading_context_and_state():
    source = (
        "# Plan\n\n## Phase one\n\n- [ ] first task\n- [x] second task\n\n"
        "## Phase two\n\n- [ ] third task\n"
    )
    tasks = extract_tasks(source, "docs/plan.md")
    assert [t.text for t in tasks] == ["first task", "second task", "third task"]
    assert [t.section for t in tasks] == ["Phase one", "Phase one", "Phase two"]
    assert [t.done for t in tasks] == [False, True, False]
    assert all(t.source == "docs/plan.md" for t in tasks)
    assert tasks[0].line == 5


def test_extract_tasks_ignores_fenced_code():
    source = "## Real\n\n- [ ] real task\n\n```\n- [ ] fake task\n```\n"
    tasks = extract_tasks(source, "docs/plan.md")
    assert [t.text for t in tasks] == ["real task"]


def test_empty_and_whitespace_documents_produce_no_blocks():
    assert markdown_to_blocks("") == []
    assert markdown_to_blocks("\n\n   \n") == []


def test_repo_docs_convert_within_api_limits():
    """Every real doc in the repo must convert to legal blocks."""

    import pathlib

    root = pathlib.Path(__file__).resolve().parent
    docs = sorted(root.glob("docs/*.md")) + [
        root / "README.md",
        root / ".omp/AGENTS.md",
        root / "mlb/data/README.md",
    ]
    checked = 0
    for path in docs:
        if not path.exists():
            continue
        blocks = markdown_to_blocks(path.read_text())
        assert blocks, f"{path} produced no blocks"
        _assert_limits(blocks, path)
        checked += 1
    assert checked >= 5


def _assert_limits(blocks: list[dict], path) -> None:
    for block in blocks:
        kind = block["type"]
        payload = block[kind]
        if kind == "table":
            width = payload["table_width"]
            for row in payload["children"]:
                assert len(row["table_row"]["cells"]) == width, f"{path}: ragged table"
            continue
        if kind == "divider":
            continue
        items = payload["rich_text"]
        assert len(items) <= 100, f"{path}: {kind} exceeds rich_text item limit"
        for obj in items:
            assert len(obj["text"]["content"]) <= MAX_RICH_TEXT_CHARS, (
                f"{path}: {kind} rich_text object over {MAX_RICH_TEXT_CHARS} chars"
            )
        if "children" in payload:
            _assert_limits(payload["children"], path)
