from __future__ import annotations

from datetime import date
from pathlib import Path

import scripts.run_daily_sim_slate as daily_sim
from src.live.publisher import PredictionPost, ResultPost


class _StubPublisher:
    def __init__(self) -> None:
        self.posts: list[PredictionPost] = []
        self.replies: list[ResultPost] = []

    def publish(self, post: PredictionPost) -> str:
        self.posts.append(post)
        return "at://root|cid"

    def publish_result(self, post: ResultPost) -> str:
        self.replies.append(post)
        return f"reply-{len(self.replies)}"


def test_publish_board_uses_single_post_for_one_page():
    publisher = _StubPublisher()
    board_paths = [Path("page1.jpg")]

    post_id = daily_sim._publish_board(
        publisher,
        board_paths,
        target_date=date(2026, 8, 10),
        games_summary="1 preview game",
        watching=True,
    )

    assert post_id == "at://root|cid"
    assert len(publisher.posts) == 1
    assert publisher.posts[0].image_path == board_paths[0]
    assert "1 preview game" in publisher.posts[0].text
    assert "Updates will follow" in publisher.posts[0].text
    assert publisher.replies == []


def test_publish_board_uses_thread_for_multiple_pages():
    publisher = _StubPublisher()
    board_paths = [Path("page1.jpg"), Path("page2.jpg"), Path("page3.jpg")]

    post_id = daily_sim._publish_board(
        publisher,
        board_paths,
        target_date=date(2026, 8, 10),
        games_summary="15 preview games",
        watching=False,
    )

    assert post_id == "at://root|cid"
    assert len(publisher.posts) == 1
    assert publisher.posts[0].image_path == board_paths[0]
    assert "Page 1 of 3." in publisher.posts[0].text
    assert len(publisher.replies) == 2
    assert publisher.replies[0].image_path == board_paths[1]
    assert publisher.replies[0].text == "Page 2 of 3."
    assert publisher.replies[0].reply_to == "at://root|cid"
    assert publisher.replies[1].image_path == board_paths[2]
    assert publisher.replies[1].text == "Page 3 of 3."
    assert publisher.replies[1].reply_to == "at://root|cid"
