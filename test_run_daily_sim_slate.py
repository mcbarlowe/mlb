from __future__ import annotations

from datetime import date
from pathlib import Path

import mlb.cli.run_daily_sim_slate as daily_sim
from mlb.live.publisher import PredictionPost, ResultPost
from mlb.sim.slate import DailySlateState, save_daily_slate_state


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



def test_posted_state_prevents_duplicate_publish(tmp_path: Path):
    state_path = tmp_path / "daily_sim_2026-08-10.json"
    save_daily_slate_state(
        state_path,
        DailySlateState(
            slate_date="2026-08-10",
            saved_at="2026-08-10T12:00:00Z",
            board_path="board.jpg",
            board_post_id="at://existing-post",
            games=[],
        ),
    )

    state = daily_sim._posted_state_for_date(
        state_path,
        date(2026, 8, 10),
        post_enabled=True,
    )

    assert state is not None
    assert state.board_post_id == "at://existing-post"
    assert (
        daily_sim._posted_state_for_date(
            state_path,
            date(2026, 8, 10),
            post_enabled=False,
        )
        is None
    )