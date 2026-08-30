"""Card renderers must reuse one warm browser instead of relaunching per card.

HtmlCardRenderer keeps a single headless Chromium alive behind a worker thread.
Constructing one per card pays the browser launch every time and defeats the
point of the worker, so every render entry point accepts a shared renderer.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Self

from mlb.sim.projection_charts import write_projection_graphics
from mlb.sim.season import SeasonProjection, TeamInfo, TeamProjection


class FakeRenderer:
    """Stands in for HtmlCardRenderer, counting worker startups and renders."""

    def __init__(self) -> None:
        self.starts = 0
        self.closes = 0
        self.renders: list[Path] = []

    def __enter__(self) -> Self:
        self.starts += 1
        return self

    def __exit__(self, *_exc: object) -> None:
        self.closes += 1

    def render(self, html: str, out_path: Path) -> Path:
        return self.render_with_size(html, out_path, width=1, height=1)

    def render_with_size(
        self, html: str, out_path: Path, *, width: int, height: int
    ) -> Path:
        assert html.lstrip().startswith("<!DOCTYPE html>")
        target = out_path.with_suffix(".jpg")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"jpeg")
        self.renders.append(target)
        return target


def _team(team_id: int, abbreviation: str) -> TeamInfo:
    return TeamInfo(
        team_id=team_id,
        abbreviation=abbreviation,
        team_name=f"Team {abbreviation}",
        league_name="AL",
        division_name="East",
    )


TEAMS = {1: _team(1, "AAA"), 2: _team(2, "BBB")}


def _projection(season: int) -> SeasonProjection:
    return SeasonProjection(
        season=season,
        as_of_date=date(season, 3, 29),
        trials=100,
        wild_cards_per_league=3,
        teams=(
            TeamProjection(1, 10, 91.2, 0.40, 0.82, 0.70, 0.42, 0.20, 0.11),
            TeamProjection(2, 8, 84.3, 0.20, 0.54, 0.31, 0.14, 0.06, 0.02),
        ),
    )


def test_a_supplied_renderer_is_reused_across_calls(tmp_path) -> None:
    fake = FakeRenderer()

    first = write_projection_graphics(
        _projection(2026), TEAMS, tmp_path, renderer=fake
    )
    second = write_projection_graphics(
        _projection(2025), TEAMS, tmp_path, renderer=fake
    )

    # Two calls, four images, and the caller's renderer was never re-entered or
    # closed: the browser stays warm, which is the whole point of passing it in.
    assert len(first) == 2
    assert len(second) == 2
    assert len(fake.renders) == 4
    assert fake.starts == 0
    assert fake.closes == 0


def test_without_a_renderer_one_is_owned_and_closed(tmp_path, monkeypatch) -> None:
    created: list[FakeRenderer] = []

    def factory() -> FakeRenderer:
        renderer = FakeRenderer()
        created.append(renderer)
        return renderer

    monkeypatch.setattr("mlb.live.card_html.HtmlCardRenderer", factory)

    write_projection_graphics(_projection(2026), TEAMS, tmp_path)

    # Backward compatible: exactly one renderer created, entered, and closed.
    assert len(created) == 1
    assert created[0].starts == 1
    assert created[0].closes == 1
    assert len(created[0].renders) == 2


def test_renderer_is_closed_even_when_a_render_raises(tmp_path, monkeypatch) -> None:
    class Exploding(FakeRenderer):
        def render_with_size(self, html, out_path, *, width, height):
            raise RuntimeError("chromium died")

    created: list[Exploding] = []

    def factory() -> Exploding:
        renderer = Exploding()
        created.append(renderer)
        return renderer

    monkeypatch.setattr("mlb.live.card_html.HtmlCardRenderer", factory)

    try:
        write_projection_graphics(_projection(2026), TEAMS, tmp_path)
    except RuntimeError:
        pass
    else:  # pragma: no cover - the fake always raises
        raise AssertionError("expected the render failure to propagate")

    # The `with` block must still tear the browser down, or the thread leaks.
    assert created[0].closes == 1
