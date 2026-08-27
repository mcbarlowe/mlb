"""Normalized completed-game results from ESPN scoreboard payloads."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

__all__ = [
    "SPORT_CONFIGS",
    "CompletedGameResult",
    "SportResultsConfig",
    "date_range",
    "event_ids_from_core_payload",
    "parse_espn_summary",
]


@dataclass(frozen=True)
class SportResultsConfig:
    code: str
    espn_core_path: str
    espn_site_path: str


SPORT_CONFIGS: dict[str, SportResultsConfig] = {
    "nba": SportResultsConfig(
        code="NBA",
        espn_core_path="basketball/leagues/nba",
        espn_site_path="basketball/nba",
    ),
    "ncaaf": SportResultsConfig(
        code="NCAAF",
        espn_core_path="football/leagues/college-football",
        espn_site_path="football/college-football",
    ),
    "nhl": SportResultsConfig(
        code="NHL",
        espn_core_path="hockey/leagues/nhl",
        espn_site_path="hockey/nhl",
    ),
}

_EVENT_ID_RE = re.compile(r"/events/(\d+)")


@dataclass(frozen=True)
class CompletedGameResult:
    sport: str
    source: str
    source_event_id: str
    season: int | None
    week: int | None
    game_date: date
    commence_time: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    home_won: bool
    neutral_site: bool
    status: str


def date_range(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError("end must be on or after start")
    days: list[date] = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return days


def event_ids_from_core_payload(payload: dict[str, Any]) -> list[str]:
    event_ids: list[str] = []
    for item in payload.get("items", []):
        ref = str(item.get("$ref", ""))
        match = _EVENT_ID_RE.search(ref)
        if match:
            event_ids.append(match.group(1))
    return sorted(dict.fromkeys(event_ids))


def _competitor_by_home_away(competitors: list[dict[str, Any]], home_away: str) -> dict[str, Any]:
    for competitor in competitors:
        if competitor.get("homeAway") == home_away:
            return competitor
    raise ValueError(f"ESPN summary missing {home_away} competitor")


def _team_name(competitor: dict[str, Any]) -> str:
    team = competitor.get("team") or {}
    for key in ("displayName", "shortDisplayName", "name", "location"):
        value = team.get(key)
        if value:
            return str(value)
    raise ValueError("ESPN competitor missing team name")


def _score(competitor: dict[str, Any]) -> int:
    value = competitor.get("score")
    if value is None or value == "":
        raise ValueError("ESPN completed competitor missing score")
    return int(float(value))


def _parse_iso_date(value: str) -> date:
    return datetime.fromisoformat(value).date()


def parse_espn_summary(
    payload: dict[str, Any], *, sport: str, include_incomplete: bool = False
) -> CompletedGameResult | None:
    """Parse one ESPN summary payload into the normalized result contract.

    Returns ``None`` for incomplete events unless ``include_incomplete`` is true.
    Incomplete events without scores are still unusable for historical settlement,
    so callers normally keep the default.
    """
    header = payload.get("header") or {}
    competitions = header.get("competitions") or []
    if not competitions:
        raise ValueError("ESPN summary missing competitions")

    competition = competitions[0]
    status_type = (competition.get("status") or {}).get("type") or {}
    status = str(status_type.get("name") or status_type.get("description") or "unknown")
    completed = bool(status_type.get("completed")) or status == "STATUS_FINAL"
    if not completed and not include_incomplete:
        return None

    competitors = competition.get("competitors") or []
    home = _competitor_by_home_away(competitors, "home")
    away = _competitor_by_home_away(competitors, "away")
    home_score = _score(home)
    away_score = _score(away)
    commence_time = str(competition.get("date") or payload.get("date") or "")
    if not commence_time:
        raise ValueError("ESPN summary missing commence_time")

    season = header.get("season") or {}
    week = header.get("week")
    if isinstance(week, dict):
        week_value = week.get("number") or week.get("week")
    else:
        week_value = week

    return CompletedGameResult(
        sport=sport.upper(),
        source="espn",
        source_event_id=str(header.get("id") or competition.get("id")),
        season=int(season["year"]) if season.get("year") is not None else None,
        week=int(week_value) if week_value is not None else None,
        game_date=_parse_iso_date(commence_time),
        commence_time=commence_time,
        home_team=_team_name(home),
        away_team=_team_name(away),
        home_score=home_score,
        away_score=away_score,
        home_won=bool(home.get("winner")) if home.get("winner") is not None else home_score > away_score,
        neutral_site=bool(competition.get("neutralSite")),
        status=status,
    )
