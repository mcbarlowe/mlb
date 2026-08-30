"""Registry of MLB-owned launchd agents and their packaged runners."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

MLFLOW_BACKEND_URI_ENV: Final = "MLB_MLFLOW_BACKEND_URI"


@dataclass(frozen=True)
class Agent:
    """One launchd agent, its packaged runner, and its load-time preflight."""

    label: str
    summary: str
    runner: str | None = None
    console_scripts: tuple[str, ...] = ()
    requires_mlflow_backend_uri: bool = False

    @property
    def plist_name(self) -> str:
        return f"{self.label}.plist"


AGENTS: Final[tuple[Agent, ...]] = (
    Agent(
        label="com.barloweanalytics.daily-random-live-game",
        summary="daily random live-game pipeline with a movement-profile refresh",
        runner="run_daily_random_live_game.sh",
        console_scripts=("mlb-live-pipeline", "mlb-build-pitcher-movement-profiles"),
    ),
    Agent(
        label="com.barloweanalytics.daily-season-projection",
        summary="daily season projection refresh, simulation, and post",
        runner="run_daily_season_projection.sh",
        console_scripts=("mlb-daily-season-projection",),
    ),
    Agent(
        label="com.barloweanalytics.daily-sim-slate",
        summary="daily slate simulation with starter watching",
        runner="run_daily_sim_slate.sh",
        console_scripts=("mlb-daily-sim-slate",),
    ),
    Agent(
        # The tracking server is a long-lived binary, not a scheduled job, so
        # launchd runs the mlflow console script directly with no runner shell.
        label="com.barloweanalytics.mlflow-server",
        summary="shared MLflow tracking and artifact server",
        console_scripts=("mlflow",),
        requires_mlflow_backend_uri=True,
    ),
)

_BY_LABEL: Final[dict[str, Agent]] = {agent.label: agent for agent in AGENTS}


def agent_by_label(label: str) -> Agent:
    """Return the registered agent for ``label``."""
    try:
        return _BY_LABEL[label]
    except KeyError:
        known = ", ".join(sorted(_BY_LABEL))
        raise KeyError(f"unknown mlb agent {label!r}; known agents: {known}") from None


def selected_agents(labels: Iterable[str] | None) -> tuple[Agent, ...]:
    """Return the requested agents, or every agent when ``labels`` is empty."""
    materialized = tuple(labels or ())
    if not materialized:
        return AGENTS
    return tuple(agent_by_label(label) for label in materialized)
