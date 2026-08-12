"""Full-game Monte Carlo simulation.

Chains matchup providers (PA outcome distributions) with the empirical
base-out engine over real lineups. v1 scope, by design:

- static 9-man lineups (no pinch hitters or mid-game substitutions)
- one pitcher change: the starter exits after a pitch limit, replaced by a
  generic league-average bullpen arm (``Pitcher(0, "R")``)
- situational features (score, inning, runners, times-through-order) are
  frozen at neutral values inside each matchup provider
- extra innings use the ghost-runner rule; a safety cap declares rare
  marathon games a tie (scored as half a win per side)
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field

from src.sim.base_out import BaseOutEngine
from src.sim.pa import PitchDistributionProvider, simulate_plate_appearance


@dataclass(frozen=True)
class Batter:
    player_id: int
    bat_side: str  # "L" | "R" | "S"


@dataclass(frozen=True)
class Pitcher:
    player_id: int
    throw_side: str  # "L" | "R"


BULLPEN_ARM = Pitcher(player_id=0, throw_side="R")


@dataclass(frozen=True)
class Lineup:
    batters: list[Batter]  # 9, in batting order
    starter: Pitcher
    # Relief corps stand-in used after the starter's pitch limit: a synthetic
    # per-team aggregate arm when available, else the league-average arm.
    bullpen: Pitcher = BULLPEN_ARM
    # Individual bullpen arms, highest-leverage first (closer at index 0);
    # empty falls back to the single aggregate ``bullpen`` arm above.
    relievers: tuple[Pitcher, ...] = ()

    def __post_init__(self) -> None:
        if len(self.batters) != 9:
            raise ValueError(f"Lineup needs 9 batters, got {len(self.batters)}")


# provider_factory(pitcher, batter, is_top_half, stretch) -> distributions
ProviderFactory = Callable[[Pitcher, Batter, bool, bool], PitchDistributionProvider]


@dataclass(frozen=True)
class GameConfig:
    innings: int = 9
    max_innings: int = 26
    starter_pitch_limit: int = 90
    ghost_runner: bool = True


@dataclass(frozen=True)
class GameResult:
    away_runs: int
    home_runs: int
    innings: int
    tie: bool = False

    @property
    def home_won(self) -> bool:
        return self.home_runs > self.away_runs


_MAX_RELIEVER_INNINGS = 2


@dataclass
class _PitchingStaff:
    starter: Pitcher
    aggregate: Pitcher
    relievers: tuple[Pitcher, ...] = ()
    current: Pitcher | None = None
    is_starter: bool = True
    pitches: int = 0
    _entered_inning: int = 0
    _reliever_innings: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.current is None:
            self.current = self.starter

    def take(self, inning: int, innings_total: int, pitch_limit: int) -> Pitcher:
        """Pitcher for the next plate appearance: starter until the pitch
        hook, then one-inning reliever rotation reserving the best arms late."""
        if self.is_starter and self.pitches >= pitch_limit:
            self.is_starter = False
            self.current = self._select(inning, innings_total)
            self._entered_inning = inning
        elif not self.is_starter and inning != self._entered_inning:
            self.current = self._select(inning, innings_total)
            self._entered_inning = inning
        return self.current

    def _select(self, inning: int, innings_total: int) -> Pitcher:
        """Reserve higher-leverage arms (earlier pool indices) for later
        innings; cap each reliever's outing and fall back to the aggregate arm."""
        pool = self.relievers
        if not pool:
            return self.aggregate
        target = max(0, min(innings_total - inning, len(pool) - 1))
        order = sorted(range(len(pool)), key=lambda k: (abs(k - target), k))
        for k in order:
            arm = pool[k]
            if self._reliever_innings.get(arm.player_id, 0) < _MAX_RELIEVER_INNINGS:
                self._reliever_innings[arm.player_id] = (
                    self._reliever_innings.get(arm.player_id, 0) + 1
                )
                return arm
        return self.aggregate


@dataclass
class _BattingTeam:
    lineup: Lineup
    slot: int = 0
    runs: int = 0

    def next_batter(self) -> Batter:
        batter = self.lineup.batters[self.slot % 9]
        self.slot += 1
        return batter


class GameSimulator:
    def __init__(
        self,
        provider_factory: ProviderFactory,
        engine: BaseOutEngine,
        rng: random.Random | None = None,
        config: GameConfig | None = None,
    ):
        self._factory = provider_factory
        self._engine = engine
        self._rng = rng or random.Random()
        self._config = config or GameConfig()

    def simulate(self, away: Lineup, home: Lineup) -> GameResult:
        cfg = self._config
        away_team = _BattingTeam(away)
        home_team = _BattingTeam(home)
        home_staff = _PitchingStaff(
            home.starter, home.bullpen, home.relievers
        )  # faces away
        away_staff = _PitchingStaff(away.starter, away.bullpen, away.relievers)

        inning = 1
        while True:
            self._play_half_inning(
                inning,
                is_top=True,
                batting=away_team,
                staff=home_staff,
                opponent=home_team,
            )
            if inning >= cfg.innings and home_team.runs > away_team.runs:
                # Home leads after the top half of a final inning: no bottom.
                return GameResult(away_team.runs, home_team.runs, inning)

            walkoff = self._play_half_inning(
                inning,
                is_top=False,
                batting=home_team,
                staff=away_staff,
                opponent=away_team,
            )
            if inning >= cfg.innings and home_team.runs != away_team.runs:
                return GameResult(away_team.runs, home_team.runs, inning)
            if walkoff:
                return GameResult(away_team.runs, home_team.runs, inning)
            if inning >= cfg.max_innings:
                return GameResult(away_team.runs, home_team.runs, inning, tie=True)
            inning += 1

    def _play_half_inning(
        self,
        inning: int,
        is_top: bool,
        batting: _BattingTeam,
        staff: _PitchingStaff,
        opponent: _BattingTeam,
    ) -> bool:
        """Play one half-inning; returns True on a walk-off end."""
        cfg = self._config
        runners = 2 if (cfg.ghost_runner and inning > cfg.innings) else 0
        outs = 0
        while outs < 3:
            pitcher = staff.take(inning, cfg.innings, cfg.starter_pitch_limit)
            provider = self._factory(
                pitcher, batting.next_batter(), is_top, runners != 0
            )
            pa = simulate_plate_appearance(provider, self._rng)
            staff.pitches += pa.n_pitches
            transition = self._engine.sample(pa.outcome, runners, outs)
            runners, outs = transition.runners_after, transition.outs_after
            batting.runs += transition.runs
            if not is_top and inning >= cfg.innings and batting.runs > opponent.runs:
                return True  # walk-off
        return False

    def simulate_many(self, away: Lineup, home: Lineup, n: int) -> list[GameResult]:
        return [self.simulate(away, home) for _ in range(n)]


def summarize(results: list[GameResult]) -> dict:
    """Aggregate Monte Carlo results; ties count half a win each way."""
    n = len(results)
    if n == 0:
        raise ValueError("No results to summarize")
    home_wins = sum(1.0 if r.home_won else 0.0 for r in results if not r.tie)
    home_wins += 0.5 * sum(1 for r in results if r.tie)
    return {
        "n": n,
        "home_win_probability": home_wins / n,
        "mean_home_runs": sum(r.home_runs for r in results) / n,
        "mean_away_runs": sum(r.away_runs for r in results) / n,
        "mean_total_runs": sum(r.home_runs + r.away_runs for r in results) / n,
        "tie_rate": sum(1 for r in results if r.tie) / n,
        "mean_innings": sum(r.innings for r in results) / n,
    }
