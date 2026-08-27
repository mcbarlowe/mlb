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


# provider_factory(pitcher, batter, is_top_half, stretch, times_through) -> provider
ProviderFactory = Callable[..., PitchDistributionProvider]


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
    _batters_faced: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.current is None:
            self.current = self.starter

    def take(
        self,
        inning: int,
        innings_total: int,
        pitch_limit: int,
        lead: int = 0,
        next_bat_side: str = "R",
    ) -> Pitcher:
        """Pitcher for the next plate appearance: starter until the pitch hook,
        then leverage-ordered one-inning reliever rotation."""
        if self.is_starter and self.pitches >= pitch_limit:
            self.is_starter = False
            self.current = self._select(inning, innings_total, lead, next_bat_side)
            self._entered_inning = inning
        elif not self.is_starter and inning != self._entered_inning:
            self.current = self._select(inning, innings_total, lead, next_bat_side)
            self._entered_inning = inning
        if self.current is None:
            self.current = self.starter
        return self.current

    def _select(
        self, inning: int, innings_total: int, lead: int, next_bat_side: str
    ) -> Pitcher:
        """Pick a reliever: reserve the closer (index 0) for save-type spots in
        the final inning(s), match leverage tier to how late it is, break ties by
        platoon handedness, cap each outing, fall back to the aggregate arm."""
        pool = self.relievers
        if not pool:
            return self.aggregate
        n = len(pool)
        available = [
            k
            for k in range(n)
            if self._reliever_innings.get(pool[k].player_id, 0) < _MAX_RELIEVER_INNINGS
        ]
        if not available:
            return self.aggregate
        # Save-type spot: final inning(s), pitching team ahead or tied by <=3.
        save_spot = inning >= innings_total and 0 <= lead <= 3
        usable = [k for k in available if not (k == 0 and not save_spot)]
        if not usable:
            usable = available  # only the closer remains
        target = max(0, min(innings_total - inning, n - 1))
        usable.sort(key=lambda k: (abs(k - target), k))
        best_distance = abs(usable[0] - target)
        tier = [k for k in usable if abs(k - target) == best_distance]
        chosen = tier[0]
        if next_bat_side in ("L", "R") and len(tier) > 1:
            for k in tier:
                if pool[k].throw_side == next_bat_side:
                    chosen = k
                    break
        arm = pool[chosen]
        self._reliever_innings[arm.player_id] = (
            self._reliever_innings.get(arm.player_id, 0) + 1
        )
        return arm


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

    def simulate_prefix(
        self, away: Lineup, home: Lineup, innings: int = 5
    ) -> GameResult:
        """Simulate a fixed regulation prefix without final-inning shortcuts.

        This is for derivative markets such as first-five totals: every inning
        in the prefix plays both top and bottom halves, including bottom five
        when the home team is already ahead. Prefix simulations never use
        extra innings or ghost runners.
        """
        if innings < 1:
            raise ValueError("innings must be positive")
        cfg = self._config
        if innings > cfg.innings:
            raise ValueError("prefix innings cannot exceed regulation innings")

        away_team = _BattingTeam(away)
        home_team = _BattingTeam(home)
        home_staff = _PitchingStaff(home.starter, home.bullpen, home.relievers)
        away_staff = _PitchingStaff(away.starter, away.bullpen, away.relievers)

        for inning in range(1, innings + 1):
            self._play_half_inning(
                inning,
                is_top=True,
                batting=away_team,
                staff=home_staff,
                opponent=home_team,
                allow_walkoff=False,
                use_ghost_runner=False,
            )
            self._play_half_inning(
                inning,
                is_top=False,
                batting=home_team,
                staff=away_staff,
                opponent=away_team,
                allow_walkoff=False,
                use_ghost_runner=False,
            )

        return GameResult(
            away_team.runs,
            home_team.runs,
            innings,
            tie=away_team.runs == home_team.runs,
        )


    def _play_half_inning(
        self,
        inning: int,
        is_top: bool,
        batting: _BattingTeam,
        staff: _PitchingStaff,
        opponent: _BattingTeam,
        *,
        allow_walkoff: bool = True,
        use_ghost_runner: bool = True,
    ) -> bool:
        """Play one half-inning; returns True on a walk-off end."""
        cfg = self._config
        runners = (
            2 if (use_ghost_runner and cfg.ghost_runner and inning > cfg.innings) else 0
        )
        outs = 0
        while outs < 3:
            due = batting.lineup.batters[batting.slot % 9]
            lead = opponent.runs - batting.runs
            pitcher = staff.take(
                inning, cfg.innings, cfg.starter_pitch_limit, lead, due.bat_side
            )
            faced = staff._batters_faced.get(pitcher.player_id, 0)
            staff._batters_faced[pitcher.player_id] = faced + 1
            provider = self._factory(
                pitcher,
                batting.next_batter(),
                is_top,
                runners != 0,
                times_through=faced // 9 + 1,
            )
            pa = simulate_plate_appearance(provider, self._rng)
            staff.pitches += pa.n_pitches
            transition = self._engine.sample(pa.outcome, runners, outs)
            runners, outs = transition.runners_after, transition.outs_after
            batting.runs += transition.runs
            if (
                allow_walkoff
                and not is_top
                and inning >= cfg.innings
                and batting.runs > opponent.runs
            ):
                return True  # walk-off
        return False

    def simulate_many(
        self,
        away: Lineup,
        home: Lineup,
        n: int,
        environment: dict[str, float] | None = None,
    ) -> list[GameResult]:
        setter = getattr(self._factory, "set_environment", None)
        if setter is not None:
            setter(environment)
        return [self.simulate(away, home) for _ in range(n)]

    def simulate_prefix_many(
        self,
        away: Lineup,
        home: Lineup,
        n: int,
        *,
        innings: int = 5,
        environment: dict[str, float] | None = None,
    ) -> list[GameResult]:
        setter = getattr(self._factory, "set_environment", None)
        if setter is not None:
            setter(environment)
        return [self.simulate_prefix(away, home, innings=innings) for _ in range(n)]


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
