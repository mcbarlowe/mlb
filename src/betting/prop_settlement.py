"""Resolution and ROI reporting for player prop paper bets.

The prop analogue of :mod:`src.betting.paper_settlement`. Pure functions only:
they resolve a single prop bet (did it hit?), compute its flat-stake profit,
and aggregate a ledger of ``mlb.prop_paper_bets`` rows into an ROI/profit
summary. The database settlement loop (which needs the market->stat mapping and
the batting join) lives in ``scripts/settle_prop_alerts.py``; this module keeps
the win/profit math and the reporting so both the standalone settler and the
daily pipeline share one implementation.

Ledger status vocabulary (written by the settler): ``open`` (game not final /
not yet graded), ``won`` / ``lost`` (graded with stake at risk), and ``void``
(player never appeared -> stake returned).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from src.betting.backtest import kelly_fraction

__all__ = [
    "PropBetSummary",
    "kelly_prop_stake_units",
    "prop_probability",
    "prop_profit",
    "resolve_prop_won",
    "summarize_prop_bet_rows",
    "summarize_prop_kelly",
]

GRADED_STATUSES = ("won", "lost")


@dataclass(frozen=True)
class PropBetSummary:
    rows: int
    open_rows: int
    settled_rows: int
    won: int
    lost: int
    void_rows: int
    total_staked: float
    profit_units: float
    roi: float
    win_rate: float


def resolve_prop_won(*, value: int, point: float, side: str) -> bool:
    """Return whether an over/under prop hit given the player's actual stat.

    The board posts half-point lines, so an exact tie is not expected; on a
    whole-number line the ``under`` resolves any non-over as a hit, matching the
    existing settler.
    """
    over_won = value > point
    if side.lower() == "over":
        return over_won
    if side.lower() == "under":
        return not over_won
    raise ValueError(f"Unknown side {side!r}")


def prop_profit(*, won: bool, decimal_odds: float, stake_units: float = 1.0) -> float:
    """Return settled profit in stake units for a graded prop bet."""
    if stake_units < 0:
        raise ValueError("stake_units must be non-negative")
    if decimal_odds <= 1.0:
        raise ValueError("decimal_odds must be greater than 1")
    return stake_units * (decimal_odds - 1.0) if won else -stake_units


def _status(row: Mapping[str, object]) -> str:
    return str(row.get("status", "") or "").lower()


def _as_float(value: object, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)  # type: ignore[arg-type]


def summarize_prop_bet_rows(
    rows: Sequence[Mapping[str, object]],
) -> PropBetSummary:
    """Aggregate a prop ledger into staked/profit/ROI, mirroring the moneyline
    :func:`summarize_paper_trade_rows`.

    ``settled_rows`` counts graded bets with stake at risk (``won`` + ``lost``);
    ``void`` bets returned the stake and are excluded from staked/ROI/win-rate.
    """
    won = [row for row in rows if _status(row) == "won"]
    lost = [row for row in rows if _status(row) == "lost"]
    void = [row for row in rows if _status(row) == "void"]
    graded = won + lost
    total_staked = sum(_as_float(row.get("stake_units"), 1.0) for row in graded)
    profit_units = sum(_as_float(row.get("profit_units")) for row in graded)
    return PropBetSummary(
        rows=len(rows),
        open_rows=len(rows) - len(graded) - len(void),
        settled_rows=len(graded),
        won=len(won),
        lost=len(lost),
        void_rows=len(void),
        total_staked=total_staked,
        profit_units=profit_units,
        roi=profit_units / total_staked if total_staked else 0.0,
        win_rate=len(won) / len(graded) if graded else 0.0,
    )


def prop_probability(row: Mapping[str, object]) -> float | None:
    """Our estimated win probability for a prop row.

    Uses the stored ``adj_prob`` when present, else recovers it from the stored
    EV and decimal odds (``ev = p*dec - 1`` -> ``p = (ev + 1) / dec``) so legacy
    rows written before ``adj_prob`` was persisted can still be sized.
    """
    prob = row.get("adj_prob")
    if prob not in (None, ""):
        return float(prob)  # type: ignore[arg-type]
    ev, dec = row.get("ev"), row.get("decimal_odds")
    if ev in (None, "") or dec in (None, "") or float(dec) <= 0.0:  # type: ignore[arg-type]
        return None
    return (float(ev) + 1.0) / float(dec)  # type: ignore[arg-type]


def _scale_within_groups(
    stakes: list[float], keys: Sequence[object], cap: float,
) -> list[float]:
    """Proportionally scale each group's stakes down so its total <= cap."""
    out = list(stakes)
    groups: dict[object, list[int]] = {}
    for i, key in enumerate(keys):
        groups.setdefault(key, []).append(i)
    for members in groups.values():
        total = sum(out[i] for i in members)
        if total > cap > 0.0:
            scale = cap / total
            for i in members:
                out[i] *= scale
    return out


def kelly_prop_stake_units(
    rows: Sequence[Mapping[str, object]],
    *,
    bankroll: float = 100.0,
    multiplier: float = 0.25,
    per_bet_cap: float = 0.05,
    player_cap: float = 0.05,
    game_cap: float = 0.10,
) -> list[float]:
    """Fractional-Kelly stake per prop row, aligned to ``rows``.

    Per bet: ``min(multiplier * fullKelly, per_bet_cap)`` of a ``bankroll``-unit
    bankroll (the moneyline settings: quarter-Kelly, 5% cap). Correlated exposure
    is then bounded by scaling each player's legs to ``player_cap`` and each
    game's legs to ``game_cap`` of bankroll -- a hot player night cashes his legs
    together, so independent per-bet Kelly would over-concentrate. Rows with no
    usable probability or price get a zero stake.
    """
    raw: list[float] = []
    for row in rows:
        prob = prop_probability(row)
        dec = row.get("decimal_odds")
        if prob is None or dec in (None, "") or float(dec) <= 1.0:  # type: ignore[arg-type]
            raw.append(0.0)
            continue
        full = kelly_fraction(prob, float(dec))  # type: ignore[arg-type]
        raw.append(min(full * multiplier, per_bet_cap) * bankroll)
    players = [str(row.get("player")) for row in rows]
    # A missing matchup must not lump unrelated rows into one game group.
    games = [row.get("matchup") or f"__row{i}" for i, row in enumerate(rows)]
    stakes = _scale_within_groups(raw, players, player_cap * bankroll)
    return _scale_within_groups(stakes, games, game_cap * bankroll)


def summarize_prop_kelly(
    rows: Sequence[Mapping[str, object]],
    stakes: Sequence[float] | None = None,
) -> PropBetSummary:
    """Kelly-weighted counterpart of :func:`summarize_prop_bet_rows`.

    Counts are identical (status-based); staked/profit/ROI use the fractional-
    Kelly stakes and derive profit from the decimal odds (won -> ``stake*(dec-1)``,
    lost -> ``-stake``). Pass ``stakes`` to reuse an already-computed sizing.
    """
    if stakes is None:
        stakes = kelly_prop_stake_units(rows)
    won = lost = void = 0
    total_staked = profit_units = 0.0
    for row, stake in zip(rows, stakes, strict=True):
        status = _status(row)
        if status == "won":
            won += 1
            total_staked += stake
            profit_units += stake * (_as_float(row.get("decimal_odds")) - 1.0)
        elif status == "lost":
            lost += 1
            total_staked += stake
            profit_units += -stake
        elif status == "void":
            void += 1
    graded = won + lost
    return PropBetSummary(
        rows=len(rows),
        open_rows=len(rows) - graded - void,
        settled_rows=graded,
        won=won,
        lost=lost,
        void_rows=void,
        total_staked=total_staked,
        profit_units=profit_units,
        roi=profit_units / total_staked if total_staked else 0.0,
        win_rate=won / graded if graded else 0.0,
    )
