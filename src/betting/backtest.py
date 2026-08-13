"""Moneyline bet selection, staking, settlement, CLV, and ROI aggregation.

The harness answers the only question that matters for betting: at real
historical prices, does a probability source make money and beat the closing
line? Proper scores (Brier/log loss) are reported alongside, but ROI and CLV
are the verdict.

Pipeline per game:
  1. De-vig the offered two-way price into a fair market probability.
  2. edge = model_prob - market_fair_prob, for each side.
  3. Bet the side whose edge clears ``edge_threshold`` (at most one side).
  4. Stake flat or by fractional Kelly on the taken decimal price.
  5. Settle against the final result; measure CLV vs the closing price.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.betting.odds import (
    american_to_decimal,
    no_vig_two_way,
)

__all__ = [
    "BacktestSummary",
    "MoneylineGame",
    "SettledBet",
    "backtest_moneyline",
    "kelly_fraction",
]


@dataclass(frozen=True)
class MoneylineGame:
    """One game's model probability, prices, and realized result.

    Prices are American odds. ``*_take`` is the price a bet would be placed at
    (opening line when available); ``*_close`` is the closing line used for CLV.
    When only closing prices exist, pass them as both take and close (CLV is
    then zero by construction, which is the honest outcome).
    """

    game_pk: int
    model_prob_home: float
    home_take: float
    away_take: float
    home_close: float
    away_close: float
    home_won: bool

    @classmethod
    def closing_only(
        cls,
        game_pk: int,
        model_prob_home: float,
        home_close: float,
        away_close: float,
        home_won: bool,
    ) -> MoneylineGame:
        return cls(
            game_pk=game_pk,
            model_prob_home=model_prob_home,
            home_take=home_close,
            away_take=away_close,
            home_close=home_close,
            away_close=away_close,
            home_won=home_won,
        )


@dataclass(frozen=True)
class SettledBet:
    game_pk: int
    side: str  # "home" | "away"
    model_prob: float
    market_prob: float
    edge: float
    take_decimal: float
    stake: float
    won: bool
    profit: float
    clv_prob: float  # closing fair prob for the side minus take fair prob
    beat_close: bool


@dataclass(frozen=True)
class BacktestSummary:
    n_games: int
    n_bets: int
    total_staked: float
    net_profit: float
    roi: float
    win_rate: float
    avg_edge: float
    avg_clv: float
    pct_beat_close: float
    model_brier_all: float
    market_brier_all: float
    model_brier_bet: float
    settings: dict[str, object] = field(default_factory=dict)

    def format_report(self) -> str:
        lines = [
            f"Games evaluated:        {self.n_games:,}",
            f"Bets placed:            {self.n_bets:,}"
            f"  ({self.n_bets / self.n_games:.1%} of games)"
            if self.n_games
            else "Bets placed:            0",
            f"Total staked (units):   {self.total_staked:.2f}",
            f"Net profit (units):     {self.net_profit:+.2f}",
            f"ROI:                    {self.roi:+.2%}",
            f"Bet win rate:           {self.win_rate:.1%}",
            f"Avg edge (model-mkt):   {self.avg_edge:+.4f}",
            f"Avg CLV (prob):         {self.avg_clv:+.4f}",
            f"Beat closing line:      {self.pct_beat_close:.1%} of bets",
            f"Model Brier (all):      {self.model_brier_all:.4f}",
            f"Market Brier (all):     {self.market_brier_all:.4f}",
            f"Model Brier (bet only): {self.model_brier_bet:.4f}",
        ]
        return "\n".join(lines)


def kelly_fraction(prob: float, decimal_odds: float) -> float:
    """Full-Kelly stake fraction of bankroll; zero when there is no edge.

    b = decimal_odds - 1 (net odds). f* = (p*b - (1-p)) / b, floored at 0.
    """
    b = decimal_odds - 1.0
    if b <= 0.0:
        return 0.0
    f = (prob * b - (1.0 - prob)) / b
    return max(f, 0.0)


def _brier(prob_home: float, home_won: bool) -> float:
    y = 1.0 if home_won else 0.0
    return (prob_home - y) ** 2


def backtest_moneyline(
    games: list[MoneylineGame],
    *,
    devig_method: str = "proportional",
    edge_threshold: float = 0.02,
    staking: str = "flat",
    flat_stake: float = 1.0,
    kelly_multiplier: float = 0.25,
    kelly_cap: float = 0.05,
) -> tuple[BacktestSummary, list[SettledBet]]:
    """Backtest a home-win probability source against historical moneylines.

    ``staking``: ``"flat"`` risks ``flat_stake`` units per bet; ``"kelly"``
    risks ``kelly_multiplier`` x full-Kelly of a 1-unit bankroll, capped at
    ``kelly_cap`` (fixed bankroll, no compounding, so ROI is stake-weighted).
    """
    if staking not in {"flat", "kelly"}:
        raise ValueError(f"Unknown staking plan {staking!r}")

    bets: list[SettledBet] = []
    total_staked = 0.0
    net_profit = 0.0
    model_brier_all = 0.0
    market_brier_all = 0.0

    for game in games:
        take_home_p, take_away_p = no_vig_two_way(
            game.home_take, game.away_take, method=devig_method
        )
        close_home_p, close_away_p = no_vig_two_way(
            game.home_close, game.away_close, method=devig_method
        )
        model_home = game.model_prob_home
        model_away = 1.0 - model_home
        model_brier_all += _brier(model_home, game.home_won)
        market_brier_all += _brier(take_home_p, game.home_won)

        home_edge = model_home - take_home_p
        away_edge = model_away - take_away_p
        if home_edge >= away_edge:
            side, edge, model_p, take_p, close_p, american = (
                "home",
                home_edge,
                model_home,
                take_home_p,
                close_home_p,
                game.home_take,
            )
        else:
            side, edge, model_p, take_p, close_p, american = (
                "away",
                away_edge,
                model_away,
                take_away_p,
                close_away_p,
                game.away_take,
            )
        if edge < edge_threshold:
            continue

        take_decimal = american_to_decimal(american)
        if staking == "flat":
            stake = flat_stake
        else:
            full = kelly_fraction(model_p, take_decimal)
            stake = min(full * kelly_multiplier, kelly_cap)
            if stake <= 0.0:
                continue

        won = (side == "home") == game.home_won
        profit = stake * (take_decimal - 1.0) if won else -stake
        clv_prob = close_p - take_p
        beat_close = take_p < close_p  # you took a longer price than the close

        total_staked += stake
        net_profit += profit
        bets.append(
            SettledBet(
                game_pk=game.game_pk,
                side=side,
                model_prob=model_p,
                market_prob=take_p,
                edge=edge,
                take_decimal=take_decimal,
                stake=stake,
                won=won,
                profit=profit,
                clv_prob=clv_prob,
                beat_close=beat_close,
            )
        )

    n_games = len(games)
    n_bets = len(bets)
    # Brier on bet games uses the game's home outcome and model home prob.
    bet_game_pks = {b.game_pk for b in bets}
    bet_games = [g for g in games if g.game_pk in bet_game_pks]
    model_brier_bet = (
        sum(_brier(g.model_prob_home, g.home_won) for g in bet_games) / len(bet_games)
        if bet_games
        else 0.0
    )

    summary = BacktestSummary(
        n_games=n_games,
        n_bets=n_bets,
        total_staked=total_staked,
        net_profit=net_profit,
        roi=(net_profit / total_staked) if total_staked > 0 else 0.0,
        win_rate=(sum(1 for b in bets if b.won) / n_bets) if n_bets else 0.0,
        avg_edge=(sum(b.edge for b in bets) / n_bets) if n_bets else 0.0,
        avg_clv=(sum(b.clv_prob for b in bets) / n_bets) if n_bets else 0.0,
        pct_beat_close=(sum(1 for b in bets if b.beat_close) / n_bets) if n_bets else 0.0,
        model_brier_all=(model_brier_all / n_games) if n_games else 0.0,
        market_brier_all=(market_brier_all / n_games) if n_games else 0.0,
        model_brier_bet=model_brier_bet,
        settings={
            "devig_method": devig_method,
            "edge_threshold": edge_threshold,
            "staking": staking,
            "flat_stake": flat_stake,
            "kelly_multiplier": kelly_multiplier,
            "kelly_cap": kelly_cap,
        },
    )
    return summary, bets
