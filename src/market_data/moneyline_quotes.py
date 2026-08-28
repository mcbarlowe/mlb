"""Normalized historical moneyline quotes for model evaluation."""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass

import psycopg

from src.database import PostgresConfig
from src.market_data.pricing import no_vig_two_way

# Books with at least 80% coverage of the last pre-game snapshot in every
# season from 2020 through 2025, ordered by historical availability.
MONEYLINE_PANEL = (
    "betonlineag",
    "draftkings",
    "fanduel",
    "bovada",
    "betrivers",
)

__all__ = [
    "MONEYLINE_PANEL",
    "NormalizedMoneylineQuote",
    "load_normalized_moneyline_quotes",
]


@dataclass(frozen=True)
class NormalizedMoneylineQuote:
    """One game's paired-book fair home probability and panel depth."""

    game_pk: int
    fair_home: float
    n_books: int


def load_normalized_moneyline_quotes(
    season: int,
    panel: tuple[str, ...] | None,
    line_type: str,
    devig_method: str,
) -> dict[int, NormalizedMoneylineQuote]:
    """Load per-game quotes normalized within each book's two-sided market.

    Each book's paired home/away prices are de-vigged before the median fair
    home probability is taken across books. This preserves the sportsbook pair
    and avoids fabricating a fair market from cross-book prices.
    """
    config = PostgresConfig.from_env()
    connection = psycopg.connect(
        dbname=config.dbname,
        user=config.user,
        password=config.password,
        host=config.host,
        port=config.port,
        connect_timeout=10,
    )
    prices_by_game: dict[int, list[tuple[int, int]]] = defaultdict(list)
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT o.game_pk, o.bookmaker, o.home_ml, o.away_ml
            FROM {config.schema}.odds o
            JOIN {config.schema}.games g ON o.game_pk = g.game_pk
            WHERE g.season::int = %s AND g.game_type = 'R' AND o.line_type = %s
              AND o.home_ml IS NOT NULL AND o.away_ml IS NOT NULL
            """,
            (season, line_type),
        )
        for game_pk, bookmaker, home_ml, away_ml in cursor.fetchall():
            if panel is not None and bookmaker not in panel:
                continue
            prices_by_game[int(game_pk)].append((int(home_ml), int(away_ml)))
    connection.close()

    return {
        game_pk: NormalizedMoneylineQuote(
            game_pk=game_pk,
            fair_home=statistics.median(
                no_vig_two_way(home_ml, away_ml, method=devig_method)[0]
                for home_ml, away_ml in prices
            ),
            n_books=len(prices),
        )
        for game_pk, prices in prices_by_game.items()
        if prices
    }
