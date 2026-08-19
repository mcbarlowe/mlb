"""Assemble one row per game: model probability, market fair probability, outcome, segments.

Cached to parquet because the walk-forward refits are the slow part and every downstream
question (blending, segment analysis, threshold tuning) reuses the same frame.

Market fair probability is the median over books of that book's own de-vigged home
probability, taken at the last strictly-pre-game snapshot, restricted to the five books with
stable coverage across 2020-2025. Cross-book dispersion and open-to-close movement are
carried through because both are candidate proxies for market uncertainty.

    uv run python scripts/build_model_vs_market_frame.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import psycopg

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.backtest_moneyline import load_finals, walkforward_home_probs
from scripts.backtest_moneyline_lineshop import PANEL_PRIORITY, load_quotes
from src.database import PostgresConfig

SEASONS = (2020, 2021, 2022, 2023, 2024, 2025)
PANEL = PANEL_PRIORITY[:5]
OUT = Path("data/analysis/model_vs_market.parquet")


def game_context(season: int) -> dict[int, dict[str, object]]:
    """Pre-game observables that could plausibly locate market weakness."""
    c = PostgresConfig.from_env()
    conn = psycopg.connect(
        dbname=c.dbname, user=c.user, password=c.password,
        host=c.host, port=c.port, connect_timeout=10,
    )
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT game_pk, game_datetime, day_night, double_header,
                   home_team_id, away_team_id, venue_id, weather_temp, weather_wind
            FROM {c.schema}.games
            WHERE season::int = %s AND game_type = 'R' AND game_datetime IS NOT NULL
            """,
            (season,),
        )
        rows = cur.fetchall()
    conn.close()
    out: dict[int, dict[str, object]] = {}
    for pk, dt, day_night, dh, home_id, away_id, venue_id, temp, wind in rows:
        out[int(pk)] = {
            "month": dt.month,
            "day_night": day_night,
            "double_header": dh,
            "home_team_id": int(home_id) if home_id is not None else None,
            "away_team_id": int(away_id) if away_id is not None else None,
            "venue_id": int(venue_id) if venue_id is not None else None,
            "weather_temp": float(temp) if temp is not None else None,
            "weather_wind": wind,
        }
    return out


def dispersion(season: int, line_type: str) -> dict[int, float]:
    """Cross-book spread of the de-vigged home probability: a market-uncertainty proxy."""
    from src.betting.odds import no_vig_two_way

    c = PostgresConfig.from_env()
    conn = psycopg.connect(
        dbname=c.dbname, user=c.user, password=c.password,
        host=c.host, port=c.port, connect_timeout=10,
    )
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT o.game_pk, o.home_ml, o.away_ml
            FROM {c.schema}.odds o JOIN {c.schema}.games g ON o.game_pk = g.game_pk
            WHERE g.season::int = %s AND g.game_type = 'R' AND o.line_type = %s
              AND o.bookmaker = ANY(%s)
              AND o.home_ml IS NOT NULL AND o.away_ml IS NOT NULL
            """,
            (season, line_type, list(PANEL)),
        )
        buf: dict[int, list[float]] = {}
        for pk, home, away in cur.fetchall():
            buf.setdefault(int(pk), []).append(
                no_vig_two_way(int(home), int(away), method="proportional")[0]
            )
    conn.close()
    return {pk: max(v) - min(v) for pk, v in buf.items() if len(v) > 1}


def main() -> None:
    frames = []
    for season in SEASONS:
        probs = walkforward_home_probs(season, list(range(2015, season))).set_index(
            "game_pk"
        )["model_prob_home"]
        finals = load_finals([season]).set_index("game_pk")
        close = load_quotes(season, PANEL, "close", "proportional")
        openq = load_quotes(season, PANEL, "open", "proportional")
        ctx = game_context(season)
        disp = dispersion(season, "close")

        rows = []
        for pk, q in close.items():
            if pk not in probs.index or pk not in finals.index or pk not in ctx:
                continue
            row = {
                "game_pk": pk,
                "season": season,
                "model_p": float(probs.loc[pk]),
                "market_p": q.fair_home,
                "home_won": bool(finals.loc[pk, "home_won"]),
                "n_books": q.n_books,
                "dispersion": disp.get(pk),
                "open_p": openq[pk].fair_home if pk in openq else None,
            }
            row.update(ctx[pk])
            rows.append(row)
        frames.append(pl.DataFrame(rows, strict=False))
        print(f"{season}: {len(rows)} games")

    frame = pl.concat(frames, how="vertical_relaxed").with_columns(
        (pl.col("market_p") - pl.col("open_p")).alias("line_move"),
        (pl.col("model_p") - pl.col("market_p")).alias("disagreement"),
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(OUT)
    print(f"\nwrote {len(frame)} rows to {OUT}")
    print(f"columns: {frame.columns}")


if __name__ == "__main__":
    main()
