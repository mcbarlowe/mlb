"""Backtest the champion win model on moneylines at OPENING prices.

Bets each game at the consensus opening line, settles on the realized result
(ROI in units), and measures closing-line value (CLV) vs the consensus close.
Reports flat and fractional-Kelly staking across edge thresholds, plus a
close-betting baseline for contrast.

    uv run python scripts/backtest_moneyline.py --season 2025
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.betting.backtest import MoneylineGame, backtest_moneyline
from src.betting.ingest import champion_home_probs, load_finals
from src.betting.odds import american_to_decimal, decimal_to_american
from src.database import PostgresConfig


def consensus_american(season: int) -> dict[str, dict[int, tuple[float, float]]]:
    """Per line_type -> {game_pk: (home_american, away_american)}.

    Consensus is taken in decimal space (median across books) to avoid the
    sign discontinuity at +/-100, then converted back to American.
    """
    c = PostgresConfig.from_env()
    conn = psycopg.connect(
        dbname=c.dbname, user=c.user, password=c.password,
        host=c.host, port=c.port, connect_timeout=10,
    )
    buf: dict[tuple[str, int], list[tuple[float, float]]] = {}
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT o.line_type, o.game_pk, o.home_ml, o.away_ml
            FROM {c.schema}.odds o JOIN {c.schema}.games g ON o.game_pk = g.game_pk
            WHERE g.season::int = %s AND g.game_type = 'R'
              AND o.home_ml IS NOT NULL AND o.away_ml IS NOT NULL
            """,
            (season,),
        )
        for lt, pk, h, a in cur.fetchall():
            buf.setdefault((lt, int(pk)), []).append(
                (american_to_decimal(int(h)), american_to_decimal(int(a)))
            )
    conn.close()

    out: dict[str, dict[int, tuple[float, float]]] = {"open": {}, "close": {}}
    for (lt, pk), rows in buf.items():
        if lt not in out:
            continue
        hd = statistics.median([r[0] for r in rows])
        ad = statistics.median([r[1] for r in rows])
        out[lt][pk] = (decimal_to_american(hd), decimal_to_american(ad))
    return out

def walkforward_home_probs(test_season: int, train_seasons):
    """Leak-free OOS home-win probs: champion feature recipe, logistic refit
    on train_seasons only (never the test season)."""
    import json as _json

    import pandas as pd
    from mlflow.artifacts import download_artifacts
    from mlflow.tracking import MlflowClient
    from sklearn.linear_model import LogisticRegression

    from src.sim.team_strength import (
        StrengthConfig,
        build_feature_frame,
        load_completed_games,
    )

    train_seasons = tuple(int(s) for s in train_seasons)
    if int(test_season) in train_seasons:
        raise SystemExit("test season must not be in train seasons")
    uri = "http://10.0.0.171:5001"
    client = MlflowClient(tracking_uri=uri)
    v = client.get_model_version_by_alias("mlb-team-strength-win", "champion")
    cp = download_artifacts(run_id=v.run_id, artifact_path="model_contract.json",
                            tracking_uri=uri)
    contract = _json.loads(Path(cp).read_text())
    features = list(contract["features"])
    sc = contract["strength_config"]
    config = StrengthConfig(**{k: sc[k] for k in (
        "initial_elo", "elo_k", "elo_home_advantage", "elo_season_regression",
        "initial_runs_per_game", "run_alpha", "run_season_regression",
        "starter_prior_ip", "starter_season_decay")})
    games = load_completed_games(start_season=2015, end_season=int(test_season))
    frame, _ = build_feature_frame(games, config)
    train = frame[frame["season"].isin(train_seasons)]
    test = frame[frame["season"] == int(test_season)]
    model = LogisticRegression(C=1.0, max_iter=1000)
    model.fit(train[features], train["home_won"])
    probs = model.predict_proba(test[features])[:, 1]
    return pd.DataFrame(
        {"game_pk": test["game_pk"].to_numpy(), "model_prob_home": probs}
    )


def build_rows(season: int, probs):
    """Games present in open + close + model + result."""
    finals = load_finals([season]).set_index("game_pk")
    probs = probs.set_index("game_pk")["model_prob_home"]
    odds = consensus_american(season)
    rows = []
    for pk, (oh, oa) in odds["open"].items():
        if pk not in odds["close"] or pk not in finals.index or pk not in probs.index:
            continue
        ch, ca = odds["close"][pk]
        rows.append((
            pk, float(probs.loc[pk]), oh, oa, ch, ca, bool(finals.loc[pk, "home_won"]),
        ))
    return rows


def games_take_open(rows) -> list[MoneylineGame]:
    return [
        MoneylineGame(pk, mp, home_take=oh, away_take=oa,
                      home_close=ch, away_close=ca, home_won=w)
        for pk, mp, oh, oa, ch, ca, w in rows
    ]


def games_take_close(rows) -> list[MoneylineGame]:
    return [
        MoneylineGame.closing_only(pk, mp, home_close=ch, away_close=ca, home_won=w)
        for pk, mp, _oh, _oa, ch, ca, w in rows
    ]


def _line(e, s):
    return (f"  edge>={e:.2f}: bets {s.n_bets:4d}  ROI {s.roi:+.2%}  "
            f"win {s.win_rate:.1%}  CLV {s.avg_clv:+.4f}  "
            f"beat-close {s.pct_beat_close:.0%}  net {s.net_profit:+.1f}u")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--edges", default="0.0,0.02,0.03,0.05")
    ap.add_argument("--devig", default="proportional", choices=("proportional", "shin"))
    ap.add_argument("--walkforward-train", default=None,
                    help="comma seasons to refit logistic on (OOS); omit=champion")
    args = ap.parse_args()

    if args.walkforward_train:
        train = [int(s) for s in args.walkforward_train.split(",")]
        probs = walkforward_home_probs(args.season, train)
        model_desc = f"walk-forward OOS (train {train})"
    else:
        probs = champion_home_probs([args.season])
        model_desc = "champion (may be in-sample)"
    rows = build_rows(args.season, probs)
    edges = [float(x) for x in args.edges.split(",")]
    open_games = games_take_open(rows)
    close_games = games_take_close(rows)

    print(f"Backtest {args.season}: {len(rows)} games | model: {model_desc}")
    print(f"De-vig: {args.devig}.  Primary: bet CONSENSUS OPEN, CLV vs consensus close.\n")

    print("=== Bet at OPEN | flat 1u ===")
    for e in edges:
        s, _ = backtest_moneyline(open_games, devig_method=args.devig,
                                  edge_threshold=e, staking="flat")
        print(_line(e, s))

    print("\n=== Bet at OPEN | quarter-Kelly (cap 5% bankroll) ===")
    for e in edges:
        s, _ = backtest_moneyline(open_games, devig_method=args.devig,
                                  edge_threshold=e, staking="kelly")
        print(_line(e, s))

    print("\n=== Baseline: bet at CLOSE | flat 1u (CLV=0 by construction) ===")
    for e in edges:
        s, _ = backtest_moneyline(close_games, devig_method=args.devig,
                                  edge_threshold=e, staking="flat")
        print(_line(e, s))


if __name__ == "__main__":
    main()
