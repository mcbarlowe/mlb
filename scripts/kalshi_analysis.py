"""Kalshi MLB edge analysis: Kalshi pre-game price vs sharp sportsbook, our
walk-forward model, and realized results, with Kalshi's fee model.

Answers: is Kalshi mispriced relative to the sharp consensus enough to bet after
fees? Segmented by favorite/dog (Kalshi fees are lowest away from pick'em).

    zsh -ic 'uv run python scripts/kalshi_analysis.py'
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import psycopg

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.backtest_moneyline import walkforward_home_probs
from scripts.kalshi_probe import first_pitch, pregame, settled
from src.betting.ingest import load_finals, team_abbrev_to_id
from src.betting.odds import no_vig_two_way
from src.database import PostgresConfig

SKIP = {"AL", "NL"}


def code_map() -> dict[str, int]:
    m = {c.upper(): i for c, i in team_abbrev_to_id().items()}
    # Kalshi Athletics code -> franchise id (A's); map via any existing A's alias
    for alias in ("OAK", "ATHLETICS", "SACRAMENTO"):
        if alias in m:
            m["ATH"] = m[alias]
            break
    return m


def sharp_home_probs(season: int) -> dict[int, float]:
    c = PostgresConfig.from_env()
    conn = psycopg.connect(dbname=c.dbname, user=c.user, password=c.password,
                           host=c.host, port=c.port, connect_timeout=15)
    with conn.cursor() as cur:
        cur.execute(f"""SELECT o.game_pk,o.home_ml,o.away_ml FROM {c.schema}.odds o
                        JOIN {c.schema}.games g USING(game_pk)
                        WHERE g.season::int=%s AND o.line_type='close'
                        AND o.home_ml IS NOT NULL""", (season,))
        rows = cur.fetchall()
    conn.close()
    agg: dict[int, list[float]] = {}
    for pk, h, a in rows:
        ph, _ = no_vig_two_way(float(h), float(a))
        agg.setdefault(int(pk), []).append(ph)
    return {pk: sum(v) / len(v) for pk, v in agg.items()}


def kalshi_home_probs() -> dict[int, float]:
    cm = code_map()
    finals = load_finals([2026])
    pair = {}
    for r in finals.to_dict("records"):
        pair[(str(r["game_date"]), frozenset({r["home_team_id"], r["away_team_id"]}))] = r
    lo = int(dt.datetime(2026, 7, 1, tzinfo=dt.UTC).timestamp())
    hi = int(dt.datetime(2026, 8, 13, tzinfo=dt.UTC).timestamp())
    events: dict[str, list[dict]] = {}
    for m in settled(lo, hi):
        events.setdefault(m["event_ticker"], []).append(m)
    out: dict[int, float] = {}
    for mks in events.values():
        teamed = []
        for m in mks:
            code = m["ticker"].rsplit("-", 1)[1]
            if code in SKIP:
                continue
            tid = cm.get(code.upper())
            if tid:
                teamed.append((tid, m))
        if len(teamed) != 2:
            continue
        fp = first_pitch(mks[0]["ticker"])
        if fp is None:
            continue
        row = pair.get((fp.date().isoformat(), frozenset(t for t, _ in teamed)))
        if row is None:
            for delta in (1, -1):
                row = pair.get(((fp.date() + dt.timedelta(days=delta)).isoformat(),
                                frozenset(t for t, _ in teamed)))
                if row:
                    break
        if row is None:
            continue
        hm = next((m for t, m in teamed if t == row["home_team_id"]), None)
        if hm is None:
            continue
        p, _ = pregame(hm["ticker"], fp)
        if p is not None:
            out[int(row["game_pk"])] = p
    return out


def kalshi_fee(price: float) -> float:
    return 0.07 * price * (1.0 - price)


def main() -> None:
    finals = load_finals([2026]).set_index("game_pk")
    won = finals["home_won"].to_dict()
    sharp = sharp_home_probs(2026)
    kal = kalshi_home_probs()
    model = walkforward_home_probs(2026, [2021, 2022, 2023, 2024, 2025])
    model = model.set_index("game_pk")["model_prob_home"].to_dict()

    pks = [pk for pk in kal if pk in won]
    print(f"Kalshi games matched: {len(kal)} | with result: {len(pks)} | "
          f"also sharp: {sum(pk in sharp for pk in pks)} | also model: {sum(pk in model for pk in pks)}")

    def brier(src):
        v = [(src[pk], won[pk]) for pk in pks if pk in src]
        p = np.array([a for a, _ in v]); y = np.array([1.0 if b else 0.0 for _, b in v])
        return len(v), float(np.mean((p - y) ** 2)), float(np.mean((p - y.mean())**2))
    for name, src in [("kalshi", kal), ("sharp", sharp), ("model", model)]:
        n, b, _ = brier(src)
        print(f"  {name:7s} n{n} Brier {b:.4f}")

    # Kalshi vs sharp deviation + fee-aware betting (sharp = fair)
    common = [pk for pk in pks if pk in sharp]
    dev = np.array([kal[pk] - sharp[pk] for pk in common])
    print(f"\nKalshi vs sharp (n={len(common)}): mean|dev| {np.abs(dev).mean():.4f} "
          f"std {dev.std():.4f} corr {np.corrcoef([kal[pk] for pk in common],[sharp[pk] for pk in common])[0,1]:.3f}")

    def bet(fair_src, label, edges=(0.02, 0.03, 0.05)):
        print(f"\nBet Kalshi vs {label} (fair={label}); Kalshi fee 0.07*p*(1-p):")
        for e in edges:
            cost = ret = 0.0; nb = 0; wins = 0
            for pk in common:
                if pk not in fair_src:
                    continue
                fair = fair_src[pk]; kp = kal[pk]; y = 1.0 if won[pk] else 0.0
                # home side price kp; away side price 1-kp
                if fair - kp > e:            # buy home YES
                    price = kp; win = y
                elif (1 - fair) - (1 - kp) > e:  # buy away YES
                    price = 1 - kp; win = 1 - y
                else:
                    continue
                f = kalshi_fee(price)
                cost += price + f; ret += win; nb += 1; wins += int(win)
                roi = (ret - cost) / cost if cost else 0.0
            roi = (ret - cost) / cost if cost else float("nan")
            print(f"  edge>={e:.2f}: bets {nb:4d} win {wins/nb:.1%} ROI {roi:+.1%}" if nb else f"  edge>={e:.2f}: no bets")

    bet(sharp, "sharp")
    bet(model, "model")


if __name__ == "__main__":
    main()
