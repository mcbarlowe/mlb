"""Kalshi MLB calibration probe (Kalshi data only, DB-independent).

First pitch is parsed from the market ticker (ET); the pre-game price is the
last hourly candle close strictly before it. Result comes from the market's own
settlement. Reports Kalshi's Brier/calibration and pre-game liquidity.
"""

from __future__ import annotations

import datetime as dt
import re
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import scripts.kalshi_client as k

SERIES = "KXMLBGAME"
ET = ZoneInfo("America/New_York")
MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}
TICK = re.compile(r"KXMLBGAME-(\d{2})([A-Z]{3})(\d{2})(\d{2})(\d{2})")


def first_pitch(ticker: str) -> dt.datetime | None:
    m = TICK.match(ticker)
    if not m:
        return None
    yy, mon, dd, hh, mm = m.groups()
    return dt.datetime(2000 + int(yy), MONTHS[mon], int(dd), int(hh), int(mm), tzinfo=ET)


def settled(lo: int, hi: int) -> list[dict]:
    out, cur = [], None
    for _ in range(200):
        p = {"series_ticker": SERIES, "status": "settled",
             "min_close_ts": lo, "max_close_ts": hi, "limit": 1000}
        if cur:
            p["cursor"] = cur
        j = k.get("/markets", p).json()
        out += j.get("markets", [])
        cur = j.get("cursor")
        if not cur or not j.get("markets"):
            break
    return out


def pregame(ticker: str, fp: dt.datetime) -> tuple[float | None, float]:
    end = int(fp.timestamp())
    start = end - 30 * 3600
    j = k.get(f"/series/{SERIES}/markets/{ticker}/candlesticks",
              {"start_ts": start, "end_ts": end, "period_interval": 60}).json()
    last, vol = None, 0.0
    for c in j.get("candlesticks", []):
        if c["end_period_ts"] > end:
            continue
        vf = c.get("volume_fp")
        if vf:
            vol += float(vf)
        cd = c.get("price", {}).get("close_dollars")
        if cd is not None:
            last = float(cd)
    return last, vol


def main() -> None:
    lo = int(dt.datetime(2026, 7, 1, tzinfo=dt.UTC).timestamp())
    hi = int(dt.datetime(2026, 8, 13, tzinfo=dt.UTC).timestamp())
    markets = settled(lo, hi)
    seen: set[str] = set()
    P, Y, V = [], [], []
    for m in markets:
        ev = m["event_ticker"]
        if ev in seen or m.get("result") not in ("yes", "no"):
            continue
        fp = first_pitch(m["ticker"])
        if fp is None:
            continue
        p, vol = pregame(m["ticker"], fp)
        if p is None:
            continue
        seen.add(ev)
        P.append(p)
        Y.append(1.0 if m["result"] == "yes" else 0.0)
        V.append(vol)
        time.sleep(0.03)
    P, Y, V = np.array(P), np.array(Y), np.array(V)
    n = len(P)
    print(f"games {n} | mean P {P.mean():.3f} std {P.std():.3f} | win rate {Y.mean():.3f}")
    print(f"pre-game price at 0/1 extreme: {int(((P<=0.02)|(P>=0.98)).sum())} ({((P<=0.02)|(P>=0.98)).mean():.0%})")
    print(f"pre-game volume: median {np.median(V):.0f} mean {V.mean():.0f} zero {float((V==0).mean()):.0%}")
    print(f"Kalshi Brier {np.mean((P-Y)**2):.4f} | coin 0.2500 | sportsbook-ref ~0.2418")
    print("calibration:")
    for lo_, hi_ in [(0, .3), (.3, .45), (.45, .55), (.55, .7), (.7, 1.01)]:
        msk = (P >= lo_) & (P < hi_)
        if msk.sum():
            print(f"  P[{lo_:.2f},{hi_:.2f}): n{int(msk.sum()):3d} pred {P[msk].mean():.3f} actual {Y[msk].mean():.3f}")


if __name__ == "__main__":
    main()
