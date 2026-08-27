"""Analyze Pinnacle-fair EV results by sport/market/horizon stratum.

Inputs are evaluated CSVs produced by sport-specific validators. The matrix keeps
sport/market/horizon separate first, then reports optional pooled rows so volume
never hides a structurally bad stratum.

Example:
    uv run python scripts/analyze_pinnacle_ev_matrix.py \
      --thresholds 0.005 0.01 0.02 \
      --output output/pinnacle_ev_matrix/matrix_2020_2025.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_INPUTS = (
    "MLB:h2h:output/mlb_pinnacle_ev/evaluated_2020_2025_h1_4_24_168.csv",
    "NFL:h2h:output/nfl_pinnacle_ev/evaluated.csv",
    "NBA:h2h:output/nba_pinnacle_ev/evaluated_2020_2025_h1_4_24_168.csv",
    "NCAAF:h2h:output/ncaaf_pinnacle_ev/evaluated_2020_2025_h1_4_24_168.csv",
    "NHL:h2h:output/nhl_pinnacle_ev/evaluated_2020_2025_h1_4_24_168.csv",
)


@dataclass(frozen=True)
class EvaluatedOffer:
    sport: str
    market: str
    horizon_hours: float
    event_id: str
    season: str
    cluster: str
    best_side: str
    home_won: bool
    decimal_odds: float
    probability: float
    ev: float
    book: str


@dataclass(frozen=True)
class SettledOffer:
    offer: EvaluatedOffer
    won: bool
    ret: float


@dataclass(frozen=True)
class MatrixRow:
    sport: str
    market: str
    horizon: str
    threshold: float
    evaluated: int
    bets: int
    win_rate: float
    avg_ev: float
    roi: float
    ci_low: float
    ci_high: float
    z: float
    per_bet_sd: float
    n_95_avg_ev: float
    n_80_avg_ev: float
    n_95_5pct: float
    n_80_5pct: float
    significant: bool


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "t", "yes", "y"}


def parse_thresholds(values: Sequence[str]) -> tuple[float, ...]:
    thresholds: list[float] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            threshold = float(part)
            if threshold < 0.0:
                raise SystemExit("thresholds must be non-negative")
            thresholds.append(threshold)
    if not thresholds:
        raise SystemExit("at least one threshold is required")
    return tuple(sorted(dict.fromkeys(thresholds)))


def parse_input_spec(value: str) -> tuple[str, str, Path]:
    parts = value.split(":", 2)
    if len(parts) != 3 or not all(parts):
        raise SystemExit("input specs must be SPORT:MARKET:path")
    sport, market, path = parts
    return sport.upper(), market.lower(), Path(path)


def event_id(row: dict[str, str]) -> str:
    return row.get("game_pk") or row.get("game_id") or row.get("event_id") or "unknown"


def cluster_key(sport: str, market: str, row: dict[str, str]) -> str:
    if row.get("game_date"):
        return f"{sport}:{market}:date:{row['game_date']}"
    if row.get("week"):
        return f"{sport}:{market}:season:{row.get('season', '?')}:week:{row['week']}"
    return f"{sport}:{market}:event:{event_id(row)}"


def read_evaluated(specs: Sequence[str]) -> list[EvaluatedOffer]:
    offers: list[EvaluatedOffer] = []
    for spec in specs:
        sport, market, path = parse_input_spec(spec)
        with path.open(newline="") as file:
            for row in csv.DictReader(file):
                offers.append(
                    EvaluatedOffer(
                        sport=sport,
                        market=market,
                        horizon_hours=float(row["horizon_hours"]),
                        event_id=event_id(row),
                        season=str(row.get("season", "")),
                        cluster=cluster_key(sport, market, row),
                        best_side=row["best_side"],
                        home_won=parse_bool(row["home_won"]),
                        decimal_odds=float(row["best_decimal"]),
                        probability=float(row["best_prob"]),
                        ev=float(row["best_ev"]),
                        book=row.get("best_book", ""),
                    )
                )
    return offers


def settle(offers: Iterable[EvaluatedOffer], threshold: float) -> list[SettledOffer]:
    settled: list[SettledOffer] = []
    for offer in offers:
        if offer.ev < threshold:
            continue
        won = offer.home_won if offer.best_side == "home" else not offer.home_won
        ret = offer.decimal_odds - 1.0 if won else -1.0
        settled.append(SettledOffer(offer=offer, won=won, ret=ret))
    return settled


def mean(values: Sequence[float]) -> float:
    if not values:
        return math.nan
    return statistics.mean(values)


def cluster_bootstrap_ci(
    settled: Sequence[SettledOffer], *, samples: int, seed: int
) -> tuple[float, float, float]:
    if not settled:
        return math.nan, math.nan, math.nan
    returns = [offer.ret for offer in settled]
    observed = mean(returns)
    if samples <= 0:
        return observed, math.nan, math.nan
    by_cluster: dict[str, list[float]] = defaultdict(list)
    for offer in settled:
        by_cluster[offer.offer.cluster].append(offer.ret)
    clusters = sorted(by_cluster)
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(samples):
        sampled = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        values = [ret for cluster in sampled for ret in by_cluster[cluster]]
        draws.append(mean(values))
    draws.sort()
    return observed, draws[int(0.025 * samples)], draws[int(0.975 * samples)]


def z_score(returns: Sequence[float]) -> float:
    if len(returns) < 2:
        return math.nan
    sd = statistics.stdev(returns)
    if sd == 0.0:
        return math.nan
    return mean(returns) / (sd / math.sqrt(len(returns)))


def required_n(sd: float, target_roi: float, *, prospective_power: bool) -> float:
    if sd <= 0.0 or target_roi <= 0.0 or not math.isfinite(sd):
        return math.nan
    z = 1.96 + (0.84 if prospective_power else 0.0)
    return (z * sd / target_roi) ** 2


def summarize_group(
    *,
    sport: str,
    market: str,
    horizon: str,
    offers: Sequence[EvaluatedOffer],
    threshold: float,
    samples: int,
    seed: int,
) -> MatrixRow:
    settled = settle(offers, threshold)
    returns = [offer.ret for offer in settled]
    evs = [offer.offer.ev for offer in settled]
    wins = [offer.won for offer in settled]
    roi, low, high = cluster_bootstrap_ci(settled, samples=samples, seed=seed)
    sd = statistics.stdev(returns) if len(returns) > 1 else math.nan
    avg_ev = mean(evs)
    return MatrixRow(
        sport=sport,
        market=market,
        horizon=horizon,
        threshold=threshold,
        evaluated=len(offers),
        bets=len(settled),
        win_rate=mean([1.0 if won else 0.0 for won in wins]),
        avg_ev=avg_ev,
        roi=roi,
        ci_low=low,
        ci_high=high,
        z=z_score(returns),
        per_bet_sd=sd,
        n_95_avg_ev=required_n(sd, avg_ev, prospective_power=False),
        n_80_avg_ev=required_n(sd, avg_ev, prospective_power=True),
        n_95_5pct=required_n(sd, 0.05, prospective_power=False),
        n_80_5pct=required_n(sd, 0.05, prospective_power=True),
        significant=bool(math.isfinite(low) and low > 0.0),
    )


def build_matrix(
    offers: Sequence[EvaluatedOffer], *, thresholds: Sequence[float], samples: int
) -> list[MatrixRow]:
    rows: list[MatrixRow] = []
    by_stratum: dict[tuple[str, str, float], list[EvaluatedOffer]] = defaultdict(list)
    for offer in offers:
        by_stratum[(offer.sport, offer.market, offer.horizon_hours)].append(offer)

    for threshold in thresholds:
        threshold_seed = int(threshold * 1_000_000)
        for (sport, market, horizon), group in sorted(by_stratum.items()):
            rows.append(
                summarize_group(
                    sport=sport,
                    market=market,
                    horizon=f"{horizon:g}h",
                    offers=group,
                    threshold=threshold,
                    samples=samples,
                    seed=1000 + threshold_seed + int(horizon * 10),
                )
            )
        rows.append(
            summarize_group(
                sport="ALL",
                market="all",
                horizon="all",
                offers=offers,
                threshold=threshold,
                samples=samples,
                seed=2000 + threshold_seed,
            )
        )
    return rows


def fmt_pct(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:+.2%}"


def fmt_n(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return str(math.ceil(value))


def print_matrix(rows: Sequence[MatrixRow]) -> None:
    print(
        f"{'sport':>5} {'market':>6} {'horizon':>7} {'edge':>7} {'eval':>6} "
        f"{'bets':>5} {'win%':>7} {'avgEV':>8} {'ROI':>8} {'95% CI':>23} {'sig':>3}"
    )
    print("-" * 103)
    for row in rows:
        win = "n/a" if not math.isfinite(row.win_rate) else f"{row.win_rate:.1%}"
        ci = f"[{fmt_pct(row.ci_low)}, {fmt_pct(row.ci_high)}]"
        print(
            f"{row.sport:>5} {row.market:>6} {row.horizon:>7} {row.threshold:7.1%} "
            f"{row.evaluated:6d} {row.bets:5d} {win:>7} {fmt_pct(row.avg_ev):>8} "
            f"{fmt_pct(row.roi):>8} {ci:>23} {row.significant!s:>3}"
        )


def write_matrix(path: Path, rows: Sequence[MatrixRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [field.name for field in MatrixRow.__dataclass_fields__.values()]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", default=list(DEFAULT_INPUTS), help="SPORT:MARKET:path")
    parser.add_argument("--thresholds", nargs="+", default=["0.005", "0.01", "0.02"])
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    parser.add_argument("--output", type=Path, default=Path("output/pinnacle_ev_matrix/matrix.csv"))
    args = parser.parse_args()

    if args.bootstrap_samples < 0:
        raise SystemExit("--bootstrap-samples must be non-negative")

    thresholds = parse_thresholds(args.thresholds)
    offers = read_evaluated(args.input)
    rows = build_matrix(offers, thresholds=thresholds, samples=args.bootstrap_samples)
    print_matrix(rows)
    write_matrix(args.output, rows)
    print(f"\nwrote matrix -> {args.output}")


if __name__ == "__main__":
    main()
