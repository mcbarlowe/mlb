"""Is the season-to-season ROI spread larger than sampling noise?

2021 (+10.06%) and 2025 (+5.55%) look like successes next to 2020 (-8.61%) and 2022
(-6.35%). Before attributing that to a cause, test whether the spread is bigger than what
identical per-bet odds would produce by chance. Every per-season interval already contains
zero, so the null of one common true ROI is not obviously wrong.

Permutation test: pool all settled bets, reassign them to seasons at random while preserving
each season's bet count, and recompute the dispersion statistic. The p-value is the share of
permutations whose dispersion matches or exceeds the observed. Two statistics are reported
because they answer slightly different questions:

  range     - max minus min season ROI, sensitive to one outlying season
  weighted sd - bet-count-weighted dispersion around the pooled ROI, sensitive to overall
                scatter

If neither is significant there is nothing to explain, and constructing an explanation would
be fitting a story to noise.

    uv run python scripts/test_season_heterogeneity.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.backtest_moneyline import load_finals, walkforward_home_probs
from scripts.backtest_moneyline_lineshop import PANEL_PRIORITY, load_quotes, run

SEASONS = (2020, 2021, 2022, 2023, 2024, 2025)
PERMUTATIONS = 20000


def roi(bets: list[tuple[float, float]]) -> float:
    return sum(p for _, p in bets) / sum(s for s, _ in bets)


def dispersion(groups: list[list[tuple[float, float]]], pooled: float):
    rois = [roi(g) for g in groups]
    counts = [len(g) for g in groups]
    total = sum(counts)
    wsd = float(
        np.sqrt(sum(n * (r - pooled) ** 2 for n, r in zip(counts, rois, strict=True)) / total)
    )
    return max(rois) - min(rois), wsd


def main() -> None:
    parser_edge = 0.05
    panel = PANEL_PRIORITY[:5]

    per_season: dict[int, list[tuple[float, float]]] = {}
    for season in SEASONS:
        probs = walkforward_home_probs(season, list(range(2015, season))).set_index(
            "game_pk"
        )["model_prob_home"]
        finals = load_finals([season]).set_index("game_pk")
        quotes = load_quotes(season, panel, "close", "proportional")
        res, _ = run(quotes, probs, finals, parser_edge, "flat", 0.25, 0.05)
        per_season[season] = res.settled

    groups = [per_season[s] for s in SEASONS]
    pool = [b for g in groups for b in g]
    pooled = roi(pool)
    obs_range, obs_wsd = dispersion(groups, pooled)

    print(f"Edge >= {parser_edge:.0%}, bet at close, panel of 5. Pooled ROI {pooled:+.2%}")
    print()
    print(f"{'Season':>6} | {'Bets':>5} | {'ROI':>8} | {'vs pooled':>9}")
    print("-" * 38)
    for s in SEASONS:
        r = roi(per_season[s])
        print(f"{s:6d} | {len(per_season[s]):5d} | {r:+7.2%} | {r - pooled:+8.2%}")
    print("-" * 38)
    print(f"observed range {obs_range:.2%}, weighted sd {obs_wsd:.2%}")
    print()

    # Expected scatter if every season shares one true ROI.
    sd_bet = float(np.std([p for _, p in pool], ddof=1))
    print("Expected under one common true ROI:")
    for s in SEASONS:
        n = len(per_season[s])
        print(f"  {s}: SE = {sd_bet / np.sqrt(n):.2%} on {n} bets")
    print()

    rng = random.Random(20260816)
    sizes = [len(g) for g in groups]
    ge_range = ge_wsd = 0
    for _ in range(PERMUTATIONS):
        shuffled = pool[:]
        rng.shuffle(shuffled)
        cut, perm = 0, []
        for n in sizes:
            perm.append(shuffled[cut : cut + n])
            cut += n
        r, w = dispersion(perm, pooled)
        ge_range += r >= obs_range
        ge_wsd += w >= obs_wsd

    p_range = (ge_range + 1) / (PERMUTATIONS + 1)
    p_wsd = (ge_wsd + 1) / (PERMUTATIONS + 1)
    print(f"Permutation test, {PERMUTATIONS:,} draws:")
    print(f"  range      p = {p_range:.3f}")
    print(f"  weighted sd p = {p_wsd:.3f}")
    print()
    if min(p_range, p_wsd) > 0.05:
        print("Neither statistic is significant. The season-to-season spread is consistent")
        print("with one common true ROI and per-bet payout variance. There is no seasonal")
        print("effect to explain, and any narrative fitted to 2021/2025 would be fitting")
        print("noise. Regime changes (2020 COVID schedule, 2022 universal DH, 2023 pitch")
        print("clock and shift ban) remain plausible a priori but are not detectable here.")
    else:
        print("At least one statistic is significant: the spread exceeds chance, so a")
        print("seasonal cause is worth pursuing. Candidate regime changes to test:")
        print("  2020 60-game schedule, empty parks, extra-innings runner rule")
        print("  2021 mid-season foreign-substance enforcement")
        print("  2022 universal DH, lockout-compressed spring")
        print("  2023 pitch clock, shift restriction, larger bases")


if __name__ == "__main__":
    main()
