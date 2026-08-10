"""Game simulation result card: win probability, projected scores, run totals.

Same dark-card look and rendering stack as the pitch prediction card
(`src/live/card_html.py`); consumes Monte Carlo `GameResult` lists from
`src.sim.game`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from src.live.card_html import (
    CARD_H,
    CARD_W,
    HtmlCardRenderer,
    _brand_footer_html,
    team_logo_data_url,
)
from src.sim.game import GameResult

ACCENT = "#F59E0B"
AWAY_COLOR = "#38BDF8"
HOME_COLOR = "#F59E0B"

TOTAL_BUCKETS: list[tuple[str, int, int]] = [
    ("≤ 5", 0, 5),
    ("6 – 7", 6, 7),
    ("8 – 9", 8, 9),
    ("10 – 11", 10, 11),
    ("12+", 12, 999),
]


@dataclass(frozen=True)
class GameSimCardData:
    away_abbrev: str
    home_abbrev: str
    away_team_id: int | None
    home_team_id: int | None
    away_starter: str
    home_starter: str
    game_date: str
    venue: str | None
    n_sims: int
    home_win_probability: float
    mean_away_runs: float
    mean_home_runs: float
    # (away_runs, home_runs, probability), most likely first
    top_scores: list[tuple[int, int, float]]
    # (bucket label, probability)
    total_runs: list[tuple[str, float]]


def card_data_from_results(
    results: list[GameResult],
    *,
    away_abbrev: str,
    home_abbrev: str,
    away_team_id: int | None,
    home_team_id: int | None,
    away_starter: str,
    home_starter: str,
    game_date: str,
    venue: str | None,
    top_n_scores: int = 5,
    home_win_probability: float | None = None,
) -> GameSimCardData:
    """Aggregate Monte Carlo results into presentation data."""
    if not results:
        raise ValueError("No simulation results")
    n = len(results)
    home_wins = sum(1.0 for r in results if not r.tie and r.home_won)
    home_wins += 0.5 * sum(1 for r in results if r.tie)

    score_counts = Counter((r.away_runs, r.home_runs) for r in results)
    top_scores = [
        (away, home, count / n)
        for (away, home), count in score_counts.most_common(top_n_scores)
    ]

    totals = Counter(r.away_runs + r.home_runs for r in results)
    total_rows = []
    for label, lo, hi in TOTAL_BUCKETS:
        p = sum(count for total, count in totals.items() if lo <= total <= hi) / n
        total_rows.append((label, p))

    return GameSimCardData(
        away_abbrev=away_abbrev,
        home_abbrev=home_abbrev,
        away_team_id=away_team_id,
        home_team_id=home_team_id,
        away_starter=away_starter,
        home_starter=home_starter,
        game_date=game_date,
        venue=venue,
        n_sims=n,
        home_win_probability=(
            home_win_probability if home_win_probability is not None else home_wins / n
        ),
        mean_away_runs=sum(r.away_runs for r in results) / n,
        mean_home_runs=sum(r.home_runs for r in results) / n,
        top_scores=top_scores,
        total_runs=total_rows,
    )


def _logo_html(team_id: int | None, abbrev: str) -> str:
    logo = team_logo_data_url(team_id)
    if logo:
        return f'<img class="team-logo" src="{logo}" alt="{abbrev}" />'
    return ""


def _score_rows_html(data: GameSimCardData) -> str:
    rows = []
    max_p = max((p for _, _, p in data.top_scores), default=1.0) or 1.0
    for away, home, p in data.top_scores:
        home_won = home > away
        winner = data.home_abbrev if home_won else data.away_abbrev
        color = HOME_COLOR if home_won else AWAY_COLOR
        hi, lo = (home, away) if home_won else (away, home)
        rows.append(
            f"""<div class="prob-row">
  <div class="prob-label"><span><b style="color:{color}">{winner}</b>
    {hi}&ndash;{lo}</span><span>{p * 100:.1f}%</span></div>
  <div class="bar"><div class="fill" style="width:{p / max_p * 100:.1f}%; background:{color}"></div></div>
</div>"""
        )
    return "\n".join(rows)


def _totals_rows_html(data: GameSimCardData) -> str:
    rows = []
    max_p = max((p for _, p in data.total_runs), default=1.0) or 1.0
    for label, p in data.total_runs:
        rows.append(
            f"""<div class="prob-row">
  <div class="prob-label"><span>{label}</span><span>{p * 100:.1f}%</span></div>
  <div class="bar"><div class="fill" style="width:{p / max_p * 100:.1f}%"></div></div>
</div>"""
        )
    return "\n".join(rows)


def build_game_sim_card_html(data: GameSimCardData) -> str:
    away_p = 1.0 - data.home_win_probability
    home_p = data.home_win_probability
    venue_html = f" &middot; {data.venue}" if data.venue else ""
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  html, body {{ width: {CARD_W}px; height: {CARD_H}px; }}
  body {{
    font-family: -apple-system, 'SF Pro Display', 'Helvetica Neue', 'Segoe UI', Roboto, sans-serif;
    background: radial-gradient(120% 140% at 18% 0%, #16283F 0%, #0B1622 58%, #081019 100%);
    color: #E8EEF6;
    font-variant-numeric: tabular-nums;
    overflow: hidden;
  }}
  .card {{ display: flex; flex-direction: column; height: 100%; padding: 18px 34px 10px; }}
  .topbar {{ display: flex; align-items: baseline; justify-content: space-between; }}
  .scoreline {{ font-size: 30px; font-weight: 800; letter-spacing: 0.5px; }}
  .score {{ display: inline-flex; align-items: center; gap: 10px; }}
  .team-logo {{ width: 30px; height: 30px; object-fit: contain; display: block; }}
  .scoreline .at {{ color: #55708F; font-size: 24px; font-weight: 600; margin: 0 14px; }}
  .chip {{ display: inline-block; background: #16283F; border: 1px solid #24405F;
    border-radius: 999px; padding: 6px 16px; font-size: 16px; font-weight: 700;
    letter-spacing: 1.2px; color: #E8EEF6; }}
  .chip b {{ color: {ACCENT}; }}
  .subline {{ margin-top: 6px; font-size: 12.5px; color: #55708F; letter-spacing: 1px; text-align: right; }}
  .rule {{ height: 2px; margin: 10px 0 14px;
           background: linear-gradient(90deg, {ACCENT} 0%, #24405F 45%, transparent 100%); }}

  .winprob {{ margin: 4px 0 6px; }}
  .winprob-nums {{ display: flex; justify-content: space-between; align-items: baseline; }}
  .side {{ display: flex; align-items: baseline; gap: 12px; }}
  .side .pct {{ font-size: 58px; font-weight: 900; }}
  .side .team {{ font-size: 20px; font-weight: 800; color: #8CA0B8; letter-spacing: 1px; }}
  .away .pct {{ color: {AWAY_COLOR}; }}
  .home .pct {{ color: {HOME_COLOR}; }}
  .split {{ display: flex; height: 22px; border-radius: 11px; overflow: hidden;
            border: 1px solid #24405F; margin-top: 8px; }}
  .split .away-fill {{ background: linear-gradient(90deg, #0C4A6E, {AWAY_COLOR}); }}
  .split .home-fill {{ background: linear-gradient(90deg, {HOME_COLOR}, #7C4A03); }}
  .winprob-caption {{ display: flex; justify-content: space-between; margin-top: 6px;
    font-size: 12.5px; color: #55708F; letter-spacing: 1px; font-weight: 700; }}

  .body {{ display: flex; flex: 1; gap: 44px; margin-top: 10px; }}
  .col {{ flex: 1; display: flex; flex-direction: column; }}
  .section-label {{ font-size: 12px; font-weight: 800; letter-spacing: 3.3px;
                    color: #8CA0B8; margin: 10px 0 8px; }}
  .prob-row {{ margin-bottom: 9px; }}
  .prob-label {{ display: flex; justify-content: space-between; align-items: baseline;
                 font-size: 15.5px; font-weight: 700; margin-bottom: 4px; }}
  .prob-label span:last-child {{ color: #8CA0B8; }}
  .bar {{ height: 9px; border-radius: 5px; background: #16283F; overflow: hidden; }}
  .fill {{ height: 100%; border-radius: 5px; background: linear-gradient(90deg, #B45309, {ACCENT}); }}

  .meanline {{ font-size: 24px; font-weight: 800; margin: 2px 0 4px; }}
  .meanline .away-mean {{ color: {AWAY_COLOR}; }}
  .meanline .home-mean {{ color: {HOME_COLOR}; }}
  .meanline .dash {{ color: #55708F; margin: 0 8px; }}
  .starters {{ font-size: 14.5px; color: #8CA0B8; line-height: 1.5; }}
  .starters b {{ color: #E8EEF6; }}

  .footer {{ display: flex; align-items: center; justify-content: space-between;
             margin-top: 8px; min-height: 44px; }}
  .brand-logo {{ height: 34px; width: auto; display: block; opacity: 0.98; }}
  .footer-note {{ font-size: 12.5px; color: #55708F; letter-spacing: 1px; }}
</style></head><body><div class="card">
  <div class="topbar">
    <div class="scoreline">
      <span class="score">{_logo_html(data.away_team_id, data.away_abbrev)}{data.away_abbrev}</span>
      <span class="at">@</span>
      <span class="score">{_logo_html(data.home_team_id, data.home_abbrev)}{data.home_abbrev}</span>
    </div>
    <div class="gamestate">
      <span class="chip">GAME <b>SIMULATION</b></span>
      <div class="subline">{data.game_date}{venue_html} &middot; {data.n_sims:,} SIMS</div>
    </div>
  </div>
  <div class="rule"></div>

  <div class="winprob">
    <div class="winprob-nums">
      <div class="side away"><span class="pct">{away_p * 100:.0f}%</span><span class="team">{data.away_abbrev}</span></div>
      <div class="side home"><span class="team">{data.home_abbrev}</span><span class="pct">{home_p * 100:.0f}%</span></div>
    </div>
    <div class="split">
      <div class="away-fill" style="width:{away_p * 100:.2f}%"></div>
      <div class="home-fill" style="width:{home_p * 100:.2f}%"></div>
    </div>
    <div class="winprob-caption"><span>WIN PROBABILITY</span><span>MODEL CHAIN &middot; PITCH &rarr; PA &rarr; GAME</span></div>
  </div>

  <div class="body">
    <div class="col">
      <div class="section-label">MOST LIKELY FINALS</div>
      {_score_rows_html(data)}
    </div>
    <div class="col">
      <div class="section-label">TOTAL RUNS</div>
      {_totals_rows_html(data)}
      <div class="section-label">PROJECTED SCORE</div>
      <div class="meanline">
        <span class="away-mean">{data.away_abbrev} {data.mean_away_runs:.1f}</span>
        <span class="dash">&ndash;</span>
        <span class="home-mean">{data.home_abbrev} {data.mean_home_runs:.1f}</span>
      </div>
      <div class="starters">
        <b>{data.away_starter}</b> vs <b>{data.home_starter}</b>
      </div>
    </div>
  </div>

  <div class="footer">
    {_brand_footer_html()}
    <span class="footer-note">MONTE CARLO FROM LIVE PITCH &middot; OUTCOME MODELS</span>
  </div>
</div></body></html>"""


def render_game_sim_card(data: GameSimCardData, out_path: Path) -> Path:
    """Render the simulation card to an image; returns the written path."""
    renderer = HtmlCardRenderer()
    try:
        return renderer.render(build_game_sim_card_html(data), out_path)
    finally:
        renderer.close()
