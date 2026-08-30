"""Daily all-games simulation board card."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from math import ceil
from pathlib import Path
from zoneinfo import ZoneInfo

from mlb.live.card_html import (
    HtmlCardRenderer,
    _brand_footer_html,
    circular_headshot_png,
    team_logo_data_url,
)

WIDE_COLUMNS = 2
NARROW_MAX_ROWS = 4
BOARD_W_WIDE = 1200
BOARD_W_NARROW = 900
BOARD_PADDING_X = 32
BOARD_PADDING_Y = 24
HEADER_HEIGHT = 146
FOOTER_HEIGHT = 52
ROW_HEIGHT_WIDE = 132
ROW_HEIGHT_NARROW = 236
GRID_GAP_X = 20
GRID_GAP_Y = 14
ACCENT = "#F59E0B"
AWAY_COLOR = "#38BDF8"
HOME_COLOR = "#F59E0B"
EASTERN = ZoneInfo("America/New_York")



def board_columns(row_count: int) -> int:
    return 1 if max(row_count, 1) <= NARROW_MAX_ROWS else WIDE_COLUMNS


def board_width(row_count: int) -> int:
    return BOARD_W_NARROW if board_columns(row_count) == 1 else BOARD_W_WIDE


def board_row_height(row_count: int) -> int:
    return ROW_HEIGHT_NARROW if board_columns(row_count) == 1 else ROW_HEIGHT_WIDE


@dataclass(frozen=True)
class SlateSimRow:
    game_pk: int
    away_abbrev: str
    home_abbrev: str
    away_team_id: int | None
    home_team_id: int | None
    away_starter: str
    home_starter: str
    away_starter_id: int | None
    home_starter_id: int | None
    game_time: str | None
    venue: str | None
    home_win_probability: float
    mean_away_runs: float
    mean_home_runs: float


@dataclass(frozen=True)
class SlateSimBoardData:
    slate_date: str
    generated_at: str
    games_summary: str
    n_sims: int
    rows: list[SlateSimRow]
    note: str | None = None


def board_height(row_count: int) -> int:
    rendered_rows = max(row_count, 1)
    columns = board_columns(rendered_rows)
    grid_rows = ceil(rendered_rows / columns)
    return (
        HEADER_HEIGHT
        + FOOTER_HEIGHT
        + BOARD_PADDING_Y * 2
        + grid_rows * board_row_height(rendered_rows)
        + max(grid_rows - 1, 0) * GRID_GAP_Y
    )

def _logo_html(team_id: int | None, abbrev: str) -> str:
    logo = team_logo_data_url(team_id)
    if logo:
        return f'<img class="team-logo" src="{logo}" alt="{abbrev}" />'
    return ""


def _time_label(raw: str | None) -> str:
    if not raw:
        return "TBD"
    try:
        parsed = datetime.fromisoformat(raw)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(EASTERN).strftime("%I:%M %p ET").lstrip("0")
    except ValueError:
        return raw


@lru_cache(maxsize=256)
def _starter_headshot(player_id: int | None) -> str | None:
    return circular_headshot_png(player_id, size=34)


def _initials(name: str) -> str:
    parts = [part for part in name.split() if part]
    return "".join(part[0].upper() for part in parts[:2]) or "?"


def _starter_html(name: str, player_id: int | None) -> str:
    headshot = _starter_headshot(player_id)
    if headshot:
        portrait = (
            f'<img class="starter-headshot" src="data:image/png;base64,{headshot}" '
            f'alt="{name}" />'
        )
    else:
        portrait = (
            f'<span class="starter-headshot starter-fallback">{_initials(name)}</span>'
        )
    return (
        '<span class="starter-entry">'
        f"{portrait}<span class=\"starter-name\">{name}</span>"
        "</span>"
    )


def _row_html(row: SlateSimRow) -> str:
    away_p = 1.0 - row.home_win_probability
    home_p = row.home_win_probability
    venue = f" · {row.venue}" if row.venue else ""
    starters = (
        f"{_starter_html(row.away_starter, row.away_starter_id)}"
        '<span class="starter-vs">vs</span>'
        f"{_starter_html(row.home_starter, row.home_starter_id)}"
    )
    return f"""<div class=\"matchup\">
  <div class=\"matchup-top\">
    <div class=\"teams\">
      <span class=\"team away\">{_logo_html(row.away_team_id, row.away_abbrev)}<b>{row.away_abbrev}</b></span>
      <span class=\"at\">@</span>
      <span class=\"team home\">{_logo_html(row.home_team_id, row.home_abbrev)}<b>{row.home_abbrev}</b></span>
    </div>
    <div class=\"meta\">{_time_label(row.game_time)}{venue}</div>
  </div>
  <div class=\"prob-head\">
    <span class=\"away-pct\">{away_p * 100:.0f}%</span>
    <span class=\"home-pct\">{home_p * 100:.0f}%</span>
  </div>
  <div class=\"prob-bar\">
    <div class=\"away-fill\" style=\"width:{away_p * 100:.2f}%\"></div>
    <div class=\"home-fill\" style=\"width:{home_p * 100:.2f}%\"></div>
  </div>
  <div class=\"proj\">Projected score <b>{row.away_abbrev} {row.mean_away_runs:.1f}</b> &ndash; <b>{row.home_abbrev} {row.mean_home_runs:.1f}</b></div>
  <div class=\"starters\">{starters}</div>
</div>"""


def build_slate_sim_card_html(data: SlateSimBoardData) -> str:
    row_count = len(data.rows)
    width = board_width(row_count)
    height = board_height(row_count)
    columns = board_columns(row_count)
    row_height = board_row_height(row_count)
    team_font = 30 if columns == 1 else 23
    meta_font = 13.5 if columns == 1 else 11.5
    prob_font = 42 if columns == 1 else 30
    proj_font = 18 if columns == 1 else 15
    starter_font = 15.5 if columns == 1 else 13
    note_html = f'<div class="note">{data.note}</div>' if data.note else ""
    rows_html = "\n".join(_row_html(row) for row in data.rows)
    if not rows_html:
        rows_html = '<div class="empty">No preview games available for this slate.</div>'
    return f"""<!DOCTYPE html>
<html><head><meta charset=\"utf-8\"><style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  html, body {{ width: {width}px; height: {height}px; }}
  body {{
    font-family: -apple-system, 'SF Pro Display', 'Helvetica Neue', 'Segoe UI', Roboto, sans-serif;
    background: radial-gradient(120% 140% at 18% 0%, #16283F 0%, #0B1622 58%, #081019 100%);
    color: #E8EEF6;
    overflow: hidden;
    font-variant-numeric: tabular-nums;
  }}
  .board {{ padding: {BOARD_PADDING_Y}px {BOARD_PADDING_X}px; height: 100%; display: flex; flex-direction: column; }}
  .topbar {{ display: flex; align-items: baseline; justify-content: space-between; gap: 24px; }}
  .title {{ font-size: 34px; font-weight: 900; letter-spacing: 0.4px; }}
  .title b {{ color: {ACCENT}; }}
  .subtitle {{ margin-top: 10px; color: #8CA0B8; font-size: 14px; letter-spacing: 1.2px; text-transform: uppercase; }}
  .note {{ margin-top: 10px; color: #E8EEF6; font-size: 14px; line-height: 1.35; max-width: 860px; }}
  .stats {{ text-align: right; }}
  .chip {{ display: inline-block; background: #16283F; border: 1px solid #24405F;
    border-radius: 999px; padding: 6px 16px; font-size: 16px; font-weight: 800;
    letter-spacing: 1.1px; color: #E8EEF6; }}
  .chip b {{ color: {ACCENT}; }}
  .stats .subline {{ margin-top: 8px; color: #8CA0B8; font-size: 13px; letter-spacing: 1px; }}
  .rule {{ height: 2px; margin: 18px 0 18px;
           background: linear-gradient(90deg, {ACCENT} 0%, #24405F 42%, transparent 100%); }}
  .grid {{ display: grid; grid-template-columns: repeat({columns}, minmax(0, 1fr));
           gap: {GRID_GAP_Y}px {GRID_GAP_X}px; flex: 1; align-content: start; }}
  .matchup {{
    min-height: {row_height}px;
    border-radius: 18px;
    border: 1px solid #24405F;
    background: linear-gradient(180deg, rgba(22,40,63,0.92) 0%, rgba(8,16,25,0.96) 100%);
    padding: 18px 20px 16px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02);
  }}
  .matchup-top {{ display: flex; justify-content: space-between; gap: 12px; align-items: baseline; }}
  .teams {{ display: flex; align-items: center; gap: 10px; font-size: {team_font}px; font-weight: 900; letter-spacing: 0.3px; }}
  .team {{ display: inline-flex; align-items: center; gap: 8px; }}
  .team-logo {{ width: 25px; height: 25px; object-fit: contain; display: block; }}
  .at {{ color: #55708F; font-size: 18px; font-weight: 700; }}
  .meta {{ color: #8CA0B8; font-size: {meta_font}px; letter-spacing: 0.8px; text-align: right; }}
  .prob-head {{ display: flex; justify-content: space-between; align-items: baseline; margin-top: 14px; }}
  .away-pct, .home-pct {{ font-size: {prob_font}px; font-weight: 900; }}
  .away-pct {{ color: {AWAY_COLOR}; }}
  .home-pct {{ color: {HOME_COLOR}; }}
  .prob-bar {{ display: flex; height: 14px; margin-top: 8px; border-radius: 999px; overflow: hidden; border: 1px solid #24405F; background: #16283F; }}
  .away-fill {{ background: linear-gradient(90deg, #0C4A6E, {AWAY_COLOR}); }}
  .home-fill {{ background: linear-gradient(90deg, {HOME_COLOR}, #7C4A03); }}
  .proj {{ margin-top: 10px; font-size: {proj_font}px; color: #E8EEF6; }}
  .proj b {{ color: #F8FAFC; }}
  .starters {{ margin-top: 10px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; color: #8CA0B8; }}
  .starter-entry {{ display: inline-flex; align-items: center; gap: 8px; min-width: 0; }}
  .starter-headshot {{ width: 34px; height: 34px; border-radius: 50%; object-fit: cover; border: 1px solid #24405F; flex: 0 0 auto; }}
  .starter-fallback {{ display: inline-flex; align-items: center; justify-content: center; background: #16283F; color: #E8EEF6; font-size: 11px; font-weight: 800; letter-spacing: 0.7px; }}
  .starter-name {{ font-size: {starter_font}px; line-height: 1.2; color: #8CA0B8; }}
  .starter-vs {{ font-size: 12px; letter-spacing: 1.2px; text-transform: uppercase; color: #55708F; }}
  .empty {{
    grid-column: 1 / -1;
    border: 1px solid #24405F;
    border-radius: 18px;
    padding: 24px;
    color: #8CA0B8;
    text-align: center;
    font-size: 18px;
  }}
  .footer {{ display: flex; align-items: center; justify-content: space-between; min-height: {FOOTER_HEIGHT}px; margin-top: 18px; gap: 16px; }}
  .brand {{ font-size: 14px; font-weight: 800; letter-spacing: 4px; color: #8CA0B8; }}
  .brand b {{ color: {ACCENT}; }}
  .brand-logo {{ height: 30px; width: auto; display: block; opacity: 0.98; }}
  .footer-note {{ color: #55708F; font-size: 12px; letter-spacing: 1px; text-transform: uppercase; margin-left: auto; text-align: right; }}
</style></head><body><div class="board">
  <div class="topbar">
    <div>
      <div class="title">MLB <b>DAILY SIM BOARD</b></div>
      <div class="subtitle">{data.slate_date} &middot; {data.games_summary} &middot; generated {data.generated_at}</div>
      {note_html}
    </div>
    <div class="stats">
      <span class="chip">GAME <b>SIMULATIONS</b></span>
      <div class="subline">{data.n_sims:,} Monte Carlo sims per game</div>
    </div>
  </div>
  <div class="rule"></div>
  <div class="grid">{rows_html}</div>
  <div class="footer">
    {_brand_footer_html()}
    <span class="footer-note">Team-strength win odds · pitch-model score projections</span>
  </div>
</div></body></html>"""


def render_slate_sim_card(
    data: SlateSimBoardData,
    out_path: Path,
    *,
    renderer: HtmlCardRenderer | None = None,
) -> Path:
    """Render one board page; pass ``renderer`` to reuse a warm browser."""

    def _render(active: HtmlCardRenderer) -> Path:
        return active.render_with_size(
            build_slate_sim_card_html(data),
            out_path,
            width=board_width(len(data.rows)),
            height=board_height(len(data.rows)),
        )

    if renderer is not None:
        return _render(renderer)
    with HtmlCardRenderer() as owned:
        return _render(owned)
