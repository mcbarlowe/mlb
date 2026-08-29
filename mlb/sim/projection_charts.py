from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape
from math import ceil
from pathlib import Path

from mlb.sim.season import SeasonProjection, TeamInfo, TeamProjection

CARD_W = 1200
CARD_H = 1500
ACCENT = "#F59E0B"
CYAN = "#38BDF8"

STAGE_COLUMNS = (
    ("playoff_prob", "PO"),
    ("division_series_prob", "DS"),
    ("league_championship_prob", "LCS"),
    ("world_series_prob", "WS"),
    ("championship_prob", "CH"),
)


def write_projection_graphics(
    projection: SeasonProjection,
    teams: Mapping[int, TeamInfo],
    output_dir: Path,
    *,
    projection_type: str = "model",
) -> tuple[Path, Path]:
    from mlb.live.card_html import HtmlCardRenderer

    output_dir.mkdir(parents=True, exist_ok=True)
    file_stem = f"season_{projection.season}_{projection_type}"
    playoff_path = output_dir / f"{file_stem}_playoff_probabilities.jpg"
    stages_path = output_dir / f"{file_stem}_playoff_stages.jpg"

    renderer = HtmlCardRenderer()
    try:
        playoff_path = renderer.render_with_size(
            _playoff_probabilities_html(
                projection,
                teams,
                projection_type=projection_type,
            ),
            playoff_path,
            width=CARD_W,
            height=CARD_H,
        )
        stages_path = renderer.render_with_size(
            _playoff_stages_html(
                projection,
                teams,
                projection_type=projection_type,
            ),
            stages_path,
            width=CARD_W,
            height=CARD_H,
        )
    finally:
        renderer.close()

    return playoff_path, stages_path


def _playoff_probabilities_html(
    projection: SeasonProjection,
    teams: Mapping[int, TeamInfo],
    *,
    projection_type: str,
) -> str:
    columns = _split_columns(_ranked_team_rows(projection))
    column_html = "\n".join(
        f'<div class="column">{_odds_rows_html(column, teams)}</div>'
        for column in columns
    )
    return _card_html(
        projection,
        projection_type=projection_type,
        chip_text="PLAYOFF <b>ODDS</b>",
        footer_note="SEASON WIN TOTALS · MONTE CARLO PLAYOFF PATHS",
        body=f'<div class="odds-grid">{column_html}</div>',
    )


def _playoff_stages_html(
    projection: SeasonProjection,
    teams: Mapping[int, TeamInfo],
    *,
    projection_type: str,
) -> str:
    columns = _split_columns(_ranked_team_rows(projection))
    column_html = "\n".join(
        f'<div class="column stage-column">{_stage_rows_html(column, teams)}</div>'
        for column in columns
    )
    return _card_html(
        projection,
        projection_type=projection_type,
        chip_text="PLAYOFF <b>STAGES</b>",
        footer_note="PO · DS · LCS · WORLD SERIES · CHAMPION",
        body=f'<div class="stage-grid">{column_html}</div>',
    )


def _card_html(
    projection: SeasonProjection,
    *,
    projection_type: str,
    chip_text: str,
    footer_note: str,
    body: str,
) -> str:
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
  .card {{ height: 100%; padding: 34px 40px 20px; display: flex; flex-direction: column; }}
  .topbar {{ display: flex; justify-content: space-between; gap: 28px; align-items: flex-start; }}
  .title {{ font-size: 42px; font-weight: 900; letter-spacing: 0.5px; line-height: 1; }}
  .title b {{ color: {ACCENT}; }}
  .subtitle {{ margin-top: 14px; color: #8CA0B8; font-size: 18px; letter-spacing: 1.35px; text-transform: uppercase; line-height: 1.35; }}
  .chip {{ display: inline-block; background: #16283F; border: 1px solid #24405F;
    border-radius: 999px; padding: 10px 26px; font-size: 19px; font-weight: 900;
    letter-spacing: 1.2px; color: #E8EEF6; text-align: center; box-shadow: inset 0 0 0 1px rgba(255,255,255,0.03); }}
  .chip b {{ color: {ACCENT}; }}
  .subline {{ margin-top: 10px; font-size: 14px; color: #8CA0B8; letter-spacing: 1px; text-align: right; }}
  .rule {{ height: 2px; margin: 24px 0 22px;
           background: linear-gradient(90deg, {ACCENT} 0%, #24405F 45%, transparent 100%); }}
  .section-label {{ font-size: 12px; font-weight: 900; letter-spacing: 3.2px; color: #8CA0B8; margin-bottom: 10px; }}
  .odds-grid, .stage-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; flex: 1; align-content: start; }}
  .column {{ display: flex; flex-direction: column; gap: 8px; min-width: 0; }}
  .odds-row {{
    min-height: 66px;
    border-radius: 14px;
    border: 1px solid #24405F;
    background: linear-gradient(180deg, rgba(22,40,63,0.92) 0%, rgba(8,16,25,0.96) 100%);
    padding: 9px 12px 10px;
    box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02);
  }}
  .odds-head {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; }}
  .team {{ display: flex; align-items: center; min-width: 0; gap: 9px; }}
  .team-logo {{ width: 26px; height: 26px; object-fit: contain; display: block; flex: 0 0 auto; }}
  .abbrev {{ font-size: 22px; font-weight: 900; letter-spacing: 0.5px; color: #F8FAFC; }}
  .wins {{ color: #8CA0B8; font-size: 13px; font-weight: 800; letter-spacing: 0.9px; text-transform: uppercase; }}
  .odds-value {{ color: {CYAN}; font-size: 28px; font-weight: 900; }}
  .track {{ height: 9px; border-radius: 999px; background: #16283F; overflow: hidden; margin-top: 8px; border: 1px solid #24405F; }}
  .fill {{ height: 100%; border-radius: 999px; background: linear-gradient(90deg, #0C4A6E, {CYAN}); }}
  .stage-header, .stage-row {{
    display: grid;
    grid-template-columns: minmax(130px, 1.45fr) repeat(5, minmax(44px, 0.7fr));
    gap: 5px;
    align-items: center;
  }}
  .stage-header {{ color: #8CA0B8; font-size: 10.5px; font-weight: 900; letter-spacing: 1.6px; text-align: center; margin-bottom: 3px; }}
  .stage-header span:first-child {{ text-align: left; }}
  .stage-row {{
    min-height: 58px;
    border-radius: 13px;
    border: 1px solid #24405F;
    background: linear-gradient(180deg, rgba(22,40,63,0.92) 0%, rgba(8,16,25,0.96) 100%);
    padding: 7px;
    box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02);
  }}
  .stage-team {{ display: flex; align-items: center; gap: 8px; min-width: 0; }}
  .stage-team .abbrev {{ font-size: 17px; }}
  .stage-team .wins {{ font-size: 10px; }}
  .stage-cell {{
    height: 44px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #E8EEF6;
    font-size: 17px;
    font-weight: 900;
    background: linear-gradient(180deg, rgba(56,189,248,var(--a)) 0%, rgba(8,16,25,0.92) 100%);
    border: 1px solid rgba(56, 112, 143, 0.45);
  }}
  .footer {{ display: flex; align-items: center; justify-content: space-between; min-height: 54px; margin-top: 18px; gap: 18px; }}
  .brand {{ font-size: 15px; font-weight: 800; letter-spacing: 4px; color: #8CA0B8; }}
  .brand b {{ color: {ACCENT}; }}
  .brand-logo {{ height: 34px; width: auto; display: block; opacity: 0.98; }}
  .footer-note {{ color: #55708F; font-size: 13px; letter-spacing: 1.2px; text-transform: uppercase; text-align: right; }}
</style></head><body><div class="card">
  <div class="topbar">
    <div>
      <div class="title">MLB <b>SEASON PROJECTION</b></div>
      <div class="subtitle">{projection.season} &middot; {escape(projection_type.upper())} &middot; AS OF {projection.as_of_date.isoformat()}</div>
    </div>
    <div>
      <span class="chip">{chip_text}</span>
      <div class="subline">{projection.trials:,} Monte Carlo seasons</div>
    </div>
  </div>
  <div class="rule"></div>
  {body}
  <div class="footer">
    {_brand_footer_html()}
    <span class="footer-note">{footer_note}</span>
  </div>
</div></body></html>"""


def _odds_rows_html(
    rows: Sequence[TeamProjection],
    teams: Mapping[int, TeamInfo],
) -> str:
    return "\n".join(
        f"""<div class="odds-row">
  <div class="odds-head">
    {_team_html(row, teams)}
    <span class="odds-value">{row.playoff_prob * 100:.0f}%</span>
  </div>
  <div class="track"><div class="fill" style="width:{max(row.playoff_prob * 100, 1.5):.1f}%"></div></div>
</div>"""
        for row in rows
    )


def _stage_rows_html(
    rows: Sequence[TeamProjection],
    teams: Mapping[int, TeamInfo],
) -> str:
    header = (
        '<div class="stage-header"><span>TEAM</span>'
        + "".join(f"<span>{label}</span>" for _field, label in STAGE_COLUMNS)
        + "</div>"
    )
    body = "\n".join(
        f"""<div class="stage-row">
  {_stage_team_html(row, teams)}
  {''.join(_stage_cell(float(getattr(row, field))) for field, _label in STAGE_COLUMNS)}
</div>"""
        for row in rows
    )
    return header + body


def _team_html(row: TeamProjection, teams: Mapping[int, TeamInfo]) -> str:
    team = teams[row.team_id]
    return (
        '<div class="team">'
        f"{_logo_html(team)}"
        f'<span class="abbrev">{escape(team.abbreviation)}</span>'
        f'<span class="wins">{row.expected_wins:.1f} W</span>'
        "</div>"
    )


def _stage_team_html(row: TeamProjection, teams: Mapping[int, TeamInfo]) -> str:
    team = teams[row.team_id]
    return (
        '<div class="stage-team">'
        f"{_logo_html(team)}"
        f'<span class="abbrev">{escape(team.abbreviation)}</span>'
        f'<span class="wins">{row.expected_wins:.1f}W</span>'
        "</div>"
    )


def _stage_cell(probability: float) -> str:
    alpha = min(max(0.12 + probability * 0.88, 0.12), 1.0)
    return (
        f'<span class="stage-cell" style="--a:{alpha:.3f}">'
        f"{probability * 100:.0f}%</span>"
    )


def _logo_html(team: TeamInfo) -> str:
    if team.team_id < 100:
        return ""
    from mlb.live.card_html import team_logo_data_url

    logo = team_logo_data_url(team.team_id)
    if logo:
        return (
            f'<img class="team-logo" src="{logo}" alt="{escape(team.abbreviation)}" />'
        )
    return ""


def _brand_footer_html() -> str:
    from mlb.live.card_html import _brand_footer_html as brand_footer_html

    return brand_footer_html()


def _split_columns(
    rows: Sequence[TeamProjection],
) -> tuple[Sequence[TeamProjection], ...]:
    midpoint = ceil(len(rows) / 2)
    return rows[:midpoint], rows[midpoint:]


def _ranked_team_rows(projection: SeasonProjection) -> list[TeamProjection]:
    return sorted(
        projection.teams,
        key=lambda row: (
            row.playoff_prob,
            row.division_series_prob,
            row.league_championship_prob,
            row.expected_wins,
        ),
        reverse=True,
    )
