"""Broadcast-style pitch card rendered with HTML/CSS in headless Chromium.

Matplotlib caps out on typography and finish; this renderer builds the
card as a dark-theme HTML page (system SF/Helvetica stack, CSS gradients,
glow-composited density layer) and screenshots it with Playwright at 2x
as a fast social-ready JPEG.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from src.ml.pitch_predictor import (
    PITCH_TYPE_FULL_NAMES,
    GameContext,
    PitchPrediction,
    fetch_mlb_headshot,
)

CARD_W = 1200
CARD_H = 675
SCALE = 2

# Zone panel geometry (CSS px), fixed square scale in px per foot.
PX_PER_FT = 100.0
PANEL_X_RANGE = (-2.1, 2.1)
PANEL_Z_RANGE = (0.55, 4.45)
PANEL_W = int((PANEL_X_RANGE[1] - PANEL_X_RANGE[0]) * PX_PER_FT)  # 462
PANEL_H = int((PANEL_Z_RANGE[1] - PANEL_Z_RANGE[0]) * PX_PER_FT)  # 429

ZONE_HALF_W_FT = (17 / 12) / 2
ZONE_BOTTOM_FT = 1.5
ZONE_TOP_FT = 3.5

# Glow colormap stops: (normalized density, R, G, B, A)
_GLOW_STOPS = [
    (0.00, 255, 171, 64, 0),
    (0.10, 255, 171, 64, 40),
    (0.30, 255, 138, 42, 110),
    (0.55, 244, 81, 30, 185),
    (0.80, 211, 47, 47, 235),
    (1.00, 255, 214, 170, 255),
]


def _panel_xy(px_ft: float, pz_ft: float) -> tuple[float, float]:
    """Map (px, pz) in feet (pitcher's view, x already flipped) to panel px."""
    x = (px_ft - PANEL_X_RANGE[0]) * PX_PER_FT
    y = (PANEL_Z_RANGE[1] - pz_ft) * PX_PER_FT
    return x, y


def density_glow_png(prediction: PitchPrediction) -> str:
    """Build the base64 PNG glow layer for the location density."""
    density = np.asarray(prediction.location_density, dtype=np.float64)
    px_grid = np.asarray(prediction.px_grid)
    pz_grid = np.asarray(prediction.pz_grid)

    peak = density.max()
    normalized = density / peak if peak > 0 else density

    # Crop the model grid to the panel extent (nearest-cell is < 0.05 ft off).
    x_keep = (px_grid >= PANEL_X_RANGE[0]) & (px_grid <= PANEL_X_RANGE[1])
    z_keep = (pz_grid >= PANEL_Z_RANGE[0]) & (pz_grid <= PANEL_Z_RANGE[1])
    cropped = normalized[np.ix_(z_keep, x_keep)]

    # Pitcher's view: mirror horizontally. Image rows run top-down: flip z.
    cropped = np.flipud(cropped[:, ::-1])

    stops = np.array([s[0] for s in _GLOW_STOPS])
    rgba = np.zeros((*cropped.shape, 4), dtype=np.uint8)
    for channel in range(4):
        values = np.array([s[channel + 1] for s in _GLOW_STOPS])
        rgba[..., channel] = np.interp(cropped, stops, values).astype(np.uint8)

    image = Image.fromarray(rgba).resize(
        (PANEL_W * SCALE, PANEL_H * SCALE), Image.Resampling.BILINEAR
    )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def circular_headshot_png(player_id: int | None, size: int = 84) -> str | None:
    """Fetch a headshot and return it circle-cropped as base64 PNG."""
    if player_id is None:
        return None
    array = fetch_mlb_headshot(player_id, size=size * SCALE)
    if array is None:
        return None

    image = Image.fromarray(array).convert("RGBA")
    side = min(image.size)
    left = (image.width - side) // 2
    top = (image.height - side) // 2
    image = image.crop((left, top, left + side, top + side))

    mask = Image.new("L", (side, side), 0)
    from PIL import ImageDraw

    ImageDraw.Draw(mask).ellipse((0, 0, side, side), fill=255)
    image.putalpha(mask)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


_TEAM_LOGO_CACHE: dict[int, str | None] = {}


def team_logo_data_url(team_id: int | None) -> str | None:
    """Fetch and cache a team logo as an inline SVG data URL."""
    if team_id is None:
        return None
    if team_id in _TEAM_LOGO_CACHE:
        return _TEAM_LOGO_CACHE[team_id]

    url = f"https://www.mlbstatic.com/team-logos/{team_id}.svg"
    try:
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=5) as response:
            image_data = response.read()
        data_url = (
            "data:image/svg+xml;base64,"
            + base64.b64encode(image_data).decode("ascii")
        )
        _TEAM_LOGO_CACHE[team_id] = data_url
        return data_url
    except (URLError, HTTPError, Exception):
        _TEAM_LOGO_CACHE[team_id] = None
        return None


_BRAND_LOGO_DATA_URL: str | None = None


def brand_logo_data_url() -> str | None:
    """Load the local Barlowe Analytics logo asset as an inline PNG."""
    global _BRAND_LOGO_DATA_URL
    if _BRAND_LOGO_DATA_URL is not None:
        return _BRAND_LOGO_DATA_URL

    asset = Path(__file__).with_name("assets") / "barlowe_analytics_logo.png"
    if not asset.exists():
        return None

    _BRAND_LOGO_DATA_URL = (
        "data:image/png;base64," + base64.b64encode(asset.read_bytes()).decode("ascii")
    )
    return _BRAND_LOGO_DATA_URL


def _brand_footer_html() -> str:
    logo = brand_logo_data_url()
    if logo:
        return f'<img class="brand-logo" src="{logo}" alt="Barlowe Analytics" />'
    return '<div class="brand">BARLOWE <b>ANALYTICS</b></div>'


def _score_team_html(
    abbreviation: str,
    score: int | None,
    team_id: int | None,
) -> str:
    logo = team_logo_data_url(team_id)
    score_html = f"<b> {score}</b>" if score is not None else ""
    if logo:
        return (
            f'<span class="score"><img class="team-logo" src="{logo}" alt="" />'
            f"{abbreviation}{score_html}</span>"
        )
    return f'<span class="score">{abbreviation}{score_html}</span>'


def _headshot_html(b64: str | None, initials: str) -> str:
    if b64:
        return f'<img class="headshot" src="data:image/png;base64,{b64}" alt="" />'
    return f'<div class="headshot headshot-fallback">{initials}</div>'


def _initials(name: str) -> str:
    parts = [part for part in name.split() if part]
    return "".join(part[0].upper() for part in parts[:2]) or "?"


def _probability_rows(prediction: PitchPrediction) -> str:
    order = np.argsort(prediction.type_probabilities)[::-1]
    from src.ml.features import PITCH_TYPE_CODES

    rows = []
    shown = 0
    for idx in order:
        prob = float(prediction.type_probabilities[idx])
        if prob <= 0.01 or shown >= 5:
            continue
        code = PITCH_TYPE_CODES[idx]
        name = PITCH_TYPE_FULL_NAMES.get(code, code)
        top = "top" if shown == 0 else ""
        rows.append(
            f'''<div class="prob-row {top}">
              <div class="prob-label"><span class="prob-name">{name}</span>
              <span class="prob-pct">{prob:.0%}</span></div>
              <div class="prob-track"><div class="prob-fill" style="width:{max(prob * 100, 1.5):.1f}%"></div></div>
            </div>'''
        )
        shown += 1
    return "\n".join(rows)


def _diamond_svg(context: GameContext) -> str:
    def base(cx: float, cy: float, occupied: bool) -> str:
        fill = "#F59E0B" if occupied else "none"
        stroke = "#F59E0B" if occupied else "#3D5876"
        glow = ' filter="url(#baseglow)"' if occupied else ""
        return (
            f'<rect x="{cx - 11}" y="{cy - 11}" width="22" height="22" rx="3.5" '
            f'transform="rotate(45 {cx} {cy})" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="2.5"{glow} />'
        )

    outs_dots = "".join(
        f'<circle cx="{78 + i * 26}" cy="133" r="7" '
        + (
            'fill="#EF4444" stroke="#B91C1C"'
            if i < context.outs
            else 'fill="none" stroke="#3D5876"'
        )
        + ' stroke-width="2" />'
        for i in range(3)
    )

    return f'''<svg width="200" height="150" viewBox="0 0 200 150">
      <defs>
        <filter id="baseglow" x="-60%" y="-60%" width="220%" height="220%">
          <feGaussianBlur stdDeviation="4" result="b"/>
          <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
      </defs>
      <path d="M 100 108 L 152 62 L 100 16 L 48 62 Z" fill="none" stroke="#3D5876" stroke-width="2.5"/>
      {base(152, 62, context.runner_on_1b)}
      {base(100, 16, context.runner_on_2b)}
      {base(48, 62, context.runner_on_3b)}
      {outs_dots}
      <text x="10" y="137" font-size="11" letter-spacing="2.5" fill="#8CA0B8" font-weight="700">OUTS</text>
    </svg>'''


def _zone_svg(
    prediction: PitchPrediction,
    actual_location: tuple[float, float] | None = None,
) -> str:
    zone_left, zone_top = _panel_xy(-ZONE_HALF_W_FT, ZONE_TOP_FT)
    zone_right, zone_bottom = _panel_xy(ZONE_HALF_W_FT, ZONE_BOTTOM_FT)
    zone_w = zone_right - zone_left
    zone_h = zone_bottom - zone_top

    plate_y = PANEL_H - 34
    plate_half = 0.708 * PX_PER_FT
    center_x = PANEL_W / 2
    plate = (
        f"M {center_x - plate_half} {plate_y} "
        f"L {center_x + plate_half} {plate_y} "
        f"L {center_x + plate_half} {plate_y + 10} "
        f"L {center_x} {plate_y + 24} "
        f"L {center_x - plate_half} {plate_y + 10} Z"
    )

    exp_x, exp_y = _panel_xy(-float(prediction.location_point[0]),
                             float(prediction.location_point[1]))

    thirds_x = [zone_left + zone_w / 3, zone_left + 2 * zone_w / 3]
    thirds_y = [zone_top + zone_h / 3, zone_top + 2 * zone_h / 3]
    grid = "".join(
        f'<line x1="{x}" y1="{zone_top}" x2="{x}" y2="{zone_bottom}" stroke="#FFFFFF" stroke-opacity="0.14" stroke-width="1.5"/>'
        for x in thirds_x
    ) + "".join(
        f'<line x1="{zone_left}" y1="{y}" x2="{zone_right}" y2="{y}" stroke="#FFFFFF" stroke-opacity="0.14" stroke-width="1.5"/>'
        for y in thirds_y
    )

    return f'''<svg class="zone-svg" width="{PANEL_W}" height="{PANEL_H}" viewBox="0 0 {PANEL_W} {PANEL_H}">
      <rect x="{zone_left}" y="{zone_top}" width="{zone_w}" height="{zone_h}"
            fill="none" stroke="#E8EEF6" stroke-width="3" rx="2"/>
      {grid}
      <path d="{plate}" fill="#E8EEF6" fill-opacity="0.9"/>
      <g transform="translate({exp_x} {exp_y})">
        <rect x="-7" y="-7" width="14" height="14" transform="rotate(45)"
              fill="#3B82F6" stroke="#0B1622" stroke-width="2.5"/>
      </g>
      {_actual_marker_svg(actual_location)}
    </svg>'''



def _actual_marker_svg(actual_location: tuple[float, float] | None) -> str:
    if actual_location is None:
        return ""
    x, y = _panel_xy(-float(actual_location[0]), float(actual_location[1]))
    return (
        f'<g transform="translate({x} {y})" stroke-linecap="round" fill="none">'
        f'<line x1="-9" y1="-9" x2="9" y2="9" stroke="#FFFFFF" stroke-width="9"/>'
        f'<line x1="-9" y1="9" x2="9" y2="-9" stroke="#FFFFFF" stroke-width="9"/>'
        f'<line x1="-9" y1="-9" x2="9" y2="9" stroke="#EF4444" stroke-width="4.5"/>'
        f'<line x1="-9" y1="9" x2="9" y2="-9" stroke="#EF4444" stroke-width="4.5"/>'
        f"</g>"
    )


def _result_bar_html(
    prediction: PitchPrediction,
    actual_pitch_type: str | None,
    pitch_result: str | None,
) -> str:
    if not actual_pitch_type and not pitch_result:
        return ""
    parts = []
    correct = actual_pitch_type == prediction.predicted_type
    if actual_pitch_type:
        name = PITCH_TYPE_FULL_NAMES.get(actual_pitch_type, actual_pitch_type)
        parts.append(f"ACTUAL: {name.upper()}")
    if pitch_result:
        parts.append(pitch_result.upper())
    if actual_pitch_type:
        parts.append("&#10003; PREDICTED" if correct else
                     f"MODEL SAID {prediction.predicted_type}")
    color = "#34D399" if correct else "#F87171"
    return (
        f'<div class="resultbar" style="color:{color};border-color:{color}33">'
        + " &nbsp;·&nbsp; ".join(parts)
        + "</div>"
    )

def build_card_html(
    prediction: PitchPrediction,
    context: GameContext,
    in_zone_probability: float,
    actual_pitch_type: str | None = None,
    actual_location: tuple[float, float] | None = None,
    pitch_result: str | None = None,
) -> str:
    """Assemble the full HTML document for one pitch card."""
    density_b64 = density_glow_png(prediction)
    pitcher_shot = _headshot_html(
        circular_headshot_png(context.pitcher_id), _initials(context.pitcher_name)
    )
    batter_shot = _headshot_html(
        circular_headshot_png(context.batter_id), _initials(context.batter_name)
    )

    score = (
        _score_team_html(
            context.away_team,
            context.score_away,
            getattr(context, "away_team_id", None),
        )
        + '<span class="at">@</span>'
        + _score_team_html(
            context.home_team,
            context.score_home,
            getattr(context, "home_team_id", None),
        )
    )

    outs_word = "OUT" if context.outs == 1 else "OUTS"
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
  .card {{ display: flex; flex-direction: column; height: 100%; padding: 22px 40px 14px; }}

  .topbar {{ display: flex; align-items: baseline; justify-content: space-between; }}
  .scoreline {{ font-size: 30px; font-weight: 800; letter-spacing: 0.5px; }}
  .score {{ display: inline-flex; align-items: center; gap: 10px; }}
  .team-logo {{ width: 30px; height: 30px; object-fit: contain; display: block; }}
  .scoreline .at {{ color: #55708F; font-size: 24px; font-weight: 600; margin: 0 14px; }}
  .scoreline b {{ color: #F59E0B; }}
  .gamestate {{ text-align: right; }}
  .chip {{
    display: inline-block; background: #16283F; border: 1px solid #24405F;
    border-radius: 999px; padding: 6px 16px; font-size: 16px; font-weight: 700;
    letter-spacing: 1.2px; color: #E8EEF6;
  }}
  .chip b {{ color: #F59E0B; }}
  .subline {{ margin-top: 6px; font-size: 12.5px; color: #55708F; letter-spacing: 1px; }}
  .rule {{ height: 2px; margin: 13px 0 16px;
           background: linear-gradient(90deg, #F59E0B 0%, #24405F 45%, transparent 100%); }}

  .body {{ display: flex; flex: 1; gap: 44px; }}
  .left {{ width: 560px; display: flex; flex-direction: column; }}

  .matchup {{ display: flex; align-items: center; gap: 16px; }}
  .headshot {{ width: 84px; height: 84px; border-radius: 50%;
               border: 2.5px solid #24405F; background: #16283F; }}
  .headshot-fallback {{ display: flex; align-items: center; justify-content: center;
                        font-size: 26px; font-weight: 800; color: #55708F; }}
  .who {{ flex: 1; }}
  .who .name {{ font-size: 23px; font-weight: 800; line-height: 1.15; }}
  .who .role {{ font-size: 12.5px; letter-spacing: 2px; color: #8CA0B8;
                font-weight: 700; margin-top: 4px; }}
  .vs {{ font-size: 15px; font-weight: 800; color: #55708F; padding: 0 2px; }}

  .section-label {{ font-size: 12.5px; font-weight: 800; letter-spacing: 3.5px;
                    color: #8CA0B8; margin: 22px 0 11px; }}
  .prob-row {{ margin-bottom: 13px; }}
  .prob-label {{ display: flex; justify-content: space-between; align-items: baseline;
                 margin-bottom: 5px; }}
  .prob-name {{ font-size: 17.5px; font-weight: 600; color: #B9C7D8; }}
  .prob-pct {{ font-size: 17.5px; font-weight: 700; color: #B9C7D8; }}
  .prob-row.top .prob-name, .prob-row.top .prob-pct {{
    color: #FFFFFF; font-weight: 800; font-size: 20px; }}
  .prob-track {{ height: 9px; border-radius: 999px; background: #16283F; overflow: hidden; }}
  .prob-fill {{ height: 100%; border-radius: 999px; background: #3D5876; }}
  .prob-row.top .prob-fill {{
    background: linear-gradient(90deg, #F59E0B, #FBBF24);
    box-shadow: 0 0 14px rgba(245, 158, 11, 0.55); }}

  .bottomleft {{ margin-top: auto; display: flex; align-items: flex-end;
                 justify-content: space-between; }}

  .right {{ flex: 1; display: flex; flex-direction: column; align-items: center; }}
  .zone-panel {{
    position: relative; width: {PANEL_W}px; height: {PANEL_H}px;
    background: linear-gradient(180deg, #0E1C2E 0%, #0A1523 100%);
    border: 1px solid #1C3352; border-radius: 14px;
    box-shadow: inset 0 0 40px rgba(0, 0, 0, 0.45); overflow: hidden;
  }}
  .zone-density {{ position: absolute; inset: 0; width: 100%; height: 100%;
                   filter: saturate(1.05); }}
  .zone-svg {{ position: absolute; inset: 0; }}
  .side-hint {{ position: absolute; bottom: 10px; font-size: 12px; font-weight: 700;
                letter-spacing: 1.5px; color: #55708F; }}
  .zone-caption {{ display: flex; align-items: center; gap: 14px; margin-top: 14px; }}
  .inzone {{ background: #16283F; border: 1px solid #24405F; border-radius: 999px;
             padding: 8px 20px; font-size: 16px; font-weight: 700; }}
  .inzone b {{ color: #F59E0B; font-size: 18px; }}
  .legend {{ font-size: 13px; color: #8CA0B8; font-weight: 600;
             display: flex; align-items: center; gap: 7px; }}
  .legend .marker {{ width: 11px; height: 11px; background: #3B82F6;
                     transform: rotate(45deg); display: inline-block;
                     border: 2px solid #0B1622; }}

  .resultbar {{ margin-top: 10px; padding: 8px 16px; border: 1px solid;
                border-radius: 10px; background: rgba(255, 255, 255, 0.03);
                font-size: 14px; font-weight: 800; letter-spacing: 1.5px;
                text-align: center; }}
  .xmark {{ color: #EF4444; font-weight: 900; }}
  .footer {{ display: flex; align-items: center; margin-top: 10px; }}
  .brand {{ font-size: 14px; font-weight: 800; letter-spacing: 4px; color: #8CA0B8; }}
  .brand b {{ color: #F59E0B; }}
  .brand-logo {{ height: 34px; width: auto; display: block; opacity: 0.98; }}
</style></head>
<body><div class="card">
  <div class="topbar">
    <div class="scoreline">{score}</div>
    <div class="gamestate">
      <span class="chip">{context.inning_half.upper()} {context.inning} &nbsp;•&nbsp; <b>{context.count_str}</b> &nbsp;•&nbsp; {context.outs} {outs_word}</span>
      <div class="subline">{context.date or ""} &nbsp;·&nbsp; PITCH #{context.pitch_number or 1} OF AT-BAT</div>
    </div>
  </div>
  <div class="rule"></div>

  <div class="body">
    <div class="left">
      <div class="matchup">
        {pitcher_shot}
        <div class="who">
          <div class="name">{context.pitcher_name}</div>
          <div class="role">{context.pitcher_hand}HP · PITCHING</div>
        </div>
        <div class="vs">VS</div>
        <div class="who" style="text-align:right">
          <div class="name">{context.batter_name}</div>
          <div class="role">{context.batter_hand}HB · BATTING</div>
        </div>
        {batter_shot}
      </div>

      <div class="section-label">NEXT PITCH PREDICTION</div>
      {_probability_rows(prediction)}

      <div class="bottomleft">
        <div>
          <div class="section-label" style="margin-bottom:2px">SITUATION</div>
          {_diamond_svg(context)}
        </div>
      </div>
    </div>

    <div class="right">
      <div class="section-label" style="margin-top:0">PREDICTED LOCATION · PITCHER'S VIEW</div>
      <div class="zone-panel">
        <img class="zone-density" src="data:image/png;base64,{density_b64}" alt="" />
        {_zone_svg(prediction, actual_location)}
        <div class="side-hint" style="left:14px">&larr; LHB</div>
        <div class="side-hint" style="right:14px">RHB &rarr;</div>
      </div>
      <div class="zone-caption">
        <div class="inzone">IN-ZONE <b>{in_zone_probability:.0%}</b></div>
        <div class="legend"><span class="marker"></span> EXPECTED{'&nbsp;&nbsp;<span class="xmark">&#10005;</span> ACTUAL' if actual_location else ' LOCATION'}</div>
      </div>
    </div>
  </div>

  {_result_bar_html(prediction, actual_pitch_type, pitch_result)}
  <div class="footer">
    {_brand_footer_html()}
  </div>
</div></body></html>"""


class HtmlCardRenderer:
    """Renders card HTML to JPEG with a persistent Chromium worker thread."""

    def __init__(self) -> None:
        self._thread: Thread | None = None
        self._requests: Queue | None = None
        self._ready = Event()
        self._startup_error: BaseException | None = None

    def _worker_main(self) -> None:
        from playwright.sync_api import sync_playwright

        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                page = browser.new_page(
                    viewport={"width": CARD_W, "height": CARD_H},
                    device_scale_factor=SCALE,
                )
                self._ready.set()
                while True:
                    assert self._requests is not None
                    item = self._requests.get()
                    if item is None:
                        break
                    html, out_path, result_q = item
                    try:
                        result_q.put(self._render_with_page(page, html, out_path))
                    except BaseException as exc:
                        result_q.put(exc)
                browser.close()
        except BaseException as exc:
            self._startup_error = exc
            self._ready.set()

    def _ensure_page(self) -> None:
        if self._thread is not None:
            return
        self._requests = Queue()
        self._ready = Event()
        self._startup_error = None
        self._thread = Thread(
            target=self._worker_main,
            name="HtmlCardRenderer",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait()
        if self._startup_error is not None:
            self.close()
            raise RuntimeError("Failed to start Playwright card renderer") from self._startup_error

    @staticmethod
    def _render_with_page(page, html: str, out_path: Path) -> Path:
        page.set_content(html, wait_until="load")
        jpeg_path = out_path.with_suffix(".jpg")
        for quality in _JPEG_QUALITY_CANDIDATES:
            page.screenshot(
                path=str(jpeg_path),
                type="jpeg",
                quality=quality,
                full_page=False,
            )
            if jpeg_path.stat().st_size <= _BLOB_LIMIT_BYTES:
                return jpeg_path
        raise RuntimeError(
            f"Card image {jpeg_path} is {jpeg_path.stat().st_size:,} bytes even "
            f"at JPEG quality {_JPEG_QUALITY_CANDIDATES[-1]}; Bluesky blobs must "
            f"stay under {_BLOB_LIMIT_BYTES:,} bytes."
        )

    def render(self, html: str, out_path: Path) -> Path:
        self._ensure_page()
        assert self._requests is not None
        result_q = Queue(maxsize=1)
        self._requests.put((html, out_path, result_q))
        result = result_q.get()
        if isinstance(result, BaseException):
            raise result
        return result

    def close(self) -> None:
        requests = self._requests
        thread = self._thread
        self._requests = None
        self._thread = None
        self._startup_error = None
        if requests is not None:
            requests.put(None)
        if thread is not None:
            thread.join(timeout=5)



# Bluesky rejects image blobs above ~976 KB; keep comfortable headroom.
_BLOB_LIMIT_BYTES = 900_000
_JPEG_QUALITY_CANDIDATES = (90, 85, 80, 75)

def render_card_png(
    prediction: PitchPrediction,
    context: GameContext,
    in_zone_probability: float,
    out_path: Path,
    renderer: HtmlCardRenderer,
    actual_pitch_type: str | None = None,
    actual_location: tuple[float, float] | None = None,
    pitch_result: str | None = None,
) -> Path:
    """Render one card (prediction or threaded result) at 2x scale."""
    html = build_card_html(
        prediction,
        context,
        in_zone_probability,
        actual_pitch_type=actual_pitch_type,
        actual_location=actual_location,
        pitch_result=pitch_result,
    )
    return renderer.render(html, out_path)
