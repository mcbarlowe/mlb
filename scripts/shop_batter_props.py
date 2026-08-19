"""Shop batter prop prices across sportsbooks and flag +EV plays.

For every batter prop on the slate (home runs, hits, total bases, RBIs,
runs, walks, stolen bases, ...), compare each book's price against the
batter's actual per-game rates from our database (current season + pooled
recent seasons) and against the no-vig consensus. Report the best price and
book per (player, market, line, side) with the EV of taking it at the
empirical rate. Optionally send an iMessage with new +EV plays (deduped per
day via a state file) so alerts arrive as soon as books post bettable lines.

EV is computed from a shrunk rate: the pooled trailing 3-season rate pulled
toward the board-pool league rate (empirical Bayes, ``--shrink-k`` games of
prior weight). A 2025 backtest showed raw trailing rates overestimate
selected bets by ~5% relative (regression to the mean); k=50 makes selected
estimates ~unbiased. Still a rate screen — no park/pitcher/platoon
adjustment; it finds prices generous relative to demonstrated frequency,
the "take +440, not +369" discipline, not a full prop model.

Usage:
  uv run python scripts/shop_batter_props.py                       # print board
  uv run python scripts/shop_batter_props.py --player alonso --show-all
  uv run python scripts/shop_batter_props.py --notify --recipient matt@barloweanalytics.com

Cost: one Odds API credit per region per market per event.
Defaults (7 markets x us,us2) cost ~14 credits per event.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import unicodedata
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import NamedTuple
from zoneinfo import ZoneInfo

import psycopg
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.betting.odds import american_to_decimal, decimal_to_american, no_vig_two_way
from src.database import PostgresConfig

EVENTS_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/events"
EVENT_ODDS_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/events/{event_id}/odds"
ET = ZoneInfo("America/New_York")
RECENT_SEASONS = 3  # pooled window for the "recent rate"

# market key -> per-game stat from a mlb.batting row
STAT_FNS: dict[str, Callable[[dict[str, int]], int]] = {
    "batter_home_runs": lambda r: r["homeruns"],
    "batter_hits": lambda r: r["hits"],
    "batter_total_bases": lambda r: r["totalbases"],
    "batter_rbis": lambda r: r["rbi"],
    "batter_runs_scored": lambda r: r["runs"],
    "batter_walks": lambda r: r["baseonballs"],
    "batter_stolen_bases": lambda r: r["stolenbases"],
    "batter_strikeouts": lambda r: r["strikeouts"],
    "batter_doubles": lambda r: r["doubles"],
    "batter_singles": lambda r: r["hits"] - r["doubles"] - r["triples"] - r["homeruns"],
    "batter_hits_runs_rbis": lambda r: r["hits"] + r["runs"] + r["rbi"],
}
STAT_COLUMNS = (
    "hits", "doubles", "triples", "homeruns", "totalbases",
    "rbi", "runs", "baseonballs", "stolenbases", "strikeouts",
)
# Markets whose estimates get conditioned on starts (PA>=3), expected PA, and
# park (calibrated 2024+2025: fixes the under-side composition bias, hits-family
# Brier improves). HR stays on the all-games estimator: conditioning measurably
# does not help there (rare event; PA leverage negligible vs estimator noise).
CONDITIONED_MARKETS = frozenset(STAT_FNS) - {"batter_home_runs"}
START_PA = 3
EXP_PA_WINDOW = 30  # trailing starts used for tonight's expected-PA estimate
STRATEGY_VERSION = "props-cond-v3"
PAPER_DDL = """
CREATE TABLE IF NOT EXISTS {schema}.prop_paper_bets (
    alert_date date NOT NULL,
    player text NOT NULL,
    market text NOT NULL,
    point double precision NOT NULL,
    side text NOT NULL,
    game_date date,
    matchup text,
    book text,
    price double precision NOT NULL,
    decimal_odds double precision,
    adj_prob double precision,
    ev double precision,
    rec_gp integer,
    stake_units double precision NOT NULL DEFAULT 1.0,
    status text NOT NULL DEFAULT 'open',
    result_value integer,
    profit_units double precision,
    strategy_version text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (alert_date, player, market, point, side)
)
"""
DEFAULT_MARKETS = (
    "batter_home_runs",
    "batter_hits",
    "batter_total_bases",
    "batter_rbis",
    "batter_runs_scored",
    "batter_walks",
    "batter_stolen_bases",
)
SHORT_MARKET = {
    "batter_home_runs": "HR",
    "batter_hits": "H",
    "batter_total_bases": "TB",
    "batter_rbis": "RBI",
    "batter_runs_scored": "R",
    "batter_walks": "BB",
    "batter_stolen_bases": "SB",
    "batter_strikeouts": "K",
    "batter_doubles": "2B",
    "batter_singles": "1B",
    "batter_hits_runs_rbis": "H+R+RBI",
}


def _norm_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return " ".join("".join(ch for ch in text.lower() if ch.isalpha() or ch.isspace()).split())


def fetch_events(api_key: str, hours_ahead: float) -> list[dict]:
    now = datetime.now(UTC)
    params = {
        "apiKey": api_key,
        "dateFormat": "iso",
        "commenceTimeFrom": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commenceTimeTo": (now + timedelta(hours=hours_ahead)).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    resp = requests.get(EVENTS_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_event_props(
    api_key: str, event_id: str, regions: str, markets: tuple[str, ...]
) -> tuple[dict, str]:
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": ",".join(markets),
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    resp = requests.get(EVENT_ODDS_URL.format(event_id=event_id), params=params, timeout=30)
    resp.raise_for_status()
    return resp.json(), resp.headers.get("x-requests-remaining", "?")


def collect_prices(event: dict) -> dict[tuple[str, str, float], dict[str, dict[str, float]]]:
    """(market, player, point) -> book -> {"over": american, "under": american}."""
    out: dict[tuple[str, str, float], dict[str, dict[str, float]]] = {}
    for book in event.get("bookmakers", []):
        for market in book.get("markets", []):
            key = market.get("key")
            if key not in STAT_FNS:
                continue
            for outcome in market.get("outcomes", []):
                player = str(outcome.get("description", "")).strip()
                side = str(outcome.get("name", "")).lower()
                point = outcome.get("point")
                if not player or side not in ("over", "under") or point is None:
                    continue
                slot = out.setdefault((key, player, float(point)), {})
                slot.setdefault(book["key"], {})[side] = float(outcome["price"])
    return out


class PlayerLines(NamedTuple):
    """Chronological per-game lines plus the player's age today."""

    lines: list[tuple[int, float, int, dict[str, int]]]  # (season, age, pa, stats)
    age_now: float


def load_game_lines(names: list[str], season: int) -> dict[str, PlayerLines]:
    """normalized name -> PlayerLines((season, age, stats) rows, age today)."""
    c = PostgresConfig.from_env()
    conn = psycopg.connect(
        dbname=c.dbname, user=c.user, password=c.password,
        host=c.host, port=c.port, connect_timeout=10,
    )
    with conn.cursor() as cur:
        cur.execute(f"SELECT player_id, full_name FROM {c.schema}.players")
        by_norm: dict[str, list[int]] = {}
        for pid, full in cur.fetchall():
            by_norm.setdefault(_norm_name(str(full)), []).append(int(pid))

        pid_to_norm: dict[int, str] = {}
        for name in names:
            for pid in by_norm.get(_norm_name(name), []):
                pid_to_norm[pid] = _norm_name(name)
        if not pid_to_norm:
            conn.close()
            return {}

        cols = ", ".join(f"COALESCE(b.{col}, 0)::int AS {col}" for col in STAT_COLUMNS)
        cur.execute(
            f"""
            SELECT b.player_id, g.season::int AS season,
                   g.game_date::date AS game_date, p.birth_date::date AS birth_date,
                   COALESCE(b.plateappearances, 0)::int AS pa,
                   {cols}
            FROM {c.schema}.batting b
            JOIN {c.schema}.games g USING (game_pk)
            LEFT JOIN {c.schema}.players p USING (player_id)
            WHERE b.player_id = ANY(%s)
              AND g.game_type = 'R' AND g.abstract_game_state = 'Final'
              AND g.season::int BETWEEN %s AND %s
              AND COALESCE(b.plateappearances, 0) > 0
            ORDER BY COALESCE(g.game_datetime, g.game_date), g.game_pk
            """,
            (list(pid_to_norm), season - RECENT_SEASONS + 1, season),
        )
        col_names = [d.name for d in cur.description]
        per_pid: dict[int, list[tuple[int, float, int, dict[str, int]]]] = {}
        births: dict[int, date | None] = {}
        for row in cur.fetchall():
            rec = dict(zip(col_names, row))
            pid = int(rec["player_id"])
            birth = rec["birth_date"]
            births.setdefault(pid, birth)
            age = (
                (rec["game_date"] - birth).days / 365.25
                if birth is not None else float("nan")
            )
            per_pid.setdefault(pid, []).append(
                (int(rec["season"]), age, int(rec["pa"]),
                 {col: int(rec[col]) for col in STAT_COLUMNS})
            )
    conn.close()

    # collapse duplicate-name player_ids: keep the one with the most games
    best_for_norm: dict[str, int] = {}
    for pid, norm in pid_to_norm.items():
        n_games = len(per_pid.get(pid, []))
        incumbent = best_for_norm.get(norm)
        if incumbent is None or n_games > len(per_pid.get(incumbent, [])):
            best_for_norm[norm] = pid
    today = datetime.now(ET).date()
    out: dict[str, PlayerLines] = {}
    for norm, pid in best_for_norm.items():
        birth = births.get(pid)
        out[norm] = PlayerLines(
            lines=per_pid.get(pid, []),
            age_now=(
                (today - birth).days / 365.25 if birth is not None else float("nan")
            ),
        )
    return out


def db_latest_final() -> date | None:
    """Date of the newest Final regular-season game in the DB (ETL freshness)."""
    c = PostgresConfig.from_env()
    conn = psycopg.connect(
        dbname=c.dbname, user=c.user, password=c.password,
        host=c.host, port=c.port, connect_timeout=10,
    )
    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT MAX(game_date)::date FROM {c.schema}.games
                WHERE abstract_game_state = 'Final' AND game_type = 'R'"""
        )
        row = cur.fetchone()
    conn.close()
    return row[0] if row else None


def team_directory() -> tuple[dict[str, str], dict[str, int]]:
    """Full team name -> (abbreviation, team_id) from mlb.teams."""
    c = PostgresConfig.from_env()
    conn = psycopg.connect(
        dbname=c.dbname, user=c.user, password=c.password,
        host=c.host, port=c.port, connect_timeout=10,
    )
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT team_name, abbreviation, team_id FROM {c.schema}.teams "
            f"WHERE sport_id = 1"
        )
        rows = cur.fetchall()
    conn.close()
    return (
        {str(n): str(a) for n, a, _ in rows},
        {str(n): int(t) for n, _, t in rows},
    )


def rate_over(
    lines: list[tuple[int, float, int, dict[str, int]]],
    stat_fn: Callable[[dict[str, int]], int],
    point: float,
    season: int | None,
    last_n: int | None = None,
) -> tuple[float, int]:
    """(share of games with stat > point, games) over the window.

    ``lines`` is chronological; ``last_n`` keeps only the most recent games.
    """
    rows = [stats for yr, _age, _pa, stats in lines if season is None or yr == season]
    if last_n:
        rows = rows[-last_n:]
    if not rows:
        return float("nan"), 0
    hits = sum(1 for stats in rows if stat_fn(stats) > point)
    return hits / len(rows), len(rows)


def decayed_over(
    lines: list[tuple[int, float, int, dict[str, int]]],
    stat_fn: Callable[[dict[str, int]], int],
    point: float,
    lam: float,
) -> tuple[float, float, float, float]:
    """Exponentially decayed (successes, games, mean age, mean PA).

    ``lines`` is chronological, so the newest game carries weight 1 and a game
    g games back carries lam**g. lam=1 reproduces plain counts.
    """
    s = w = a = p = 0.0
    for _, age, pa, stats in lines:
        s = lam * s + (1.0 if stat_fn(stats) > point else 0.0)
        w = lam * w + 1.0
        a = lam * a + (0.0 if math.isnan(age) else age)
        p = lam * p + pa
    if w <= 0.0:
        return 0.0, 0.0, float("nan"), float("nan")
    has_age = bool(lines) and not math.isnan(lines[0][1])
    return s, w, (a / w if has_age else float("nan")), p / w


def expected_pa(lines: list[tuple[int, float, int, dict[str, int]]]) -> float:
    """Mean PA over the player's most recent starts (tonight's role proxy)."""
    recent = [pa for _, _, pa, _ in lines[-EXP_PA_WINDOW:]]
    return sum(recent) / len(recent) if recent else float("nan")


def load_aging_curves(path: Path) -> dict[str, dict[int, float]]:
    """market|point -> {integer age -> cumulative logit offset (ref age 27)}."""
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return {
        key: {int(age): float(v) for age, v in curve.items()}
        for key, curve in payload.get("curves", {}).items()
    }


def load_park_factors(path: Path) -> dict[str, dict[int, float]]:
    """market|point -> {home_team_id -> shrunk logit offset vs league}."""
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return {
        key: {int(t): float(v) for t, v in per_venue.items()}
        for key, per_venue in payload.get("factors", {}).items()
    }


MODEL_NAME = "mlb-prop-rate-estimator"
MLFLOW_CACHE = Path(__file__).parent.parent / "models/mlflow_cache/prop_rate_estimator"
DEFAULT_TRACKING_URI = "http://10.0.0.171:5001"


def default_tracking_uri() -> str:
    """Shared HTTP tracking server; env honored only when it is an HTTP URI
    (the artifact resolver needs the REST API, not a raw DB store URI)."""
    env = os.environ.get("MLFLOW_TRACKING_URI", "")
    return env if env.startswith(("http://", "https://")) else DEFAULT_TRACKING_URI


def resolve_registry_artifacts(tracking_uri: str) -> tuple[Path | None, Path | None, str]:
    """Resolve the @champion generation's artifacts from the MLflow registry.

    Returns (curves_path, parks_path, provenance). Downloads the champion's
    registered aging-curve/park-factor artifacts into a per-version cache
    (models/mlflow_cache/prop_rate_estimator/v<N>/). Server unreachable ->
    newest cached champion. Champion generation != this code's
    STRATEGY_VERSION, or nothing cached -> (None, None, reason) and the
    caller falls back to the repo-local artifact files.
    """
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
    try:
        from mlflow.artifacts import download_artifacts
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=tracking_uri)
        v = client.get_model_version_by_alias(MODEL_NAME, "champion")
        strategy = v.tags.get("strategy_version", "?")
        if strategy != STRATEGY_VERSION:
            return None, None, (
                f"registry champion v{v.version} is {strategy} but this code is "
                f"{STRATEGY_VERSION}; using repo-local artifacts"
            )
        cache = MLFLOW_CACHE / f"v{v.version}"
        curves = cache / "estimator/aging_curves.json"
        parks = cache / "estimator/park_factors.json"
        if not (curves.exists() and parks.exists()):
            cache.mkdir(parents=True, exist_ok=True)
            download_artifacts(run_id=v.run_id, artifact_path="estimator",
                               dst_path=str(cache), tracking_uri=tracking_uri)
        return (
            curves if curves.exists() else None,
            parks if parks.exists() else None,
            f"mlflow @champion v{v.version} ({strategy})",
        )
    except Exception as exc:
        if MLFLOW_CACHE.exists():
            for cdir in sorted(MLFLOW_CACHE.glob("v*"),
                               key=lambda p: int(p.name[1:]), reverse=True):
                curves = cdir / "estimator/aging_curves.json"
                parks = cdir / "estimator/park_factors.json"
                if curves.exists() and parks.exists():
                    return curves, parks, (
                        f"mlflow cache {cdir.name} "
                        f"(registry unreachable: {type(exc).__name__})"
                    )
        return None, None, (
            f"registry unavailable ({type(exc).__name__}); using repo-local artifacts"
        )


def curve_at(curve: dict[int, float], age: float) -> float:
    ages = sorted(curve)
    a = min(max(age, ages[0]), ages[-1])
    lo = math.floor(a)
    hi = min(lo + 1, ages[-1])
    frac = a - lo
    return curve[lo] * (1.0 - frac) + curve[hi] * frac


def send_imessage(recipient: str, text: str) -> None:
    script = (
        'on run argv\n'
        'set theHandle to item 1 of argv\n'
        'set theText to item 2 of argv\n'
        'tell application "Messages"\n'
        'set svc to first account whose service type = iMessage\n'
        'send theText to participant theHandle of svc\n'
        'end tell\n'
        'end run'
    )
    subprocess.run(
        ["osascript", "-e", script, recipient, text],
        check=True, capture_output=True, timeout=60,
    )


def log_paper_bets(alert_date: str, plays: list) -> int:
    """Record alerted plays as open bets in mlb.prop_paper_bets (flat 1u)."""
    c = PostgresConfig.from_env()
    conn = psycopg.connect(
        dbname=c.dbname, user=c.user, password=c.password,
        host=c.host, port=c.port, connect_timeout=10,
    )
    inserted = 0
    with conn.cursor() as cur:
        cur.execute(PAPER_DDL.format(schema=c.schema))
        for _key, r, _prior in plays:
            cur.execute(
                f"""
                INSERT INTO {c.schema}.prop_paper_bets
                    (alert_date, player, market, point, side, game_date, matchup,
                     book, price, decimal_odds, adj_prob, ev, rec_gp,
                     strategy_version)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (alert_date, player, market, point, side) DO NOTHING
                """,
                (
                    alert_date, r["player"], r["market"], r["point"], r["side"],
                    f"{r['start_et']:%Y-%m-%d}", r["matchup"], r["best_book"],
                    r["best_price"], american_to_decimal(r["best_price"]),
                    round(r["adj_rate"], 5), round(r["ev_adj"], 5), r["rec_gp"],
                    STRATEGY_VERSION,
                ),
            )
            inserted += cur.rowcount
    conn.commit()
    conn.close()
    return inserted


def send_ntfy(topic: str, text: str) -> None:
    """Instant push via ntfy.sh (install the ntfy app and subscribe to topic)."""
    resp = requests.post(
        f"https://ntfy.sh/{topic}",
        data=text.encode("utf-8"),
        headers={"Title": "MLB Props", "Priority": "high", "Tags": "baseball"},
        timeout=15,
    )
    resp.raise_for_status()


def load_state(path: Path) -> dict[str, dict]:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_state(path: Path, state: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--markets", default=",".join(DEFAULT_MARKETS),
                    help="comma-separated Odds API batter market keys")
    ap.add_argument("--regions", default="us,us2")
    ap.add_argument("--hours-ahead", type=float, default=24.0)
    ap.add_argument("--min-books", type=int, default=2,
                    help="min books quoting a side before it is shopped")
    ap.add_argument("--player", default=None, help="substring filter on batter name")
    ap.add_argument("--season", type=int, default=datetime.now(ET).year)
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--show-all", action="store_true",
                    help="print every row, not just EV > 0")
    ap.add_argument("--min-ev", type=float, default=0.11,
                    help="min shrunk-rate EV to text an alert")
    ap.add_argument("--market-min-ev", default="batter_home_runs=0.15",
                    help="per-market min-EV overrides for alerts, "
                         "e.g. batter_home_runs=0.15,batter_hits=0.08")
    ap.add_argument("--shrink-k", type=float, default=50.0,
                    help="empirical-Bayes prior strength (in games): trailing rates "
                         "are shrunk toward the board-pool league rate before EV. "
                         "Calibrated on 2025 (k=50 makes max-pick estimates ~unbiased); "
                         "0 disables.")
    ap.add_argument("--recency-half-life", type=float, default=400.0,
                    help="within-window exponential decay half-life in games; "
                         "calibrated jointly with aging on 2024+2025; <=0 disables")
    ap.add_argument("--model-source", choices=("registry", "local"),
                    default="registry",
                    help="registry = resolve @champion artifacts from MLflow "
                         "(per-version cache, offline fallback); local = repo files")
    ap.add_argument("--mlflow-tracking-uri", default=default_tracking_uri())
    ap.add_argument("--aging-curves", default=None,
                    help="explicit aging-curve JSON (overrides registry resolution)")
    ap.add_argument("--park-factors", default=None,
                    help="explicit park-factor JSON (overrides registry resolution)")
    ap.add_argument("--min-gp", type=int, default=150,
                    help="min pooled recent games before a rate is alert-worthy")
    ap.add_argument("--max-alerts", type=int, default=8, help="max plays per text")
    ap.add_argument("--max-fair-ratio", type=float, default=1.5,
                    help="suppress alerts where our prob exceeds the no-vig market "
                         "prob by more than this ratio (market-sanity anchor)")
    ap.add_argument("--max-fair-diff", type=float, default=0.15,
                    help="suppress alerts where our prob exceeds the no-vig market "
                         "prob by more than this many probability points")
    ap.add_argument("--max-decimal", type=float, default=15.0,
                    help="suppress alerts priced longer than this decimal (~+1400)")
    ap.add_argument("--max-ev", type=float, default=0.50,
                    help="suppress alerts with EV above this (implausible-edge backstop)")
    ap.add_argument("--notify", action="store_true",
                    help="push new +EV plays to the configured channels")
    ap.add_argument("--notify-method", choices=("imessage", "ntfy", "both"),
                    default="both",
                    help="ntfy = instant push via ntfy.sh app; imessage = Messages relay")
    ap.add_argument("--ntfy-topic", default="barlowe-props-c47d9e2a51b3",
                    help="ntfy.sh topic to publish to")
    ap.add_argument("--recipient", default="matt@barloweanalytics.com")
    ap.add_argument("--state-file",
                    default=str(Path.home() / "Library/Application Support/BarloweAnalytics/prop_shop_state.json"))
    ap.add_argument("--dry-run-notify", action="store_true",
                    help="print the would-be text instead of sending")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress the board table (for high-frequency agent runs)")
    ap.add_argument("--realert-improve", type=float, default=0.03,
                    help="re-alert an already-texted play when its best-price EV "
                         "improves by at least this much; <=0 disables")
    args = ap.parse_args()

    markets = tuple(m.strip() for m in args.markets.split(",") if m.strip())
    unknown = [m for m in markets if m not in STAT_FNS]
    if unknown:
        raise SystemExit(f"unsupported markets (no stat mapping): {unknown}")

    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        raise SystemExit("ODDS_API_KEY not set in environment")

    events = fetch_events(api_key, args.hours_ahead)
    if args.max_events:
        events = events[: args.max_events]
    if not events:
        print("no upcoming events in window")
        return
    print(f"slate: {len(events)} events | regions {args.regions} | markets "
          f"{', '.join(SHORT_MARKET[m] for m in markets)}")

    rows: list[dict] = []
    remaining = "?"
    for ev in events:
        data, remaining = fetch_event_props(api_key, ev["id"], args.regions, markets)
        matchup = f"{ev['away_team']} @ {ev['home_team']}"
        start_et = datetime.fromisoformat(ev["commence_time"]).astimezone(ET)
        for (market, player, point), books in collect_prices(data).items():
            if args.player and args.player.lower() not in player.lower():
                continue
            for side in ("over", "under"):
                prices = {bk: p[side] for bk, p in books.items() if side in p}
                if len(prices) < args.min_books:
                    continue
                best_book, best_price = max(
                    prices.items(), key=lambda kv: american_to_decimal(kv[1])
                )
                med_dec = statistics.median(american_to_decimal(p) for p in prices.values())
                pairs = [
                    (p["over"], p["under"]) for p in books.values()
                    if "over" in p and "under" in p
                ]
                fair_p = float("nan")
                if pairs:
                    med_o = statistics.median(american_to_decimal(o) for o, _ in pairs)
                    med_u = statistics.median(american_to_decimal(u) for _, u in pairs)
                    over_fair, under_fair = no_vig_two_way(
                        decimal_to_american(med_o), decimal_to_american(med_u)
                    )
                    fair_p = over_fair if side == "over" else under_fair
                rows.append({
                    "market": market, "player": player, "point": point, "side": side,
                    "matchup": matchup, "start_et": start_et,
                    "away": ev["away_team"], "home": ev["home_team"],
                    "n_books": len(prices), "best_price": best_price,
                    "best_book": best_book, "median_dec": med_dec, "fair_p": fair_p,
                })

    if not rows:
        print("no props matched the filters")
        return

    lines_by_norm = load_game_lines(sorted({r["player"] for r in rows}), args.season)
    repo = Path(__file__).parent.parent
    curves_path = Path(args.aging_curves) if args.aging_curves else repo / "models/props/aging_curves.json"
    parks_path = Path(args.park_factors) if args.park_factors else repo / "models/props/park_factors.json"
    provenance = "repo-local artifacts"
    if args.model_source == "registry" and not (args.aging_curves or args.park_factors):
        reg_curves, reg_parks, provenance = resolve_registry_artifacts(
            args.mlflow_tracking_uri
        )
        if reg_curves is not None:
            curves_path = reg_curves
        if reg_parks is not None:
            parks_path = reg_parks
    print(f"estimator artifacts: {provenance}")
    curves = load_aging_curves(curves_path)
    if not curves:
        print("note: aging curves unavailable - aging adjustment disabled")
    parks = load_park_factors(parks_path)
    if not parks:
        print("note: park factors unavailable - park adjustment disabled")
    team_abbrev, team_ids = team_directory()
    lam = 1.0 if args.recency_half_life <= 0 else 0.5 ** (1.0 / args.recency_half_life)
    latest = db_latest_final()
    stale_days = (datetime.now(ET).date() - latest).days if latest else 99
    if stale_days > 2:
        print(f"WARNING: DB stale - latest Final game is {latest} "
              f"({stale_days}d old); rates are frozen until the daily ETL runs")

    # Per-player streams: conditioned markets estimate from STARTS only.
    starts_by_norm = {
        norm: PlayerLines(
            lines=[ln for ln in e.lines if ln[2] >= START_PA], age_now=e.age_now
        )
        for norm, e in lines_by_norm.items()
    }

    # League rates per (market, point) over the family-appropriate board pool:
    # the empirical-Bayes prior that trailing rates get shrunk toward.
    pool_all = [stats for e in lines_by_norm.values() for _, _, _, stats in e.lines]
    pool_starts = [
        stats for e in starts_by_norm.values() for _, _, _, stats in e.lines
    ]
    league_over: dict[tuple[str, float], float] = {}
    for market, point in {(r["market"], r["point"]) for r in rows}:
        fn = STAT_FNS[market]
        pool = pool_starts if market in CONDITIONED_MARKETS else pool_all
        if pool:
            league_over[(market, point)] = sum(1 for s in pool if fn(s) > point) / len(pool)

    for r in rows:
        conditioned = r["market"] in CONDITIONED_MARKETS
        source = starts_by_norm if conditioned else lines_by_norm
        entry = source.get(_norm_name(r["player"]))
        lines = entry.lines if entry else []
        age_now = entry.age_now if entry else float("nan")
        stat_fn = STAT_FNS[r["market"]]
        over_szn, szn_gp = rate_over(lines, stat_fn, r["point"], args.season)
        over_rec, rec_gp = rate_over(lines, stat_fn, r["point"], None)
        over_l30, l30_gp = rate_over(lines, stat_fn, r["point"], None, last_n=30)
        lg = league_over.get((r["market"], r["point"]), float("nan"))
        over_adj = over_rec
        if rec_gp and not math.isnan(lg):
            s_w, n_w, age_wmean, mean_pa = decayed_over(lines, stat_fn, r["point"], lam)
            p_w = (s_w + 0.5) / (n_w + 1.0)
            curve = curves.get(f"{r['market']}|{r['point']:g}")
            delta = 0.0
            if curve and not math.isnan(age_now) and not math.isnan(age_wmean):
                delta = curve_at(curve, age_now) - curve_at(curve, age_wmean)
                delta = max(-0.75, min(0.75, delta))
            p_aged = 1.0 / (1.0 + math.exp(-(math.log(p_w / (1.0 - p_w)) + delta)))
            if args.shrink_k > 0:
                over_adj = (n_w * p_aged + args.shrink_k * lg) / (n_w + args.shrink_k)
            else:
                over_adj = p_aged
            if conditioned:
                # expected-PA rescale (0.5 lines: per-PA equivalent rate model)
                exp_pa = expected_pa(lines)
                if (
                    r["point"] == 0.5
                    and not math.isnan(exp_pa)
                    and not math.isnan(mean_pa)
                    and mean_pa >= 1.0
                ):
                    q = 1.0 - (1.0 - over_adj) ** (1.0 / mean_pa)
                    over_adj = 1.0 - (1.0 - q) ** min(max(exp_pa, 1.0), 6.0)
                # tonight's park (home team of the event)
                park_map = parks.get(f"{r['market']}|{r['point']:g}")
                home_id = team_ids.get(r["home"])
                if park_map and home_id is not None and home_id in park_map:
                    clipped = min(max(over_adj, 1e-4), 1.0 - 1e-4)
                    over_adj = 1.0 / (1.0 + math.exp(
                        -(math.log(clipped / (1.0 - clipped)) + park_map[home_id])
                    ))
        p_szn = over_szn if r["side"] == "over" else 1.0 - over_szn
        p_rec = over_rec if r["side"] == "over" else 1.0 - over_rec
        p_l30 = over_l30 if r["side"] == "over" else 1.0 - over_l30
        p_adj = over_adj if r["side"] == "over" else 1.0 - over_adj
        dec = american_to_decimal(r["best_price"])
        r["szn_rate"], r["szn_gp"] = p_szn, szn_gp
        r["rec_rate"], r["rec_gp"] = p_rec, rec_gp
        r["l30_rate"], r["l30_gp"] = p_l30, l30_gp
        r["adj_rate"] = p_adj
        r["ev_adj"] = p_adj * dec - 1.0 if rec_gp else float("nan")

    rows.sort(key=lambda r: (r["ev_adj"] != r["ev_adj"], -(r["ev_adj"] or 0.0)))
    shown = [
        r for r in rows
        if args.show_all or (r["ev_adj"] == r["ev_adj"] and r["ev_adj"] > 0)
    ]

    if not args.quiet:
        hdr = (f"{'mkt':<8} {'line':<6} {'batter':<22} {'best':>6} {'book':<14} "
               f"{'#bk':>3} {'med':>6} {'fair%':>6} {'szn%':>6} {'rec%':>6} "
               f"{'L30%':>6} {'adj%':>6} {'recGP':>5} {'EV@adj':>7}  game")
        print("\n" + hdr)
        print("-" * len(hdr))
        for r in shown:
            med_am = decimal_to_american(r["median_dec"])
            fair = f"{r['fair_p']:.1%}" if r["fair_p"] == r["fair_p"] else "  -  "
            szn = f"{r['szn_rate']:.1%}" if r["szn_gp"] else "  -  "
            rec = f"{r['rec_rate']:.1%}" if r["rec_gp"] else "  -  "
            l30 = f"{r['l30_rate']:.1%}" if r["l30_gp"] else "  -  "
            adj = f"{r['adj_rate']:.1%}" if r["rec_gp"] else "  -  "
            ev = f"{r['ev_adj']:+.1%}" if r["ev_adj"] == r["ev_adj"] else "   -   "
            line = f"{'o' if r['side'] == 'over' else 'u'}{r['point']:g}"
            print(
                f"{SHORT_MARKET[r['market']]:<8} {line:<6} {r['player']:<22.22} "
                f"{r['best_price']:>+6.0f} {r['best_book']:<14.14} {r['n_books']:>3d} "
                f"{med_am:>+6.0f} {fair:>6} {szn:>6} {rec:>6} {l30:>6} {adj:>6} "
                f"{r['rec_gp']:>5d} {ev:>7}  {r['start_et']:%-I:%M%p} {r['matchup']}"
            )
        if not shown:
            print("(no +EV rows at current prices)")

    print(f"API credits remaining: {remaining}")
    if not args.quiet:
        print(f"rates: szn% = {args.season}; rec% = pooled "
              f"{args.season - RECENT_SEASONS + 1}-{args.season}; L30% = last 30 games; "
              f"adj% = recency-decayed (H={args.recency_half_life:g}g), age-projected, "
              f"league-shrunk (k={args.shrink_k:g}) rate; "
              f"EV@adj = best-price EV at adj%. No matchup adjustment.")

    if not (args.notify or args.dry_run_notify):
        return

    market_min_ev: dict[str, float] = {}
    for part in (args.market_min_ev or "").split(","):
        part = part.strip()
        if part:
            mk, _, val = part.partition("=")
            market_min_ev[mk.strip()] = float(val)
    alerts = [
        r for r in rows
        if r["ev_adj"] == r["ev_adj"]
        and r["ev_adj"] >= market_min_ev.get(r["market"], args.min_ev)
        and r["ev_adj"] <= args.max_ev
        and r["rec_gp"] >= args.min_gp
        # market anchor: a two-sided fair prob must exist, and our estimate may
        # not exceed it implausibly - when we disagree with the market by this
        # much, the market usually knows the role/matchup and we do not.
        and r["fair_p"] == r["fair_p"]
        and r["adj_rate"] <= r["fair_p"] * args.max_fair_ratio
        and r["adj_rate"] - r["fair_p"] <= args.max_fair_diff
        and american_to_decimal(r["best_price"]) <= args.max_decimal
    ]
    state_path = Path(args.state_file)
    state = load_state(state_path)
    today = datetime.now(ET).strftime("%Y-%m-%d")
    state = {k: v for k, v in state.items() if k.startswith(today)}  # drop stale days
    fresh = []
    for r in alerts:
        key = f"{today}|{r['player']}|{r['market']}|{r['point']:g}|{r['side']}"
        prior = state.get(key)
        if prior is None:
            fresh.append((key, r, None))
        elif (
            args.realert_improve > 0
            and r["ev_adj"] >= float(prior.get("ev", 0.0)) + args.realert_improve
        ):
            fresh.append((key, r, float(prior.get("price", 0.0))))
    if not fresh:
        print("notify: no new alert-worthy plays")
        save_state(state_path, state)
        return

    fresh = fresh[: args.max_alerts]
    ab = team_abbrev
    entries = [f"MLB props {datetime.now(ET):%-m/%-d %-I:%M%p}"]
    if stale_days > 2:
        entries.append(f"WARNING: rates stale - DB last updated {latest}")
    for _, r, prior_price in fresh:
        side = "Over" if r["side"] == "over" else "Under"
        szn = f"{r['szn_rate']:.0%}" if r["szn_gp"] else "-"
        l30 = f"{r['l30_rate']:.0%}" if r["l30_gp"] else "-"
        game = f"{ab.get(r['away'], r['away'])}@{ab.get(r['home'], r['home'])}"
        improved = f" — IMPROVED (was {prior_price:+.0f})" if prior_price else ""
        entries.append(
            f"{SHORT_MARKET[r['market']]} {side} {r['point']:g} — {r['player']} — "
            f"{r['best_price']:+.0f} @ {r['best_book']} — 3yr {r['rec_rate']:.0%}, "
            f"1yr {szn}, L30 {l30}, adj {r['adj_rate']:.0%} — EV {r['ev_adj']:+.0%} — "
            f"{game} {r['start_et']:%-I:%M%p}{improved}"
        )
    text = "\n\n".join(entries)
    if args.dry_run_notify:
        print("\n--- would send ---\n" + text)
        return
    sent_to = []
    if args.notify_method in ("ntfy", "both") and args.ntfy_topic:
        try:
            send_ntfy(args.ntfy_topic, text)
            sent_to.append(f"ntfy:{args.ntfy_topic}")
        except requests.RequestException as exc:
            print(f"notify: ntfy failed: {exc}")
    if args.notify_method in ("imessage", "both"):
        try:
            send_imessage(args.recipient, text)
            sent_to.append(f"imessage:{args.recipient}")
        except (subprocess.SubprocessError, OSError) as exc:
            print(f"notify: imessage failed: {exc}")
    if not sent_to:
        raise SystemExit("notify: every channel failed; not recording state")
    print(f"notify: sent {len(fresh)} plays via {', '.join(sent_to)}")
    for key, r, _prior in fresh:
        state[key] = {"price": r["best_price"], "book": r["best_book"],
                      "ev": round(r["ev_adj"], 4),
                      "game_date": f"{r['start_et']:%Y-%m-%d}",
                      "matchup": r["matchup"]}
    save_state(state_path, state)
    try:
        logged = log_paper_bets(today, fresh)
        print(f"paper ledger: {logged} new bets in prop_paper_bets")
    except psycopg.Error as exc:
        print(f"paper ledger: insert failed ({exc}); alerts already sent")


if __name__ == "__main__":
    main()
