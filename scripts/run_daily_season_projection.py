from __future__ import annotations

import argparse
import asyncio
import json
import math
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database import PostgresConfig, PostgresHandler
from src.endpoints.game_feed import GameFeed
from src.etl.postgres_backfill import run_postgres_backfill
from src.live.publisher import POST_PROVIDER_CHOICES, build_publisher

EASTERN = ZoneInfo("America/New_York")
DEFAULT_RAW_DATA_PATH = Path("data/raw/livefeeds")
DEFAULT_REFRESH_LOOKBACK_DAYS = 3
DEFAULT_MAX_REFRESH_GAMES = 500
DEFAULT_TRIALS = 5_000
DEFAULT_TUNE_TRIALS = 1_000
MONTH_LABELS = {
    1: "Jan.",
    2: "Feb.",
    3: "Mar.",
    4: "Apr.",
    5: "May",
    6: "Jun.",
    7: "Jul.",
    8: "Aug.",
    9: "Sep.",
    10: "Oct.",
    11: "Nov.",
    12: "Dec.",
}


@dataclass(frozen=True)
class ScheduleSnapshot:
    total_games: int
    final_games: int
    stale_before_as_of: tuple[int, ...]
    refresh_game_pks: tuple[int, ...]
    status_counts: dict[str, int]


@dataclass(frozen=True)
class ProjectionOutputs:
    output_dir: Path
    projection_csv: Path
    calibration_csv: Path
    summary_csv: Path
    playoff_probabilities: Path
    playoff_stages: Path


async def _download_live_feeds(
    *,
    game_pks: Sequence[int],
    season: int,
    raw_data_path: Path,
    concurrency_limit: int,
) -> None:
    output_dir = raw_data_path / str(season)
    output_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    async with GameFeed(concurrency_limit=max(1, min(concurrency_limit, len(game_pks)))) as game_feed:
        async def fetch_and_write(game_pk: int) -> None:
            try:
                data = await game_feed.get_async(game_pk)
                output_path = output_dir / f"{game_pk}.json"
                output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            except Exception as exc:
                errors.append(f"{game_pk}: {exc}")

        await asyncio.gather(*(fetch_and_write(game_pk) for game_pk in game_pks))

    if errors:
        preview = "; ".join(errors[:5])
        suffix = "" if len(errors) <= 5 else f"; ... {len(errors) - 5} more"
        raise RuntimeError(f"Failed to refresh {len(errors)} live feeds: {preview}{suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh current-season MLB data, render season-projection graphics, and optionally post them.",
    )
    today = datetime.now(EASTERN).date()
    parser.add_argument("--season", type=int, default=today.year)
    parser.add_argument(
        "--as-of",
        type=str,
        default=today.isoformat(),
        help="Projection cutoff date. Defaults to today's America/New_York date.",
    )
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--tune-trials", type=int, default=DEFAULT_TUNE_TRIALS)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--raw-data-path", type=Path, default=DEFAULT_RAW_DATA_PATH)
    parser.add_argument(
        "--refresh-lookback-days",
        type=int,
        default=DEFAULT_REFRESH_LOOKBACK_DAYS,
        help="Always refresh games from this many days before --as-of through --as-of.",
    )
    parser.add_argument(
        "--max-refresh-games",
        type=int,
        default=DEFAULT_MAX_REFRESH_GAMES,
        help="Safety cap for one run's forced live-feed refresh.",
    )
    parser.add_argument("--concurrency-limit", type=int, default=15)
    parser.add_argument(
        "--skip-refresh",
        action="store_true",
        help="Skip live-feed refresh/backfill; useful for local smoke checks only.",
    )
    parser.add_argument(
        "--allow-stale-before-as-of",
        action="store_true",
        help="Allow projection when pre-as-of games remain non-final after refresh.",
    )
    parser.add_argument(
        "--post",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Publish the generated graphics. Defaults to dry run.",
    )
    parser.add_argument(
        "--post-provider",
        choices=POST_PROVIDER_CHOICES,
        default="x",
        help="Posting backend used when --post is enabled.",
    )
    parser.add_argument(
        "--caption",
        type=str,
        default=None,
        help="Override social post text.",
    )
    parser.add_argument(
        "--force-post",
        action="store_true",
        help="Ignore the same-day state file and publish another post.",
    )
    parser.add_argument(
        "--market-win-totals",
        type=Path,
        default=None,
        help="Optional preseason market win totals CSV passed to the projection model.",
    )
    parser.add_argument(
        "--calibrate-playoff-probs",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--no-tune-simulation-params",
        action="store_true",
        help="Use fixed projection parameters instead of prior-season tuning.",
    )
    return parser.parse_args()


def _date_from_arg(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise SystemExit(f"--as-of must be YYYY-MM-DD; got {value!r}") from exc


def _coerce_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _has_value(value: object) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def _is_final_row(row: object) -> bool:
    return (
        str(getattr(row, "status", "")) == "Final"
        and _has_value(getattr(row, "away_runs", None))
        and _has_value(getattr(row, "home_runs", None))
    )


def _schedule_snapshot_from_rows(
    rows: Iterable[object],
    *,
    as_of: date,
    refresh_lookback_days: int,
) -> ScheduleSnapshot:
    refresh_start = as_of - timedelta(days=refresh_lookback_days)
    status_counts: Counter[str] = Counter()
    stale_game_pks: set[int] = set()
    refresh_game_pks: set[int] = set()
    total_games = 0
    final_games = 0

    for row in rows:
        total_games += 1
        game_pk = int(row.game_pk)
        game_date = _coerce_date(row.game_date)
        status = str(row.status) or "unknown"
        status_counts[status] += 1
        is_final = _is_final_row(row)
        if is_final:
            final_games += 1
        if game_date < as_of and not is_final:
            stale_game_pks.add(game_pk)
        if refresh_start <= game_date <= as_of:
            refresh_game_pks.add(game_pk)

    refresh_game_pks.update(stale_game_pks)
    return ScheduleSnapshot(
        total_games=total_games,
        final_games=final_games,
        stale_before_as_of=tuple(sorted(stale_game_pks)),
        refresh_game_pks=tuple(sorted(refresh_game_pks)),
        status_counts=dict(sorted(status_counts.items())),
    )


def _load_schedule_snapshot(
    *,
    db_config: PostgresConfig,
    season: int,
    as_of: date,
    refresh_lookback_days: int,
) -> ScheduleSnapshot:
    query = f"""
        WITH scores AS (
            SELECT
                game_pk,
                SUM(runs) FILTER (WHERE team_type = 'away')::int AS away_runs,
                SUM(runs) FILTER (WHERE team_type = 'home')::int AS home_runs
            FROM linescore
            GROUP BY game_pk
        )
        SELECT
            g.game_pk,
            g.game_date,
            COALESCE(g.abstract_game_state, '') AS status,
            s.away_runs,
            s.home_runs
        FROM games AS g
        LEFT JOIN scores AS s USING (game_pk)
        WHERE g.game_type = 'R'
          AND g.season::int = {int(season)}
        ORDER BY g.game_date, g.game_pk
    """
    with PostgresHandler(db_config) as db:
        frame = db.query(query)
    return _schedule_snapshot_from_rows(
        frame.itertuples(index=False),
        as_of=as_of,
        refresh_lookback_days=refresh_lookback_days,
    )


def _print_snapshot(label: str, snapshot: ScheduleSnapshot) -> None:
    statuses = ", ".join(f"{key}={value}" for key, value in snapshot.status_counts.items())
    print(
        f"{label}: total={snapshot.total_games} finals={snapshot.final_games} "
        f"stale_before_as_of={len(snapshot.stale_before_as_of)} "
        f"refresh_candidates={len(snapshot.refresh_game_pks)} statuses=[{statuses}]"
    )


def _ensure_fresh_inputs(
    *,
    db_config: PostgresConfig,
    season: int,
    as_of: date,
    raw_data_path: Path,
    refresh_lookback_days: int,
    max_refresh_games: int,
    concurrency_limit: int,
    skip_refresh: bool,
    allow_stale_before_as_of: bool,
) -> ScheduleSnapshot:
    before = _load_schedule_snapshot(
        db_config=db_config,
        season=season,
        as_of=as_of,
        refresh_lookback_days=refresh_lookback_days,
    )
    _print_snapshot("before_refresh", before)
    if before.total_games == 0:
        raise RuntimeError(f"No regular-season games found in PostgreSQL for {season}")

    if skip_refresh:
        after = before
    elif before.refresh_game_pks:
        if len(before.refresh_game_pks) > max_refresh_games:
            raise RuntimeError(
                f"Refusing to refresh {len(before.refresh_game_pks)} games; "
                f"raise --max-refresh-games above {max_refresh_games} after checking the scope."
            )
        print(f"refreshing_live_feeds: {len(before.refresh_game_pks)} game(s)")
        asyncio.run(
            _download_live_feeds(
                game_pks=before.refresh_game_pks,
                season=season,
                raw_data_path=raw_data_path,
                concurrency_limit=concurrency_limit,
            )
        )
        summary = run_postgres_backfill(
            db_config,
            raw_data_path,
            force_game_pks=before.refresh_game_pks,
        )
        if summary.failed_games:
            raise RuntimeError(f"Postgres refresh failed for {summary.failed_games} game(s)")
        after = _load_schedule_snapshot(
            db_config=db_config,
            season=season,
            as_of=as_of,
            refresh_lookback_days=refresh_lookback_days,
        )
    else:
        after = before

    _print_snapshot("after_refresh", after)
    if after.stale_before_as_of and not allow_stale_before_as_of:
        sample = ", ".join(str(game_pk) for game_pk in after.stale_before_as_of[:10])
        raise RuntimeError(
            f"{len(after.stale_before_as_of)} games before {as_of} are still non-final "
            f"after refresh; sample game_pk(s): {sample}"
        )
    return after


def _projection_outputs(season: int, output_dir: Path | None) -> ProjectionOutputs:
    resolved_dir = output_dir or Path("output") / f"season_projection_{season}"
    return ProjectionOutputs(
        output_dir=resolved_dir,
        projection_csv=resolved_dir / f"season_{season}_model_projection.csv",
        calibration_csv=resolved_dir / f"season_{season}_model_projection_calibration.csv",
        summary_csv=resolved_dir / f"season_{season}_model_projection_summary.csv",
        playoff_probabilities=resolved_dir / f"season_{season}_model_playoff_probabilities.jpg",
        playoff_stages=resolved_dir / f"season_{season}_model_playoff_stages.jpg",
    )


def _projection_command(
    *,
    args: argparse.Namespace,
    as_of: date,
    outputs: ProjectionOutputs,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/backtest_season_projections.py",
        "--seasons",
        str(args.season),
        "--as-of",
        as_of.isoformat(),
        "--trials",
        str(args.trials),
        "--tune-trials",
        str(args.tune_trials),
        "--out",
        str(outputs.projection_csv),
        "--calibration-out",
        str(outputs.calibration_csv),
        "--summary-out",
        str(outputs.summary_csv),
        "--graphics-out-dir",
        str(outputs.output_dir),
    ]
    if args.no_tune_simulation_params:
        command.append("--no-tune-simulation-params")
    if args.calibrate_playoff_probs:
        command.append("--calibrate-playoff-probs")
    if args.market_win_totals is not None:
        command.extend(["--market-win-totals", str(args.market_win_totals)])
    return command


def _run_projection(
    *,
    args: argparse.Namespace,
    as_of: date,
    outputs: ProjectionOutputs,
) -> None:
    outputs.output_dir.mkdir(parents=True, exist_ok=True)
    command = _projection_command(args=args, as_of=as_of, outputs=outputs)
    print("projection_command:", " ".join(command))
    subprocess.run(command, check=True)
    for path in (outputs.projection_csv, outputs.playoff_probabilities, outputs.playoff_stages):
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Projection output missing or empty: {path}")
        print(f"output: {path} bytes={path.stat().st_size}")


def _caption_date(as_of: date) -> str:
    return f"{MONTH_LABELS[as_of.month]} {as_of.day}"


def _default_caption(season: int, as_of: date) -> str:
    return (
        f"{season} MLB season projection as of {_caption_date(as_of)}.\n\n"
        "Playoff odds + playoff stage view."
    )


def _state_path(outputs: ProjectionOutputs, as_of: date) -> Path:
    return outputs.output_dir / f"season_{as_of.year}_{as_of.isoformat()}_post.json"


def _x_url_from_post_id(post_id: str) -> str | None:
    if post_id.isdigit():
        return f"https://x.com/i/web/status/{post_id}"
    if post_id.startswith("multi:"):
        try:
            values = json.loads(post_id.removeprefix("multi:"))
        except json.JSONDecodeError:
            return None
        x_id = values.get("x") if isinstance(values, dict) else None
        if isinstance(x_id, str) and x_id.isdigit():
            return f"https://x.com/i/web/status/{x_id}"
    return None


def _publish_outputs(
    *,
    args: argparse.Namespace,
    as_of: date,
    outputs: ProjectionOutputs,
) -> None:
    image_paths = [outputs.playoff_probabilities, outputs.playoff_stages]
    text = args.caption or _default_caption(args.season, as_of)
    state_path = _state_path(outputs, as_of)

    if not args.post:
        print("dry_run_post: would publish")
        print(text)
        for path in image_paths:
            print(f"dry_run_image: {path}")
        return

    if state_path.exists() and not args.force_post:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        existing_post_id = str(state.get("post_id", ""))
        if existing_post_id:
            print(f"post_skipped_existing_state: {state_path}")
            print(f"existing_post_id: {existing_post_id}")
            existing_url = _x_url_from_post_id(existing_post_id)
            if existing_url:
                print(f"existing_url: {existing_url}")
            return

    publisher = build_publisher(post=True, provider=args.post_provider)
    post_id = publisher.publish_images(text, image_paths)
    state = {
        "season": args.season,
        "as_of": as_of.isoformat(),
        "post_provider": args.post_provider,
        "post_id": post_id,
        "url": _x_url_from_post_id(post_id),
        "caption": text,
        "image_paths": [str(path) for path in image_paths],
        "created_at": datetime.now(UTC).isoformat(),
    }
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"post_id: {post_id}")
    if state["url"]:
        print(f"post_url: {state['url']}")
    print(f"post_state: {state_path}")


def main() -> None:
    args = parse_args()
    as_of = _date_from_arg(args.as_of)
    if args.trials < 1:
        raise SystemExit("--trials must be positive")
    if args.tune_trials < 1:
        raise SystemExit("--tune-trials must be positive")
    if args.refresh_lookback_days < 0:
        raise SystemExit("--refresh-lookback-days must be non-negative")
    if args.max_refresh_games < 1:
        raise SystemExit("--max-refresh-games must be positive")
    if args.concurrency_limit < 1:
        raise SystemExit("--concurrency-limit must be positive")

    db_config = PostgresConfig.from_env()
    print(f"database: {db_config.describe()}")
    _ensure_fresh_inputs(
        db_config=db_config,
        season=args.season,
        as_of=as_of,
        raw_data_path=args.raw_data_path,
        refresh_lookback_days=args.refresh_lookback_days,
        max_refresh_games=args.max_refresh_games,
        concurrency_limit=args.concurrency_limit,
        skip_refresh=args.skip_refresh,
        allow_stale_before_as_of=args.allow_stale_before_as_of,
    )
    outputs = _projection_outputs(args.season, args.output_dir)
    _run_projection(args=args, as_of=as_of, outputs=outputs)
    _publish_outputs(args=args, as_of=as_of, outputs=outputs)


if __name__ == "__main__":
    main()
