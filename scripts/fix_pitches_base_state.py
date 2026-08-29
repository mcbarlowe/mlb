"""In-place repair of at-bat base/out state columns in ``mlb.pitches``.

The legacy ETL wrote a dead ``outs`` column (always 0) and mover-only
``is_runner_on_*`` flags. The extraction is fixed for new loads
(``src/data/base_state.py``); this script back-fills the EXISTING rows by
reconstructing true at-bat start state from the archived live feed JSONs
and updating each game's rows in place — no truncation, idempotent, safe
to re-run or resume by season.

    uv run python scripts/fix_pitches_base_state.py                # all seasons on disk
    uv run python scripts/fix_pitches_base_state.py --seasons 2024 2025
    uv run python scripts/fix_pitches_base_state.py --game-pk 745340 --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mlb.data.base_state import compute_at_bat_states

LIVEFEED_ROOT = Path("data/raw/livefeeds")

UPDATE_TEMPLATE = """
UPDATE {schema}.pitches AS p SET
    outs = v.outs,
    is_runner_on_first = v.r1,
    runner_on_first_id = v.rid1,
    is_runner_on_second = v.r2,
    runner_on_second_id = v.rid2,
    is_runner_on_third = v.r3,
    runner_on_third_id = v.rid3
FROM (VALUES {values}) AS v(at_bat_index, outs, r1, rid1, r2, rid2, r3, rid3)
WHERE p.game_pk = %s AND p.at_bat_index = v.at_bat_index
"""

VALUE_ROW = "(%s::int, %s::int, %s::bool, %s::float8, %s::bool, %s::float8, %s::bool, %s::float8)"


def extract_game_states(feed_path: Path) -> tuple[int, list[tuple]] | None:
    """(game_pk, per-at-bat state rows) for one archived feed."""
    try:
        feed = json.loads(feed_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    game_pk = feed.get("gameData", {}).get("game", {}).get("pk")
    plays = feed.get("liveData", {}).get("plays", {}).get("allPlays", [])
    if game_pk is None or not plays:
        return None
    states = compute_at_bat_states(plays)
    rows = []
    for play, state in zip(plays, states):
        at_bat_index = play.get("about", {}).get("atBatIndex")
        if at_bat_index is None:
            continue
        rows.append(
            (
                int(at_bat_index),
                int(state["outs_before"]),
                bool(state["is_runner_on_first"]),
                float(state["runner_on_first_id"]) if state["runner_on_first_id"] else None,
                bool(state["is_runner_on_second"]),
                float(state["runner_on_second_id"]) if state["runner_on_second_id"] else None,
                bool(state["is_runner_on_third"]),
                float(state["runner_on_third_id"]) if state["runner_on_third_id"] else None,
            )
        )
    return int(game_pk), rows


def update_game(cursor, schema: str, game_pk: int, rows: list[tuple]) -> int:
    values_sql = ", ".join([VALUE_ROW] * len(rows))
    params: list = []
    for row in rows:
        params.extend(row)
    params.append(game_pk)
    cursor.execute(
        UPDATE_TEMPLATE.format(schema=schema, values=values_sql).encode("utf-8"),
        params,
    )
    return cursor.rowcount


def season_files(seasons: list[int] | None) -> list[Path]:
    if seasons:
        dirs = [LIVEFEED_ROOT / str(season) for season in seasons]
    else:
        dirs = sorted(p for p in LIVEFEED_ROOT.iterdir() if p.is_dir())
    files: list[Path] = []
    for directory in dirs:
        if not directory.is_dir():
            raise SystemExit(f"Missing live feed directory {directory}")
        files.extend(sorted(directory.glob("*.json")))
    return files


def main() -> None:
    import psycopg

    from mlb.database import PostgresConfig

    parser = argparse.ArgumentParser(description="Repair pitches base/out state.")
    parser.add_argument("--seasons", type=int, nargs="+", default=None)
    parser.add_argument("--game-pk", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--commit-every", type=int, default=100)
    args = parser.parse_args()

    config = PostgresConfig.from_env()
    conninfo = {
        "dbname": config.dbname,
        "user": config.user,
        "password": config.password,
        "host": config.host,
        "port": config.port,
    }
    conninfo = {k: v for k, v in conninfo.items() if v is not None}

    if args.game_pk:
        path = next(
            (
                p
                for p in (
                    LIVEFEED_ROOT / d.name / f"{args.game_pk}.json"
                    for d in sorted(LIVEFEED_ROOT.iterdir())
                )
                if p.exists()
            ),
            None,
        )
        if path is None:
            raise SystemExit(f"No feed for game {args.game_pk}")
        extracted = extract_game_states(path)
        if extracted is None:
            raise SystemExit(f"Feed for game {args.game_pk} has no plays")
        game_pk, rows = extracted
        print(f"Game {game_pk}: {len(rows)} at-bats reconstructed")
        for row in rows[:12]:
            print("  ", row)
        if args.dry_run:
            return
        with psycopg.connect(**conninfo) as connection, connection.cursor() as cursor:
            updated = update_game(cursor, config.schema, game_pk, rows)
            connection.commit()
        print(f"Updated {updated:,} pitch rows")
        return

    files = season_files(args.seasons)
    print(f"Repairing base/out state from {len(files):,} feeds...")
    start = time.perf_counter()
    total_rows = 0
    games_updated = 0
    games_skipped = 0

    with psycopg.connect(**conninfo) as connection, connection.cursor() as cursor:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for index, extracted in enumerate(
                pool.map(extract_game_states, files, chunksize=25), start=1
            ):
                if extracted is None:
                    games_skipped += 1
                else:
                    game_pk, rows = extracted
                    if rows:
                        updated = update_game(cursor, config.schema, game_pk, rows)
                        total_rows += updated
                        games_updated += 1 if updated else 0
                if index % args.commit_every == 0:
                    connection.commit()
                if index % 2000 == 0:
                    elapsed = time.perf_counter() - start
                    print(
                        f"  {index:,}/{len(files):,} feeds; {games_updated:,} games,"
                        f" {total_rows:,} rows updated; {elapsed:.0f}s"
                    )
        connection.commit()

    print(
        f"Done in {time.perf_counter() - start:.0f}s: {games_updated:,} games,"
        f" {total_rows:,} pitch rows updated; {games_skipped:,} feeds skipped"
    )


if __name__ == "__main__":
    main()
