"""In-place repair of spin columns in ``mlb.pitches``.

The legacy extraction read ``pitchData.spinRate`` / ``pitchData.spinDirection``,
but the live feed carries spin under ``pitchData.breaks`` — so ``spin_rate``
was never populated and ``spin_direction`` was stored as the literal string
``'None'`` for every row. The extraction is fixed for new loads
(``src/data/game_feed_data.py``); this script back-fills the EXISTING rows
from the archived live feed JSONs, updating each game's pitch rows in place
(real value where the feed has one, true NULL otherwise — which also scrubs
the ``'None'`` strings). Idempotent and safe to re-run or resume by season.

    uv run python scripts/fix_pitches_spin.py                 # all seasons on disk
    uv run python scripts/fix_pitches_spin.py --seasons 2024 2025
    uv run python scripts/fix_pitches_spin.py --game-pk 745340 --dry-run
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

LIVEFEED_ROOT = Path("data/raw/livefeeds")

UPDATE_TEMPLATE = """
UPDATE {schema}.pitches AS p SET
    spin_rate = v.spin_rate,
    spin_direction = v.spin_direction
FROM (VALUES {values}) AS v(at_bat_index, pitch_number, spin_rate, spin_direction)
WHERE p.game_pk = %s
  AND p.at_bat_index = v.at_bat_index
  AND p.pitch_number = v.pitch_number
"""

VALUE_ROW = "(%s::int, %s::int, %s::float8, %s::text)"


def extract_game_spin(feed_path: Path) -> tuple[int, list[tuple]] | None:
    """(game_pk, per-pitch spin rows) for one archived feed."""
    try:
        feed = json.loads(feed_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    game_pk = feed.get("gameData", {}).get("game", {}).get("pk")
    plays = feed.get("liveData", {}).get("plays", {}).get("allPlays", [])
    if game_pk is None or not plays:
        return None
    rows: list[tuple] = []
    for play in plays:
        at_bat_index = play.get("about", {}).get("atBatIndex")
        if at_bat_index is None:
            continue
        for event in play.get("playEvents", []):
            if not event.get("isPitch"):
                continue
            pitch_number = event.get("pitchNumber")
            if pitch_number is None:
                continue
            pitch_data = event.get("pitchData", {})
            breaks = pitch_data.get("breaks", {})
            spin = breaks.get("spinRate", pitch_data.get("spinRate"))
            direction = breaks.get(
                "spinDirection", pitch_data.get("spinDirection")
            )
            rows.append(
                (
                    int(at_bat_index),
                    int(pitch_number),
                    float(spin) if spin is not None else None,
                    str(direction) if direction is not None else None,
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

    from src.database import PostgresConfig

    parser = argparse.ArgumentParser(description="Repair pitches spin columns.")
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
        extracted = extract_game_spin(path)
        if extracted is None:
            raise SystemExit(f"Feed for game {args.game_pk} has no plays")
        game_pk, rows = extracted
        with_spin = sum(1 for r in rows if r[2] is not None)
        print(f"Game {game_pk}: {len(rows)} pitches, {with_spin} with spin")
        for row in rows[:10]:
            print("  ", row)
        if args.dry_run:
            return
        with psycopg.connect(**conninfo) as connection, connection.cursor() as cursor:
            updated = update_game(cursor, config.schema, game_pk, rows)
            connection.commit()
        print(f"Updated {updated:,} pitch rows")
        return

    files = season_files(args.seasons)
    print(f"Repairing spin columns from {len(files):,} feeds...")
    start = time.perf_counter()
    total_rows = 0
    games_updated = 0
    games_skipped = 0

    with psycopg.connect(**conninfo) as connection, connection.cursor() as cursor:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for index, extracted in enumerate(
                pool.map(extract_game_spin, files, chunksize=25), start=1
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
