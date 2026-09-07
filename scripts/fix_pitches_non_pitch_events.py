"""Repair pitch rows corrupted by the non-pitch-event extraction bug.

The legacy extraction walked ``play["pitchIndex"]`` and emitted a row for every
referenced ``playEvents`` entry. But ``pitchIndex`` also points at events with
``isPitch: False`` — pickoff attempts, pitcher step-offs, and the "Automatic
Ball/Strike" events produced by intentional walks and pitch-timer violations.
Those events either carry no ``pitchNumber`` (so ``pitch.get("pitchNumber", 0)``
stored 0) or REUSE the preceding real pitch's ``pitchNumber``. Combined with
``pitches_pk (season, game_pk, at_bat_index, pitch_number)`` and
``ON CONFLICT DO NOTHING``, that produced two defects:

  * junk rows parked at ``pitch_number = 0`` (one per collided at-bat), and
  * real pitches EVICTED by an "Automatic ..." event that stole their key.

The extraction is fixed for new loads (``mlb/data/game_feed_data.py`` skips
``isPitch is not True``), so games loaded from 2026-08-25 onward are clean.
This script repairs the games loaded before that by re-fetching each live feed
and rewriting that game's pitch rows from the corrected extraction.

Targets are detected from the database, not from local files:

  * games whose pitch rows stop before the linescore's final inning,
  * games with gaps in ``at_bat_index``,
  * final games with no pitch rows at all,
  * games holding an ``Automatic %`` row at ``pitch_number > 0``.

Each game is rewritten inside its own transaction, scoped by ``game_pk``.
Idempotent and safe to re-run or interrupt.

    uv run python scripts/fix_pitches_non_pitch_events.py --dry-run
    uv run python scripts/fix_pitches_non_pitch_events.py
    uv run python scripts/fix_pitches_non_pitch_events.py --game-pk 822942
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

FEED_URL = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
FETCH_CHUNK = 16

TARGET_SQL = """
WITH p AS (
    SELECT game_pk,
           MAX(inning) AS last_pitch_inning,
           COUNT(DISTINCT at_bat_index) AS at_bats_seen,
           MAX(at_bat_index) AS max_at_bat,
           COUNT(*) FILTER (
               WHERE pitch_number > 0 AND pitch_call_description LIKE 'Automatic %%'
           ) AS evicted
    FROM {schema}.pitches
    GROUP BY 1
), l AS (
    SELECT game_pk, MAX(inning) AS last_inning FROM {schema}.linescore GROUP BY 1
)
SELECT g.game_pk,
       g.season,
       COALESCE(p.evicted, 0) AS evicted,
       (p.game_pk IS NULL) AS no_pitches,
       (p.last_pitch_inning < l.last_inning) AS truncated,
       (p.at_bats_seen <> p.max_at_bat + 1) AS at_bat_gaps
FROM {schema}.games g
LEFT JOIN p ON p.game_pk = g.game_pk
LEFT JOIN l ON l.game_pk = g.game_pk
WHERE g.coded_game_state = 'F'
  AND (
        p.game_pk IS NULL
     OR p.last_pitch_inning < l.last_inning
     OR p.at_bats_seen <> p.max_at_bat + 1
     OR COALESCE(p.evicted, 0) > 0
  )
ORDER BY g.season, g.game_pk
"""

CHILD_COUNT_SQL = """
SELECT (SELECT COUNT(*) FROM {schema}.linescore WHERE game_pk = %(pk)s) AS linescore,
       (SELECT COUNT(*) FROM {schema}.batting   WHERE game_pk = %(pk)s) AS batting,
       (SELECT COUNT(*) FROM {schema}.pitching  WHERE game_pk = %(pk)s) AS pitching,
       (SELECT COUNT(*) FROM {schema}.fielding  WHERE game_pk = %(pk)s) AS fielding
"""


def fetch_feed(game_pk: int, attempts: int = 4) -> dict:
    """Fetch one live feed, retrying transient HTTP/network failures."""
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(
                FEED_URL.format(game_pk=game_pk), timeout=120
            ) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"feed fetch failed for {game_pk}: {last_error}")


def iter_feeds(
    pool: ThreadPoolExecutor, targets: list[dict]
) -> Iterator[tuple[dict, dict]]:
    """Yield (target, feed) pairs, prefetching at most FETCH_CHUNK feeds at a time.

    A parsed feed is ~10 MB, so submitting every target to the pool at once would
    let completed downloads pile up far faster than the database consumes them.
    """
    for start in range(0, len(targets), FETCH_CHUNK):
        chunk = targets[start : start + FETCH_CHUNK]
        yield from zip(chunk, pool.map(lambda t: fetch_feed(t["game_pk"]), chunk))


PURGE_SAFETY_SQL = """
SELECT count(*) AS rows,
       count(*) FILTER (
           WHERE pitch_start_speed IS NOT NULL
              OR pitch_end_speed IS NOT NULL
              OR pitch_type_code IS NOT NULL
              OR px IS NOT NULL
              OR pz IS NOT NULL
       ) AS tracked,
       count(*) FILTER (
           WHERE pitch_call_description NOT LIKE 'Pickoff%%'
             AND pitch_call_description NOT LIKE 'Automatic%%'
             AND pitch_call_description <> 'Pitcher Step Off'
             AND pitch_call_description <> 'None'
             AND pitch_call_description IS NOT NULL
       ) AS unexpected
FROM {schema}.pitches
WHERE pitch_number = 0
"""


def purge_non_pitch_rows(db, schema: str, dry_run: bool) -> None:
    """Delete the ``pitch_number = 0`` rows left behind by the legacy extraction.

    Real pitches are always numbered from 1, so every row at 0 is a non-pitch
    event (pickoff attempt, step-off, automatic ball/strike) that the fixed
    extraction no longer emits. The delete refuses to run unless every such row
    is provably a non-pitch: no tracking measurements and no description outside
    the known non-pitch set.
    """
    with db.connection.cursor() as cursor:
        cursor.execute(PURGE_SAFETY_SQL.format(schema=schema).encode("utf-8"))
        rows, tracked, unexpected = cursor.fetchone() or (0, 0, 0)
        print(
            f"pitch_number=0 rows: {rows} "
            f"(with tracking data: {tracked}, unexpected description: {unexpected})"
        )
        if not rows:
            print("nothing to purge")
            return
        if tracked or unexpected:
            raise SystemExit(
                "refusing to purge: found rows at pitch_number=0 that look like real "
                f"pitches (tracked={tracked}, unexpected_description={unexpected})"
            )
        if dry_run:
            print(f"dry run: would delete {rows} non-pitch rows")
            return
        cursor.execute(
            f"DELETE FROM {schema}.pitches WHERE pitch_number = 0".encode()
        )
        deleted = cursor.rowcount
        cursor.execute(
            f"SELECT count(*) FROM {schema}.pitches WHERE pitch_number = 0".encode()
        )
        remaining = (cursor.fetchone() or (0,))[0]
        cursor.execute(f"SELECT count(*) FROM {schema}.pitches".encode())
        total = (cursor.fetchone() or (0,))[0]
    print(f"deleted {deleted} non-pitch rows; remaining at pitch_number=0: {remaining}")
    print(f"mlb.pitches now holds {total} rows")


def main() -> None:
    from mlb.data import BoxscoreData, GameFeedData, LinescoreData, PlayerData, TeamData
    from mlb.data.venue_data import VenueData
    from mlb.database import PostgresConfig, PostgresHandler

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report targets, write nothing")
    parser.add_argument("--game-pk", type=int, nargs="*", help="repair only these games")
    parser.add_argument("--limit", type=int, help="cap the number of games repaired")
    parser.add_argument("--workers", type=int, default=8, help="concurrent feed downloads")
    parser.add_argument(
        "--purge-non-pitch-rows",
        action="store_true",
        help="delete the leftover pitch_number=0 non-pitch rows and exit",
    )
    args = parser.parse_args()

    config = PostgresConfig.from_env()
    db = PostgresHandler(config)
    schema = db.schema
    print(f"database: {config.describe()}")

    if args.purge_non_pitch_rows:
        purge_non_pitch_rows(db, schema, args.dry_run)
        return

    pitch_transformer = GameFeedData()
    linescore_transformer = LinescoreData()
    boxscore_transformer = BoxscoreData()
    player_transformer = PlayerData()
    team_transformer = TeamData()
    venue_transformer = VenueData()

    with db.connection.cursor() as cursor:
        cursor.execute(TARGET_SQL.format(schema=schema).encode("utf-8"))
        targets = [
            {
                "game_pk": int(row[0]),
                "season": int(row[1]),
                "evicted": int(row[2]),
                "no_pitches": bool(row[3]),
                "truncated": bool(row[4]),
                "at_bat_gaps": bool(row[5]),
            }
            for row in cursor.fetchall()
        ]

    if args.game_pk:
        wanted = set(args.game_pk)
        targets = [t for t in targets if t["game_pk"] in wanted]
    if args.limit:
        targets = targets[: args.limit]

    print(
        f"targets: {len(targets)} games "
        f"(no pitches={sum(t['no_pitches'] for t in targets)}, "
        f"truncated={sum(t['truncated'] for t in targets)}, "
        f"at-bat gaps={sum(t['at_bat_gaps'] for t in targets)}, "
        f"evicted rows={sum(t['evicted'] for t in targets)})"
    )
    if args.dry_run or not targets:
        print("dry run: nothing written" if args.dry_run else "nothing to repair")
        return

    existing_players = db.query(f"SELECT player_id FROM {schema}.players")
    known_players = {int(v) for v in existing_players["player_id"].dropna().tolist()}
    existing_teams = db.query(f"SELECT team_id FROM {schema}.teams")
    known_teams = {int(v) for v in existing_teams["team_id"].dropna().tolist()}
    existing_venues = db.query(f"SELECT venue_id FROM {schema}.venues")
    known_venues = {int(v) for v in existing_venues["venue_id"].dropna().tolist()}

    stats = {"repaired": 0, "failed": 0, "pitches_before": 0, "pitches_after": 0, "children": 0}
    started = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for target, feed in iter_feeds(pool, targets):
            game_pk = target["game_pk"]
            season = target["season"]
            try:
                pitches_df = pitch_transformer.transform(feed, game_id=game_pk, season=season)
                if pitches_df.empty:
                    print(f"  ! {game_pk}: feed has no pitches, skipping")
                    continue

                teams_df = team_transformer.transform(feed)
                new_teams = teams_df[~teams_df["team_id"].isin(known_teams)]
                venue_df = venue_transformer.transform(feed)
                new_venue = venue_df[~venue_df["venue_id"].isin(known_venues)]
                players_df = player_transformer.transform(feed)
                new_players = players_df[~players_df["player_id"].isin(known_players)]

                with db.connection.cursor() as cursor:
                    cursor.execute(
                        f"SELECT COUNT(*) FROM {schema}.pitches WHERE game_pk = %s".encode(),
                        (game_pk,),
                    )
                    before_row = cursor.fetchone()
                    cursor.execute(
                        CHILD_COUNT_SQL.format(schema=schema).encode("utf-8"), {"pk": game_pk}
                    )
                    child_row = cursor.fetchone()

                before = int(before_row[0]) if before_row else 0
                needs_children = any(count == 0 for count in (child_row or ()))

                with db.connection.transaction():
                    if not new_teams.empty:
                        team_transformer.save_to_db(new_teams, db)
                    if not new_venue.empty:
                        venue_transformer.save_to_db(new_venue, db)
                    if not new_players.empty:
                        player_transformer.save_to_db(new_players, db)

                    with db.connection.cursor() as cursor:
                        cursor.execute(
                            f"DELETE FROM {schema}.pitches WHERE game_pk = %s".encode(),
                            (game_pk,),
                        )
                    pitch_transformer.save_to_db(pitches_df, db)

                    if needs_children:
                        linescore_df = linescore_transformer.transform(feed, game_pk=game_pk)
                        boxscore_data = boxscore_transformer.transform_all(feed, game_pk=game_pk)
                        with db.connection.cursor() as cursor:
                            for table in ("linescore", "batting", "pitching", "fielding"):
                                cursor.execute(
                                    f"DELETE FROM {schema}.{table} WHERE game_pk = %s".encode(),
                                    (game_pk,),
                                )
                        if not linescore_df.empty:
                            linescore_transformer.save_to_db(linescore_df, db)
                        for table_name, frame in boxscore_data.items():
                            if not frame.empty:
                                boxscore_transformer.save_to_db(
                                    frame, table_name.split("_")[1], db
                                )
                        stats["children"] += 1

                known_teams.update(int(v) for v in new_teams["team_id"].tolist())
                known_venues.update(int(v) for v in new_venue["venue_id"].tolist())
                known_players.update(int(v) for v in new_players["player_id"].tolist())

                stats["repaired"] += 1
                stats["pitches_before"] += before
                stats["pitches_after"] += len(pitches_df)
            except Exception as exc:  # keep repairing the remaining games
                stats["failed"] += 1
                print(f"  ! {game_pk}: {type(exc).__name__}: {exc}")

            done = stats["repaired"] + stats["failed"]
            if done % 100 == 0:
                rate = done / max(time.time() - started, 1e-6)
                print(
                    f"  {done}/{len(targets)} games  "
                    f"({rate:.1f}/s, {stats['pitches_after'] - stats['pitches_before']:+d} rows)"
                )

    print(
        f"\nrepaired {stats['repaired']} games ({stats['failed']} failed) in "
        f"{time.time() - started:.0f}s\n"
        f"pitch rows: {stats['pitches_before']} -> {stats['pitches_after']} "
        f"({stats['pitches_after'] - stats['pitches_before']:+d})\n"
        f"games whose linescore/boxscore were also reloaded: {stats['children']}"
    )


if __name__ == "__main__":
    main()
