"""Live next-pitch prediction pipeline.

Ties together schedule-driven game monitoring, per-poll snapshot
building, model prediction, card rendering, and publishing.
"""

from __future__ import annotations

import asyncio
import random
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt

from src.etl.daily_pipeline import DailyPipeline, GameState, extract_game_status
from src.live.game_state import LiveSnapshot, build_live_snapshot
from src.live.predictor import LiveNextPitchPredictor
from src.live.publisher import PredictionPost, Publisher, build_post_text


def parse_game_start(game: dict) -> datetime | None:
    """Parse a schedule entry's gameDate into an aware UTC datetime."""
    raw = game.get("gameDate")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw))
    except ValueError:
        return None


def earliest_start(games: list[dict]) -> datetime | None:
    """Earliest scheduled first pitch among a day's games."""
    starts = [start for game in games if (start := parse_game_start(game))]
    return min(starts) if starts else None


def seconds_until_monitoring(
    games: list[dict],
    now: datetime,
    lead: timedelta,
) -> float:
    """Seconds to sleep before monitoring should begin (0 when due)."""
    start = earliest_start(games)
    if start is None:
        return 0.0
    return max((start - lead - now).total_seconds(), 0.0)


class LiveGamePredictionService:
    """Per-poll handler: detect new pitch state, predict, render, publish."""

    def __init__(
        self,
        predictor: LiveNextPitchPredictor,
        publisher: Publisher,
        output_dir: str | Path = Path("output/live_cards"),
        post_cadence: str = "at_bat",
        max_posts_per_game: int = 40,
    ):
        if post_cadence not in {"at_bat", "pitch"}:
            raise ValueError("post_cadence must be 'at_bat' or 'pitch'")
        self.predictor = predictor
        self.publisher = publisher
        self.output_dir = Path(output_dir)
        self.post_cadence = post_cadence
        self.max_posts_per_game = max_posts_per_game
        self._last_pitch_key: dict[int, tuple[int, int, int, int]] = {}
        self._last_posted_at_bat: dict[int, int] = {}
        self._posts_per_game: dict[int, int] = {}

    def should_post(self, snapshot: LiveSnapshot) -> bool:
        """Posting policy: every pitch, or once per new at-bat."""
        if self._posts_per_game.get(snapshot.game_pk, 0) >= self.max_posts_per_game:
            return False
        if self.post_cadence == "pitch":
            return True
        return (
            self._last_posted_at_bat.get(snapshot.game_pk) != snapshot.at_bat_index
        )

    def handle_feed(self, game_pk: int, feed: dict) -> dict | None:
        """Process one poll of one game; returns a summary when predicted."""
        snapshot = build_live_snapshot(feed)
        if snapshot is None:
            return None
        if self._last_pitch_key.get(game_pk) == snapshot.pitch_key:
            return None
        self._last_pitch_key[game_pk] = snapshot.pitch_key

        prediction = self.predictor.predict(snapshot)

        card_dir = self.output_dir / str(game_pk)
        card_dir.mkdir(parents=True, exist_ok=True)
        card_path = card_dir / (
            f"ab{snapshot.at_bat_index:03d}_pitch{snapshot.next_pitch_number:02d}"
            f"_{snapshot.balls}-{snapshot.strikes}.png"
        )
        figure = self.predictor.pitch_predictor.create_pitch_card(
            prediction,
            snapshot.context,
            save_path=str(card_path),
        )
        plt.close(figure)

        summary = {
            "game_pk": game_pk,
            "at_bat_index": snapshot.at_bat_index,
            "next_pitch_number": snapshot.next_pitch_number,
            "count": f"{snapshot.balls}-{snapshot.strikes}",
            "predicted_type": prediction.predicted_type,
            "top_3": [
                (code, float(prob)) for code, prob in prediction.top_3_types
            ],
            "card_path": str(card_path),
            "posted": False,
        }

        if self.should_post(snapshot):
            post = PredictionPost(
                text=build_post_text(snapshot, prediction),
                image_path=card_path,
            )
            post_id = self.publisher.publish(post)
            self._last_posted_at_bat[snapshot.game_pk] = snapshot.at_bat_index
            self._posts_per_game[snapshot.game_pk] = (
                self._posts_per_game.get(snapshot.game_pk, 0) + 1
            )
            summary["posted"] = True
            summary["post_id"] = post_id

        print(
            f"[game {game_pk}] AB {snapshot.at_bat_index} pitch "
            f"{snapshot.next_pitch_number} ({snapshot.balls}-{snapshot.strikes}): "
            f"{prediction.predicted_type} "
            f"{prediction.top_3_types[0][1]:.0%}"
            + (" [posted]" if summary["posted"] else "")
        )
        return summary


async def run_live_day(
    target_date: date,
    service: LiveGamePredictionService,
    poll_interval: float = 20.0,
    lead_minutes: float = 15.0,
    concurrency_limit: int = 8,
) -> dict:
    """Monitor a full day: wait for first pitch, then poll and predict.

    Uses the MLB schedule to sleep until `lead_minutes` before the
    earliest scheduled start, then polls every game through completion.
    """

    def on_game_update(game_pk: int, feed: dict, state: GameState) -> None:
        if state != GameState.LIVE:
            return
        try:
            service.handle_feed(game_pk, feed)
        except Exception as exc:
            print(f"[game {game_pk}] prediction failed: {exc}")

    async with DailyPipeline(
        concurrency_limit=concurrency_limit,
        poll_interval=poll_interval,
        on_game_update=on_game_update,
    ) as pipeline:
        games = await pipeline.get_games_for_date(target_date)
        if not games:
            print(f"No games scheduled for {target_date.isoformat()}")
            return {"date": target_date.isoformat(), "games": 0}

        delay = seconds_until_monitoring(
            games,
            now=datetime.now(tz=UTC),
            lead=timedelta(minutes=lead_minutes),
        )
        if delay > 0:
            first = earliest_start(games)
            print(
                f"{len(games)} games scheduled; first pitch at {first}. "
                f"Sleeping {delay/60:.1f} minutes before monitoring."
            )
            await asyncio.sleep(delay)

        return await pipeline.monitor_all_games(target_date, skip_existing=False)


async def run_live_game(
    game_pk: int,
    service: LiveGamePredictionService,
    poll_interval: float = 20.0,
    concurrency_limit: int = 4,
) -> dict:
    """Poll a single game until it reaches a terminal state."""
    from src.endpoints.game_feed import GameFeed

    predictions = 0
    async with GameFeed(concurrency_limit=concurrency_limit) as game_feed:
        while True:
            feed = await game_feed.get_async(game_pk)
            _, state = extract_game_status(feed)

            if state == GameState.LIVE:
                try:
                    if service.handle_feed(game_pk, feed) is not None:
                        predictions += 1
                except Exception as exc:
                    print(f"[game {game_pk}] prediction failed: {exc}")
            elif state in (
                GameState.FINAL,
                GameState.CANCELLED,
                GameState.POSTPONED,
                GameState.SUSPENDED,
            ):
                print(f"[game {game_pk}] reached state {state.value}; stopping")
                return {
                    "game_pk": game_pk,
                    "final_state": state.value,
                    "predictions": predictions,
                }
            else:
                print(f"[game {game_pk}] state {state.value}; waiting for start")

            await asyncio.sleep(poll_interval)



def eligible_games(games: list[dict]) -> list[dict]:
    """Games that can still be followed today (not final/postponed/cancelled)."""
    from src.etl.daily_pipeline import classify_game_state

    keep = (GameState.SCHEDULED, GameState.LIVE, GameState.UNKNOWN)
    return [
        game
        for game in games
        if classify_game_state(game.get("status", {}).get("statusCode", "")) in keep
    ]


def choose_random_game(games: list[dict], rng: random.Random) -> dict | None:
    """Pick one followable game at random from a day's schedule."""
    candidates = eligible_games(games)
    if not candidates:
        return None
    return rng.choice(candidates)


def describe_game(game: dict) -> str:
    away = game.get("teams", {}).get("away", {}).get("team", {}).get("name", "?")
    home = game.get("teams", {}).get("home", {}).get("team", {}).get("name", "?")
    return f"{away} @ {home} (gamePk {game.get('gamePk')}, start {game.get('gameDate')})"


async def run_random_live_game(
    target_date: date,
    service: LiveGamePredictionService,
    poll_interval: float = 20.0,
    lead_minutes: float = 15.0,
    seed: int | None = None,
) -> dict:
    """Pick one game at random from the schedule and follow only that game.

    Waits until `lead_minutes` before the chosen game's scheduled start,
    then polls it through completion.
    """
    rng = random.Random(seed)

    async with DailyPipeline(concurrency_limit=4) as pipeline:
        games = await pipeline.get_games_for_date(target_date)

    chosen = choose_random_game(games, rng)
    if chosen is None:
        print(f"No followable games on {target_date.isoformat()}")
        return {"date": target_date.isoformat(), "games": 0}

    print(f"Randomly selected: {describe_game(chosen)}")

    delay = seconds_until_monitoring(
        [chosen],
        now=datetime.now(tz=UTC),
        lead=timedelta(minutes=lead_minutes),
    )
    if delay > 0:
        print(f"Sleeping {delay/60:.1f} minutes until game time...")
        await asyncio.sleep(delay)

    game_pk = int(chosen["gamePk"])
    result = await run_live_game(game_pk, service, poll_interval=poll_interval)
    result["selected_game"] = describe_game(chosen)
    return result