"""Live next-pitch prediction pipeline.

Ties together schedule-driven game monitoring, per-poll snapshot
building, model prediction, card rendering, and publishing.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt

from mlb.etl.daily_pipeline import DailyPipeline, GameState, extract_game_status
from mlb.live.game_state import LiveSnapshot, build_live_snapshot
from mlb.live.predictor import LiveNextPitchPredictor
from mlb.live.publisher import (
    PredictionPost,
    Publisher,
    ResultPost,
    build_post_text,
    build_result_text,
)
from mlb.ml.pitch_predictor import GameContext, PitchPrediction


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


@dataclass(frozen=True)
class PendingPostedPrediction:
    game_pk: int
    at_bat_index: int
    pitch_number: int
    inning_key: tuple[str, int]
    prediction: PitchPrediction
    prediction_context: GameContext
    post_id: str
    card_path: Path


class LiveGamePredictionService:
    """Per-poll handler: detect new pitch state, predict, render, publish."""

    def __init__(
        self,
        predictor: LiveNextPitchPredictor,
        publisher: Publisher,
        output_dir: str | Path = Path("output/live_cards"),
        post_cadence: str = "at_bat",
        max_posts_per_game: int = 40,
        random_pitch_ceiling: int = 4,
        seed: int | None = None,
        card_style: str = "html",
        outcome_predictor=None,
    ):
        if post_cadence not in {"at_bat", "pitch", "random_pitch", "half_inning"}:
            raise ValueError(
                "post_cadence must be 'at_bat', 'pitch', 'random_pitch', or 'half_inning'"
            )
        if random_pitch_ceiling < 1:
            raise ValueError("random_pitch_ceiling must be >= 1")
        if card_style not in {"html", "matplotlib"}:
            raise ValueError("card_style must be 'html' or 'matplotlib'")
        self.predictor = predictor
        self.publisher = publisher
        self.output_dir = Path(output_dir)
        self.post_cadence = post_cadence
        self.max_posts_per_game = max_posts_per_game
        self.random_pitch_ceiling = random_pitch_ceiling
        self.card_style = card_style
        self.outcome_predictor = outcome_predictor
        self._rng = random.Random(seed)
        self._html_renderer = None
        self._last_pitch_key: dict[int, tuple[int, int, int, int]] = {}
        self._last_posted_at_bat: dict[int, int] = {}
        self._last_posted_inning: dict[int, tuple[str, int]] = {}
        self._posts_per_game: dict[int, int] = {}
        self._ab_target_pitch: dict[tuple[int, int], int] = {}
        self._pending_posted_predictions: dict[int, PendingPostedPrediction] = {}

    def _predict_outcome(self, snapshot: LiveSnapshot, prediction) -> dict | None:
        """Marginal outcome odds for the predicted pitch (None on any failure)."""
        if self.outcome_predictor is None:
            return None
        try:
            from mlb.ml.features import PITCH_TYPE_CODES
            from mlb.outcome.inference import (
                OutcomeGameState,
                sample_locations_from_grid,
            )

            context = snapshot.context
            frame = snapshot.frame

            pitcher_abs = (
                frame.filter(frame["pitcher_id"] == context.pitcher_id)
                .get_column("at_bat_index")
                .n_unique()
            )
            season_value = frame.get_column("season").tail(1).item()
            sz_top = frame.get_column("pitch_strike_zone_top").drop_nulls()
            sz_bottom = frame.get_column("pitch_strike_zone_bottom").drop_nulls()

            is_top = context.inning_half == "Top"
            score_home = context.score_home or 0
            score_away = context.score_away or 0
            state = OutcomeGameState(
                balls=snapshot.balls,
                strikes=snapshot.strikes,
                outs=snapshot.outs,
                runner_on_first=context.runner_on_1b,
                runner_on_second=context.runner_on_2b,
                runner_on_third=context.runner_on_3b,
                inning=context.inning,
                is_top_half=is_top,
                score_diff=(score_away - score_home) if is_top else (score_home - score_away),
                season=int(str(season_value)),
                times_through_order=((max(pitcher_abs, 1) - 1) // 9) + 1,
                pitcher_id=context.pitcher_id,
                batter_id=context.batter_id,
                throw_side=context.pitcher_hand,
                bat_side=context.batter_hand,
                sz_top=float(sz_top.tail(1).item()) if len(sz_top) else 3.4,
                sz_bottom=float(sz_bottom.tail(1).item()) if len(sz_bottom) else 1.6,
            )
            type_probabilities = {
                code: float(prob)
                for code, prob in zip(
                    PITCH_TYPE_CODES, prediction.type_probabilities
                )
            }
            locations = sample_locations_from_grid(
                prediction.px_grid,
                prediction.pz_grid,
                prediction.location_density,
                n_samples=40,
                seed=snapshot.pitch_key[0] * 100 + snapshot.pitch_key[1],
            )
            return self.outcome_predictor.predict(state, type_probabilities, locations)
        except Exception as exc:
            print(f"Outcome prediction failed ({exc}); rendering without it")
            return None

    def _render_card(
        self,
        prediction,
        context,
        out_path: Path,
        actual_pitch_type: str | None = None,
        actual_location: tuple[float, float] | None = None,
        pitch_result: str | None = None,
        outcome: dict | None = None,
    ) -> Path:
        """Render a card, preferring the HTML renderer with mpl fallback."""
        if self.card_style == "html":
            try:
                from mlb.live.card_html import HtmlCardRenderer, render_card_png

                if self._html_renderer is None:
                    self._html_renderer = HtmlCardRenderer()
                in_zone = self.predictor.pitch_predictor.get_strike_zone_probability(
                    prediction
                )
                return render_card_png(
                    prediction,
                    context,
                    in_zone,
                    out_path,
                    self._html_renderer,
                    actual_pitch_type=actual_pitch_type,
                    actual_location=actual_location,
                    pitch_result=pitch_result,
                    outcome=outcome,
                )
            except Exception as exc:
                print(f"HTML card renderer failed ({exc}); using matplotlib")

        figure = self.predictor.pitch_predictor.create_pitch_card(
            prediction,
            context,
            actual_pitch_type=actual_pitch_type,
            actual_location=actual_location,
            save_path=str(out_path),
        )
        plt.close(figure)
        return out_path

    def _target_pitch_for(self, snapshot: LiveSnapshot) -> int:
        """Draw (once per at-bat) the pitch number this at-bat will post on.

        The draw happens the first time we see the at-bat, uniform between
        the first pitch we can still catch and the ceiling. At-bats that
        end before the target simply do not post.
        """
        key = (snapshot.game_pk, snapshot.at_bat_index)
        if key not in self._ab_target_pitch:
            low = min(snapshot.next_pitch_number, self.random_pitch_ceiling)
            self._ab_target_pitch[key] = self._rng.randint(
                low, self.random_pitch_ceiling
            )
        return self._ab_target_pitch[key]

    def should_post(self, snapshot: LiveSnapshot) -> bool:
        """Posting policy: every pitch, random pitch, at-bat, or half-inning."""
        if self._posts_per_game.get(snapshot.game_pk, 0) >= self.max_posts_per_game:
            return False
        if self.post_cadence == "pitch":
            return True
        if self.post_cadence == "random_pitch":
            if self._last_posted_at_bat.get(snapshot.game_pk) == snapshot.at_bat_index:
                return False
            return snapshot.next_pitch_number == self._target_pitch_for(snapshot)
        if self.post_cadence == "half_inning":
            return self._last_posted_inning.get(snapshot.game_pk) != snapshot.inning_key
        return (
            self._last_posted_at_bat.get(snapshot.game_pk) != snapshot.at_bat_index
        )

    def _resolve_actual_pitch(
        self,
        snapshot: LiveSnapshot,
        pending: PendingPostedPrediction,
    ) -> tuple[str | None, str | None, tuple[float, float] | None]:
        actual_rows = (
            snapshot.frame.filter(
                (snapshot.frame["at_bat_index"] == pending.at_bat_index)
                & (snapshot.frame["pitch_number"] == pending.pitch_number)
                & snapshot.frame["pitch_type_code"].is_not_null()
            )
            .sort("pitch_number")
        )
        if actual_rows.is_empty():
            return None, None, None
        actual = actual_rows.row(-1, named=True)
        location = None
        if actual.get("px") is not None and actual.get("pz") is not None:
            location = (float(actual["px"]), float(actual["pz"]))
        result = actual.get("pitch_call_description")
        actual_type = actual.get("pitch_type_code")
        return (
            str(actual_type) if actual_type is not None else None,
            str(result) if result is not None else None,
            location,
        )

    def _publish_result_if_ready(
        self, game_pk: int, snapshot: LiveSnapshot
    ) -> str | None:
        pending = self._pending_posted_predictions.get(game_pk)
        if pending is None:
            return None

        actual_pitch_type, pitch_result, actual_location = self._resolve_actual_pitch(
            snapshot,
            pending,
        )
        if actual_pitch_type is None and actual_location is None:
            return None

        card_dir = self.output_dir / str(game_pk)
        card_dir.mkdir(parents=True, exist_ok=True)
        result_card_path = card_dir / (
            f"ab{pending.at_bat_index:03d}_pitch{pending.pitch_number:02d}_result.png"
        )

        result_context = pending.prediction_context
        result_context.pitch_result = pitch_result
        result_card_path = self._render_card(
            pending.prediction,
            result_context,
            result_card_path,
            actual_pitch_type=actual_pitch_type,
            actual_location=actual_location,
            pitch_result=pitch_result,
        )

        post = ResultPost(
            text=build_result_text(
                result_context,
                pending.prediction,
                actual_pitch_type,
                pitch_result,
            ),
            image_path=result_card_path,
            reply_to=pending.post_id,
        )
        reply_id = self.publisher.publish_result(post)
        del self._pending_posted_predictions[game_pk]
        return reply_id

    def handle_feed(self, game_pk: int, feed: dict) -> dict | None:
        """Process one poll of one game; returns a summary when predicted."""
        snapshot = build_live_snapshot(feed)
        if snapshot is None:
            return None

        reply_id = self._publish_result_if_ready(game_pk, snapshot)

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
        outcome = self._predict_outcome(snapshot, prediction)
        card_path = self._render_card(
            prediction, snapshot.context, card_path, outcome=outcome
        )

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
            "result_replied": bool(reply_id),
        }

        if self.should_post(snapshot):
            post = PredictionPost(
                text=build_post_text(snapshot, prediction),
                image_path=card_path,
            )
            post_id = self.publisher.publish(post)
            self._last_posted_at_bat[snapshot.game_pk] = snapshot.at_bat_index
            self._last_posted_inning[snapshot.game_pk] = snapshot.inning_key
            self._posts_per_game[snapshot.game_pk] = (
                self._posts_per_game.get(snapshot.game_pk, 0) + 1
            )
            self._pending_posted_predictions[game_pk] = PendingPostedPrediction(
                game_pk=game_pk,
                at_bat_index=snapshot.at_bat_index,
                pitch_number=snapshot.next_pitch_number,
                inning_key=snapshot.inning_key,
                prediction=prediction,
                prediction_context=snapshot.context,
                post_id=post_id,
                card_path=card_path,
            )
            summary["posted"] = True
            summary["post_id"] = post_id

        print(
            f"[game {game_pk}] AB {snapshot.at_bat_index} pitch "
            f"{snapshot.next_pitch_number} ({snapshot.balls}-{snapshot.strikes}): "
            f"{prediction.predicted_type} "
            f"{prediction.top_3_types[0][1]:.0%}"
            + (" [posted]" if summary["posted"] else "")
            + (" [result]" if summary["result_replied"] else "")
        )
        return summary


async def run_live_day(
    target_date: date,
    service: LiveGamePredictionService,
    poll_interval: float = 3.0,
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
    poll_interval: float = 3.0,
    concurrency_limit: int = 4,
) -> dict:
    """Poll a single game until it reaches a terminal state."""
    from mlb.endpoints.game_feed import GameFeed

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
    from mlb.etl.daily_pipeline import classify_game_state

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
    poll_interval: float = 3.0,
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