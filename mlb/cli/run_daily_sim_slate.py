from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from mlb.live.publisher import (
    POST_PROVIDER_CHOICES,
    PredictionPost,
    ResultPost,
    build_publisher,
)
from mlb.live.slate_sim_card import (
    SlateSimBoardData,
    SlateSimRow,
    render_slate_sim_card,
)
from mlb.sim.slate import (
    DailySlateState,
    SlateGame,
    SlatePrediction,
    build_daily_board_caption,
    build_day_ahead_simulator,
    build_update_caption,
    changed_games,
    fetch_slate_games,
    load_daily_slate_state,
    render_prediction_card,
    save_daily_slate_state,
    simulate_slate_game,
    snapshot_state,
)
from mlb.sim.team_strength import (
    DEFAULT_REGISTERED_STRENGTH_MODEL,
    TeamStrengthPredictor,
    build_live_strength_predictor,
)

MAX_ROWS_PER_BOARD = 4


EASTERN = ZoneInfo("America/New_York")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and optionally post an all-games morning simulation board, "
            "then watch for probable-starter changes before games begin."
        ),
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Slate date in YYYY-MM-DD format. Defaults to today in local time.",
    )
    parser.add_argument("--sims", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--post", action="store_true")
    parser.add_argument(
        "--post-provider",
        type=str,
        choices=POST_PROVIDER_CHOICES,
        default="bluesky",
        help="Posting backend to use when --post is set (default: bluesky)",
    )
    parser.add_argument(
        "--all-games",
        action="store_true",
        help="Include every scheduled game on the date, not just preview games.",
    )
    parser.add_argument(
        "--watch-starters",
        action="store_true",
        help="Keep polling preview games and refresh one-game sims when probable starters change.",
    )
    parser.add_argument(
        "--poll-interval-minutes",
        type=float,
        default=15.0,
        help="Minutes between probable-starter polls while --watch-starters is enabled.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/sim_cards/daily",
        help="Directory for the daily board image and any refreshed one-game cards.",
    )
    parser.add_argument(
        "--state-dir",
        type=str,
        default="output/sim_state",
        help="Directory for persisted probable-starter snapshots.",
    )
    parser.add_argument(
        "--outcome-run-dir",
        type=str,
        default="auto",
        help=(
            "Outcome model run directory. Default 'auto' prefers shared MLflow production runs when "
            "MLFLOW_TRACKING_URI is set, then falls back to models/outcome/latest_run.txt or the newest local run_* directory."
        ),
    )
    parser.add_argument(
        "--win-model-name",
        type=str,
        default=DEFAULT_REGISTERED_STRENGTH_MODEL,
        help="Registered MLflow model containing the champion win estimator.",
    )
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None)
    return parser.parse_args(argv)


def resolve_target_date(date_arg: str | None) -> date:
    if date_arg is None:
        return datetime.now(tz=UTC).astimezone().date()
    return date.fromisoformat(date_arg)


def _state_path(state_dir: Path, target_date: date) -> Path:
    return state_dir / f"daily_sim_{target_date.isoformat()}.json"


def _posted_state_for_date(
    state_path: Path,
    target_date: date,
    *,
    post_enabled: bool,
) -> DailySlateState | None:
    if not post_enabled:
        return None
    state = load_daily_slate_state(state_path)
    if (
        state is None
        or state.slate_date != target_date.isoformat()
        or not state.board_post_id
    ):
        return None
    return state


def _build_board_rows(predictions: list[SlatePrediction]) -> list[SlateSimRow]:
    return [
        SlateSimRow(
            game_pk=prediction.game.game_pk,
            away_abbrev=prediction.game.away_abbrev,
            home_abbrev=prediction.game.home_abbrev,
            away_team_id=prediction.game.away_team_id,
            home_team_id=prediction.game.home_team_id,
            away_starter=prediction.away_starter,
            home_starter=prediction.home_starter,
            away_starter_id=prediction.game.away_probable.player_id,
            home_starter_id=prediction.game.home_probable.player_id,
            game_time=prediction.game.game_datetime,
            venue=prediction.game.venue,
            home_win_probability=prediction.stats["home_win_probability"],
            mean_away_runs=prediction.stats["mean_away_runs"],
            mean_home_runs=prediction.stats["mean_home_runs"],
        )
        for prediction in predictions
    ]

def _games_summary(game_count: int, *, preview_only: bool) -> str:
    noun = "preview game" if preview_only else "game"
    if game_count != 1:
        noun += "s"
    return f"{game_count} {noun}"


def _page_games_summary(
    total_games: int,
    *,
    preview_only: bool,
    page_index: int,
    page_count: int,
) -> str:
    summary = _games_summary(total_games, preview_only=preview_only)
    if page_count == 1:
        return summary
    return f"{summary} · page {page_index} of {page_count}"


def _prediction_pages(
    predictions: list[SlatePrediction],
    *,
    max_rows: int = MAX_ROWS_PER_BOARD,
) -> list[list[SlatePrediction]]:
    return [
        predictions[index:index + max_rows]
        for index in range(0, len(predictions), max_rows)
    ]




def _generated_at_label() -> str:
    return datetime.now(tz=EASTERN).strftime("%Y-%m-%d %I:%M %p ET").replace(" 0", " ")



def _simulate_preview_games(
    preview_games: list[SlateGame],
    *,
    simulator,
    win_predictor: TeamStrengthPredictor,
    season: int,
    sims: int,
) -> tuple[list[SlatePrediction], list[tuple[SlateGame, str]]]:
    predictions: list[SlatePrediction] = []
    skipped: list[tuple[SlateGame, str]] = []
    for game in preview_games:
        try:
            prediction = simulate_slate_game(
                game,
                simulator,
                season=season,
                n_sims=sims,
                win_predictor=win_predictor,
            )
        except (ValueError, KeyError) as exc:
            skipped.append((game, str(exc)))
            print(f"{game.label:12s} SKIPPED ({exc})")
            continue
        predictions.append(prediction)
        stats = prediction.stats
        print(
            f"{prediction.game.label:12s} p(home)={stats['home_win_probability']:.2f}  "
            f"proj {stats['mean_away_runs']:.1f}-{stats['mean_home_runs']:.1f}  "
            f"{prediction.away_starter} vs {prediction.home_starter}"
        )
    return predictions, skipped


def _board_note(
    skipped: list[tuple[SlateGame, str]],
    watching: bool,
    *,
    preview_only: bool,
) -> str | None:
    notes: list[str] = []
    if skipped:
        labels = ", ".join(game.label for game, _ in skipped[:3])
        extra = "" if len(skipped) <= 3 else f" (+{len(skipped) - 3} more)"
        scope = "preview game(s)" if preview_only else "game(s)"
        notes.append(f"Skipped {len(skipped)} {scope}: {labels}{extra}.")
    if watching:
        notes.append("Monitoring preview games for probable-starter changes.")
    return " ".join(notes) if notes else None


def _initial_board_path(output_dir: Path, target_date: date) -> Path:
    return output_dir / f"daily_sim_{target_date.isoformat()}.jpg"


def _board_path(
    output_dir: Path,
    target_date: date,
    *,
    page_index: int,
    page_count: int,
) -> Path:
    if page_count == 1:
        return _initial_board_path(output_dir, target_date)
    return output_dir / f"daily_sim_{target_date.isoformat()}_p{page_index}.jpg"



def _updated_card_path(output_dir: Path, target_date: date, game_pk: int) -> Path:
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    return output_dir / "updates" / f"daily_sim_{target_date.isoformat()}_{game_pk}_{timestamp}.jpg"


def _save_state(
    state_path: Path,
    *,
    target_date: date,
    board_path: Path,
    board_post_id: str | None,
    games: list[SlateGame],
) -> DailySlateState:
    state = snapshot_state(
        target_date.isoformat(),
        board_path=board_path,
        board_post_id=board_post_id,
        games=games,
    )
    save_daily_slate_state(state_path, state)
    return state


def _render_board_pages(
    predictions: list[SlatePrediction],
    *,
    output_dir: Path,
    target_date: date,
    preview_only: bool,
    watching: bool,
    skipped: list[tuple[SlateGame, str]],
    sims: int,
) -> list[Path]:
    pages = _prediction_pages(predictions)
    generated_at = _generated_at_label()
    paths: list[Path] = []
    note = _board_note(skipped, watching, preview_only=preview_only)
    for page_index, page_predictions in enumerate(pages, start=1):
        board_data = SlateSimBoardData(
            slate_date=target_date.isoformat(),
            generated_at=generated_at,
            games_summary=_page_games_summary(
                len(predictions),
                preview_only=preview_only,
                page_index=page_index,
                page_count=len(pages),
            ),
            n_sims=sims,
            rows=_build_board_rows(page_predictions),
            note=note if page_index == 1 else None,
        )
        board_path = _board_path(
            output_dir,
            target_date,
            page_index=page_index,
            page_count=len(pages),
        )
        paths.append(render_slate_sim_card(board_data, board_path))
    return paths



def _publish_board(
    publisher,
    board_paths: list[Path],
    *,
    target_date: date,
    games_summary: str,
    watching: bool,
) -> str:
    caption = build_daily_board_caption(
        target_date.isoformat(),
        games_summary=games_summary,
        include_update_note=watching,
    )
    if len(board_paths) == 1:
        return publisher.publish(
            PredictionPost(text=caption, image_path=board_paths[0])
        )

    total_pages = len(board_paths)
    root_text = f"{caption} Page 1 of {total_pages}."
    root_post_id = publisher.publish(
        PredictionPost(text=root_text[:300], image_path=board_paths[0])
    )
    for page_index, image_path in enumerate(board_paths[1:], start=2):
        publisher.publish_result(
            ResultPost(
                text=f"Page {page_index} of {total_pages}.",
                image_path=image_path,
                reply_to=root_post_id,
            )
        )
    return root_post_id


def _poll_probable_starters(
    *,
    publisher,
    target_date: date,
    simulator,
    win_predictor: TeamStrengthPredictor,
    season: int,
    sims: int,
    state_path: Path,
    board_path: Path,
    board_post_id: str | None,
    previous_games: list[SlateGame],
    poll_interval_minutes: float,
) -> None:
    previous_by_pk = {game.game_pk: game for game in previous_games}
    while True:
        sleep_seconds = max(poll_interval_minutes * 60.0, 1.0)
        print(
            f"Waiting {poll_interval_minutes:.1f} minute(s) before the next probable-starter check..."
        )
        time.sleep(sleep_seconds)

        current_games = fetch_slate_games(target_date, abstract_states={"Preview"})
        if not current_games:
            print("No preview games remain; stopping probable-starter watch.")
            _save_state(
                state_path,
                target_date=target_date,
                board_path=board_path,
                board_post_id=board_post_id,
                games=[],
            )
            return

        updates = changed_games(previous_by_pk, current_games)
        if not updates:
            print("No probable-starter changes detected.")
            previous_by_pk = {game.game_pk: game for game in current_games}
            _save_state(
                state_path,
                target_date=target_date,
                board_path=board_path,
                board_post_id=board_post_id,
                games=current_games,
            )
            continue

        print(f"Detected {len(updates)} probable-starter update(s).")
        for game, changes in updates:
            try:
                prediction = simulate_slate_game(
                    game,
                    simulator,
                    season=season,
                    n_sims=sims,
                    win_predictor=win_predictor,
                )
            except (ValueError, KeyError) as exc:
                print(f"{game.label:12s} UPDATE SKIPPED ({exc})")
                continue

            card_path = render_prediction_card(
                prediction,
                _updated_card_path(Path(board_path).parent, target_date, game.game_pk),
            )
            caption = build_update_caption(prediction, changes)
            post_id = publisher.publish(
                PredictionPost(text=caption, image_path=card_path)
            )
            reasons = "; ".join(
                f"{game.abbrev_for(change.side)} {change.previous} -> {change.current}"
                for change in changes
            )
            print(f"Updated {game.label}: {reasons}")
            print(f"             card: {card_path}")
            print(f"             posted: {post_id}")

        previous_by_pk = {game.game_pk: game for game in current_games}
        _save_state(
            state_path,
            target_date=target_date,
            board_path=board_path,
            board_post_id=board_post_id,
            games=current_games,
        )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    target_date = resolve_target_date(args.date)
    season = target_date.year
    output_dir = Path(args.output_dir)
    state_path = _state_path(Path(args.state_dir), target_date)

    preview_only = not args.all_games
    slate_states = {"Preview"} if preview_only else None
    slate_games = fetch_slate_games(target_date, abstract_states=slate_states)
    if not slate_games:
        scope = "preview games" if preview_only else "games"
        raise SystemExit(f"No {scope} on {target_date.isoformat()}")

    simulator, run_dir = build_day_ahead_simulator(
        season=season,
        seed=args.seed,
        outcome_run_dir=args.outcome_run_dir,
        tracking_uri=args.mlflow_tracking_uri,
    )
    win_predictor = build_live_strength_predictor(
        target_date,
        tracking_uri=args.mlflow_tracking_uri,
        registered_model_name=args.win_model_name,
    )
    source = win_predictor.source
    if source is None:
        raise RuntimeError("Win predictor has no registered-model provenance")
    print(f"Loaded outcome models from {run_dir}")
    print(
        f"Loaded win model {source.registered_model_name} "
        f"v{source.version} from MLflow run {source.run_id}"
    )
    posted_state = _posted_state_for_date(
        state_path,
        target_date,
        post_enabled=args.post,
    )
    if posted_state is not None:
        print(
            f"Board already posted as {posted_state.board_post_id}; "
            "resuming without a duplicate post."
        )
        if args.watch_starters:
            publisher = build_publisher(
                post=True,
                provider=args.post_provider,
            )
            _poll_probable_starters(
                publisher=publisher,
                target_date=target_date,
                simulator=simulator,
                win_predictor=win_predictor,
                season=season,
                sims=args.sims,
                state_path=state_path,
                board_path=Path(posted_state.board_path),
                board_post_id=posted_state.board_post_id,
                previous_games=posted_state.games,
                poll_interval_minutes=args.poll_interval_minutes,
            )
        return
    print(
        f"Simulating {_games_summary(len(slate_games), preview_only=preview_only)} "
        f"for {target_date.isoformat()}..."
    )

    predictions, skipped = _simulate_preview_games(
        slate_games,
        simulator=simulator,
        win_predictor=win_predictor,
        season=season,
        sims=args.sims,
    )
    if not predictions:
        raise SystemExit("No simulations produced for the selected slate")

    games_summary = _games_summary(len(predictions), preview_only=preview_only)
    board_paths = _render_board_pages(
        predictions,
        output_dir=output_dir,
        target_date=target_date,
        preview_only=preview_only,
        watching=args.watch_starters,
        skipped=skipped,
        sims=args.sims,
    )
    for board_path in board_paths:
        print(f"Board image: {board_path}")

    publisher = build_publisher(post=args.post, provider=args.post_provider)
    board_post_id = _publish_board(
        publisher,
        board_paths,
        target_date=target_date,
        games_summary=games_summary,
        watching=args.watch_starters,
    )
    print(f"Board published: {board_post_id}")

    state_games = (
        slate_games
        if preview_only
        else fetch_slate_games(target_date, abstract_states={"Preview"})
    )
    _save_state(
        state_path,
        target_date=target_date,
        board_path=board_paths[0],
        board_post_id=board_post_id,
        games=state_games,
    )
    print(f"State saved: {state_path}")

    if args.watch_starters:
        _poll_probable_starters(
            publisher=publisher,
            target_date=target_date,
            simulator=simulator,
            win_predictor=win_predictor,
            season=season,
            sims=args.sims,
            state_path=state_path,
            board_path=board_paths[0],
            board_post_id=board_post_id,
            previous_games=state_games,
            poll_interval_minutes=args.poll_interval_minutes,
        )


if __name__ == "__main__":
    main()
