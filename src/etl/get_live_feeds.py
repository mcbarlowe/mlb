import json
from pathlib import Path

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from src.endpoints.game_feed import GameFeed


def live_feed_etl() -> None:
    """
    Extract live feed data and save it to a JSON file.
    """

    game_file_path = Path("data/raw/schedules/")
    game_feed = GameFeed()

    for file in game_file_path.glob("*.json"):
        season = file.stem.split("_")[1]

        live_feeds_path = Path(f"data/raw/livefeeds/{season}")
        live_feeds_path.mkdir(parents=True, exist_ok=True)

        with open(file, "r") as f:
            games = json.load(f)

        dates = games["dates"]

        for date in tqdm(dates, desc=f"Processing season {season}"):
            for game in date["games"]:
                game_live_feed = game_feed.get(game["gamePk"])

                with open(live_feeds_path / f"{game['gamePk']}.json", "w") as gf:
                    json.dump(game_live_feed, gf, indent=4)


def process_live_feed_data():
    """
    Process raw live feed data and transform to parquet files.
    """
    from src.data.game_feed_data import GameFeedData

    game_feed_data = GameFeedData()
    live_feed_raw_path = Path("data/raw/livefeeds/")

    with logging_redirect_tqdm():
        for file in tqdm(
            list(live_feed_raw_path.glob("**/*.json")),
            desc="Processing live feed files",
        ):
            season = file.parent.stem
            game_pk = int(file.stem)

            live_feeds_path = Path(f"data/processed/livefeeds/{season}")
            output_file = live_feeds_path / f"{game_pk}.parquet"

            with open(file, "r") as f:
                live_feed_json = json.load(f)

            try:
                data_df = game_feed_data.transform(live_feed_json, game_pk, season)
                game_feed_data.save(data_df, output_file, format="parquet")
            except Exception as e:
                tqdm.write(f"Error processing file {file}: {e}")


if __name__ == "__main__":
    process_live_feed_data()
