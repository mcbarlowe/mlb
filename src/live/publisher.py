"""Publishing predicted-pitch graphics to Twitter (or a local dry run)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from src.live.game_state import LiveSnapshot
from src.ml.pitch_predictor import PitchPrediction

REQUIRED_TWITTER_ENV_VARS = (
    "TWITTER_API_KEY",
    "TWITTER_API_SECRET",
    "TWITTER_ACCESS_TOKEN",
    "TWITTER_ACCESS_TOKEN_SECRET",
)


@dataclass(frozen=True)
class PredictionPost:
    """One publishable prediction: caption text plus rendered card path."""

    text: str
    image_path: Path


def build_post_text(snapshot: LiveSnapshot, prediction: PitchPrediction) -> str:
    """Build the tweet caption for a next-pitch prediction."""
    context = snapshot.context
    top = prediction.top_3_types[:2]
    top_text = " | ".join(f"{code} {prob:.0%}" for code, prob in top)
    location = prediction.location_point
    return (
        f"Next pitch: {context.pitcher_name} to {context.batter_name}\n"
        f"{top_text}\n"
        f"Expected location: ({location[0]:.2f}, {location[1]:.2f}) ft\n"
        f"{context.inning_half} {context.inning}, count {context.count_str}, "
        f"{context.outs} out\n"
        f"{context.game_str}"
    )


class Publisher(Protocol):
    """Anything that can publish one prediction post."""

    def publish(self, post: PredictionPost) -> str:
        """Publish and return an identifier (tweet id or file path)."""
        ...


class DryRunPublisher:
    """Log-only publisher used for local testing and development."""

    def __init__(self) -> None:
        self.published: list[PredictionPost] = []

    def publish(self, post: PredictionPost) -> str:
        self.published.append(post)
        print(f"[dry-run] would tweet card {post.image_path}")
        print(post.text)
        return str(post.image_path)


class TwitterPublisher:
    """Posts the card image plus caption to Twitter/X.

    Requires TWITTER_API_KEY, TWITTER_API_SECRET, TWITTER_ACCESS_TOKEN,
    and TWITTER_ACCESS_TOKEN_SECRET in the environment. Media upload uses
    the v1.1 endpoint; the tweet itself is created through the v2 API.
    """

    def __init__(self) -> None:
        missing = [name for name in REQUIRED_TWITTER_ENV_VARS if not os.getenv(name)]
        if missing:
            raise RuntimeError(
                "Missing Twitter credentials in environment: " + ", ".join(missing)
            )

        import tweepy

        auth = tweepy.OAuth1UserHandler(
            os.environ["TWITTER_API_KEY"],
            os.environ["TWITTER_API_SECRET"],
            os.environ["TWITTER_ACCESS_TOKEN"],
            os.environ["TWITTER_ACCESS_TOKEN_SECRET"],
        )
        self._media_api = tweepy.API(auth)
        self._client = tweepy.Client(
            consumer_key=os.environ["TWITTER_API_KEY"],
            consumer_secret=os.environ["TWITTER_API_SECRET"],
            access_token=os.environ["TWITTER_ACCESS_TOKEN"],
            access_token_secret=os.environ["TWITTER_ACCESS_TOKEN_SECRET"],
        )

    def publish(self, post: PredictionPost) -> str:
        media = self._media_api.media_upload(str(post.image_path))
        response = self._client.create_tweet(
            text=post.text,
            media_ids=[media.media_id],
        )
        response_data = getattr(response, "data", None) or {}
        tweet_id = str(response_data.get("id", ""))
        print(f"Posted tweet {tweet_id} with card {post.image_path}")
        return tweet_id
