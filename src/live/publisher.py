"""Publishing predicted-pitch graphics to Bluesky (or a local dry run)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from src.live.game_state import LiveSnapshot
from src.ml.pitch_predictor import PitchPrediction

REQUIRED_BLUESKY_ENV_VARS = ("BLUESKY_HANDLE", "BLUESKY_APP_PASSWORD")
DEFAULT_PDS_URL = "https://bsky.social"

# Bluesky rejects blobs larger than ~976 KB and posts over 300 graphemes.
MAX_IMAGE_BYTES = 950_000
MAX_POST_CHARS = 300


@dataclass(frozen=True)
class PredictionPost:
    """One publishable prediction: caption text plus rendered card path."""

    text: str
    image_path: Path


def build_post_text(snapshot: LiveSnapshot, prediction: PitchPrediction) -> str:
    """Build the post caption for a next-pitch prediction."""
    context = snapshot.context
    top = prediction.top_3_types[:2]
    top_text = " | ".join(f"{code} {prob:.0%}" for code, prob in top)
    location = prediction.location_point
    text = (
        f"Next pitch: {context.pitcher_name} to {context.batter_name}\n"
        f"{top_text}\n"
        f"Expected location: ({location[0]:.2f}, {location[1]:.2f}) ft\n"
        f"{context.inning_half} {context.inning}, count {context.count_str}, "
        f"{context.outs} out\n"
        f"{context.game_str}"
    )
    return text[:MAX_POST_CHARS]


def build_image_alt_text(post_text: str) -> str:
    """Alt text for the card image, derived from the caption."""
    return "Pitch prediction card. " + post_text.replace("\n", " ")


class Publisher(Protocol):
    """Anything that can publish one prediction post."""

    def publish(self, post: PredictionPost) -> str:
        """Publish and return an identifier (post URI or file path)."""
        ...


class DryRunPublisher:
    """Log-only publisher used for local testing and development."""

    def __init__(self) -> None:
        self.published: list[PredictionPost] = []

    def publish(self, post: PredictionPost) -> str:
        self.published.append(post)
        print(f"[dry-run] would post card {post.image_path}")
        print(post.text)
        return str(post.image_path)


class BlueskyPublisher:
    """Posts the card image plus caption to Bluesky via the AT Protocol.

    Requires BLUESKY_HANDLE and BLUESKY_APP_PASSWORD in the environment.
    BLUESKY_PDS_URL optionally overrides the default https://bsky.social
    endpoint for self-hosted PDS setups.
    """

    def __init__(self) -> None:
        missing = [name for name in REQUIRED_BLUESKY_ENV_VARS if not os.getenv(name)]
        if missing:
            raise RuntimeError(
                "Missing Bluesky credentials in environment: "
                + ", ".join(missing)
                + ". Create an app password at bsky.app -> Settings -> App Passwords."
            )

        from atproto import Client

        self._client = Client(os.getenv("BLUESKY_PDS_URL", DEFAULT_PDS_URL))
        self._client.login(
            os.environ["BLUESKY_HANDLE"],
            os.environ["BLUESKY_APP_PASSWORD"],
        )

    def publish(self, post: PredictionPost) -> str:
        image_bytes = post.image_path.read_bytes()
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise RuntimeError(
                f"Card image {post.image_path} is {len(image_bytes):,} bytes; "
                f"Bluesky blobs must stay under {MAX_IMAGE_BYTES:,} bytes. "
                "Lower the card DPI or figure size."
            )

        response = self._client.send_image(
            text=post.text,
            image=image_bytes,
            image_alt=build_image_alt_text(post.text),
        )
        post_uri = str(getattr(response, "uri", ""))
        print(f"Posted to Bluesky {post_uri} with card {post.image_path}")
        return post_uri
