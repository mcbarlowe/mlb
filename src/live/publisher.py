"""Publishing predicted-pitch graphics to Bluesky (or a local dry run)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from src.live.game_state import LiveSnapshot
from src.ml.pitch_predictor import PITCH_TYPE_FULL_NAMES, GameContext, PitchPrediction

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


@dataclass(frozen=True)
class ResultPost:
    """Outcome follow-up for a previously posted pitch prediction."""

    text: str
    image_path: Path
    reply_to: str


def build_post_text(snapshot: LiveSnapshot, prediction: PitchPrediction) -> str:
    """Build the post caption for a next-pitch prediction."""
    context = snapshot.context
    top = prediction.top_3_types[:2]
    top_text = " | ".join(
        f"{PITCH_TYPE_FULL_NAMES.get(code, code)} {prob:.0%}"
        for code, prob in top
    )
    location = prediction.location_point
    text = (
        f"Next pitch: {context.pitcher_name} to {context.batter_name}\n"
        f"Top calls: {top_text}\n"
        f"Expected location: ({location[0]:.2f}, {location[1]:.2f}) ft\n"
        f"{context.inning_half} {context.inning}, count {context.count_str}, "
        f"{context.outs} out\n"
        f"{context.game_str}"
    )

    return text[:MAX_POST_CHARS]

def build_result_text(
    context: GameContext,
    prediction: PitchPrediction,
    actual_pitch_type: str | None,
    pitch_result: str | None,
) -> str:
    """Build the threaded follow-up text for the actual pitch outcome."""
    actual_code = actual_pitch_type or "UNK"
    actual_name = PITCH_TYPE_FULL_NAMES.get(actual_code, actual_code)
    predicted_name = PITCH_TYPE_FULL_NAMES.get(
        prediction.predicted_type, prediction.predicted_type
    )
    outcome = pitch_result or "Result unavailable"
    accuracy = "matched" if actual_pitch_type == prediction.predicted_type else "missed"
    text = (
        f"Result: {actual_name} — {outcome}\n"
        f"Prediction {accuracy}: expected {predicted_name}\n"
        f"{context.inning_half} {context.inning}, count {context.count_str}, "
        f"{context.outs} out\n"
        f"{context.game_str}"
    )
    return text[:MAX_POST_CHARS]


def _image_aspect_ratio(image_path: Path):
    from atproto import models
    from PIL import Image

    with Image.open(image_path) as image:
        width, height = image.size
    return models.AppBskyEmbedDefs.AspectRatio(width=width, height=height)



def build_image_alt_text(post_text: str) -> str:
    """Alt text for the card image, derived from the caption."""
    return "Pitch prediction card. " + post_text.replace("\n", " ")


class Publisher(Protocol):
    """Anything that can publish prediction posts and threaded results."""

    def publish(self, post: PredictionPost) -> str:
        """Publish and return an identifier (post URI or file path)."""
        ...

    def publish_result(self, post: ResultPost) -> str:
        """Publish a threaded result reply and return its identifier."""
        ...


class DryRunPublisher:
    """Log-only publisher used for local testing and development."""

    def __init__(self) -> None:
        self.published: list[PredictionPost | ResultPost] = []

    def publish(self, post: PredictionPost) -> str:
        self.published.append(post)
        print(f"[dry-run] would post card {post.image_path}")
        print(post.text)
        return str(post.image_path)

    def publish_result(self, post: ResultPost) -> str:
        self.published.append(post)
        print(f"[dry-run] would reply with card {post.image_path} to {post.reply_to}")
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

    def _reply_ref(self, post_uri: str, post_cid: str):
        from atproto import models

        strong_ref = models.ComAtprotoRepoStrongRef.Main(cid=post_cid, uri=post_uri)
        return models.AppBskyFeedPost.ReplyRef(
            parent=strong_ref,
            root=strong_ref,
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
            image_aspect_ratio=_image_aspect_ratio(post.image_path),
        )
        post_uri = str(getattr(response, "uri", ""))
        post_cid = str(getattr(response, "cid", ""))
        print(f"Posted to Bluesky {post_uri} with card {post.image_path}")
        return f"{post_uri}|{post_cid}"

    def publish_result(self, post: ResultPost) -> str:
        image_bytes = post.image_path.read_bytes()
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise RuntimeError(
                f"Card image {post.image_path} is {len(image_bytes):,} bytes; "
                f"Bluesky blobs must stay under {MAX_IMAGE_BYTES:,} bytes. "
                "Lower the card DPI or figure size."
            )

        if "|" not in post.reply_to:
            raise RuntimeError(
                f"Reply target {post.reply_to!r} is missing the stored CID payload"
            )
        post_uri, post_cid = post.reply_to.split("|", 1)
        response = self._client.send_image(
            text=post.text,
            image=image_bytes,
            image_alt=build_image_alt_text(post.text),
            reply_to=self._reply_ref(post_uri, post_cid),
            image_aspect_ratio=_image_aspect_ratio(post.image_path),
        )
        reply_uri = str(getattr(response, "uri", ""))
        print(
            f"Posted result reply to Bluesky {reply_uri} in thread {post_uri} "
            f"with card {post.image_path}"
        )
        return reply_uri
