"""Publishing predicted-pitch graphics to social platforms or a local dry run."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import mimetypes
import os
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import parse_qsl, quote, urlsplit

import requests
from requests import Response, Session
from requests.auth import HTTPBasicAuth

from src.live.game_state import LiveSnapshot
from src.ml.pitch_predictor import PITCH_TYPE_FULL_NAMES, GameContext, PitchPrediction

REQUIRED_BLUESKY_ENV_VARS = ("BLUESKY_HANDLE", "BLUESKY_APP_PASSWORD")
DEFAULT_PDS_URL = "https://bsky.social"

DEFAULT_X_API_BASE_URL = "https://api.x.com"
X_OAUTH2_ENV_ALIASES = {
    "client_id": ("X_API_CLIENT_ID", "X_CLIENT_ID"),
    "client_secret": ("X_API_CLIENT_SECRET", "X_CLIENT_SECRET"),
    "access_token": (
        "X_API_ACCESS_TOKEN",
        "X_API_OAUTH2_ACCESS_TOKEN",
        "X_ACCESS_TOKEN",
    ),
    "refresh_token": (
        "X_API_REFRESH_TOKEN",
        "X_API_OAUTH2_REFRESH_TOKEN",
        "X_REFRESH_TOKEN",
    ),
}
X_OAUTH1_ENV_ALIASES = {
    "api_key": ("X_API_KEY", "X_KEY"),
    "api_key_secret": ("X_API_KEY_SECRET", "X_KEY_SECRET"),
    "access_token": ("X_ACCESS_TOKEN", "X_API_ACCESS_TOKEN"),
    "access_token_secret": ("X_ACCESS_TOKEN_SECRET", "X_API_ACCESS_TOKEN_SECRET"),
}

POST_PROVIDER_CHOICES = ("bluesky", "x", "both")
MULTI_PUBLISHER_PREFIX = "multi:"
REQUEST_TIMEOUT_SECONDS = 30.0

# Base caption builders keep the richer 300-character form. X trims to 280.
DEFAULT_POST_TEXT_CHARS = 300
BLUESKY_MAX_IMAGE_BYTES = 950_000
X_MAX_IMAGE_BYTES = 5_000_000
X_MAX_MEDIA_IDS = 4
X_MAX_POST_CHARS = 280


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
    return _truncate_text(text, DEFAULT_POST_TEXT_CHARS)


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
    return _truncate_text(text, DEFAULT_POST_TEXT_CHARS)


def _truncate_text(text: str, max_chars: int) -> str:
    return text[:max_chars]


def _image_aspect_ratio(image_path: Path):
    from atproto import models
    from PIL import Image

    with Image.open(image_path) as image:
        width, height = image.size
    return models.AppBskyEmbedDefs.AspectRatio(width=width, height=height)


def build_image_alt_text(post_text: str) -> str:
    """Alt text for the card image, derived from the caption."""
    return "Pitch prediction card. " + post_text.replace("\n", " ")


def _image_bytes(image_path: Path, *, max_bytes: int, platform_name: str) -> bytes:
    image_bytes = image_path.read_bytes()
    if len(image_bytes) > max_bytes:
        raise RuntimeError(
            f"Card image {image_path} is {len(image_bytes):,} bytes; "
            f"{platform_name} uploads must stay under {max_bytes:,} bytes."
        )
    return image_bytes


def _gallery_alt_text(post_text: str, index: int, total: int) -> str:
    prefix = "Prediction graphic"
    if total > 1:
        prefix += f" ({index} of {total})"
    return prefix + ". " + post_text.replace("\n", " ")


def _env_value(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _require_x_env(key: str) -> str:
    names = X_OAUTH2_ENV_ALIASES[key]
    value = _env_value(*names)
    if value:
        return value
    raise RuntimeError(
        "Missing X API credentials in environment: "
        + " or ".join(names)
        + ". Configure OAuth 2.0 user-context credentials in your shell or launcher."
    )


def _require_x_oauth1_env(key: str) -> str:
    names = X_OAUTH1_ENV_ALIASES[key]
    value = _env_value(*names)
    if value:
        return value
    raise RuntimeError(
        "Missing X OAuth 1.0a credentials in environment: "
        + " or ".join(names)
        + ". Configure the app key/secret and user token/secret in your shell or launcher."
    )


def _oauth1_env_present() -> bool:
    return all(_env_value(*names) for names in X_OAUTH1_ENV_ALIASES.values())


def _oauth1_percent_encode(value: str) -> str:
    return quote(value, safe="-._~")


def _response_body(response: Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip()
    if isinstance(payload, (dict, list)):
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)
    return str(payload)


def _json_dict(response: Response, *, context: str) -> dict:
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(f"{context} returned non-JSON response") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"{context} returned unexpected payload: {payload!r}")
    return payload


class Publisher(Protocol):
    """Anything that can publish prediction posts and threaded results."""

    def publish(self, post: PredictionPost) -> str:
        """Publish and return an identifier (post URI or file path)."""
        ...

    def publish_images(self, text: str, image_paths: list[Path]) -> str:
        """Publish a multi-image board and return its identifier."""
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

    def publish_images(self, text: str, image_paths: list[Path]) -> str:
        for image_path in image_paths:
            self.published.append(PredictionPost(text=text, image_path=image_path))
            print(f"[dry-run] would post board image {image_path}")
        print(text)
        return str(image_paths[0])


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
        text = _truncate_text(post.text, DEFAULT_POST_TEXT_CHARS)
        image_bytes = _image_bytes(
            post.image_path,
            max_bytes=BLUESKY_MAX_IMAGE_BYTES,
            platform_name="Bluesky",
        )
        response = self._client.send_image(
            text=text,
            image=image_bytes,
            image_alt=build_image_alt_text(text),
            image_aspect_ratio=_image_aspect_ratio(post.image_path),
        )
        post_uri = str(getattr(response, "uri", ""))
        post_cid = str(getattr(response, "cid", ""))
        print(f"Posted to Bluesky {post_uri} with card {post.image_path}")
        return f"{post_uri}|{post_cid}"

    def publish_images(self, text: str, image_paths: list[Path]) -> str:
        text = _truncate_text(text, DEFAULT_POST_TEXT_CHARS)
        image_bytes = [
            _image_bytes(
                path,
                max_bytes=BLUESKY_MAX_IMAGE_BYTES,
                platform_name="Bluesky",
            )
            for path in image_paths
        ]
        response = self._client.send_images(
            text=text,
            images=image_bytes,
            image_alts=[
                _gallery_alt_text(text, index + 1, len(image_paths))
                for index in range(len(image_paths))
            ],
            image_aspect_ratios=[
                _image_aspect_ratio(path) for path in image_paths
            ],
        )
        post_uri = str(getattr(response, "uri", ""))
        post_cid = str(getattr(response, "cid", ""))
        print(f"Posted to Bluesky {post_uri} with {len(image_paths)} image(s)")
        return f"{post_uri}|{post_cid}"

    def publish_result(self, post: ResultPost) -> str:
        text = _truncate_text(post.text, DEFAULT_POST_TEXT_CHARS)
        image_bytes = _image_bytes(
            post.image_path,
            max_bytes=BLUESKY_MAX_IMAGE_BYTES,
            platform_name="Bluesky",
        )

        if "|" not in post.reply_to:
            raise RuntimeError(
                f"Reply target {post.reply_to!r} is missing the stored CID payload"
            )
        post_uri, post_cid = post.reply_to.split("|", 1)
        response = self._client.send_image(
            text=text,
            image=image_bytes,
            image_alt=build_image_alt_text(text),
            reply_to=self._reply_ref(post_uri, post_cid),
            image_aspect_ratio=_image_aspect_ratio(post.image_path),
        )
        reply_uri = str(getattr(response, "uri", ""))
        print(
            f"Posted result reply to Bluesky {reply_uri} in thread {post_uri} "
            f"with card {post.image_path}"
        )
        return reply_uri


class XPublisher:
    """Posts card images to X via the v2 media upload and post endpoints."""

    def __init__(self, session: Session | None = None) -> None:
        self._api_base_url = os.getenv("X_API_BASE_URL", DEFAULT_X_API_BASE_URL).rstrip(
            "/"
        )
        self._session = session or requests.Session()
        if _oauth1_env_present():
            self._auth_mode = "oauth1"
            self._api_key = _require_x_oauth1_env("api_key")
            self._api_key_secret = _require_x_oauth1_env("api_key_secret")
            self._access_token = _require_x_oauth1_env("access_token")
            self._access_token_secret = _require_x_oauth1_env("access_token_secret")
            return

        self._auth_mode = "oauth2"
        self._client_id = _require_x_env("client_id")
        self._client_secret = _require_x_env("client_secret")
        self._access_token = _require_x_env("access_token")
        self._refresh_token = _env_value(*X_OAUTH2_ENV_ALIASES["refresh_token"])

    def _refresh_access_token_or_raise(self) -> None:
        if not self._refresh_token:
            raise RuntimeError(
                "X access token was rejected and X_API_REFRESH_TOKEN is not set. "
                "Refresh the token manually or add offline.access credentials."
            )
        response = self._session.post(
            f"{self._api_base_url}/2/oauth2/token",
            auth=HTTPBasicAuth(self._client_id, self._client_secret),
            data={
                "grant_type": "refresh_token",
                "refresh_token": self._refresh_token,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Refreshing X access token failed ({response.status_code}): "
                f"{_response_body(response)}"
            )
        payload = _json_dict(response, context="X token refresh")
        access_token = str(payload.get("access_token", "")).strip()
        if not access_token:
            raise RuntimeError("X token refresh response did not include access_token")
        self._access_token = access_token
        refresh_token = str(payload.get("refresh_token", "")).strip()
        if refresh_token:
            self._refresh_token = refresh_token

    def _oauth1_authorization_header(self, method: str, url: str) -> str:
        """Build an OAuth 1.0a header for JSON or multipart X API requests."""
        split = urlsplit(url)
        base_url = f"{split.scheme}://{split.netloc}{split.path}"
        oauth_params = {
            "oauth_consumer_key": self._api_key,
            "oauth_nonce": secrets.token_hex(16),
            "oauth_signature_method": "HMAC-SHA1",
            "oauth_timestamp": str(int(time.time())),
            "oauth_token": self._access_token,
            "oauth_version": "1.0",
        }
        signature_params = list(parse_qsl(split.query, keep_blank_values=True))
        signature_params.extend(oauth_params.items())
        encoded_params = [
            (_oauth1_percent_encode(str(key)), _oauth1_percent_encode(str(value)))
            for key, value in signature_params
        ]
        encoded_params.sort()
        normalized_params = "&".join(
            f"{key}={value}" for key, value in encoded_params
        )
        signature_base = "&".join(
            _oauth1_percent_encode(part)
            for part in (method.upper(), base_url, normalized_params)
        )
        signing_key = "&".join(
            (
                _oauth1_percent_encode(self._api_key_secret),
                _oauth1_percent_encode(self._access_token_secret),
            )
        )
        oauth_params["oauth_signature"] = base64.b64encode(
            hmac.new(
                signing_key.encode("utf-8"),
                signature_base.encode("utf-8"),
                hashlib.sha1,
            ).digest()
        ).decode("utf-8")
        return "OAuth " + ", ".join(
            f'{_oauth1_percent_encode(key)}="{_oauth1_percent_encode(value)}"'
            for key, value in oauth_params.items()
        )

    def _request(self, method: str, path: str, **kwargs) -> Response:
        extra_headers = dict(kwargs.pop("headers", {}))
        full_url = f"{self._api_base_url}{path}"
        headers = {"Accept": "application/json", **extra_headers}
        if self._auth_mode == "oauth1":
            headers["Authorization"] = self._oauth1_authorization_header(
                method, full_url
            )
            response = self._session.request(
                method,
                full_url,
                headers=headers,
                timeout=REQUEST_TIMEOUT_SECONDS,
                **kwargs,
            )
            if response.status_code >= 400:
                raise RuntimeError(
                    f"X API request failed for {path} ({response.status_code}): "
                    f"{_response_body(response)}"
                )
            return response

        allow_refresh = kwargs.pop("allow_refresh", True)
        headers["Authorization"] = f"Bearer {self._access_token}"
        response = self._session.request(
            method,
            full_url,
            headers=headers,
            timeout=REQUEST_TIMEOUT_SECONDS,
            **kwargs,
        )
        if response.status_code == 401 and allow_refresh:
            self._refresh_access_token_or_raise()
            return self._request(
                method,
                path,
                headers=extra_headers,
                allow_refresh=False,
                **kwargs,
            )
        if response.status_code >= 400:
            raise RuntimeError(
                f"X API request failed for {path} ({response.status_code}): "
                f"{_response_body(response)}"
            )
        return response

    def _upload_media(self, image_path: Path) -> str:
        image_bytes = _image_bytes(
            image_path,
            max_bytes=X_MAX_IMAGE_BYTES,
            platform_name="X",
        )
        mime_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
        response = self._request(
            "POST",
            "/2/media/upload",
            data={"media_category": "tweet_image"},
            files={"media": (image_path.name, image_bytes, mime_type)},
        )
        payload = _json_dict(response, context="X media upload")
        data = payload.get("data")
        if not isinstance(data, dict):
            raise TypeError(f"X media upload returned unexpected payload: {payload!r}")
        media_id = str(data.get("id", "")).strip()
        if not media_id:
            raise RuntimeError(f"X media upload did not return a media id: {payload!r}")
        return media_id

    def _create_post(
        self,
        text: str,
        media_ids: list[str],
        *,
        reply_to: str | None = None,
    ) -> str:
        payload: dict[str, object] = {
            "text": _truncate_text(text, X_MAX_POST_CHARS),
            "media": {"media_ids": media_ids},
        }
        if reply_to is not None:
            payload["reply"] = {"in_reply_to_tweet_id": reply_to}
        response = self._request("POST", "/2/tweets", json=payload)
        body = _json_dict(response, context="X post creation")
        data = body.get("data")
        if not isinstance(data, dict):
            raise TypeError(f"X post creation returned unexpected payload: {body!r}")
        post_id = str(data.get("id", "")).strip()
        if not post_id:
            raise RuntimeError(f"X post creation did not return a post id: {body!r}")
        return post_id

    def publish(self, post: PredictionPost) -> str:
        media_id = self._upload_media(post.image_path)
        post_id = self._create_post(post.text, [media_id])
        print(f"Posted to X {post_id} with card {post.image_path}")
        return post_id

    def publish_images(self, text: str, image_paths: list[Path]) -> str:
        if len(image_paths) > X_MAX_MEDIA_IDS:
            raise RuntimeError(
                f"X posts accept at most {X_MAX_MEDIA_IDS} images; got {len(image_paths)}"
            )
        media_ids = [self._upload_media(path) for path in image_paths]
        post_id = self._create_post(text, media_ids)
        print(f"Posted to X {post_id} with {len(image_paths)} image(s)")
        return post_id

    def publish_result(self, post: ResultPost) -> str:
        media_id = self._upload_media(post.image_path)
        reply_id = self._create_post(post.text, [media_id], reply_to=post.reply_to)
        print(
            f"Posted result reply to X {reply_id} in thread {post.reply_to} "
            f"with card {post.image_path}"
        )
        return reply_id


class MultiPublisher:
    """Fan out posts to multiple publishers while keeping reply ids opaque."""

    def __init__(self, publishers: dict[str, Publisher]) -> None:
        self._publishers = publishers

    def _encode(self, results: dict[str, str]) -> str:
        return MULTI_PUBLISHER_PREFIX + json.dumps(
            results,
            separators=(",", ":"),
            sort_keys=True,
        )

    def _decode(self, value: str) -> dict[str, str]:
        if not value.startswith(MULTI_PUBLISHER_PREFIX):
            raise RuntimeError(
                f"Reply target {value!r} is missing the multi-publisher payload"
            )
        try:
            payload = json.loads(value[len(MULTI_PUBLISHER_PREFIX) :])
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Reply target {value!r} is not valid multi-publisher JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise TypeError(f"Reply target {value!r} is not a multi-publisher mapping")
        return {str(name): str(result) for name, result in payload.items()}

    def _collect(
        self,
        action: str,
        operation: Callable[[str, Publisher], str],
        *,
        names: list[str] | None = None,
    ) -> str:
        results: dict[str, str] = {}
        errors: list[str] = []
        target_names = names or list(self._publishers)
        for name in target_names:
            publisher = self._publishers.get(name)
            if publisher is None:
                errors.append(f"{name}: publisher not configured")
                continue
            try:
                results[name] = operation(name, publisher)
            except Exception as exc:
                errors.append(f"{name}: {exc}")
        if not results:
            raise RuntimeError(
                f"All publishers failed during {action}: " + "; ".join(errors)
            )
        if errors:
            print(
                f"[multi-publisher] partial failure during {action}: "
                + "; ".join(errors)
            )
        return self._encode(results)


    def publish(self, post: PredictionPost) -> str:
        return self._collect(
            "publish",
            lambda _name, publisher: publisher.publish(post),
        )

    def publish_images(self, text: str, image_paths: list[Path]) -> str:
        return self._collect(
            "publish_images",
            lambda _name, publisher: publisher.publish_images(text, image_paths),
        )

    def publish_result(self, post: ResultPost) -> str:
        reply_targets = self._decode(post.reply_to)
        return self._collect(
            "publish_result",
            lambda name, publisher: publisher.publish_result(
                ResultPost(
                    text=post.text,
                    image_path=post.image_path,
                    reply_to=reply_targets[name],
                )
            ),
            names=list(reply_targets),
        )


def build_publisher(*, post: bool, provider: str) -> Publisher:
    """Build the configured posting backend while preserving dry-run behavior."""
    if not post:
        return DryRunPublisher()
    if provider == "bluesky":
        return BlueskyPublisher()
    if provider == "x":
        return XPublisher()
    if provider == "both":
        return MultiPublisher(
            {
                "bluesky": BlueskyPublisher(),
                "x": XPublisher(),
            }
        )
    raise ValueError(
        f"provider must be one of {', '.join(POST_PROVIDER_CHOICES)}; got {provider!r}"
    )
