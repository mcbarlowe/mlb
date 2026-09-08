"""Email an alert when a scheduled MLB agent exits nonzero.

Every scheduled agent already records its exit status, but nothing read it: the
live pitch pipeline exited 1 at 09:00 for nine consecutive days after the
``src`` to ``mlb`` rename and the only trace was a launchd status column nobody
was watching. The runner templates now call this from their EXIT trap so a
failing run reports itself, with the tail of its stderr log in the body so the
cause is visible without opening a shell.

Credentials follow the convention already used on the deployment host: SMTP
settings and an app password in a ``KEY=value`` env file. ``MLB_ALERT_*`` wins,
then the shared ``BETTING_ARB_*`` values, so this works today against the
existing file and can be split onto MLB-owned credentials later by dropping an
``alerts.env`` in place. Only SMTP settings are read; nothing here touches the
betting schema or package.
"""

from __future__ import annotations

import argparse
import os
import platform
import smtplib
import socket
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Final

ALERT_ENV_VAR: Final = "MLB_ALERT_ENV"
DEFAULT_ENV_FILES: Final = (
    Path(".config/barlowe/alerts.env"),
    Path(".config/betting/arbitrage.env"),
)
DEFAULT_SENDER: Final = "sabresbot@gmail.com"
DEFAULT_RECIPIENT: Final = "MCBarlowe@gmail.com"
DEFAULT_SMTP_HOST: Final = "smtp.gmail.com"
DEFAULT_SMTP_PORT: Final = 587
DEFAULT_LOG_LINES: Final = 40
SMTP_TIMEOUT: Final = 20.0


@dataclass(frozen=True)
class AlertConfig:
    sender: str
    username: str
    recipient: str
    host: str
    port: int
    password: str


def parse_env_file(text: str) -> dict[str, str]:
    """Parse a shell-style ``KEY=value`` credentials file.

    Only assignments are honored; anything else in the file is ignored, so a
    file that also holds shell logic cannot inject values.
    """
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").lstrip()
        key, separator, value = line.partition("=")
        if not separator:
            continue
        key = key.strip()
        if not key.replace("_", "").isalnum():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def env_file_candidates(
    environ: Mapping[str, str], home: Path | None = None
) -> list[Path]:
    override = environ.get(ALERT_ENV_VAR, "").strip()
    if override:
        return [Path(override).expanduser()]
    root = home or Path(environ.get("HOME") or Path.home())
    return [root / relative for relative in DEFAULT_ENV_FILES]


def load_settings(
    environ: Mapping[str, str] | None = None, home: Path | None = None
) -> dict[str, str]:
    """Process environment first, then each credentials file that exists."""
    env = dict(os.environ if environ is None else environ)
    merged: dict[str, str] = {}
    for candidate in reversed(env_file_candidates(env, home)):
        if candidate.is_file():
            merged.update(parse_env_file(candidate.read_text(encoding="utf-8")))
    merged.update({key: value for key, value in env.items() if value})
    return merged


def _first(settings: Mapping[str, str], *names: str) -> str:
    for name in names:
        value = settings.get(name, "").strip()
        if value:
            return value
    return ""


def resolve_config(
    environ: Mapping[str, str] | None = None, home: Path | None = None
) -> AlertConfig:
    """Resolve SMTP settings, raising an actionable error when unusable."""
    settings = load_settings(environ, home)
    password = _first(settings, "MLB_ALERT_EMAIL_PASSWORD", "EMAIL_PASSWORD")
    if not password:
        searched = ", ".join(
            str(path) for path in env_file_candidates(dict(os.environ), home)
        )
        raise RuntimeError(
            "No SMTP password: set MLB_ALERT_EMAIL_PASSWORD or EMAIL_PASSWORD, "
            f"in the environment or in one of {searched} "
            f"(override the file with {ALERT_ENV_VAR})"
        )
    sender = _first(
        settings, "MLB_ALERT_EMAIL_FROM", "BETTING_ARB_EMAIL_FROM"
    ) or DEFAULT_SENDER
    port = _first(settings, "MLB_ALERT_SMTP_PORT", "BETTING_ARB_SMTP_PORT")
    return AlertConfig(
        sender=sender,
        username=_first(
            settings, "MLB_ALERT_EMAIL_USERNAME", "BETTING_ARB_EMAIL_USERNAME"
        )
        or sender,
        recipient=_first(settings, "MLB_ALERT_EMAIL_TO", "BETTING_ARB_EMAIL_TO")
        or DEFAULT_RECIPIENT,
        host=_first(settings, "MLB_ALERT_SMTP_HOST", "BETTING_ARB_SMTP_HOST")
        or DEFAULT_SMTP_HOST,
        port=int(port) if port.isdigit() else DEFAULT_SMTP_PORT,
        password=password,
    )


def tail(path: Path, lines: int) -> str:
    """Return the last ``lines`` of a log, or a note explaining why it cannot."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"<could not read {path}: {exc}>"
    kept = content.splitlines()[-lines:]
    return "\n".join(kept) if kept else f"<{path} is empty>"


def build_message(
    *,
    label: str,
    exit_code: int,
    log_path: Path | None,
    lines: int = DEFAULT_LOG_LINES,
    hostname: str | None = None,
    when: datetime | None = None,
) -> tuple[str, str]:
    """Return the (subject, body) of a failure alert."""
    host = hostname or platform.node() or socket.gethostname()
    stamp = (when or datetime.now().astimezone()).strftime("%Y-%m-%d %H:%M:%S %Z")
    subject = f"[mlb] {label} FAILED (exit {exit_code}) on {host}"
    body = [
        f"agent:     {label}",
        f"exit code: {exit_code}",
        f"host:      {host}",
        f"when:      {stamp}",
        f"log:       {log_path if log_path else '<not supplied>'}",
        "",
    ]
    if log_path is not None:
        body += [f"--- last {lines} lines of {log_path.name} ---", tail(log_path, lines)]
    else:
        body.append("No log path was supplied, so no output is attached.")
    return subject, "\n".join(body)


def send(subject: str, body: str, config: AlertConfig) -> None:
    message = EmailMessage()
    message["From"] = config.sender
    message["To"] = config.recipient
    message["Subject"] = subject
    message.set_content(body)
    with smtplib.SMTP(config.host, config.port, timeout=SMTP_TIMEOUT) as smtp:
        smtp.starttls()
        smtp.login(config.username, config.password)
        smtp.send_message(message)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="launchd agent label")
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--log", type=str, default=None, help="stderr log to excerpt")
    parser.add_argument("--lines", type=int, default=DEFAULT_LOG_LINES)
    parser.add_argument(
        "--print",
        action="store_true",
        dest="print_only",
        help="render the alert without sending it",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    log_path = Path(args.log).expanduser() if args.log else None
    subject, body = build_message(
        label=args.label,
        exit_code=args.exit_code,
        log_path=log_path,
        lines=args.lines,
    )
    if args.print_only:
        print(subject)
        print()
        print(body)
        return 0
    try:
        send(subject, body, resolve_config())
    except Exception as exc:
        # The caller is already failing; say why the alert could not go out
        # rather than replacing its exit status with this one.
        print(f"failure alert NOT sent ({type(exc).__name__}: {exc})", file=sys.stderr)
        return 1
    print(f"failure alert sent: {subject}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
