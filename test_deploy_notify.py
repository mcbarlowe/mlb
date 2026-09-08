"""Guards on the scheduled-agent failure alert.

This exists because a failing agent used to report nothing but an exit code, so
the tests here defend the properties that make the alert worth having: it names
the agent and status, it carries enough log to diagnose, it finds credentials
the way the deployment host supplies them, and it never converts a job failure
into a different failure.
"""

from __future__ import annotations

import os
import smtplib
import subprocess
from pathlib import Path
from typing import ClassVar

import pytest

from mlb.deploy import notify
from mlb.deploy.agents import AGENTS, agent_by_label
from mlb.deploy.install import RUNNER_SUBDIR, read_resource


def _config() -> notify.AlertConfig:
    return notify.AlertConfig(
        sender="from@example.com",
        username="user@example.com",
        recipient="to@example.com",
        host="smtp.example.com",
        port=587,
        password="secret",
    )


class _FakeSMTP:
    sent: ClassVar[list[tuple[str, str, str]]] = []

    def __init__(self, host, port, timeout=None):
        self.host, self.port, self.timeout = host, port, timeout

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starttls(self):
        self.tls = True

    def login(self, username, password):
        self.credentials = (username, password)

    def send_message(self, message):
        type(self).sent.append(
            (message["To"], message["Subject"], message.get_content())
        )


def test_the_alert_names_the_agent_status_and_carries_the_failing_output(tmp_path):
    log = tmp_path / "daily-random-live-game.err.log"
    log.write_text(
        "\n".join(
            [*(f"noise line {i}" for i in range(200)), "ModuleNotFoundError: no src"]
        )
    )

    subject, body = notify.build_message(
        label="com.barloweanalytics.daily-random-live-game",
        exit_code=1,
        log_path=log,
        lines=5,
        hostname="imac",
    )

    assert "com.barloweanalytics.daily-random-live-game" in subject
    assert "FAILED" in subject and "exit 1" in subject and "imac" in subject
    # The nine-day outage was diagnosable from the last line of stderr alone.
    assert "ModuleNotFoundError: no src" in body
    assert str(log) in body
    assert "noise line 0" not in body


def test_a_missing_log_still_produces_a_usable_alert(tmp_path):
    _, body = notify.build_message(
        label="com.barloweanalytics.daily-sim-slate",
        exit_code=2,
        log_path=tmp_path / "absent.log",
    )
    assert "could not read" in body

    _, body = notify.build_message(
        label="com.barloweanalytics.daily-sim-slate", exit_code=2, log_path=None
    )
    assert "No log path was supplied" in body


def test_env_files_are_parsed_but_shell_logic_in_them_is_not():
    parsed = notify.parse_env_file(
        "# comment\n"
        "export EMAIL_PASSWORD='app pw'\n"
        'BETTING_ARB_EMAIL_TO="owner@example.com"\n'
        "BETTING_ARB_SMTP_PORT=2525\n"
        "if [[ -n $X ]]; then\n"
        "echo nope\n"
        "  \n"
    )
    assert parsed == {
        "EMAIL_PASSWORD": "app pw",
        "BETTING_ARB_EMAIL_TO": "owner@example.com",
        "BETTING_ARB_SMTP_PORT": "2525",
    }


def test_mlb_settings_win_over_the_shared_ones_and_the_process_wins_over_files(tmp_path):
    shared = tmp_path / ".config/betting/arbitrage.env"
    shared.parent.mkdir(parents=True)
    shared.write_text(
        "EMAIL_PASSWORD=filepw\n"
        "BETTING_ARB_EMAIL_TO=shared@example.com\n"
        "BETTING_ARB_SMTP_PORT=2525\n"
        "BETTING_ARB_EMAIL_FROM=shared-from@example.com\n"
    )

    from_file = notify.resolve_config({"HOME": str(tmp_path)}, home=tmp_path)
    assert from_file.password == "filepw"
    assert from_file.recipient == "shared@example.com"
    assert from_file.port == 2525
    assert from_file.username == "shared-from@example.com"

    overridden = notify.resolve_config(
        {
            "HOME": str(tmp_path),
            "MLB_ALERT_EMAIL_TO": "mlb@example.com",
            "MLB_ALERT_EMAIL_PASSWORD": "envpw",
        },
        home=tmp_path,
    )
    assert overridden.recipient == "mlb@example.com"
    assert overridden.password == "envpw"


def test_an_alert_with_no_credentials_says_where_it_looked(tmp_path):
    with pytest.raises(RuntimeError, match="MLB_ALERT_EMAIL_PASSWORD"):
        notify.resolve_config({"HOME": str(tmp_path)}, home=tmp_path)


def test_sending_uses_starttls_and_the_resolved_recipient(monkeypatch):
    _FakeSMTP.sent = []
    monkeypatch.setattr(smtplib, "SMTP", _FakeSMTP)

    notify.send("subject", "body", _config())

    assert _FakeSMTP.sent == [("to@example.com", "subject", "body\n")]


def test_an_unsendable_alert_reports_itself_instead_of_raising(monkeypatch, capsys):
    def explode(*args, **kwargs):
        raise OSError("smtp unreachable")

    monkeypatch.setattr(smtplib, "SMTP", explode)
    monkeypatch.setattr(notify, "resolve_config", _config)

    code = notify.main(["--label", "agent", "--exit-code", "3"])

    assert code == 1
    assert "failure alert NOT sent" in capsys.readouterr().err


@pytest.mark.parametrize("agent", [a for a in AGENTS if a.runner])
def test_every_scheduled_runner_alerts_on_a_nonzero_exit(agent):
    body = read_resource(f"{RUNNER_SUBDIR}/{agent.runner}")
    short_label = agent.label.removeprefix("com.barloweanalytics.")

    assert f'AGENT_LABEL="{agent.label}"' in body
    assert f'ERR_LOG="__MLB_LOG_DIR__/{short_label}.err.log"' in body
    # The alert must hang off the existing EXIT trap: bash keeps only one, so a
    # second trap would silently drop the lock cleanup.
    assert body.count("trap cleanup EXIT") == 1
    assert 'if [[ "$status" != "0" ]]; then' in body
    assert '"$BIN_DIR/mlb-notify-failure" --label "$AGENT_LABEL"' in body
    # A failing alert must not replace the job's own exit status.
    assert body.index("mlb-notify-failure") < body.index('exit "$status"')
    assert "mlb-notify-failure" in agent.console_scripts


def _fake_bin(directory: Path, name: str, script: str) -> Path:
    path = directory / name
    path.write_text(f"#!/bin/bash\n{script}\n")
    path.chmod(0o755)
    return path


@pytest.mark.skipif(
    not Path("/usr/bin/caffeinate").exists(), reason="runner requires macOS caffeinate"
)
def test_a_failing_run_actually_fires_the_alert_and_keeps_its_own_exit_status(tmp_path):
    """Bash keeps one EXIT trap, so this has to be proved by running the script."""
    from mlb.deploy.install import InstallPaths, render_runner

    agent = agent_by_label("com.barloweanalytics.daily-random-live-game")
    bin_dir = tmp_path / "tools"
    log_dir = tmp_path / "logs"
    for directory in (bin_dir, log_dir, tmp_path / "state"):
        directory.mkdir(parents=True)
    (log_dir / "daily-random-live-game.err.log").write_text("boom: the real cause\n")

    recorded = tmp_path / "alert.txt"
    _fake_bin(bin_dir, "mlb-build-pitcher-movement-profiles", "exit 0")
    _fake_bin(bin_dir, "mlb-live-pipeline", "exit 7")
    _fake_bin(bin_dir, "mlb-notify-failure", f'printf "%s\\n" "$@" > {recorded}')

    runner = tmp_path / agent.runner
    runner.write_text(
        render_runner(
            agent,
            InstallPaths(
                home=tmp_path,
                state_root=tmp_path / "state",
                runner_dir=tmp_path,
                bin_dir=bin_dir,
                log_dir=log_dir,
                social_env=tmp_path / "absent.env",
                launch_agents_dir=tmp_path / "agents",
            ),
        )
    )
    runner.chmod(0o755)

    result = subprocess.run(
        ["/bin/bash", str(runner)],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "BARLOWE_RANDOM_GAME_POST": "0",
            "TMPDIR": str(tmp_path),
        },
        check=False,
    )

    # The job's own status survives the alert.
    assert result.returncode == 7
    assert recorded.is_file(), result.stdout + result.stderr
    arguments = recorded.read_text().split("\n")
    assert "--label" in arguments
    assert agent.label in arguments
    assert arguments[arguments.index("--exit-code") + 1] == "7"
    assert arguments[arguments.index("--log") + 1] == str(
        log_dir / "daily-random-live-game.err.log"
    )


@pytest.mark.skipif(
    not Path("/usr/bin/caffeinate").exists(), reason="runner requires macOS caffeinate"
)
def test_a_successful_run_sends_nothing(tmp_path):
    from mlb.deploy.install import InstallPaths, render_runner

    agent = agent_by_label("com.barloweanalytics.daily-random-live-game")
    bin_dir = tmp_path / "tools"
    log_dir = tmp_path / "logs"
    for directory in (bin_dir, log_dir, tmp_path / "state"):
        directory.mkdir(parents=True)

    recorded = tmp_path / "alert.txt"
    _fake_bin(bin_dir, "mlb-build-pitcher-movement-profiles", "exit 0")
    _fake_bin(bin_dir, "mlb-live-pipeline", "exit 0")
    _fake_bin(bin_dir, "mlb-notify-failure", f'printf "%s\\n" "$@" > {recorded}')

    runner = tmp_path / agent.runner
    runner.write_text(
        render_runner(
            agent,
            InstallPaths(
                home=tmp_path,
                state_root=tmp_path / "state",
                runner_dir=tmp_path,
                bin_dir=bin_dir,
                log_dir=log_dir,
                social_env=tmp_path / "absent.env",
                launch_agents_dir=tmp_path / "agents",
            ),
        )
    )
    runner.chmod(0o755)

    result = subprocess.run(
        ["/bin/bash", str(runner)],
        capture_output=True,
        text=True,
        env={**os.environ, "BARLOWE_RANDOM_GAME_POST": "0", "TMPDIR": str(tmp_path)},
        check=False,
    )

    assert result.returncode == 0
    assert not recorded.exists()
