"""Install MLB-owned launchd agents from packaged runners.

The wheel carries every runner and plist template, so this installer never
reads a source checkout. It materializes runners into a stable support
directory, renders plists against the console scripts that were installed with
this distribution, and optionally loads the agents through launchd.
"""

from __future__ import annotations

import argparse
import os
import plistlib
import re
import subprocess
import sysconfig
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Final

from mlb.deploy.agents import (
    AGENTS,
    MLFLOW_BACKEND_URI_ENV,
    Agent,
    selected_agents,
)
from mlb.paths import STATE_ROOT_ENV, state_root

RESOURCE_PACKAGE: Final = "mlb.deploy"
RUNNER_SUBDIR: Final = "resources"
LAUNCHD_SUBDIR: Final = "resources/launchd"
PLACEHOLDER_PATTERN: Final = re.compile(r"__[A-Z0-9_]+__")
RUNNER_MODE: Final = 0o755
PLIST_MODE: Final = 0o644

DEFAULT_SUPPORT_DIR: Final = Path("Library/Application Support/BarloweAnalytics")
DEFAULT_LOG_DIR: Final = Path("Library/Logs/BarloweAnalytics")
DEFAULT_SOCIAL_ENV: Final = Path(".config/barlowe/social.env")
LAUNCH_AGENTS_DIR: Final = Path("Library/LaunchAgents")


@dataclass(frozen=True)
class InstallPaths:
    """Absolute locations the rendered agents refer to."""

    home: Path
    state_root: Path
    runner_dir: Path
    bin_dir: Path
    log_dir: Path
    social_env: Path
    launch_agents_dir: Path

    def substitutions(self) -> dict[str, str]:
        return {
            "__HOME__": str(self.home),
            "__MLB_STATE_ROOT__": str(self.state_root),
            "__MLB_RUNNER_DIR__": str(self.runner_dir),
            "__MLB_BIN_DIR__": str(self.bin_dir),
            "__MLB_LOG_DIR__": str(self.log_dir),
            "__MLB_SOCIAL_ENV__": str(self.social_env),
        }


def default_bin_dir() -> Path:
    """Return the directory holding the console scripts of this distribution.

    ``sys.executable`` must not be resolved through symlinks: a uv-managed
    tool environment links its interpreter to a shared CPython install, and
    following that link points away from the installed console scripts.
    """
    return Path(sysconfig.get_path("scripts"))


def resolve_paths(
    *,
    home: Path | None = None,
    mlb_state_root: Path | None = None,
    runner_dir: Path | None = None,
    bin_dir: Path | None = None,
    log_dir: Path | None = None,
    social_env: Path | None = None,
    launch_agents_dir: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> InstallPaths:
    """Resolve every install location from arguments, environment, then defaults."""
    env = os.environ if environ is None else environ
    resolved_home = (home or Path(env.get("HOME") or Path.home())).expanduser()
    support_dir = resolved_home / DEFAULT_SUPPORT_DIR
    return InstallPaths(
        home=resolved_home,
        state_root=(
            mlb_state_root or _env_path(env, STATE_ROOT_ENV) or state_root()
        ).expanduser(),
        runner_dir=(
            runner_dir or _env_path(env, "MLB_RUNNER_DIR") or support_dir / "bin"
        ).expanduser(),
        bin_dir=(
            bin_dir or _env_path(env, "MLB_BIN_DIR") or default_bin_dir()
        ).expanduser(),
        log_dir=(
            log_dir or _env_path(env, "MLB_LOG_DIR") or resolved_home / DEFAULT_LOG_DIR
        ).expanduser(),
        social_env=(
            social_env
            or _env_path(env, "BARLOWE_SOCIAL_ENV")
            or resolved_home / DEFAULT_SOCIAL_ENV
        ).expanduser(),
        launch_agents_dir=(
            launch_agents_dir or resolved_home / LAUNCH_AGENTS_DIR
        ).expanduser(),
    )


def _env_path(environ: Mapping[str, str], name: str) -> Path | None:
    value = environ.get(name, "").strip()
    return Path(value) if value else None


def read_resource(relative_path: str) -> str:
    """Return one packaged resource as text."""
    return (
        resources.files(RESOURCE_PACKAGE)
        .joinpath(relative_path)
        .read_text(encoding="utf-8")
    )


def render(text: str, substitutions: Mapping[str, str], *, source: str) -> str:
    """Substitute every placeholder, refusing to emit an unresolved template."""
    rendered = text
    for placeholder, value in substitutions.items():
        rendered = rendered.replace(placeholder, value)
    leftover = sorted(set(PLACEHOLDER_PATTERN.findall(rendered)))
    if leftover:
        raise ValueError(f"{source} has unresolved placeholders: {', '.join(leftover)}")
    return rendered


def render_runner(agent: Agent, paths: InstallPaths) -> str:
    """Return the runner script body for ``agent``."""
    if agent.runner is None:
        raise ValueError(f"{agent.label} has no packaged runner")
    return render(
        read_resource(f"{RUNNER_SUBDIR}/{agent.runner}"),
        paths.substitutions(),
        source=agent.runner,
    )


def plist_substitutions(
    agent: Agent, paths: InstallPaths, *, environ: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Return the substitutions for ``agent``'s plist.

    The MLflow backend store URI carries a database password, so it is never
    committed to a template; it is supplied by the environment at install time.
    """
    substitutions = paths.substitutions()
    if agent.requires_mlflow_backend_uri:
        env = os.environ if environ is None else environ
        backend_uri = env.get(MLFLOW_BACKEND_URI_ENV, "").strip()
        if not backend_uri:
            raise SystemExit(
                f"{agent.label} needs the backend store URI in "
                f"${MLFLOW_BACKEND_URI_ENV}; export it and rerun"
            )
        substitutions["__MLB_MLFLOW_BACKEND_URI__"] = backend_uri
    return substitutions


def render_plist(
    agent: Agent, paths: InstallPaths, *, environ: Mapping[str, str] | None = None
) -> bytes:
    """Return the plist bytes for ``agent``."""
    text = render(
        read_resource(f"{LAUNCHD_SUBDIR}/{agent.plist_name}"),
        plist_substitutions(agent, paths, environ=environ),
        source=agent.plist_name,
    )
    return plistlib.dumps(plistlib.loads(text.encode("utf-8")))


def materialize_runners(
    agents: Sequence[Agent], paths: InstallPaths, *, dry_run: bool = False
) -> list[Path]:
    """Write each runner-backed agent's runner into the support directory."""
    with_runners = [agent for agent in agents if agent.runner is not None]
    written: list[Path] = []
    if with_runners and not dry_run:
        paths.runner_dir.mkdir(parents=True, exist_ok=True)
    for agent in with_runners:
        target = paths.runner_dir / str(agent.runner)
        if not dry_run:
            target.write_text(render_runner(agent, paths), encoding="utf-8")
            target.chmod(RUNNER_MODE)
        written.append(target)
    return written


def lint_plist(path: Path) -> None:
    """Validate a written plist with plutil."""
    subprocess.run(
        ["/usr/bin/plutil", "-lint", str(path)],
        check=True,
        capture_output=True,
    )


def launchctl_domain() -> str:
    return f"gui/{os.getuid()}"


def agent_is_loaded(label: str) -> bool:
    """Return whether launchd already knows ``label``."""
    result = subprocess.run(
        ["/bin/launchctl", "print", f"{launchctl_domain()}/{label}"],
        check=False,
        capture_output=True,
    )
    return result.returncode == 0


def preflight(agents: Sequence[Agent], paths: InstallPaths) -> None:
    """Refuse to load agents whose console scripts are not installed.

    A launchd job that cannot find its program fails silently on a schedule,
    so the missing binary is reported here instead of in a log nobody reads.
    """
    missing = sorted(
        {
            str(paths.bin_dir / script)
            for agent in agents
            for script in agent.console_scripts
            if not (paths.bin_dir / script).exists()
        }
    )
    if missing:
        raise SystemExit("Missing installed console scripts: " + ", ".join(missing))


def install(
    agents: Sequence[Agent],
    paths: InstallPaths,
    *,
    load: bool = False,
    dry_run: bool = False,
) -> list[Path]:
    """Materialize runners and plists, optionally loading each agent."""
    if load and not dry_run:
        preflight(agents, paths)

    # Every payload is rendered before anything is written, so a template that
    # cannot be resolved leaves no half-installed set of agents behind.
    payloads = [(agent, render_plist(agent, paths)) for agent in agents]

    materialize_runners(agents, paths, dry_run=dry_run)
    if not dry_run:
        paths.log_dir.mkdir(parents=True, exist_ok=True)
        paths.launch_agents_dir.mkdir(parents=True, exist_ok=True)

    installed: list[Path] = []
    domain = launchctl_domain()
    for agent, payload in payloads:
        target = paths.launch_agents_dir / agent.plist_name
        if not dry_run:
            target.write_bytes(payload)
            target.chmod(PLIST_MODE)
            lint_plist(target)
            if load:
                if agent_is_loaded(agent.label):
                    subprocess.run(
                        ["/bin/launchctl", "bootout", domain, str(target)],
                        check=True,
                    )
                subprocess.run(
                    ["/bin/launchctl", "bootstrap", domain, str(target)],
                    check=True,
                )
        installed.append(target)
    return installed


def uninstall(agents: Sequence[Agent], paths: InstallPaths) -> list[Path]:
    """Boot out and remove each agent's plist, leaving runners in place."""
    removed: list[Path] = []
    domain = launchctl_domain()
    for agent in agents:
        target = paths.launch_agents_dir / agent.plist_name
        if agent_is_loaded(agent.label):
            subprocess.run(
                ["/bin/launchctl", "bootout", domain, str(target)],
                check=False,
            )
        if target.exists():
            target.unlink()
            removed.append(target)
    return removed


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        action="append",
        metavar="LABEL",
        help="install one agent; repeat to select several (default: every agent)",
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="run preflight checks and load the agents through launchd",
    )
    parser.add_argument("--state-root", type=Path)
    parser.add_argument("--bin-dir", type=Path)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--runner-dir", type=Path)
    parser.add_argument("--launch-agents-dir", type=Path)
    parser.add_argument("--uninstall", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report the resolved plan without writing files or calling launchctl",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list every registered agent and exit",
    )
    args = parser.parse_args(argv)

    if args.list:
        for agent in AGENTS:
            print(f"{agent.label}\t{agent.runner or '-'}\t{agent.summary}")
        return

    if args.uninstall and args.load:
        parser.error("--uninstall cannot be combined with --load")

    try:
        agents = selected_agents(args.only)
    except KeyError as exc:
        parser.error(str(exc))
        raise AssertionError("unreachable") from exc

    paths = resolve_paths(
        mlb_state_root=args.state_root,
        runner_dir=args.runner_dir,
        bin_dir=args.bin_dir,
        log_dir=args.log_dir,
        launch_agents_dir=args.launch_agents_dir,
    )

    if args.uninstall:
        for path in uninstall(agents, paths):
            print(f"removed {path}", flush=True)
        return

    installed = install(agents, paths, load=args.load, dry_run=args.dry_run)
    prefix = "would install" if args.dry_run else "installed"
    print(f"state root: {paths.state_root}", flush=True)
    print(f"runners: {paths.runner_dir}", flush=True)
    print(f"console scripts: {paths.bin_dir}", flush=True)
    print(f"logs: {paths.log_dir}", flush=True)
    for path in installed:
        print(f"{prefix} {path}", flush=True)
    if args.load:
        print(f"loaded {len(installed)} agent(s) into {launchctl_domain()}", flush=True)
    elif not args.dry_run:
        print("launchd unchanged; rerun with --load to activate", flush=True)


if __name__ == "__main__":
    main()
