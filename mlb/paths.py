"""Filesystem roots for generated data, models, and reports.

Installed as a wheel, this package no longer sits inside the directory holding
its generated state, so nothing may derive that state's location from
``__file__``. Most call sites reference state with working-directory-relative
paths such as ``Path("models/sim")`` and keep working because every scheduled
agent runs with its working directory set to the state root. The helpers here
exist for the remaining call sites that need a root explicitly.
"""

from __future__ import annotations

import os
import sysconfig
from pathlib import Path
from typing import Final

STATE_ROOT_ENV: Final = "MLB_STATE_ROOT"
DEFAULT_STATE_ROOT: Final = Path("code/python/mlb")

__all__ = ["DEFAULT_STATE_ROOT", "STATE_ROOT_ENV", "console_script", "state_root"]


def state_root() -> Path:
    """Return the directory holding ``data/``, ``models/``, and ``output/``.

    Resolution order is the ``MLB_STATE_ROOT`` environment variable, then the
    conventional location under the user's home directory. The result is never
    derived from this package's install location or the current working
    directory, so it is stable whether the caller runs from a checkout or from
    an installed distribution.
    """
    configured = os.environ.get(STATE_ROOT_ENV, "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / DEFAULT_STATE_ROOT


def console_script(name: str) -> Path:
    """Return the path to a console script installed with this distribution.

    Resolved through ``sysconfig`` rather than ``PATH`` so that a subprocess
    re-enters the same environment that is currently running. Note that
    ``sys.executable`` must not be resolved through symlinks for this purpose: a
    uv-managed tool environment links its interpreter to a shared CPython
    install, and following that link points away from the installed scripts.
    """
    return Path(sysconfig.get_path("scripts")) / name
