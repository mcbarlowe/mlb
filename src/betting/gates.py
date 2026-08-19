"""Betting-gate artifact helpers for paper-trade runners."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

__all__ = ["BettingGate", "load_betting_gate", "require_open_gate"]

GateStatus = Literal["open", "closed"]


@dataclass(frozen=True)
class BettingGate:
    status: GateStatus
    reason: str
    artifact: str | None = None
    metrics: Mapping[str, object] = field(default_factory=dict)

    @property
    def is_open(self) -> bool:
        return self.status == "open"


def _payload_gate(payload: object) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise TypeError("Betting gate artifact must be a JSON object")
    gate = payload.get("betting_gate", payload)
    if not isinstance(gate, Mapping):
        raise TypeError("betting_gate must be a JSON object")
    return gate


def load_betting_gate(path: str | Path) -> BettingGate:
    """Load a betting gate artifact.

    The accepted schema is either the gate object itself or a top-level object with
    a ``betting_gate`` object. Missing status defaults to closed.
    """
    gate_path = Path(path)
    payload = json.loads(gate_path.read_text())
    gate = _payload_gate(payload)
    status_text = str(gate.get("status", "closed")).lower()
    if status_text not in {"open", "closed"}:
        raise ValueError(f"Unsupported betting gate status {status_text!r}")
    reason = str(gate.get("reason") or "No gate reason supplied")
    metrics = gate.get("metrics", {})
    if not isinstance(metrics, Mapping):
        metrics = {}
    return BettingGate(
        status=status_text,  # type: ignore[arg-type]
        reason=reason,
        artifact=str(gate_path),
        metrics=metrics,
    )


def require_open_gate(path: str | Path) -> BettingGate:
    """Return an open gate or raise SystemExit with the closed-gate reason."""
    gate = load_betting_gate(path)
    if not gate.is_open:
        raise SystemExit(f"Betting gate is CLOSED: {gate.reason} ({gate.artifact})")
    return gate
