"""Packaged launchd agents, runners, and their installer.

The mlb package is deployed as a wheel, so every operational file a scheduler
needs must travel inside the distribution rather than being read out of a
source checkout.
"""

from __future__ import annotations

from mlb.deploy.agents import (
    AGENTS,
    Agent,
    agent_by_label,
    selected_agents,
)

__all__ = [
    "AGENTS",
    "Agent",
    "agent_by_label",
    "selected_agents",
]
