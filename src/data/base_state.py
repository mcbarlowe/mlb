"""Reconstruct true base/out state per at-bat from GUMBO play data.

The archived live feeds do not carry a base-state snapshot per play: the
``runners`` array only lists runners who moved. Correct at-bat start state
therefore requires walking each half-inning and applying runner movements
in order. Post-play outs come from the authoritative ``play.count.outs``.
"""

from __future__ import annotations

_BASES = ("1B", "2B", "3B")


def _apply_runner_movements(play: dict, bases: dict[str, int | None]) -> None:
    """Apply a play's runner movements to the base occupancy map.

    Each movement re-places its runner: he is removed from any base he
    currently occupies, then added to ``movement.end`` when that is a base
    (outs and scored runs simply leave him off the map). Entries arrive in
    chronological order within a play; a stable sort by ``details.playIndex``
    preserves that while grouping multi-event plays correctly. Re-placement
    makes duplicate movement entries idempotent.
    """
    entries = []
    for order, runner in enumerate(play.get("runners", [])):
        details = runner.get("details", {})
        runner_id = details.get("runner", {}).get("id")
        if runner_id is None:
            continue
        play_index = details.get("playIndex")
        movement = runner.get("movement", {})
        entries.append((play_index if play_index is not None else -1, order, runner_id, movement))
    entries.sort(key=lambda item: (item[0], item[1]))

    for _, _, runner_id, movement in entries:
        for base in _BASES:
            if bases[base] == runner_id:
                bases[base] = None
        if movement.get("isOut"):
            continue
        end = movement.get("end")
        if end in bases:
            bases[end] = runner_id


def compute_at_bat_states(all_plays: list[dict]) -> list[dict]:
    """Per-play base/out state, aligned index-for-index with ``all_plays``.

    Returns one dict per play:
    - ``outs_before`` / ``is_runner_on_*`` / ``runner_on_*_id``: state when
      the at-bat began
    - ``outs_after``: authoritative post-play outs from ``play.count.outs``
    """
    states: list[dict] = []
    bases: dict[str, int | None] = dict.fromkeys(_BASES)
    outs = 0
    current_half: tuple[int | None, str | None] | None = None

    for play in all_plays:
        about = play.get("about", {})
        half = (about.get("inning"), about.get("halfInning"))
        if half != current_half:
            current_half = half
            bases = dict.fromkeys(_BASES)
            outs = 0

        states.append(
            {
                "outs_before": outs,
                "is_runner_on_first": bases["1B"] is not None,
                "runner_on_first_id": bases["1B"],
                "is_runner_on_second": bases["2B"] is not None,
                "runner_on_second_id": bases["2B"],
                "is_runner_on_third": bases["3B"] is not None,
                "runner_on_third_id": bases["3B"],
                "outs_after": play.get("count", {}).get("outs", outs),
            }
        )

        _apply_runner_movements(play, bases)
        outs = play.get("count", {}).get("outs", outs)

    return states
