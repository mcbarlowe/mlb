from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

import scripts.publish_totals_simulations as publisher
from mlb.sim.totals_artifact import (
    ARTIFACT_TYPE,
    CONTRACT_VERSION,
    TotalsSimulationArtifact,
    TotalsSimulationGame,
    game_totals_from_simulations,
    read_totals_artifact,
    write_totals_artifact,
)

GENERATED_AT = datetime(2026, 8, 28, 12, 0, tzinfo=UTC)


def totals_artifact(
    *,
    games: tuple[TotalsSimulationGame, ...] | None = None,
) -> TotalsSimulationArtifact:
    return TotalsSimulationArtifact(
        generated_at=GENERATED_AT.isoformat(),
        season=2025,
        sims_per_game=4,
        model_name="mlb-game-simulator-totals",
        model_version="outcome-a1-b2",
        games=games if games is not None else (game_totals_from_simulations(777, [7, 8, 8, 10]),),
    )


def test_totals_artifact_round_trip_and_atomic_replace(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "totals.json"
    path.parent.mkdir()
    path.write_text("partial", encoding="utf-8")

    write_totals_artifact(path, totals_artifact())

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload) == {
        "contract_version",
        "artifact_type",
        "generated_at",
        "season",
        "sims_per_game",
        "model_name",
        "model_version",
        "games",
    }
    assert payload["contract_version"] == CONTRACT_VERSION
    assert payload["artifact_type"] == ARTIFACT_TYPE
    assert payload["games"] == [
        {
            "game_pk": 777,
            "total_counts": {"7": 1, "8": 2, "10": 1},
            "sim_mean_total": 8.25,
            "sim_total_stdev": pytest.approx(1.0897247358851685),
        }
    ]
    assert read_totals_artifact(path) == totals_artifact()
    assert list(path.parent.iterdir()) == [path]


def test_totals_artifact_rejects_invalid_counts() -> None:
    with pytest.raises(ValueError, match="non-negative integer strings"):
        TotalsSimulationGame(
            game_pk=1,
            total_counts={"-1": 1},
            sim_mean_total=0.0,
            sim_total_stdev=0.0,
        )

    with pytest.raises(ValueError, match="positive integer"):
        TotalsSimulationGame(
            game_pk=1,
            total_counts={"8": 0},
            sim_mean_total=8.0,
            sim_total_stdev=0.0,
        )

    with pytest.raises(ValueError, match="sum must equal sims_per_game"):
        totals_artifact(
            games=(
                TotalsSimulationGame(
                    game_pk=1,
                    total_counts={"8": 3},
                    sim_mean_total=8.0,
                    sim_total_stdev=0.0,
                ),
            )
        )


def test_publisher_main_writes_injected_producer_output(tmp_path: Path) -> None:
    output_json = tmp_path / "totals.json"
    requests: list[publisher.TotalsSimulationRequest] = []

    def producer(
        request: publisher.TotalsSimulationRequest,
    ) -> TotalsSimulationArtifact:
        requests.append(request)
        return TotalsSimulationArtifact(
            generated_at=GENERATED_AT.isoformat(),
            season=request.season,
            sims_per_game=request.sims_per_game,
            model_name=request.model_name,
            model_version=request.model_version or "smoke-version",
            games=(game_totals_from_simulations(123, [3, 4, 5]),),
        )

    artifact = publisher.main(
        [
            "--season",
            "2025",
            "--games",
            "1",
            "--sims",
            "3",
            "--seed",
            "11",
            "--pa-calibration",
            "models/sim/pa.json",
            "--mlflow-tracking-uri",
            "file:mlruns",
            "--model-name",
            "totals-smoke-model",
            "--model-version",
            "v-smoke",
            "--output-json",
            str(output_json),
        ],
        producer=producer,
    )

    assert requests == [
        publisher.TotalsSimulationRequest(
            season=2025,
            games=1,
            sims_per_game=3,
            seed=11,
            pa_calibration=Path("models/sim/pa.json"),
            mlflow_tracking_uri="file:mlruns",
            model_name="totals-smoke-model",
            model_version="v-smoke",
        )
    ]
    assert artifact == read_totals_artifact(output_json)
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["games"][0]["total_counts"] == {"3": 1, "4": 1, "5": 1}
