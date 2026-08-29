"""Tests for training-time player identity dropout."""

import polars as pl
import torch

from mlb.ml.dataset import PitchSequenceDataset, PlayerDropoutSpec

FEATURE_COLUMNS = [
    "pitcher_idx",
    "batter_idx",
    "pitcher_ff_pct",
    "pitcher_repertoire",
    "balls",
]
TARGET_COLUMNS = ["pitch_type_idx"]


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_pk": [1, 1, 1, 1],
            "at_bat_index": [0, 0, 1, 1],
            "pitch_number": [1, 2, 1, 2],
            "pitcher_idx": [7.0, 7.0, 9.0, 9.0],
            "batter_idx": [3.0, 3.0, 5.0, 5.0],
            "pitcher_ff_pct": [0.61, 0.61, 0.44, 0.44],
            "pitcher_repertoire": [5 / 7, 5 / 7, 3 / 7, 3 / 7],
            "balls": [0.0, 1.0, 0.0, 1.0],
            "pitch_type_idx": [0.0, 4.0, 1.0, 0.0],
        }
    )


def _spec(rate: float) -> PlayerDropoutSpec:
    return PlayerDropoutSpec.build(
        rate=rate,
        feature_columns=FEATURE_COLUMNS,
        n_known_pitchers=100,
        n_known_batters=200,
    )


def test_full_dropout_rewrites_identity_features_only():
    dataset = PitchSequenceDataset(
        _frame(), FEATURE_COLUMNS, TARGET_COLUMNS, player_dropout=_spec(1.0)
    )
    sample = dataset[0]
    features = sample["features"]
    assert torch.all(features[:, 0] == 100.0)  # pitcher -> unknown slot
    assert torch.all(features[:, 1] == 200.0)  # batter -> unknown slot
    assert torch.all(features[:, 2] == 0.5)  # ff_pct -> transform default
    assert torch.allclose(features[:, 3], torch.tensor(4.0 / 7.0))
    # Non-identity features are untouched.
    assert features[0, 4] == 0.0 and features[1, 4] == 1.0


def test_dropout_never_mutates_cached_tensors():
    dataset = PitchSequenceDataset(
        _frame(), FEATURE_COLUMNS, TARGET_COLUMNS, player_dropout=_spec(1.0)
    )
    _ = dataset[0]
    cached = dataset.at_bats[0]["features"]
    assert torch.all(cached[:, 0] == 7.0)
    assert torch.all(cached[:, 2] == torch.tensor(0.61))


def test_zero_rate_is_identity():
    plain = PitchSequenceDataset(_frame(), FEATURE_COLUMNS, TARGET_COLUMNS)
    dropped = PitchSequenceDataset(
        _frame(), FEATURE_COLUMNS, TARGET_COLUMNS, player_dropout=_spec(0.0)
    )
    assert torch.equal(plain[1]["features"], dropped[1]["features"])
