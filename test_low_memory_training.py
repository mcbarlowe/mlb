from __future__ import annotations

import polars as pl
import torch

from src.ml.dataset import PitchSequenceIterableDataset, collate_pitch_sequences
from src.ml.features import (
    PitchFeatureEngine,
    compute_class_weights,
    compute_class_weights_from_counts,
)
from src.ml.pitch_type_location_model import PitchTypeLocationBatchIterableDataset


class IdentityFeatureEngine:
    def __init__(self, feature_columns: list[str], target_columns: list[str]):
        self._feature_columns = feature_columns
        self._target_columns = target_columns

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        return df

    def get_feature_columns(self) -> list[str]:
        return self._feature_columns

    def get_target_columns(self) -> list[str]:
        return self._target_columns


def test_fit_frames_matches_concat_fit():
    season_one = pl.DataFrame(
        {
            "pitcher_id": [10, 10, 20],
            "batter_id": [100, 101, 100],
            "pitch_type_code": ["FF", "SL", "FF"],
        }
    )
    season_two = pl.DataFrame(
        {
            "pitcher_id": [10, 30],
            "batter_id": [102, 103],
            "pitch_type_code": ["CH", "CU"],
        }
    )

    concat_engine = PitchFeatureEngine()
    concat_engine.fit(pl.concat([season_one, season_two], how="diagonal"))

    streaming_engine = PitchFeatureEngine()
    streaming_engine.fit_frames([season_one, season_two])

    assert streaming_engine.pitcher_to_idx == concat_engine.pitcher_to_idx
    assert streaming_engine.batter_to_idx == concat_engine.batter_to_idx
    assert streaming_engine.pitcher_ff_pct == concat_engine.pitcher_ff_pct
    assert streaming_engine.pitcher_repertoire_size == concat_engine.pitcher_repertoire_size



def test_compute_class_weights_from_counts_matches_dataframe_helper():
    df = pl.DataFrame({"pitch_type_idx": [0, 0, 1, 2, 2, 2]})
    weights_from_df = compute_class_weights(df, smoothing=0.5)
    weights_from_counts = compute_class_weights_from_counts({0: 2, 1: 1, 2: 3}, smoothing=0.5)

    assert torch.allclose(weights_from_df, weights_from_counts)



def test_pitch_sequence_iterable_dataset_streams_sequences():
    season_frames = {
        "2021": pl.DataFrame(
            {
                "game_pk": [1, 1, 1, 1],
                "at_bat_index": [1, 1, 2, 2],
                "pitch_number": [1, 2, 1, 2],
                "feat_a": [0.1, 0.2, 0.3, 0.4],
                "feat_b": [1.0, 1.1, 1.2, 1.3],
                "pitch_type_idx": [0.0, 1.0, 2.0, 2.0],
                "px": [0.5, 0.6, 0.7, 0.8],
                "pz": [2.0, 2.1, 2.2, 2.3],
            }
        ),
        "2022": pl.DataFrame(
            {
                "game_pk": [2, 2, 2],
                "at_bat_index": [1, 1, 1],
                "pitch_number": [1, 2, 3],
                "feat_a": [0.9, 1.0, 1.1],
                "feat_b": [1.4, 1.5, 1.6],
                "pitch_type_idx": [3.0, 3.0, 4.0],
                "px": [0.9, 1.0, 1.1],
                "pz": [2.4, 2.5, 2.6],
            }
        ),
    }
    dataset = PitchSequenceIterableDataset(
        seasons=["2021", "2022"],
        load_season=season_frames.__getitem__,
        transform_season=lambda df: df,
        feature_columns=["feat_a", "feat_b"],
        target_columns=["pitch_type_idx", "px", "pz"],
        max_seq_len=3,
        shuffle=False,
    )

    samples = list(dataset)

    assert len(samples) == 3
    assert samples[0]["features"].shape == (2, 2)
    assert samples[2]["features"].shape == (3, 2)

    batch = collate_pitch_sequences(samples[:2])
    assert batch["features"].shape == (2, 2, 2)
    assert batch["targets"].shape == (2, 2, 3)
    assert batch["lengths"].tolist() == [2, 2]



def test_pitch_type_location_batch_iterable_dataset_streams_batched_rows():
    season_frames = {
        "2021": pl.DataFrame(
            {
                "feat_a": [0.1, 0.2, 0.3],
                "feat_b": [1.0, 1.1, 1.2],
                "pitch_type_idx": [0, 1, 2],
                "px": [0.5, 0.6, 0.7],
                "pz": [2.0, 2.1, 2.2],
            }
        ),
        "2022": pl.DataFrame(
            {
                "feat_a": [0.4, 0.5],
                "feat_b": [1.3, 1.4],
                "pitch_type_idx": [3, 4],
                "px": [0.8, 0.9],
                "pz": [2.3, 2.4],
            }
        ),
    }
    dataset = PitchTypeLocationBatchIterableDataset(
        seasons=["2021", "2022"],
        load_season=season_frames.__getitem__,
        transform_season=lambda df: df,
        feature_columns=["feat_a", "feat_b"],
        batch_size=2,
        shuffle=False,
    )

    batches = list(dataset)

    assert [batch[0].shape[0] for batch in batches] == [2, 1, 2]
    assert batches[0][0].shape[1] == 2
    assert batches[0][1].dtype == torch.int64
    assert batches[0][2].shape[1] == 2
