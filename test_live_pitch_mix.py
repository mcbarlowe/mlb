"""Tests for the unknown-pitcher empirical pitch-mix backoff."""

import numpy as np

from src.live.pitch_mix import (
    blend_with_empirical_mix,
    counts_to_vector,
)
from src.ml.features import PITCH_TYPE_CODES, PITCH_TYPE_TO_IDX


def test_counts_to_vector_maps_codes_and_folds_unknown_into_other():
    vector = counts_to_vector({"FF": 10, "ST": 5, "SV": 3, "XX": 2, "CU": 0})
    assert vector[PITCH_TYPE_TO_IDX["FF"]] == 10
    assert vector[PITCH_TYPE_TO_IDX["ST"]] == 5
    # SV and XX are not canonical codes and fold into OTHER.
    assert vector[PITCH_TYPE_TO_IDX["OTHER"]] == 5
    assert vector[PITCH_TYPE_TO_IDX["CU"]] == 0
    assert vector.sum() == 20


def test_blend_without_history_returns_model_distribution():
    model = np.zeros(len(PITCH_TYPE_CODES))
    model[PITCH_TYPE_TO_IDX["FF"]] = 0.95
    model[PITCH_TYPE_TO_IDX["SI"]] = 0.05
    blended = blend_with_empirical_mix(model, np.zeros(len(PITCH_TYPE_CODES)))
    np.testing.assert_allclose(blended, model, atol=1e-9)


def test_blend_pulls_overconfident_model_toward_empirical_mix():
    # The Dylan Smith miss: model said FF 95% for a pitcher it never saw.
    model = np.full(len(PITCH_TYPE_CODES), 1e-6)
    model[PITCH_TYPE_TO_IDX["FF"]] = 0.95
    model[PITCH_TYPE_TO_IDX["SI"]] = 0.04
    model = model / model.sum()

    counts = counts_to_vector({"FF": 149, "ST": 98, "SI": 25, "FS": 18, "CU": 17})
    blended = blend_with_empirical_mix(model, counts)

    assert abs(blended.sum() - 1.0) < 1e-9
    ff = blended[PITCH_TYPE_TO_IDX["FF"]]
    st = blended[PITCH_TYPE_TO_IDX["ST"]]
    cu = blended[PITCH_TYPE_TO_IDX["CU"]]
    # FF stays the top call but loses its hallucinated certainty.
    assert blended.argmax() == PITCH_TYPE_TO_IDX["FF"]
    assert ff < 0.7
    # The sweeper becomes a visible second option and the curveball is live.
    assert st > 0.2
    assert cu > 0.03


def test_blend_converges_to_empirical_with_large_samples():
    model = np.full(len(PITCH_TYPE_CODES), 1e-6)
    model[PITCH_TYPE_TO_IDX["FF"]] = 1.0
    model = model / model.sum()
    counts = counts_to_vector({"ST": 5000})
    blended = blend_with_empirical_mix(model, counts)
    assert blended.argmax() == PITCH_TYPE_TO_IDX["ST"]
    assert blended[PITCH_TYPE_TO_IDX["ST"]] > 0.9
