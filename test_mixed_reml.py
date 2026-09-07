"""Validation for the Woodbury-space REML estimator in ``mlb.analysis.mixed``.

The manager-effect conclusions rest entirely on the variance components this
module reports, and it evaluates the likelihood on the ``q x q`` random-effect
space rather than the ``n x n`` observation space. A sign error or a dropped
determinant term there produces a plausible-looking but wrong sigma, so the
dense likelihood and the dense BLUP posterior are recomputed here from the
definition and compared.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlb.analysis.mixed import (
    REMLProblem,
    between_group_statistic,
    fit_without_block,
    moment_variance_from_permutation,
    permutation_test,
    residualize,
)


def _design(seed: int = 11, n: int = 90, n_a: int = 7, n_b: int = 5):
    rng = np.random.default_rng(seed)
    codes_a = rng.integers(0, n_a, n)
    codes_b = rng.integers(0, n_b, n)
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    weights = rng.uniform(0.5, 4.0, n)
    y = (
        X @ np.array([2.0, -0.8, 0.35])
        + rng.normal(0.0, 1.3, n_a)[codes_a]
        + rng.normal(0.0, 0.7, n_b)[codes_b]
        + rng.normal(0.0, 1.0, n) / np.sqrt(weights)
    )
    return y, X, weights, {"a": codes_a, "b": codes_b}


def _indicator(codes: np.ndarray) -> np.ndarray:
    Z = np.zeros((codes.size, int(codes.max()) + 1))
    Z[np.arange(codes.size), codes] = 1.0
    return Z


def _dense_reml(y, X, weights, blocks, variances, resid_variance):
    """-2 REML and GLS beta computed directly in the n x n observation space."""

    n, p = X.shape
    V = np.diag(resid_variance / weights)
    for name, codes in blocks.items():
        Z = _indicator(codes)
        V = V + variances[name] * (Z @ Z.T)
    Vi = np.linalg.inv(V)
    XtViX = X.T @ Vi @ X
    beta = np.linalg.solve(XtViX, X.T @ Vi @ y)
    resid = y - X @ beta
    value = (
        np.linalg.slogdet(V)[1]
        + np.linalg.slogdet(XtViX)[1]
        + float(resid @ Vi @ resid)
        + (n - p) * np.log(2.0 * np.pi)
    )
    return value, beta


def test_woodbury_likelihood_matches_the_dense_observation_space_form():
    y, X, weights, blocks = _design()
    problem = REMLProblem(y, X, weights, blocks)

    value, beta, _ = problem.evaluate(np.array([1.7, 0.4]), 0.9)
    expected_value, expected_beta = _dense_reml(
        y, X, weights, blocks, {"a": 1.7, "b": 0.4}, 0.9
    )

    assert value == pytest.approx(expected_value, rel=1e-10)
    assert beta == pytest.approx(expected_beta, rel=1e-9)


def test_blups_match_the_dense_posterior_mean_and_sd():
    y, X, weights, blocks = _design()
    fit = REMLProblem(y, X, weights, blocks).fit()

    V = np.diag(fit.resid_variance / weights)
    indicators = {name: _indicator(codes) for name, codes in blocks.items()}
    for name, Z in indicators.items():
        V = V + fit.variances[name] * (Z @ Z.T)
    Vi = np.linalg.inv(V)
    XtViX = X.T @ Vi @ X
    P = Vi - Vi @ X @ np.linalg.solve(XtViX, X.T @ Vi)
    resid = y - X @ fit.beta

    for name, Z in indicators.items():
        s = fit.variances[name]
        assert fit.blups[name] == pytest.approx(s * (Z.T @ Vi @ resid), rel=1e-7, abs=1e-9)
        expected_sd = np.sqrt(np.maximum(s - s * np.diag(Z.T @ P @ Z) * s, 0.0))
        assert fit.blup_sd[name] == pytest.approx(expected_sd, rel=1e-6, abs=1e-9)


def test_removing_a_component_is_the_sigma_zero_null_and_never_fits_better():
    y, X, weights, blocks = _design()
    full = REMLProblem(y, X, weights, blocks).fit()
    reduced = fit_without_block(y, X, weights, blocks, "a")

    assert "a" not in reduced.variances
    # Planted sigma_a = 1.3 dwarfs sigma_b = 0.7, so dropping it must cost real
    # likelihood rather than land on the boundary.
    assert reduced.neg2_reml - full.neg2_reml > 3.84


def test_reml_recovers_a_planted_variance_component():
    rng = np.random.default_rng(5)
    n_groups, per_group = 60, 25
    codes = np.repeat(np.arange(n_groups), per_group)
    n = codes.size
    X = np.ones((n, 1))
    weights = np.ones(n)
    y = 4.0 + np.repeat(rng.normal(0.0, 2.0, n_groups), per_group) + rng.normal(0.0, 1.0, n)

    fit = REMLProblem(y, X, weights, {"g": codes}).fit()

    assert fit.beta[0] == pytest.approx(4.0, abs=0.6)
    assert fit.variances["g"] == pytest.approx(4.0, rel=0.35)
    assert fit.resid_variance == pytest.approx(1.0, rel=0.1)


def test_permutation_test_separates_a_planted_group_effect_from_noise():
    rng = np.random.default_rng(3)
    n_groups, per_group = 40, 30
    codes = np.repeat(np.arange(n_groups), per_group)
    n = codes.size
    X = np.ones((n, 1))
    weights = np.ones(n)
    noise = rng.normal(0.0, 1.0, n)

    flat = residualize(4.0 + noise, X, weights)
    signal = residualize(
        4.0 + np.repeat(rng.normal(0.0, 1.5, n_groups), per_group) + noise, X, weights
    )

    flat_result = permutation_test(flat, weights, codes, None, 400, np.random.default_rng(7))
    signal_result = permutation_test(signal, weights, codes, None, 400, np.random.default_rng(7))

    assert signal_result["p_value"] < 0.01
    assert flat_result["p_value"] > 0.05
    assert signal_result["statistic"] == pytest.approx(
        between_group_statistic(signal, weights, codes, n_groups)
    )
    # Taking the null from the permutation is what makes the moment estimate
    # usable on data whose fixed effects were already projected out.
    assert moment_variance_from_permutation(signal_result, weights, codes) == pytest.approx(
        1.5**2, rel=0.4
    )


def test_permutation_test_is_reproducible_for_a_fixed_seed():
    y, X, weights, blocks = _design()
    resid = residualize(y, X, weights)

    first = permutation_test(resid, weights, blocks["a"], None, 200, np.random.default_rng(42))
    second = permutation_test(resid, weights, blocks["a"], None, 200, np.random.default_rng(42))

    assert first == second
