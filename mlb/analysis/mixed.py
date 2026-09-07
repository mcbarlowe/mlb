"""Variance-component REML with heteroskedastic known weights.

Fits ``y = X beta + Z a + e`` where ``a`` collects one or more crossed
grouping factors, ``a_g ~ N(0, sigma^2_g)``, and ``Var(e_i) = sigma^2_e / w_i``
for known per-observation weights ``w_i`` (a rate observed over ``w_i``
trials has exactly this variance structure).

The likelihood is evaluated through the Woodbury and matrix-determinant
identities on the ``q x q`` random-effect space (``q`` = total group levels)
rather than the ``n x n`` observation space, and every term that does not
depend on the variance parameters is precomputed once. That makes one
likelihood evaluation a single ``q x q`` Cholesky, which is what allows
profile-likelihood intervals and thousands of permutation replicates.

Verified against ``statsmodels.regression.mixed_linear_model.MixedLM``
(identical -2 REML, fixed effects, and variance components at an interior
optimum with unit weights) and against the dense ``n x n`` likelihood with
heteroskedastic weights (agreement to 1e-12).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, sparse
from scipy.linalg import cho_factor, cho_solve

__all__ = [
    "MixedFit",
    "REMLProblem",
    "between_group_statistic",
    "fit_without_block",
    "moment_variance_estimate",
    "moment_variance_from_permutation",
    "permutation_test",
    "profile_ci_lower",
    "profile_ci_upper",
    "residualize",
]


@dataclass
class MixedFit:
    names: list[str]
    variances: dict[str, float]
    resid_variance: float
    beta: np.ndarray
    neg2_reml: float
    converged: bool
    blups: dict[str, np.ndarray] = field(default_factory=dict)
    blup_sd: dict[str, np.ndarray] = field(default_factory=dict)


class REMLProblem:
    def __init__(
        self,
        y: np.ndarray,
        X: np.ndarray,
        weights: np.ndarray,
        blocks: dict[str, np.ndarray],
    ) -> None:
        self.y = y
        self.X = X
        self.w = weights
        self.names = list(blocks)
        self.codes = blocks
        self.sizes = [int(blocks[name].max()) + 1 for name in self.names]
        self.n, self.p = X.shape
        self.q = int(sum(self.sizes))
        self.block_slices: dict[str, slice] = {}
        offset = 0
        columns = np.empty(self.n * len(self.names), dtype=np.int64)
        rows = np.empty_like(columns)
        for i, name in enumerate(self.names):
            self.block_slices[name] = slice(offset, offset + self.sizes[i])
            columns[i * self.n : (i + 1) * self.n] = blocks[name] + offset
            rows[i * self.n : (i + 1) * self.n] = np.arange(self.n)
            offset += self.sizes[i]
        Z = sparse.csr_matrix((np.ones(rows.size), (rows, columns)), shape=(self.n, self.q))
        wy = weights * y
        wX = weights[:, None] * X
        self.ytDy1 = float(y @ wy)
        self.XtDy1 = X.T @ wy
        self.XtDX1 = X.T @ wX
        self.ZtDy1 = np.asarray(Z.T @ wy).ravel()
        self.ZtDX1 = np.asarray(Z.T @ wX)
        self.M = np.asarray((Z.T @ sparse.diags(weights) @ Z).todense())
        self.log_w_sum = float(np.sum(np.log(weights)))
        self.Z = Z
        self._rep = np.concatenate([np.full(size, i) for i, size in enumerate(self.sizes)])

    def _expand(self, variances: np.ndarray) -> np.ndarray:
        return variances[self._rep]

    def evaluate(
        self, variances: np.ndarray, resid_variance: float
    ) -> tuple[float, np.ndarray, dict]:
        """Return (-2 REML, beta, workspace) at the given variance parameters."""

        ve = resid_variance
        s = self._expand(variances)
        if np.any(s <= 0) or ve <= 0:
            return np.inf, np.zeros(self.p), {}
        A = self.M / ve
        A[np.diag_indices(self.q)] += 1.0 / s
        try:
            chol = cho_factor(A, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            return np.inf, np.zeros(self.p), {}
        logdet_A = 2.0 * float(np.sum(np.log(np.diag(chol[0]))))
        logdet_D = self.n * np.log(ve) - self.log_w_sum
        logdet_S = float(np.sum(np.log(s)))
        B = self.M / ve
        C = self.ZtDX1 / ve
        d = self.ZtDy1 / ve
        Ai_C = cho_solve(chol, C, check_finite=False)
        Ai_d = cho_solve(chol, d, check_finite=False)
        XtViX = self.XtDX1 / ve - C.T @ Ai_C
        XtViy = self.XtDy1 / ve - C.T @ Ai_d
        ytViy = self.ytDy1 / ve - float(d @ Ai_d)
        sign, logdet_XtViX = np.linalg.slogdet(XtViX)
        if sign <= 0:
            return np.inf, np.zeros(self.p), {}
        beta = np.linalg.solve(XtViX, XtViy)
        quad = ytViy - float(beta @ XtViy)
        value = (
            logdet_D
            + logdet_S
            + logdet_A
            + logdet_XtViX
            + quad
            + (self.n - self.p) * np.log(2.0 * np.pi)
        )
        return float(value), beta, {"chol": chol, "B": B, "C": C, "XtViX": XtViX, "s": s}

    def fit(self, pinned: dict[str, float] | None = None) -> MixedFit:
        """REML fit. ``pinned`` fixes a component's variance (profile likelihood)."""

        pinned = pinned or {}
        free = [name for name in self.names if name not in pinned]

        def unpack(theta: np.ndarray) -> tuple[np.ndarray, float]:
            variances = np.empty(len(self.names))
            values = np.exp(theta)
            for i, name in enumerate(self.names):
                variances[i] = pinned[name] if name in pinned else values[free.index(name)]
            return variances, float(values[-1])

        # Initialize the residual scale from weighted OLS and start every random
        # effect at 1% of it; a flat start is orders of magnitude off and costs
        # Nelder-Mead hundreds of iterations. Search bounds are relative to that
        # scale: a fixed absolute ceiling silently binds for responses whose
        # natural variance is not O(1).
        sw = np.sqrt(self.w)
        coef, *_ = np.linalg.lstsq(self.X * sw[:, None], self.y * sw, rcond=None)
        resid0 = self.y - self.X @ coef
        ve0 = max(float(np.sum(self.w * resid0**2) / max(self.n - self.p, 1)), 1e-12)
        lo_clip = np.log(ve0) - 32.0
        hi_clip = np.log(ve0) + 12.0

        def objective(theta: np.ndarray) -> float:
            variances, ve = unpack(np.clip(theta, lo_clip, hi_clip))
            return self.evaluate(variances, ve)[0]

        theta0 = np.concatenate([np.full(len(free), np.log(0.01 * ve0)), [np.log(ve0)]])
        result = optimize.minimize(
            objective,
            theta0,
            method="Nelder-Mead",
            options={"maxiter": 5000, "maxfev": 5000, "xatol": 1e-8, "fatol": 1e-10},
        )
        variances, ve = unpack(np.clip(result.x, lo_clip, hi_clip))
        value, beta, workspace = self.evaluate(variances, ve)
        fit = MixedFit(
            names=list(self.names),
            variances=dict(zip(self.names, variances)),
            resid_variance=ve,
            beta=beta,
            neg2_reml=value,
            converged=bool(result.success),
        )
        self._attach_blups(fit, beta, workspace)
        return fit

    def _attach_blups(self, fit: MixedFit, beta: np.ndarray, workspace: dict) -> None:
        if not workspace:
            return
        chol, B, C, XtViX = workspace["chol"], workspace["B"], workspace["C"], workspace["XtViX"]
        s = workspace["s"]
        resid = self.y - self.X @ beta
        ZtDr = np.asarray(self.Z.T @ (self.w * resid)).ravel() / fit.resid_variance
        ZtVir = ZtDr - B @ cho_solve(chol, ZtDr, check_finite=False)
        blup = s * ZtVir
        # Var(a_hat - a) = S - S Z'PZ S, P = V^-1 - V^-1 X (X'V^-1 X)^-1 X'V^-1
        ZtViZ = B - B @ cho_solve(chol, B, check_finite=False)
        ZtViX = C - B @ cho_solve(chol, C, check_finite=False)
        ZtPZ = ZtViZ - ZtViX @ np.linalg.solve(XtViX, ZtViX.T)
        post_var = np.maximum(s - s * np.diag(ZtPZ) * s, 0.0)
        for name in self.names:
            sl = self.block_slices[name]
            fit.blups[name] = blup[sl]
            fit.blup_sd[name] = np.sqrt(post_var[sl])


def fit_without_block(
    y: np.ndarray,
    X: np.ndarray,
    weights: np.ndarray,
    blocks: dict[str, np.ndarray],
    drop: str,
) -> MixedFit:
    """REML fit with one variance component removed (the sigma^2 = 0 null)."""

    reduced = {name: codes for name, codes in blocks.items() if name != drop}
    if not reduced:
        # Only the residual term remains: weighted OLS is the REML fit.
        sw = np.sqrt(weights)
        coef, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
        resid = y - X @ coef
        n, p = X.shape
        ve = float(np.sum(weights * resid**2) / (n - p))
        XtDX = X.T @ (weights[:, None] * X)
        _, logdet_XtDX = np.linalg.slogdet(XtDX / ve)
        value = (
            n * np.log(ve)
            - float(np.sum(np.log(weights)))
            + logdet_XtDX
            + float(np.sum(weights * resid**2) / ve)
            + (n - p) * np.log(2.0 * np.pi)
        )
        return MixedFit([], {}, ve, coef, value, True)
    return REMLProblem(y, X, weights, reduced).fit()


def profile_ci_upper(
    problem: REMLProblem, base: MixedFit, target: str, deviance_threshold: float
) -> float:
    """Upper end of the profile-likelihood interval for ``sigma^2_target``."""

    point = base.variances[target]
    hi = max(point, 1e-10) * 4.0
    for _ in range(30):
        if problem.fit(pinned={target: hi}).neg2_reml > deviance_threshold:
            break
        hi *= 4.0
    else:
        return float("nan")
    lo = point
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        if problem.fit(pinned={target: mid}).neg2_reml > deviance_threshold:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def profile_ci_lower(
    problem: REMLProblem,
    y: np.ndarray,
    X: np.ndarray,
    weights: np.ndarray,
    blocks: dict[str, np.ndarray],
    base: MixedFit,
    target: str,
    deviance_threshold: float,
) -> float:
    """Lower end; returns 0 when ``sigma^2 = 0`` lies inside the interval."""

    if fit_without_block(y, X, weights, blocks, target).neg2_reml <= deviance_threshold:
        return 0.0
    lo, hi = 0.0, base.variances[target]
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        if problem.fit(pinned={target: mid}).neg2_reml > deviance_threshold:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def residualize(y: np.ndarray, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted least-squares residuals."""

    sw = np.sqrt(weights)
    coef, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    return y - X @ coef


def between_group_statistic(
    resid: np.ndarray, weights: np.ndarray, codes: np.ndarray, n_levels: int
) -> float:
    """Weighted between-group sum of squares of the residuals."""

    w_sum = np.bincount(codes, weights=weights, minlength=n_levels)
    wy_sum = np.bincount(codes, weights=weights * resid, minlength=n_levels)
    grand = wy_sum.sum() / w_sum.sum()
    means = np.divide(wy_sum, w_sum, out=np.zeros_like(wy_sum), where=w_sum > 0)
    return float(np.sum(w_sum * (means - grand) ** 2))


def moment_variance_estimate(
    resid: np.ndarray,
    weights: np.ndarray,
    codes: np.ndarray,
    resid_variance: float,
) -> float:
    """Weighted one-way ANOVA estimate of the group variance on RAW data.

    Unlike REML this may come out negative, which is the information a boundary
    estimate of exactly 0 hides: negative means the observed between-group
    spread is *smaller* than the sampling-noise expectation.

    ``E[SSB] = (M - 1) sigma^2_e + sigma^2_g (sum_g W_g - sum_g W_g^2 / sum W)``
    with ``W_g`` the total weight of group ``g``.

    WARNING: the ``(M - 1) sigma^2_e`` null term assumes ``resid`` is the raw
    response, not the residual of a projection that absorbed other factors.
    Absorbing e.g. franchise and season removes between-group variance the term
    does not account for, which biases this estimate DOWNWARD — measured at
    99% negative on simulated null data for the 2009-2025 design. For
    residualized input use :func:`moment_variance_from_permutation`.
    """

    n_levels = int(codes.max()) + 1
    ssb = between_group_statistic(resid, weights, codes, n_levels)
    w_sum = np.bincount(codes, weights=weights, minlength=n_levels)
    total = w_sum.sum()
    n_groups = int(np.sum(w_sum > 0))
    denominator = total - float(np.sum(w_sum**2)) / total
    if denominator <= 0:
        return float("nan")
    return float((ssb - (n_groups - 1) * resid_variance) / denominator)


def moment_variance_from_permutation(
    permutation: dict[str, float], weights: np.ndarray, codes: np.ndarray
) -> float:
    """Group-variance moment estimate that is valid on residualized data.

    Takes ``E[SSB | sigma^2_g = 0]`` from the permutation null mean instead of
    an analytic degrees-of-freedom term. The permutation reshuffles labels
    through the same projection that produced ``resid``, so whatever variance
    the projection removed is removed from the null in the same way. May be
    negative, with the same meaning as in :func:`moment_variance_estimate`.
    """

    n_levels = int(codes.max()) + 1
    w_sum = np.bincount(codes, weights=weights, minlength=n_levels)
    total = w_sum.sum()
    denominator = total - float(np.sum(w_sum**2)) / total
    if denominator <= 0:
        return float("nan")
    return float((permutation["statistic"] - permutation["null_mean"]) / denominator)


def permutation_test(
    resid: np.ndarray,
    weights: np.ndarray,
    codes: np.ndarray,
    blocks: np.ndarray | None,
    n_permutations: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Permutation test on the weighted between-group sum of squares.

    ``blocks`` restricts shuffling to within-block, which preserves a
    confounded structure (e.g. franchise) under the null.
    """

    n_levels = int(codes.max()) + 1
    observed = between_group_statistic(resid, weights, codes, n_levels)
    null = np.empty(n_permutations)
    if blocks is None:
        shuffled = codes.copy()
        for i in range(n_permutations):
            rng.shuffle(shuffled)
            null[i] = between_group_statistic(resid, weights, shuffled, n_levels)
    else:
        block_index = [
            idx for idx in (np.flatnonzero(blocks == b) for b in np.unique(blocks)) if idx.size > 1
        ]
        for i in range(n_permutations):
            shuffled = codes.copy()
            for idx in block_index:
                shuffled[idx] = rng.permutation(shuffled[idx])
            null[i] = between_group_statistic(resid, weights, shuffled, n_levels)
    return {
        "statistic": observed,
        "null_mean": float(null.mean()),
        "null_p95": float(np.quantile(null, 0.95)),
        "p_value": float((np.sum(null >= observed) + 1) / (n_permutations + 1)),
    }
