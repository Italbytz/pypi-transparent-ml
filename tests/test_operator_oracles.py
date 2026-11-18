"""Tier 1 verification: deterministic operator oracles (< 1e-12 deviation)."""

import math
import numpy as np
from scipy.stats import norm

from transparent_ml.algorithms.furia.antecedent import NumericAntecedent
from transparent_ml.algorithms.furia.heuristics import foil_info_gain
from transparent_ml.algorithms.part.heuristics import (
    c45_add_errors,
    c45_estimated_errors_for_leaf,
    entropy,
    info_gain_ratio,
)

TOL = 1e-12


def test_furia_foil_info_gain_oracle():
    cases = [
        (10.0, 2.0, 20.0, 20.0),
        (0.0, 5.0, 10.0, 10.0),
        (15.0, 0.0, 15.0, 15.0),
        (7.5, 2.5, 30.0, 10.0),
    ]
    for p1, n1, p0, n0 in cases:
        expected = p1 * (
            math.log2((p1 + 1.0) / (p1 + n1 + 1.0))
            - math.log2((p0 + 1.0) / (p0 + n0 + 1.0))
        )
        actual = float(foil_info_gain(p1, n1, p0, n0))
        assert abs(actual - expected) < TOL


def test_furia_trapezoidal_membership_oracle():
    # direction=0: <= 4.0 with fuzzy support bound 8.0
    ant_le = NumericAntecedent(feature_idx=0, split_point=4.0, direction=0)
    ant_le.fuzzy_boundary = 8.0
    X = np.array([[2.0], [4.0], [5.0], [6.0], [7.0], [8.0], [10.0]])
    expected_le = np.array([1.0, 1.0, 0.75, 0.5, 0.25, 0.0, 0.0])
    actual_le = ant_le.covers(X)
    assert np.max(np.abs(actual_le - expected_le)) < TOL
    # Scalar check matches vectorized check
    for row, exp in zip(X, expected_le):
        assert abs(float(ant_le.covers(row)) - float(exp)) < TOL

    # direction=1: >= 10.0 with fuzzy support bound 6.0
    ant_ge = NumericAntecedent(feature_idx=0, split_point=10.0, direction=1)
    ant_ge.fuzzy_boundary = 6.0
    X2 = np.array([[4.0], [6.0], [7.0], [8.0], [9.0], [10.0], [12.0]])
    expected_ge = np.array([0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0])
    actual_ge = ant_ge.covers(X2)
    assert np.max(np.abs(actual_ge - expected_ge)) < TOL


def test_part_entropy_and_gain_ratio_oracle():
    y_pure = np.array([1, 1, 1, 1])
    assert abs(float(entropy(y_pure)) - 0.0) < TOL
    assert abs(float(entropy(np.array([]))) - 0.0) < TOL

    y_bal = np.array([0, 0, 1, 1])
    assert abs(float(entropy(y_bal)) - 1.0) < TOL

    y = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    mask_l = np.array([True, True, True, True, False, False, False, False])
    mask_r = ~mask_l
    ig, gr = info_gain_ratio(y, mask_l, mask_r)

    def _H(arr):
        _, c = np.unique(arr, return_counts=True)
        p = c / len(arr)
        return float(-np.sum(p * np.log2(p)))

    exp_ig = _H(y) - 0.5 * _H(y[mask_l]) - 0.5 * _H(y[mask_r])
    exp_gr = exp_ig / 1.0  # balanced 4/4 split -> split_info == 1.0
    assert abs(float(ig) - exp_ig) < TOL
    assert abs(float(gr) - exp_gr) < TOL

    # Degenerate split (empty side) returns (0.0, 0.0)
    ig_deg, gr_deg = info_gain_ratio(y, np.ones(len(y), dtype=bool), np.zeros(len(y), dtype=bool))
    assert ig_deg == 0.0 and gr_deg == 0.0


def test_part_c45_add_errors_oracle():
    # CF > 0.5 -> 0.0
    assert c45_add_errors(10, 2, CF=0.6) == 0.0

    # e == 0 branch: N * (1 - CF^(1/N))
    N, cf = 12.0, 0.25
    exp_zero = N * (1.0 - cf ** (1.0 / N))
    assert abs(float(c45_add_errors(N, 0.0, CF=cf)) - exp_zero) < TOL

    # e >= 1 Wilson upper bound branch
    N, e = 20.0, 3.0
    z = float(norm.ppf(1.0 - cf))
    f = (e + 0.5) / N
    r = (f + (z * z) / (2 * N) + z * math.sqrt((f / N) - (f * f / N) + (z * z / (4 * N * N)))) / (1 + (z * z) / N)
    exp_err = r * N - e
    assert abs(float(c45_add_errors(N, e, CF=cf)) - exp_err) < TOL
    assert abs(float(c45_estimated_errors_for_leaf(N, e, CF=cf)) - (e + exp_err)) < TOL
