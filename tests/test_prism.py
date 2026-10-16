"""Tests for PrismClassifier in transparent-ml."""

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.utils.estimator_checks import parametrize_with_checks

from transparent_ml.algorithms.prism import PrismClassifier, PrismRule, PrismAtom


def test_prism_atoms_and_rules():
    """Verify PrismAtom and PrismRule masks and string representation."""
    a1 = PrismAtom(feature_idx=0, op="<=", value=2.5)
    a2 = PrismAtom(feature_idx=1, op="==", value=1.0)

    X = np.array([
        [1.0, 1.0],
        [3.0, 1.0],
        [2.0, 0.0],
    ])

    assert np.array_equal(a1.covers(X), np.array([True, False, True]))
    assert np.array_equal(a2.covers(X), np.array([True, True, False]))

    rule = PrismRule(atoms=[a1, a2], predicted_class="A", class_idx=0, support=1, confidence=1.0)
    assert np.array_equal(rule.covers(X), np.array([True, False, False]))
    assert "class = A" in str(rule)


def test_prism_fit_predict_binary():
    """Test PrismClassifier on binary classification."""
    X, y = load_breast_cancer(return_X_y=True)
    clf = PrismClassifier(max_rules=30, random_state=42)
    clf.fit(X, y)

    preds = clf.predict(X)
    assert preds.shape == y.shape
    assert set(preds).issubset(set(clf.classes_))

    proba = clf.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    acc = clf.score(X, y)
    assert acc > 0.85

    rules_str = clf.export_rules()
    assert "--- Induced PRISM Rules ---" in rules_str
    assert "IF " in rules_str
    assert "DEFAULT: class =" in rules_str


def test_prism_fit_predict_multiclass():
    """Test PrismClassifier on multi-class classification."""
    X, y = load_iris(return_X_y=True)
    clf = PrismClassifier(max_rules=20, random_state=42)
    clf.fit(X, y)

    preds = clf.predict(X)
    assert preds.shape == y.shape
    proba = clf.predict_proba(X)
    assert proba.shape == (X.shape[0], 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    acc = clf.score(X, y)
    assert acc > 0.80


@parametrize_with_checks([PrismClassifier(max_rules=10, random_state=42)])
def test_sklearn_compliance(estimator, check):
    """Verify scikit-learn standard compliance contract suite."""
    check(estimator)
