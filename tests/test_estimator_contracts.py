"""Tier 2 verification: estimator and schema contracts across integrated models."""

import numpy as np
import pytest
from imodels import RuleFitClassifier
from scoredrulesets import RuleGPClassifier
from sklearn.datasets import make_classification

from transparent_ml import (
    BaseInterpretableModel,
    FarcHDClassifier,
    FuriaClassifier,
    IModelsWrapper,
    PartClassifier,
    PrismClassifier,
    ScoredRuleSetsWrapper,
)


@pytest.fixture(scope="module")
def binary_data():
    X, y = make_classification(
        n_samples=120,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=42,
    )
    return X, y


def _make_estimators():
    return [
        ("prism", PrismClassifier(max_rules=10, random_state=42), True),
        ("furia", FuriaClassifier(seed=42), True),
        ("part", PartClassifier(min_samples_leaf=5), True),
        ("farc_hd", FarcHDClassifier(max_depth=1, pop_size=5, max_generations=2), True),
        ("imodels_rulefit", IModelsWrapper(RuleFitClassifier(max_rules=8, random_state=42)), True),
        (
            "scoredrulesets_rulegp",
            ScoredRuleSetsWrapper(
                RuleGPClassifier(population_size=15, max_generations=10, max_rules=4, random_state=42)
            ),
            True,
        ),
    ]


@pytest.mark.parametrize("name,est,has_proba", _make_estimators(), ids=lambda x: x if isinstance(x, str) else "")
def test_estimator_and_export_contracts(name, est, has_proba, binary_data):
    X, y = binary_data
    assert isinstance(est, BaseInterpretableModel)

    fitted = est.fit(X, y)
    assert fitted is est

    preds = est.predict(X)
    assert preds.shape == (len(X),)
    assert set(np.unique(preds)).issubset(set(np.unique(y)))

    rules = est.export_rules()
    assert rules is not None and len(rules) > 0

    if has_proba:
        proba = est.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.all(proba >= -1e-9) and np.all(proba <= 1.0 + 1e-9)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_deterministic_seeding_for_seeded_learners(binary_data):
    X, y = binary_data
    for cls, kwargs in [
        (PrismClassifier, {"max_rules": 10, "random_state": 7}),
        (FuriaClassifier, {"seed": 7}),
    ]:
        m1 = cls(**kwargs).fit(X, y)
        m2 = cls(**kwargs).fit(X, y)
        np.testing.assert_array_equal(m1.predict(X), m2.predict(X))
        assert m1.export_rules() == m2.export_rules()
