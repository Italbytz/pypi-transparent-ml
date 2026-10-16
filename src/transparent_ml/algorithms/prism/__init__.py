"""PRISM (modular rule induction) algorithm package."""

from transparent_ml.algorithms.prism.rule import PrismAtom, PrismRule
from transparent_ml.algorithms.prism.estimator import PrismClassifier

__all__ = [
    "PrismAtom",
    "PrismRule",
    "PrismClassifier",
]
