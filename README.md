# TransparentML (`transparent-ml`)

**TransparentML** is a unified Python toolkit for interpretable, rule-based machine learning. It provides native, vectorized implementations of classic and fuzzy rule learners alongside standardized wrappers for the broader interpretable ML ecosystem.

---

## Implemented Algorithms & Interpretable Models

Every model inherits from `BaseInterpretableModel`, providing full **scikit-learn** compatibility (`fit`, `predict`, `predict_proba`) and a standardized `export_rules() -> str` interface for human inspection:

- **`PrismClassifier`**: Modular rule induction algorithm (Cendrowska, 1987) using vectorized candidate evaluation, quantile continuous split generation, and deterministic tie-breaking.
- **`PartClassifier`**: Projective Adaptive Resonance Theory rule induction based on partial C4.5 decision tree generation (Frank & Witten, 1998).
- **`FuriaClassifier`**: Fuzzy Unordered Rule Induction Algorithm with trapezoidal and triangular fuzzy intervals (Hühn & Hüllermeier, 2009).
- **`FarcHDClassifier`**: Fuzzy Association Rule-based Classification for High-Dimensional domains with Apriori frequent itemset generation (Alcalá-Fdez et al., 2011).
- **`IModelsWrapper`**: Standardized adapter for rule estimators from the `imodels` library.
- **`ScoredRuleSetsWrapper`**: Unified adapter interfacing `scoredrulesets` estimators.

---

## Quickstart

### PRISM Rule Classifier

```python
from sklearn.datasets import load_breast_cancer
from transparent_ml.algorithms.prism import PrismClassifier

X, y = load_breast_cancer(return_X_y=True)

# Fit PRISM modular rule learner
clf = PrismClassifier(max_rules=30, random_state=42)
clf.fit(X, y)

print(f"Accuracy: {clf.score(X, y):.3f}")
print(clf.export_rules())
```

Example exported rules:
```text
--- Induced PRISM Rules ---
IF feature_27 <= 0.1165 AND feature_22 <= 114.3 THEN class = 1 (conf=1.00, n=330)
IF feature_20 > 16.82 AND feature_27 > 0.1333 THEN class = 0 (conf=1.00, n=140)
DEFAULT: class = 1
```

---

## Installation & Development

```bash
uv sync --extra dev
uv run pytest
```
