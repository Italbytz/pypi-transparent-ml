# TransparentML (`transparent-iml`)

**TransparentML** is a unified Python ecosystem for interpretable, rule-based machine learning. It provides native, vectorized implementations of classic and fuzzy rule learners alongside standardized wrappers for the broader interpretable ML ecosystem.

- **PyPI package:** [`transparent-iml`](https://pypi.org/project/transparent-iml/) (`import transparent_ml`)
- **Requires:** Python $\ge 3.14$

---

## Installation

Install the official release from PyPI:

```bash
pip install transparent-iml
```

Or with `uv`:

```bash
uv add transparent-iml
```

---

## Implemented Algorithms & Interpretable Models

Every model inherits from `BaseInterpretableModel`, providing full **scikit-learn** compatibility (`fit`, `predict`, `predict_proba`) and a standardized `export_rules() -> str` interface for human inspection:

- **`FuriaClassifier`**: Fuzzy Unordered Rule Induction Algorithm with trapezoidal and triangular fuzzy intervals (Hühn & Hüllermeier, 2009).
- **`PartClassifier`**: Partial Decision Trees (PART) rule induction based on partial C4.5 decision tree generation (Frank & Witten, 1998).
- **`FarcHDClassifier`**: Fuzzy Association Rule-based Classification for High-Dimensional domains with Apriori frequent itemset generation (Alcalá-Fdez et al., 2011).
- **`PrismClassifier`**: Modular rule induction algorithm (Cendrowska, 1987) using vectorized candidate evaluation, quantile continuous split generation, and deterministic tie-breaking.
- **`IModelsWrapper`**: Standardized adapter for rule estimators from the `imodels` library.
- **`ScoredRuleSetsWrapper`**: Unified adapter interfacing `scoredrulesets` estimators.

---

## Quickstart

After installing `transparent-iml`, import the learners from `transparent_ml`:

```python
from sklearn.datasets import load_breast_cancer
from transparent_ml import FuriaClassifier

X, y = load_breast_cancer(return_X_y=True)

# Fit natively translated FURIA fuzzy rule learner
clf = FuriaClassifier(seed=42)
clf.fit(X, y)

print(f"Accuracy: {clf.score(X, y):.3f}")
print(clf.export_rules())
```

Example exported rules:
```text
IF feature_27 >= 0.1418 (fuzzy drop-off to 0.1414) AND feature_20 >= 16.31 (fuzzy drop-off to 16.25) THEN class = 0 (weight: 1.00)
IF feature_27 >= 0.1099 (fuzzy drop-off to 0.1096) THEN class = 0 (weight: 1.00)
...
```

---

## Development

```bash
uv sync --extra dev
uv run pytest
```
