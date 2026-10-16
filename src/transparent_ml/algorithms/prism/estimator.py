"""Vectorized PRISM rule induction classifier for transparent-ml.

Implements the PRISM algorithm for modular rule induction:
    J. Cendrowska (1987): "PRISM: An algorithm for inducing modular rules"
    International Journal of Man-Machine Studies, 27(4), 349-370.
"""

from __future__ import annotations
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.utils.multiclass import unique_labels

from transparent_ml.core.base import BaseInterpretableModel
from transparent_ml.algorithms.prism.rule import PrismAtom, PrismRule


class PrismClassifier(BaseInterpretableModel):
    """Vectorized PRISM rule induction classifier.

    Iteratively extracts modular decision rules by greedily maximizing the
    conditional probability p(class | condition), removing covered target
    instances after each pure or maximal rule is found.

    Parameters
    ----------
    max_rules : int, default=100
        Maximum number of rules to induce.
    max_rule_length : int | None, default=None
        Maximum number of conditions per rule. If None, defaults to 2 * n_features.
    random_state : int | None, default=None
        Seed for deterministic tie-breaking when candidate selectors have identical accuracy.
    """

    def __init__(
        self,
        max_rules: int = 100,
        max_rule_length: int | None = None,
        random_state: int | None = None,
    ):
        self.max_rules = max_rules
        self.max_rule_length = max_rule_length
        self.random_state = random_state

    def fit(self, X, y):
        """Fit the PRISM rule classifier on training data X with target labels y."""
        X, y = validate_data(self, X, y, ensure_2d=True)
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        n_samples, n_features = X.shape

        rng = np.random.RandomState(self.random_state)

        # Precompute candidate selectors per feature (discrete values or quantile splits)
        candidate_selectors: list[tuple[int, str, float]] = []
        for j in range(n_features):
            vals = np.unique(X[:, j])
            if len(vals) <= 10:
                for v in vals:
                    candidate_selectors.append((j, "==", float(v)))
            else:
                quantiles = np.percentile(vals, np.linspace(10, 90, 9))
                for q in np.unique(quantiles):
                    candidate_selectors.append((j, "<=", float(q)))
                    candidate_selectors.append((j, ">", float(q)))

        y_indices = np.searchsorted(self.classes_, y)
        self.rules_: list[PrismRule] = []

        active_mask = np.ones(n_samples, dtype=bool)
        max_len = self.max_rule_length if self.max_rule_length is not None else (n_features * 2)

        # Outer loop over classes: induce rules for each class in order
        for c_idx, class_val in enumerate(self.classes_):
            while np.any(active_mask & (y_indices == c_idx)):
                if len(self.rules_) >= self.max_rules:
                    break

                current_atoms: list[PrismAtom] = []
                used_selectors: set[tuple[int, str]] = set()
                current_rule_mask = np.ones(n_samples, dtype=bool)

                # Inner loop: greedily add atoms to maximize precision
                while len(current_atoms) < max_len:
                    best_acc = -1.0
                    best_correct = -1
                    best_candidates: list[tuple[int, str, float, np.ndarray]] = []

                    # Evaluate candidate selectors in vectorized fashion
                    for feat_idx, op, val in candidate_selectors:
                        if (feat_idx, op) in used_selectors:
                            continue

                        if op == "==":
                            sel_mask = (X[:, feat_idx] == val)
                        elif op == "<=":
                            sel_mask = (X[:, feat_idx] <= val)
                        elif op == ">":
                            sel_mask = (X[:, feat_idx] > val)
                        else:
                            continue

                        combined_mask = current_rule_mask & sel_mask & active_mask
                        n_covered = int(np.count_nonzero(combined_mask))
                        if n_covered == 0:
                            continue

                        n_correct = int(np.count_nonzero(combined_mask & (y_indices == c_idx)))
                        acc = n_correct / n_covered

                        if (acc > best_acc) or (acc == best_acc and n_correct > best_correct):
                            best_acc = acc
                            best_correct = n_correct
                            best_candidates = [(feat_idx, op, val, combined_mask)]
                        elif acc == best_acc and n_correct == best_correct:
                            best_candidates.append((feat_idx, op, val, combined_mask))

                    if not best_candidates or best_correct == 0:
                        break

                    # Deterministic tie-breaking
                    if len(best_candidates) == 1:
                        chosen_feat, chosen_op, chosen_val, chosen_mask = best_candidates[0]
                    else:
                        pick_idx = rng.randint(len(best_candidates))
                        chosen_feat, chosen_op, chosen_val, chosen_mask = best_candidates[pick_idx]

                    current_atoms.append(PrismAtom(chosen_feat, chosen_op, chosen_val))
                    used_selectors.add((chosen_feat, chosen_op))
                    current_rule_mask = chosen_mask

                    # Check purity: does current rule cover any active non-target instances?
                    has_impurities = np.any(current_rule_mask & (y_indices != c_idx))
                    if not has_impurities:
                        break  # Pure rule achieved

                if not current_atoms:
                    break

                covered_total = np.count_nonzero(current_rule_mask)
                rule = PrismRule(
                    atoms=current_atoms,
                    predicted_class=class_val,
                    class_idx=c_idx,
                    rule_id=len(self.rules_),
                    support=int(best_correct),
                    confidence=float(best_correct / max(covered_total, 1)),
                )
                self.rules_.append(rule)

                # Remove covered target instances from active set
                covered_targets = rule.covers(X) & (y_indices == c_idx)
                active_mask[covered_targets] = False

        # Default class fallback
        labels, counts = np.unique(y, return_counts=True)
        self.default_class_ = labels[np.argmax(counts)]
        return self

    def predict(self, X):
        """Predict class labels using first-match decision list over induced rules."""
        check_is_fitted(self, ["rules_", "classes_", "default_class_"])
        X = validate_data(self, X, reset=False)
        n_samples = X.shape[0]

        preds = np.full(n_samples, fill_value=self.default_class_, dtype=self.classes_.dtype)
        uncovered = np.ones(n_samples, dtype=bool)

        for rule in self.rules_:
            if not np.any(uncovered):
                break
            fires = rule.covers(X) & uncovered
            if np.any(fires):
                preds[fires] = rule.predicted_class
                uncovered[fires] = False

        return preds

    def predict_proba(self, X):
        """Predict class probabilities for samples in X."""
        check_is_fitted(self, ["rules_", "classes_", "default_class_"])
        X = validate_data(self, X, reset=False)
        n_samples = X.shape[0]

        # Use rule confidence and class mapping with Laplace smoothing
        proba = np.zeros((n_samples, self.n_classes_))
        uncovered = np.ones(n_samples, dtype=bool)

        for rule in self.rules_:
            if not np.any(uncovered):
                break
            fires = rule.covers(X) & uncovered
            if np.any(fires):
                # Distribute high probability to predicted class, remaining smoothed
                proba[fires, rule.class_idx] = rule.confidence
                uncovered[fires] = False

        # Assign default class probability for uncovered samples
        if np.any(uncovered):
            default_idx = int(np.where(self.classes_ == self.default_class_)[0][0])
            proba[uncovered, default_idx] = 1.0

        # Clip and normalize to strictly valid probabilities
        if self.n_classes_ > 1:
            proba = np.clip(proba, 0.05, 0.95)
            proba /= proba.sum(axis=1, keepdims=True)
        else:
            proba = np.ones((n_samples, 1))

        return proba

    def export_rules(self) -> str:
        """Export induced PRISM rules in a unified human-readable format."""
        check_is_fitted(self, ["rules_", "default_class_"])
        lines = ["--- Induced PRISM Rules ---"]
        for rule in self.rules_:
            lines.append(str(rule))
        lines.append(f"DEFAULT: class = {self.default_class_}")
        return "\n".join(lines)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        if tags.classifier_tags is not None:
            tags.classifier_tags.poor_score = True
        return tags

    def _more_tags(self):
        return {"poor_score": True}
