import numpy as np
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

from transparent_ml.core.base import BaseInterpretableModel
from transparent_ml.algorithms.part.tree import PartialTreeBuilder
from transparent_ml.algorithms.part.rule import PartRule

class PartClassifier(BaseInterpretableModel):
    """
    Projective Adaptive Resonance Theory (PART) Rule Learner.
    Builds a list of rules by repeatedly generating partial C4.5 decision trees
    and extracting their best leaf.
    """
    
    def __init__(self, min_samples_leaf=2, cf=0.25):
        self.min_samples_leaf = min_samples_leaf
        self.cf = cf
        
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.rules_ = []
        
        # We process until all instances are covered
        remaining_mask = np.ones(X.shape[0], dtype=bool)
        
        while np.any(remaining_mask):
            builder = PartialTreeBuilder(
                X, y, 
                min_samples_leaf=self.min_samples_leaf, 
                cf=self.cf
            )
            
            # 1. Build partial tree and extract best rule
            best_rule = builder.build_and_extract_best_rule(remaining_mask)
            
            # 2. Add rule
            self.rules_.append(best_rule)
            
            # 3. Remove covered instances
            covered_mask = best_rule.covers(X)
            
            # Note: We must only consider instances currently in remaining_mask
            newly_covered = covered_mask & remaining_mask
            
            if not np.any(newly_covered):
                # Failsafe: if the best rule covers no *new* instances, stop to prevent infinite loops.
                # Just add a default rule for the remaining instances and break.
                majority_class = builder._get_majority_class(y[remaining_mask])
                self.rules_.append(PartRule(conditions=[], predicted_class=majority_class, weight=np.sum(remaining_mask)))
                break
                
            remaining_mask &= ~covered_mask
            
        return self
        
    def predict(self, X):
        check_is_fitted(self, "rules_")
        X = check_array(X)
        
        predictions = np.zeros(X.shape[0], dtype=int)
        uncovered_mask = np.ones(X.shape[0], dtype=bool)
        
        # Predict uses a decision list approach (first match wins)
        for rule in self.rules_:
            if not np.any(uncovered_mask):
                break
                
            rule_mask = rule.covers(X)
            new_matches = rule_mask & uncovered_mask
            
            if np.any(new_matches):
                predictions[new_matches] = rule.predicted_class
                uncovered_mask &= ~new_matches
                
        # If any remain uncovered, assign the class of the last rule (which usually acts as a default)
        if np.any(uncovered_mask) and len(self.rules_) > 0:
            predictions[uncovered_mask] = self.rules_[-1].predicted_class
            
        return predictions
        
    def predict_proba(self, X):
        check_is_fitted(self, "rules_")
        X = check_array(X)
        preds = self.predict(X)
        
        # Simple one-hot for rules.
        probas = np.zeros((X.shape[0], len(self.classes_)))
        for i, c in enumerate(self.classes_):
            probas[:, i] = (preds == c).astype(float)
        return probas
        
    def export_rules(self) -> str:
        check_is_fitted(self, "rules_")
        lines = ["--- Induced PART Rules ---"]
        for rule in self.rules_:
            lines.append(str(rule))
        return "\n".join(lines)
