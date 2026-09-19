from transparent_ml.core.base import BaseInterpretableModel
from sklearn.base import clone

class ScoredRuleSetsWrapper(BaseInterpretableModel):
    """
    Wrapper for algorithms from the `scoredrulesets` package.
    Ensures that the models conform to the unified transparent-ml interface,
    specifically the `export_rules` method.
    """
    
    def __init__(self, base_estimator):
        """
        Parameters
        ----------
        base_estimator : estimator object
            An instance of a scoredrulesets classifier (e.g., RuleGPEstimator, EvoSubgroup).
        """
        self.base_estimator = base_estimator
        self.estimator_ = None
        
    def fit(self, X, y, **kwargs):
        self.estimator_ = clone(self.base_estimator)
        self.estimator_.fit(X, y, **kwargs)
        return self
        
    def predict(self, X):
        if self.estimator_ is None:
            raise ValueError("Model is not fitted yet.")
        return self.estimator_.predict(X)
        
    def predict_proba(self, X):
        if self.estimator_ is None:
            raise ValueError("Model is not fitted yet.")
        return self.estimator_.predict_proba(X)
        
    def export_rules(self) -> str:
        """
        Extracts rules from the underlying scoredrulesets estimator.
        scoredrulesets provides a `to_ruleset()` method that returns a ScoredRuleSet object.
        """
        if self.estimator_ is None:
            return "No rules induced (model not fitted)."
            
        if hasattr(self.estimator_, 'to_ruleset'):
            ruleset = self.estimator_.to_ruleset()
            # Depending on the schema, __str__ or similar should exist
            return str(ruleset)
        else:
            return str(self.estimator_)
