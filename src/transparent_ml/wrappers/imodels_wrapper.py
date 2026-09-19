import numpy as np
from transparent_ml.core.base import BaseInterpretableModel
from sklearn.base import clone

class IModelsWrapper(BaseInterpretableModel):
    """
    Wrapper for algorithms from the `imodels` package.
    Ensures that the models conform to the unified transparent-ml interface,
    specifically the `export_rules` method.
    """
    
    def __init__(self, base_estimator):
        """
        Parameters
        ----------
        base_estimator : estimator object
            An instance of an imodels classifier (e.g., RuleFitClassifier, SkopeRulesClassifier).
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
        Extracts rules from the underlying imodels estimator.
        Many imodels classifiers implement a `rules_` attribute or a `__str__` method
        that prints the rules. We use `__str__` as a fallback.
        """
        if self.estimator_ is None:
            return "No rules induced (model not fitted)."
            
        # imodels estimators usually have a nice string representation
        # Or we can extract from .rules_ if available
        if hasattr(self.estimator_, 'rules_'):
            rules_str = [str(rule) for rule in self.estimator_.rules_]
            return "\n".join(rules_str)
        else:
            return str(self.estimator_)
