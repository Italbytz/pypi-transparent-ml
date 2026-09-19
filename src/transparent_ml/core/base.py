from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, ClassifierMixin

class BaseInterpretableModel(BaseEstimator, ClassifierMixin, ABC):
    """
    Base class for all models in the transparent-ml ecosystem.
    Mandates the implementation of a standard Scikit-Learn API
    and a unified rule export mechanism.
    """
    
    @abstractmethod
    def fit(self, X, y):
        """Fit the model to the training data."""
        pass

    @abstractmethod
    def predict(self, X):
        """Predict class labels."""
        pass

    @abstractmethod
    def predict_proba(self, X):
        """Predict class probabilities."""
        pass

    @abstractmethod
    def export_rules(self) -> str:
        """
        Export the induced rules in a unified, human-readable format.
        This provides a standardized output for both native algorithms (like FURIA)
        and wrapped algorithms (like imodels).
        """
        pass
