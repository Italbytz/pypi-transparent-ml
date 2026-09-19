from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

@dataclass
class PartCondition:
    feature_idx: int
    operator: str  # '<=', '>', '=='
    value: float
    feature_name: str = None
    
    def __str__(self):
        fname = self.feature_name if self.feature_name else f"feature_{self.feature_idx}"
        return f"{fname} {self.operator} {self.value}"
        
    def matches(self, X: np.ndarray) -> np.ndarray:
        if self.operator == '<=':
            return X[:, self.feature_idx] <= self.value
        elif self.operator == '>':
            return X[:, self.feature_idx] > self.value
        elif self.operator == '==':
            return X[:, self.feature_idx] == self.value
        else:
            raise ValueError(f"Unknown operator {self.operator}")

class PartRule:
    def __init__(self, conditions: List[PartCondition], predicted_class: int, weight: float = 1.0):
        self.conditions = conditions
        self.predicted_class = predicted_class
        self.weight = weight
        
    def covers(self, X: np.ndarray) -> np.ndarray:
        """Return boolean mask of instances covered by this rule."""
        if not self.conditions:
            return np.ones(X.shape[0], dtype=bool)
        
        mask = np.ones(X.shape[0], dtype=bool)
        for cond in self.conditions:
            mask &= cond.matches(X)
        return mask
        
    def __str__(self):
        if not self.conditions:
            return f"DEFAULT THEN class = {self.predicted_class} (weight: {self.weight:.2f})"
            
        cond_strs = [str(c) for c in self.conditions]
        return f"IF {' AND '.join(cond_strs)} THEN class = {self.predicted_class} (weight: {self.weight:.2f})"
