"""Rule and condition atom data structures for PRISM."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List
import numpy as np


@dataclass
class PrismAtom:
    """Canonical condition atom representing (feature op value)."""
    feature_idx: int
    op: str
    value: float
    feature_name: str | None = None

    def covers(self, X: np.ndarray) -> np.ndarray:
        """Return boolean mask of samples satisfying this atomic condition."""
        col = X[:, self.feature_idx]
        if self.op == "==":
            return col == self.value
        elif self.op == "<=":
            return col <= self.value
        elif self.op == ">":
            return col > self.value
        return np.ones(X.shape[0], dtype=bool)

    def fires_mask(self, X: np.ndarray) -> np.ndarray:
        """Alias for covers."""
        return self.covers(X)

    def __str__(self) -> str:
        name = self.feature_name if self.feature_name is not None else f"feature_{self.feature_idx}"
        return f"{name} {self.op} {self.value:.4g}"


@dataclass
class PrismRule:
    """Canonical rule representing an antecedent conjunction predicting a target class."""
    atoms: List[PrismAtom]
    predicted_class: Any
    class_idx: int
    rule_id: int = 0
    support: int = 0
    confidence: float = 1.0

    def covers(self, X: np.ndarray) -> np.ndarray:
        """Return boolean mask of samples satisfying all atoms in this rule."""
        if not self.atoms:
            return np.ones(X.shape[0], dtype=bool)
        mask = np.ones(X.shape[0], dtype=bool)
        for atom in self.atoms:
            mask &= atom.covers(X)
        return mask

    def fires_mask(self, X: np.ndarray) -> np.ndarray:
        """Alias for covers."""
        return self.covers(X)

    def __str__(self) -> str:
        if not self.atoms:
            return f"IF TRUE THEN class = {self.predicted_class}"
        conds = " AND ".join(str(a) for a in self.atoms)
        return f"IF {conds} THEN class = {self.predicted_class} (conf={self.confidence:.2f}, n={self.support})"
