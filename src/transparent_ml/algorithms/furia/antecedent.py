class Antecedent:
    """
    Base class for FURIA rule antecedents.
    Represents a single condition in a fuzzy rule (e.g. feature_X > 5.0).
    """
    def __init__(self, feature_idx, value):
        self.feature_idx = feature_idx
        self.value = value
        
        # Stats tracked during rule growth
        self.max_info_gain = 0.0
        self.accu_rate = 0.0
        self.coverage = 0.0

    def covers(self, x):
        """Returns the degree of coverage (0.0 to 1.0) for a given instance x."""
        raise NotImplementedError("Subclasses must implement covers()")

class NominalAntecedent(Antecedent):
    """
    Antecedent for nominal (categorical) features.
    """
    def __init__(self, feature_idx, value):
        super().__init__(feature_idx, value)
        
    def covers(self, x):
        # Nominal is hard boolean logic even in fuzzy rules
        return 1.0 if x[self.feature_idx] == self.value else 0.0
        
    def __str__(self):
        return f"feature_{self.feature_idx} == {self.value}"

class NumericAntecedent(Antecedent):
    """
    Antecedent for numeric features with fuzzy boundaries.
    """
    def __init__(self, feature_idx, split_point, direction):
        super().__init__(feature_idx, split_point)
        # direction is 0 for <=, 1 for >=
        self.direction = direction
        
        # Fuzzy boundaries (fuzzification parameters)
        self.fuzzy_boundary = None 
        
    def fuzzify(self, X, y, target_class):
        """
        Calculates the fuzzy boundary (support_bound) for this antecedent.
        Tests potential support bounds and selects the one that maximizes rule purity.
        """
        import numpy as np
        
        # We only consider values of the feature that are outside the crisp boundary
        X_feature = X[:, self.feature_idx]
        
        if self.direction == 0:
            # Crisp rule is <= split_point. Support bound must be > split_point
            candidates = np.unique(X_feature[X_feature > self.value])
            candidates = np.sort(candidates)
        else:
            # Crisp rule is >= split_point. Support bound must be < split_point
            candidates = np.unique(X_feature[X_feature < self.value])
            candidates = np.sort(candidates)[::-1] # descending
            
        best_purity = -1.0
        best_support_bound = self.value
        
        for candidate in candidates:
            # Temporarily set the fuzzy boundary
            self.fuzzy_boundary = candidate
            
            # Calculate coverage for all instances
            # Utilizing vectorized covering logic
            coverage = self.covers(X)
            
            # Calculate purity
            covered_mask = coverage > 0
            if not np.any(covered_mask):
                continue
                
            accu = np.sum(coverage[y == target_class])
            total_cover = np.sum(coverage)
            
            purity = accu / total_cover if total_cover > 0 else 0
            
            if purity >= best_purity:
                best_purity = purity
                best_support_bound = candidate
                
        # Lock in the best boundary
        self.fuzzy_boundary = best_support_bound

    def covers(self, x):
        """
        Returns the degree of coverage (0.0 to 1.0) for an instance or array of instances.
        """
        import numpy as np
        
        # Handle both single instance and array of instances (vectorized)
        is_array = isinstance(x, np.ndarray) and x.ndim > 1
        val = x[:, self.feature_idx] if is_array else x[self.feature_idx]
        
        if self.fuzzy_boundary is None or self.fuzzy_boundary == self.value:
            # Hard boundary
            if self.direction == 0:
                mask = val <= self.value
            else:
                mask = val >= self.value
            return mask.astype(float) if is_array else float(mask)
        
        # Fuzzy membership (linear interpolation)
        if self.direction == 0:
            # val <= value -> 1.0
            # val >= fuzzy_boundary -> 0.0
            membership = (self.fuzzy_boundary - val) / (self.fuzzy_boundary - self.value)
            membership = np.clip(membership, 0.0, 1.0)
        else:
            # val >= value -> 1.0
            # val <= fuzzy_boundary -> 0.0
            membership = (val - self.fuzzy_boundary) / (self.value - self.fuzzy_boundary)
            membership = np.clip(membership, 0.0, 1.0)
            
        return membership if is_array else float(membership)

    def __str__(self):
        op = "<=" if self.direction == 0 else ">="
        if self.fuzzy_boundary is not None and self.fuzzy_boundary != self.value:
            if self.direction == 0:
                return f"feature_{self.feature_idx} <= {self.value} (fuzzy drop-off to {self.fuzzy_boundary})"
            else:
                return f"feature_{self.feature_idx} >= {self.value} (fuzzy drop-off to {self.fuzzy_boundary})"
        return f"feature_{self.feature_idx} {op} {self.value}"
