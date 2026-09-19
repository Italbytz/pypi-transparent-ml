import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

from transparent_ml.algorithms.furia.heuristics import foil_info_gain, find_best_numeric_split

class FuriaClassifier(BaseEstimator, ClassifierMixin):
    """
    Native Python implementation of the Fuzzy Unordered Rule Induction Algorithm (FURIA).
    Translated via AI from the original WEKA Java implementation.
    
    Parameters
    ----------
    optimizations : int, default=2
        The number of optimizations to run during rule induction.
    folds : int, default=3
        The number of folds to use for internal pruning/validation.
    seed : int, default=None
        Random seed for reproducibility.
    """
    def __init__(self, optimizations=2, folds=3, seed=None):
        self.optimizations = optimizations
        self.folds = folds
        self.seed = seed

    def fit(self, X, y):
        """
        Builds the FURIA rule set from the training data.
        
        Translated algorithm steps:
        1. Separate-and-conquer (IREP/RIPPER logic) to grow rules per class.
        2. Prune rules using Minimum Description Length (MDL) or internal CV.
        3. Fuzzification: Transform crisp bounds into fuzzy boundaries using 
           the nearest negative examples.
        """
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.rules_ = []
        
        for c in self.classes_:
            # Java: rulesetForOneClass(expFPRate, data, classIndex, defDL)
            class_rules = self._induce_rules_for_class(X, y, target_class=c)
            
            # Java: Fuzzification happens after rule growth and pruning
            for rule in class_rules:
                self._fuzzify_rule(rule, X, y)
                
            self.rules_.extend(class_rules)
            
        return self

    def _induce_rules_for_class(self, X, y, target_class):
        """
        Implements the 'separate and conquer' RIPPER strategy using FOIL heuristics.
        Java Reference: FURIA.java -> rulesetForOneClass()
        """
        from transparent_ml.algorithms.furia.rule import FuzzyRule
        
        rules = []
        
        # We work on a copy of data (newData in Java) since we remove covered instances
        uncovered_indices = np.arange(X.shape[0])
        stop = False
        
        # Building stage (Java line 2221)
        while not stop:
            # Check if there are any positive instances left
            pos_remaining = np.sum(y[uncovered_indices] == target_class)
            if pos_remaining == 0:
                break
                
            # IREP Data Split: 2/3 for growing, 1/3 for pruning
            # Shuffle indices to ensure random split
            np.random.seed(self.seed)
            shuffled_indices = np.random.permutation(uncovered_indices)
            split_idx = int(len(shuffled_indices) * 2 / 3)
            grow_indices = shuffled_indices[:split_idx]
            prune_indices = shuffled_indices[split_idx:]
            
            # If the pruning set has no positive instances, just use all for growing (fallback)
            if np.sum(y[prune_indices] == target_class) == 0:
                grow_indices = uncovered_indices
                prune_indices = np.array([], dtype=int)
                
            # Initialize a new rule for the target class
            rule = FuzzyRule(target_class=target_class)
            
            # 1. Grow rule
            rule = self._grow_rule(rule, X[grow_indices], y[grow_indices])
            
            # 2. Prune rule (IREP)
            if len(prune_indices) > 0 and rule.antecedents:
                rule = self._prune_rule(rule, X[prune_indices], y[prune_indices])
            
            # 3. Description Length (DL) check to decide if we keep the rule or stop
            # TODO: Implement MDL (Minimum Description Length) stopping criteria
            # For this scaffolding, we simulate stopping if no antecedents remain after pruning
            if not rule.antecedents:
                stop = True
                break
                
            rules.append(rule)
            
            # Remove instances covered by the new rule
            # In FURIA, a rule 'covers' an instance if firing strength > 0 (or = 1 for crisp)
            # For the crisp building stage, we treat coverage as a boolean mask
            covered_mask = np.array([rule.calculate_firing_strength(x) > 0 for x in X[uncovered_indices]])
            uncovered_indices = uncovered_indices[~covered_mask]
            
        return rules

    def _grow_rule(self, rule, X, y):
        """
        Grows a rule by iteratively adding conditions that maximize FOIL info gain.
        Java Reference: FURIA.java -> RipperRule.grow()
        """
        from transparent_ml.algorithms.furia.antecedent import NumericAntecedent
        
        # Calculate initial default accuracy rate for FOIL
        target_class = rule.target_class
        p0 = np.sum(y == target_class)
        n0 = len(y) - p0
        def_ac_rt = (p0 + 1.0) / (p0 + n0 + 1.0)
        
        active_indices = np.arange(X.shape[0])
        
        # Keep adding antecedents until the rule covers only positive instances (acc = 1.0)
        # or we can't find any info gain
        while len(active_indices) > 0:
            current_y = y[active_indices]
            p_current = np.sum(current_y == target_class)
            
            # Stop growing if rule is pure
            if p_current == len(current_y):
                break
                
            best_gain = 0.0
            best_antecedent = None
            
            # Iterate over all features to find the best split
            for feature_idx in range(X.shape[1]):
                # Determine feature type (assuming numeric for this scaffolding)
                X_feature = X[active_indices, feature_idx]
                
                # Use our vectorized heuristic to find the best numeric split
                split_val, direction, gain = find_best_numeric_split(
                    X_feature, current_y, target_class, def_ac_rt
                )
                
                if gain > best_gain:
                    best_gain = gain
                    best_antecedent = NumericAntecedent(feature_idx, split_val, direction)
            
            if best_antecedent is None:
                # No condition yielded positive information gain
                break
                
            rule.add_antecedent(best_antecedent)
            
            # Update the dataset to only include instances covered by the new antecedent
            # This is the "conquer" part of the growing phase
            covered = np.array([best_antecedent.covers(x) > 0 for x in X[active_indices]])
            active_indices = active_indices[covered]
            
            # Update default accuracy for the next iteration of FOIL
            p0 = np.sum(y[active_indices] == target_class)
            n0 = len(active_indices) - p0
            def_ac_rt = (p0 + 1.0) / (p0 + n0 + 1.0)
            
        return rule

    def _prune_rule(self, rule, X_prune, y_prune):
        """
        Prunes the rule using Incremental Reduced Error Pruning (IREP) on the pruning set.
        Evaluates the sequence of antecedents and cuts off antecedents that cause overfitting.
        Java Reference: FURIA.java -> prune(Instances pruneData, boolean useWhole)
        """
        if not rule.antecedents or len(X_prune) == 0:
            return rule
            
        target_class = rule.target_class
        total_instances = len(y_prune)
        
        # Default accuracy of empty rule on prune data
        def_accu = np.sum(y_prune == target_class)
        max_worth = (def_accu + 1.0) / (total_instances + 2.0)
        best_cutoff_idx = -1 # -1 means prune all (keep empty rule)
        
        # Track which instances are still covered as we apply antecedents sequentially
        active_mask = np.ones(total_instances, dtype=bool)
        
        for i, antecedent in enumerate(rule.antecedents):
            # Apply the current antecedent to filter active instances
            # We use covers > 0 for crisp pruning evaluation
            current_cover = antecedent.covers(X_prune) > 0
            active_mask = active_mask & current_cover
            
            # Evaluate the worth of the rule up to this antecedent
            covered_instances = np.sum(active_mask)
            if covered_instances == 0:
                break
                
            true_positives = np.sum(active_mask & (y_prune == target_class))
            
            # IREP worth formula with Laplace smoothing
            worth = (true_positives + 1.0) / (covered_instances + 2.0)
            
            if worth > max_worth:
                max_worth = worth
                best_cutoff_idx = i
                
        # If best_cutoff_idx is -1, the rule is worse than default (empty), so prune everything
        # Otherwise, keep antecedents up to best_cutoff_idx (inclusive)
        if best_cutoff_idx == -1:
            rule.antecedents = []
        else:
            rule.antecedents = rule.antecedents[:best_cutoff_idx + 1]
            
        return rule

    def _fuzzify_rule(self, rule, X, y):
        """
        Transforms crisp rules into fuzzy rules.
        Java Reference: FURIA.java -> fuzzify(Rule rule, Instances data)
        Finds the nearest instances of the negative class to establish the fuzzy overlap.
        """
        from transparent_ml.algorithms.furia.antecedent import NumericAntecedent
        
        # In FURIA, each antecedent is fuzzified independently by looking at instances
        # that are covered by all OTHER antecedents in the rule.
        for i, antecedent in enumerate(rule.antecedents):
            if isinstance(antecedent, NumericAntecedent):
                # Filter data to instances covered by all other antecedents
                relevant_mask = np.ones(X.shape[0], dtype=bool)
                for j, other_ant in enumerate(rule.antecedents):
                    if i != j:
                        # Only keep instances covered by the other antecedents
                        # Since we are fuzzifying, we use the crisp coverage (covers > 0)
                        relevant_mask = relevant_mask & (other_ant.covers(X) > 0)
                        
                if np.any(relevant_mask):
                    antecedent.fuzzify(X[relevant_mask], y[relevant_mask], rule.target_class)

    def predict(self, X):
        """Predict class labels for samples in X."""
        check_is_fitted(self)
        X = check_array(X)
        
        # Java Reference: FURIA.java -> distributionForInstance()
        # FURIA uses a competitive voting scheme based on rule firing strengths.
        proba = self.predict_proba(X)
        
        # In case of ties, or if no rule fires, FURIA falls back to 'rule stretching'
        # to generalize rules until one fires.
        # Here we just take the argmax of the calculated probabilities/strengths.
        predictions = np.argmax(proba, axis=1)
        return self.classes_[predictions]

    def predict_proba(self, X):
        """Predict class probabilities (firing strengths) for samples in X."""
        check_is_fitted(self)
        X = check_array(X)
        
        # Initialize support matrix: (n_samples, n_classes)
        support = np.zeros((X.shape[0], len(self.classes_)))
        
        # Evaluate firing strength of each rule
        for rule in self.rules_:
            class_idx = np.where(self.classes_ == rule.target_class)[0][0]
            for i, x in enumerate(X):
                # Calculate the membership degree (0.0 to 1.0)
                # Java: rule.predictFuzzy(instance)
                activation = rule.calculate_firing_strength(x)
                
                # In FURIA, certainty is weighted by the rule's support/accuracy
                support[i, class_idx] += activation * rule.weight
                
        # Normalize to get probabilities
        row_sums = support.sum(axis=1, keepdims=True)
        # Avoid division by zero for instances where no rule fired (rule stretching needed)
        # TODO: Implement Rule Stretching for row_sums == 0
        proba = np.divide(support, row_sums, out=np.zeros_like(support), where=row_sums!=0)
        return proba

    def export_rules(self):
        """Export the induced rules in a human-readable format."""
        check_is_fitted(self)
        if not self.rules_:
            return "No rules induced."
        return "\n".join(str(rule) for rule in self.rules_)
