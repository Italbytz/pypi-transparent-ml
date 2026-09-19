import numpy as np
from typing import List, Tuple, Optional
from transparent_ml.algorithms.part.heuristics import info_gain_ratio, c45_estimated_errors_for_leaf
from transparent_ml.algorithms.part.rule import PartCondition, PartRule

class TreeNode:
    def __init__(self, mask: np.ndarray, depth: int):
        self.mask = mask
        self.depth = depth
        self.is_leaf = False
        
        self.feature_idx: Optional[int] = None
        self.threshold: Optional[float] = None
        self.left: Optional['TreeNode'] = None   # <= threshold
        self.right: Optional['TreeNode'] = None  # > threshold
        
        self.predicted_class: Optional[int] = None
        self.errors: float = 0.0
        self.total_weight: float = 0.0

class PartialTreeBuilder:
    def __init__(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None, 
                 min_samples_leaf: int = 2, cf: float = 0.25):
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.min_samples_leaf = min_samples_leaf
        self.cf = cf
        
    def build_and_extract_best_rule(self, mask: np.ndarray) -> PartRule:
        """
        Builds a pruned C4.5 tree on the subset defined by `mask`
        and extracts the "best" leaf (typically the one covering the most instances) as a Rule.
        """
        root = TreeNode(mask, depth=0)
        self._build_tree(root)
        self._prune_tree(root)
        
        # Extract all root-to-leaf paths
        leaves_info = []
        self._extract_leaves(root, [], leaves_info)
        
        if not leaves_info:
            # Fallback if everything is empty
            majority_class = self._get_majority_class(self.y[mask])
            return PartRule(conditions=[], predicted_class=majority_class, weight=np.sum(mask))
            
        # Select the best leaf: in PART, this is usually the leaf covering the most instances
        # or the one with the highest coverage/purity ratio.
        # We pick the one with the highest total_weight (coverage).
        best_leaf = max(leaves_info, key=lambda x: x['total_weight'])
        
        return PartRule(
            conditions=best_leaf['conditions'],
            predicted_class=best_leaf['predicted_class'],
            weight=best_leaf['total_weight']
        )
        
    def _build_tree(self, node: TreeNode):
        y_subset = self.y[node.mask]
        node.total_weight = len(y_subset)
        
        if len(y_subset) == 0:
            node.is_leaf = True
            node.predicted_class = -1
            node.errors = 0.0
            return
            
        node.predicted_class = self._get_majority_class(y_subset)
        node.errors = np.sum(y_subset != node.predicted_class)
        
        # Stop condition: pure node or too few samples
        if node.errors == 0 or len(y_subset) < 2 * self.min_samples_leaf:
            node.is_leaf = True
            return
            
        # Find best split
        best_gain_ratio = -1.0
        best_split = None
        
        for f in range(self.X.shape[1]):
            X_f = self.X[node.mask, f]
            unique_vals = np.unique(X_f)
            
            # Optimization: only consider midpoints between distinct values
            if len(unique_vals) > 1:
                thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0
                for thresh in thresholds:
                    mask_left_local = X_f <= thresh
                    mask_right_local = X_f > thresh
                    
                    if np.sum(mask_left_local) < self.min_samples_leaf or np.sum(mask_right_local) < self.min_samples_leaf:
                        continue
                        
                    _, gain_ratio = info_gain_ratio(y_subset, mask_left_local, mask_right_local)
                    
                    if gain_ratio > best_gain_ratio:
                        best_gain_ratio = gain_ratio
                        best_split = (f, thresh, mask_left_local, mask_right_local)
                        
        if best_split is None:
            node.is_leaf = True
            return
            
        # Apply split
        f, thresh, mask_left_local, mask_right_local = best_split
        node.feature_idx = f
        node.threshold = thresh
        
        # Reconstruct global masks
        global_mask_left = np.zeros_like(node.mask)
        global_mask_left[node.mask] = mask_left_local
        
        global_mask_right = np.zeros_like(node.mask)
        global_mask_right[node.mask] = mask_right_local
        
        node.left = TreeNode(global_mask_left, node.depth + 1)
        node.right = TreeNode(global_mask_right, node.depth + 1)
        
        # Depth first on largest subset first (similar to PART's most promising branch)
        if np.sum(mask_left_local) >= np.sum(mask_right_local):
            self._build_tree(node.left)
            self._build_tree(node.right)
        else:
            self._build_tree(node.right)
            self._build_tree(node.left)
            
    def _prune_tree(self, node: TreeNode) -> float:
        """
        Recursively prune the tree bottom-up using C4.5 Error-Based Pruning.
        Returns the estimated errors for the tree rooted at this node.
        """
        if node.is_leaf:
            return c45_estimated_errors_for_leaf(node.total_weight, node.errors, self.cf)
            
        errs_left = self._prune_tree(node.left)
        errs_right = self._prune_tree(node.right)
        errs_tree = errs_left + errs_right
        
        errs_leaf = c45_estimated_errors_for_leaf(node.total_weight, node.errors, self.cf)
        
        if errs_leaf <= errs_tree + 0.1: # C4.5 precision heuristic
            node.is_leaf = True
            node.left = None
            node.right = None
            return errs_leaf
            
        return errs_tree
        
    def _extract_leaves(self, node: TreeNode, current_conditions: List[PartCondition], leaves: List[dict]):
        if node.is_leaf:
            if node.total_weight > 0:
                leaves.append({
                    'conditions': list(current_conditions),
                    'predicted_class': node.predicted_class,
                    'total_weight': node.total_weight,
                    'errors': node.errors
                })
            return
            
        fname = self.feature_names[node.feature_idx] if self.feature_names else f"feature_{node.feature_idx}"
        
        # Left branch
        cond_left = PartCondition(node.feature_idx, '<=', node.threshold, fname)
        current_conditions.append(cond_left)
        self._extract_leaves(node.left, current_conditions, leaves)
        current_conditions.pop()
        
        # Right branch
        cond_right = PartCondition(node.feature_idx, '>', node.threshold, fname)
        current_conditions.append(cond_right)
        self._extract_leaves(node.right, current_conditions, leaves)
        current_conditions.pop()
        
    def _get_majority_class(self, y: np.ndarray) -> int:
        if len(y) == 0:
            return -1
        vals, counts = np.unique(y, return_counts=True)
        return vals[np.argmax(counts)]
