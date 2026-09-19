import numpy as np

def foil_info_gain(p1, n1, p0, n0):
    """
    Calculates the FOIL (First Order Inductive Learner) Information Gain.
    This is the core heuristic used in RIPPER and FURIA to evaluate the quality of a new condition.
    
    Parameters
    ----------
    p1 : float
        Number of positive instances covered by the rule AFTER adding the condition.
        (In FURIA source: fstAccu / sndAccu)
    n1 : float
        Number of negative instances covered by the rule AFTER adding the condition.
    p0 : float
        Number of positive instances covered by the rule BEFORE adding the condition.
    n0 : float
        Number of negative instances covered by the rule BEFORE adding the condition.
        
    Returns
    -------
    info_gain : float
        The FOIL information gain. Higher is better.
        
    Notes
    -----
    Translated from Weka's FURIA implementation:
    ```java
    // fstAccuRate = (fstAccu + 1.0) / (fstCover + 1.0)
    // fstInfoGain = fstAccu * (Utils.log2(fstAccuRate) - Utils.log2(defAcRt))
    ```
    Note the Laplace correction (+1.0) used in the Weka source to avoid log(0).
    """
    
    # Laplace correction for probabilities, exactly as in FURIA.java line 970
    acc_rate_after = (p1 + 1.0) / (p1 + n1 + 1.0)
    acc_rate_before = (p0 + 1.0) / (p0 + n0 + 1.0)
    
    # Information gain formula
    info_gain = p1 * (np.log2(acc_rate_after) - np.log2(acc_rate_before))
    
    return info_gain

def find_best_numeric_split(X_feature, y, target_class, def_ac_rt):
    """
    Finds the best split point for a numeric antecedent based on FOIL information gain.
    
    Parameters
    ----------
    X_feature : array-like of shape (n_samples,)
        The values of a single numeric feature.
    y : array-like of shape (n_samples,)
        The target class labels.
    target_class : int or string
        The class label we are predicting.
    def_ac_rt : float
        The default accuracy rate before adding this condition.
        
    Returns
    -------
    best_split : float
        The optimal split threshold.
    best_direction : int
        0 for '<=', 1 for '>='
    max_info_gain : float
        The information gain achieved by this split.
    """
    # This function represents the vectorized translation of FURIA.java's NumericAntd.splitData()
    # It replaces the instance-by-instance iteration with NumPy array operations.
    
    # Sort values for efficient splitting
    sorted_indices = np.argsort(X_feature)
    X_sorted = X_feature[sorted_indices]
    y_sorted = y[sorted_indices]
    
    is_target = (y_sorted == target_class)
    
    # Cumulative counts for fast evaluation
    cum_target = np.cumsum(is_target)
    total_target = cum_target[-1]
    total_samples = len(y_sorted)
    
    max_info_gain = 0.0
    best_split = None
    best_direction = 0
    
    # Iterate over possible split points (ignoring duplicates)
    for i in range(total_samples - 1):
        if X_sorted[i] == X_sorted[i+1]:
            continue
            
        split_val = X_sorted[i]
        
        # Split <= split_val (First Bag in FURIA)
        p1_le = cum_target[i]
        n1_le = (i + 1) - p1_le
        
        # Split >= split_val (Second Bag in FURIA) - NOTE: Weka uses strictly > for the second bag, 
        # but the semantics effectively evaluate the upper bound.
        p1_gt = total_target - p1_le
        n1_gt = (total_samples - (i + 1)) - p1_gt
        
        # Evaluate <= direction
        acc_le = (p1_le + 1.0) / (p1_le + n1_le + 1.0)
        gain_le = p1_le * (np.log2(acc_le) - np.log2(def_ac_rt))
        
        # Evaluate > direction
        acc_gt = (p1_gt + 1.0) / (p1_gt + n1_gt + 1.0)
        gain_gt = p1_gt * (np.log2(acc_gt) - np.log2(def_ac_rt))
        
        # Check if <= is better
        if gain_le > max_info_gain:
            max_info_gain = gain_le
            best_split = split_val
            best_direction = 0 # <=
            
        # Check if > is better
        if gain_gt > max_info_gain:
            max_info_gain = gain_gt
            best_split = X_sorted[i+1] # The next value represents the boundary for >
            best_direction = 1 # >=
            
    return best_split, best_direction, max_info_gain
