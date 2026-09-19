import numpy as np
from scipy.stats import norm

def entropy(y):
    """Calculate the Shannon entropy of a label array."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs))

def info_gain_ratio(y, mask_left, mask_right):
    """
    Calculate the information gain ratio of a split.
    """
    if len(y) == 0:
        return 0.0, 0.0
    
    total_entropy = entropy(y)
    
    n = len(y)
    n_left = np.sum(mask_left)
    n_right = np.sum(mask_right)
    
    if n_left == 0 or n_right == 0:
        return 0.0, 0.0
        
    entropy_left = entropy(y[mask_left])
    entropy_right = entropy(y[mask_right])
    
    # Information Gain
    rem_entropy = (n_left / n) * entropy_left + (n_right / n) * entropy_right
    ig = total_entropy - rem_entropy
    
    # Split Info
    p_left = n_left / n
    p_right = n_right / n
    split_info = - (p_left * np.log2(p_left) + p_right * np.log2(p_right))
    
    # Gain Ratio
    gain_ratio = ig / split_info if split_info > 0 else 0.0
    
    return ig, gain_ratio

def c45_add_errors(N, e, CF=0.25):
    """
    Calculates the statistical upper bound on the error according to C4.5 Error-Based Pruning.
    This translates Weka's `Stats.addErrs` method.
    
    Parameters
    ----------
    N : float
        Total number of instances covered.
    e : float
        Number of incorrectly classified instances.
    CF : float
        Confidence factor (default: 0.25).
        
    Returns
    -------
    float
        The estimated additional error due to statistical uncertainty.
    """
    if CF > 0.5:
        return 0.0
        
    if e < 1:
        # Base case
        base = N * (1 - np.power(CF, 1 / N))
        if e == 0:
            return base
        # Linear interpolation
        return base + e * (c45_add_errors(N, 1, CF) - base)
        
    if e + 0.5 >= N:
        return max(N - e, 0.0)
        
    # Get z-score
    z = norm.ppf(1 - CF)
    
    f = (e + 0.5) / N
    r = (f + (z * z) / (2 * N) +
         z * np.sqrt((f / N) - (f * f / N) + (z * z / (4 * N * N)))) / (1 + (z * z) / N)
         
    return (r * N) - e

def c45_estimated_errors_for_leaf(N, e, CF=0.25):
    """
    Computes total estimated errors for a leaf.
    """
    return e + c45_add_errors(N, e, CF)
