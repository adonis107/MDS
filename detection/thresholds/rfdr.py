import numpy as np
from scipy import stats
from collections import deque

class RollingFalseDiscoveryRate:
    def __init__(self, window_size=500, alpha=0.05):
        """
        window_size: How many past scores to look at (the "memory").
        alpha: The FDR significance level.
        """
        self.window_size = window_size
        self.alpha = alpha
        self.history = deque(maxlen=window_size)
        
    def process_new_score(self, new_score):
        """
        Add one new score, determine if it's an anomaly, and return the dynamic threshold used.
        """
        self.history.append(new_score)
        
        if len(self.history) < 20:
            return False, 0.0
            
        scores = np.array(self.history)
        
        min_val = np.min(scores)
        if min_val <= 0:
            shift = np.abs(min_val) + 1e-6
            scores_shifted = scores + shift
        else:
            scores_shifted = scores
            shift = 0

        scores_log = np.log(scores_shifted)
        
        median_val = np.median(scores_log)
        diff = np.abs(scores_log - median_val)
        mad = np.median(diff)
        
        if mad == 0:
            mad = np.mean(diff) + 1e-10

        robust_sigma = mad * 1.4826
        z_scores = (scores_log - median_val) / robust_sigma
        
        p_values = stats.norm.sf(z_scores)
        
        n = len(scores)
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        ranks = np.arange(1, n + 1)
        
        critical_values = (ranks / n) * self.alpha
        below_threshold = sorted_p <= critical_values
        
        if np.any(below_threshold):
            k_index = np.max(np.where(below_threshold))
            threshold_idx = sorted_indices[k_index]
            threshold_raw = scores[threshold_idx]
        else:
            threshold_raw = np.max(scores)

        is_anomaly = new_score > threshold_raw
        
        return is_anomaly, threshold_raw
    


