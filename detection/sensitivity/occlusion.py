import numpy as np
import pandas as pd
import re
import torch


def parse_feature_attributes(feature_name):
    """
    Parses a feature name to extract its attributes: side, level, and feature type.
    
    Args:
        feature_name (str): The feature name to parse.
        
    Returns:
        dict: Dictionary with 'side', 'level', and 'type' keys.
    """
    name_lower = feature_name.lower()
    
    # Side
    if 'bid' in name_lower: side = 'bid'
    elif 'ask' in name_lower: side = 'ask'
    else: side = 'neutral'
    
    # Level
    level = None
    
    # Pattern for features
    lob_match = re.search(r'(bid|ask)[-_]?(price|volume)[-_]?(\d+)', name_lower)
    if lob_match:
        level = int(lob_match.group(3))
    
    # Pattern for OFI features
    ofi_match = re.search(r'level[-_]?(\d+)', name_lower)
    if ofi_match:
        level = int(ofi_match.group(1))
    
    # Determine feature type
    if 'order_flow_imbalance' in name_lower or name_lower.startswith('ofi'):
        ftype = 'ofi'
    elif 'hawkes' in name_lower:
        ftype = 'hawkes'
    elif 'price' in name_lower and ('bid-price' in name_lower or 'ask-price' in name_lower):
        ftype = 'lob_price'
    elif 'volume' in name_lower and ('bid-volume' in name_lower or 'ask-volume' in name_lower):
        ftype = 'lob_volume'
    elif 'imbalance' in name_lower:
        ftype = 'imbalance'
    elif 'spread' in name_lower:
        ftype = 'spread'
    elif 'mid_price' in name_lower or 'micro_price' in name_lower:
        ftype = 'mid_price'
    elif 'depth' in name_lower:
        ftype = 'depth'
    elif 'volatility' in name_lower:
        ftype = 'volatility'
    elif 'velocity' in name_lower or 'acceleration' in name_lower:
        ftype = 'dynamics'
    elif 'flow' in name_lower:
        ftype = 'flow'
    elif 'return' in name_lower:
        ftype = 'return'
    elif 'sweep' in name_lower:
        ftype = 'sweep'
    elif 'slope' in name_lower:
        ftype = 'slope'
    elif 'cancel' in name_lower:
        ftype = 'cancel'
    elif 'trade' in name_lower:
        ftype = 'trade'
    elif 'sma' in name_lower:
        ftype = 'sma'
    elif 'rapidity' in name_lower:
        ftype = 'rapidity'
    elif 'speed' in name_lower:
        ftype = 'speed'
    elif 'deep_order' in name_lower:
        ftype = 'deep_order'
    elif 'dt' in name_lower:
        ftype = 'time'
    else:
        ftype = 'other'
    
    return {'side': side, 'level': level, 'type': ftype}


def group_features(feature_names, group_by='side'):
    """
    Groups features based on the specified grouping criterion.
    
    Args:
        feature_names (list): List of feature names.
        group_by (str): Grouping criterion - 'side', 'level', 'type', or combinations 
                        like 'side_level', 'side_type', 'level_type', 'side_level_type'.
        
    Returns:
        dict: Dictionary mapping group names to lists of (feature_name, feature_index) tuples.
    """
    groups = {}
    
    for idx, fname in enumerate(feature_names):
        attrs = parse_feature_attributes(fname)
        
        # Build group key based on grouping criterion
        if group_by == 'side':
            key = attrs['side']
        elif group_by == 'level':
            key = f"level_{attrs['level']}" if attrs['level'] is not None else 'no_level'
        elif group_by == 'type':
            key = attrs['type']
        elif group_by == 'side_level':
            level_str = f"level_{attrs['level']}" if attrs['level'] is not None else 'no_level'
            key = f"{attrs['side']}_{level_str}"
        elif group_by == 'side_type':
            key = f"{attrs['side']}_{attrs['type']}"
        elif group_by == 'level_type':
            level_str = f"level_{attrs['level']}" if attrs['level'] is not None else 'no_level'
            key = f"{level_str}_{attrs['type']}"
        elif group_by == 'side_level_type':
            level_str = f"level_{attrs['level']}" if attrs['level'] is not None else 'no_level'
            key = f"{attrs['side']}_{level_str}_{attrs['type']}"
        else:
            raise ValueError(f"Invalid group_by: {group_by}. Must be one of: "
                           "'side', 'level', 'type', 'side_level', 'side_type', 'level_type', 'side_level_type'")
        
        if key not in groups:
            groups[key] = []
        groups[key].append((fname, idx))
    
    return groups


def GroupedOcclusion(detector, x_seq, feature_names, group_by='side', baseline_mode='mean'):
    """
    Performs Grouped Feature Ablation (Occlusion) to explain Transformer+OCSVM anomalies.
    Instead of occluding one feature at a time, this function occludes groups of features together.
    
    Args:
        detector: The trained TransformerOCSVM detector.
        x_seq (torch.Tensor): The target sequence of shape (1, Seq_Len, Num_Features).
        feature_names (list): List of feature names.
        group_by (str): Grouping criterion - 'side', 'level', 'type', or combinations 
                        like 'side_level', 'side_type', 'level_type', 'side_level_type'.
        baseline_mode (str): 'mean' to use mean values as baseline, 'zero' to use zeros.
        
    Returns:
        pd.DataFrame: Group importance sorted by contribution to the anomaly.
        dict: Dictionary mapping group names to their constituent features.
    """
    # Set transformer to eval mode
    detector.transformer.eval()
    
    # Setup Data
    if x_seq.dim() != 3 or x_seq.size(0) != 1:
        raise ValueError(f"Input must be (1, Seq, Feat), got {x_seq.shape}")
    
    # Get feature groups
    groups = group_features(feature_names, group_by)
    group_names = list(groups.keys())
    num_groups = len(group_names)
    
    # Determine mask value
    if baseline_mode == 'mean':
        baseline_values = x_seq.mean(dim=1, keepdim=True)
    else:
        baseline_values = torch.zeros_like(x_seq[:, 0:1, :])
    
    # Create batch: original + one for each group
    batch_tensor = x_seq.repeat(num_groups + 1, 1, 1).clone()
    
    # Occlusion by group
    for g_idx, group_name in enumerate(group_names):
        feature_indices = [f[1] for f in groups[group_name]]  # Get indices from tuples
        for feat_idx in feature_indices:
            # Set all features in the group to baseline
            if baseline_mode == 'mean':
                batch_tensor[g_idx + 1, :, feat_idx] = baseline_values[0, 0, feat_idx]
            else:
                batch_tensor[g_idx + 1, :, feat_idx] = 0.0
    
    # Batch Inference
    # Process through transformer to get latent representations
    with torch.no_grad():
        batch_tensor = batch_tensor.to(detector.device)
        latent_representations = detector.transformer.get_representation(batch_tensor)
        
        # Get anomaly scores from OCSVM (negative decision function = anomaly score)
        # decision_function accepts CUDA tensors directly via the Nyström OC-SVM
        scores = -detector.ocsvm.decision_function(latent_representations)
    
    # Importance
    original_score = scores[0]
    occluded_scores = scores[1:]
    
    importances = original_score - occluded_scores
    
    # Feature list strings for each group
    feature_lists = {g: [f[0] for f in groups[g]] for g in group_names}
    feature_counts = [len(groups[g]) for g in group_names]
    
    # Format Results
    importance_df = pd.DataFrame({
        'Group': group_names,
        'Importance': importances,
        'Importance_Per_Feature': importances / np.array(feature_counts),
        'Num_Features': feature_counts,
        'Original_Score': original_score,
        'New_Score': occluded_scores
    }).sort_values(by='Importance', ascending=False)
    
    return importance_df, feature_lists

