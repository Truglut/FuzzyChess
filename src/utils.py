import itertools
import torch
import pandas as pd

def build_heuristic_rules_and_consequents(
    var_configs: list[dict[str, bool]], 
    antecedent_length: int, 
    feature_values: list[float]
) -> tuple[list[tuple[int, ...]], list[float]]:
    """
    Builds rule permutations and initializes consequents, automatically 
    handling symmetry for paired variables.
    
    Args:
        var_configs: The same list of dicts passed to SymmetricEvaluator.
        antecedent_length: How many variables are active per rule.
        feature_values: Advantage for White when the *unique* feature is "High".
                        Must match the length of `var_configs`.
    """
    if len(feature_values) != len(var_configs):
        raise ValueError("Provide exactly one 'high' value per unique feature configuration.")

    # 1. Dynamically expand the variable values based on pairings
    expanded_values = []
    for config, val in zip(var_configs, feature_values):
        expanded_values.append(val)
        if config.get("paired", False):
            # The opponent's version of the feature gets the exact opposite evaluation
            expanded_values.append(-val)
            
    n_vars = len(expanded_values)
    rules = []
    consequents = []
    
    # 2. Generate combinations and compute consequents
    for active_vars in itertools.combinations(range(n_vars), antecedent_length):
        for labels in itertools.product(range(3), repeat=antecedent_length):
            rule = [-1] * n_vars  
            consequent_sum = 0.0
            
            for var_idx, label in zip(active_vars, labels):
                rule[var_idx] = label
                val = expanded_values[var_idx]
                
                if label == 2:    # High
                    consequent_sum += val
                elif label == 0:  # Low
                    consequent_sum -= val
                # Medium (label == 1) does nothing
                
            rules.append(tuple(rule)) # Note: Using tuple to match your internal class logic
            consequents.append(consequent_sum / antecedent_length)
            
    return rules, consequents


def get_scaling_factors(X: torch.Tensor, var_configs: list[dict]):
    """
    Calculates combined standard deviations for paired features.
    
    Returns:
        unique_stds: Tensor of length len(var_configs) for `absorb_scaling_factors`.
        expanded_stds: Tensor of length X.shape[1] to actually divide the dataset.
    """
    unique_stds = []
    expanded_stds = []
    
    col_idx = 0
    for config in var_configs:
        is_paired = config.get("paired", False)
        
        if is_paired:
            # Combine White and Black data (col_idx and col_idx + 1)
            combined_data = torch.cat([X[:, col_idx], X[:, col_idx + 1]])
            
            # Add a tiny epsilon to prevent division by zero on constant features
            std = combined_data.std().item() + 1e-8 
            
            unique_stds.append(std)
            expanded_stds.extend([std, std])  # Apply same std to both columns
            col_idx += 2
        else:
            std = X[:, col_idx].std().item() + 1e-8
            
            unique_stds.append(std)
            expanded_stds.append(std)
            col_idx += 1
            
    return (
        torch.tensor(unique_stds, dtype=torch.float32), 
        torch.tensor(expanded_stds, dtype=torch.float32)
    )