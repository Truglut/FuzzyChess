import numpy as np
import torch
import matplotlib.pyplot as plt
from .evaluation import SymmetricEvaluator


def plot_membership_functions(
    model: SymmetricEvaluator,
    feature_idx: int,
    feature_name: str,
    x_min: float,
    x_max: float,
) -> None:
    model.eval()

    x = torch.linspace(x_min, x_max, 500).unsqueeze(1)

    # Scale x as in the model
    var_idx = (model.var_indices == feature_idx).nonzero(as_tuple=True)[0][0]
    x_scaled = (x - model.X_mean[var_idx]) / model.X_std[var_idx]

    # TODO: complete this function
    with torch.no_grad():
        # Extract membership parameters for this feature
        center_sigma_sq = torch.exp(model.log_center_sigma_sq_raw[feature_idx]) + 1e-8
        sigmoid_slope = torch.exp(model.log_sigmoid_slope[feature_idx])
        sigmoid_center = model.sigmoid_center_raw[feature_idx]

        if model.has_learnable_centers:
            centers = (
                model.learnable_centers[model.center_indices] * model.center_multipliers
            )
            center = centers[var_idx]
        else:
            center = 0

        # Evaluate parameters on the scaled coordinates
        mu_center = torch.exp(
            -0.5 * torch.divide((x_scaled - center) ** 2, center_sigma_sq)
        )
        mu_right = torch.sigmoid(sigmoid_slope * (x_scaled - sigmoid_center))
        mu_left = torch.sigmoid(-sigmoid_slope * (x_scaled + sigmoid_center))

    # Plot membership functions with corresponding labels
    x_np = x.numpy()
    plt.figure(figsize=(8, 4))
    plt.plot(x_np, mu_left.numpy(), label="Left / Negative / Low", color="red")
    plt.plot(x_np, mu_center.numpy(), label="Center / Zero / Medium", color="blue")
    plt.plot(x_np, mu_right.numpy(), label="Right / Positive / High", color="green")

    plt.title(f"Learned Membership Functions: {feature_name}")
    plt.xlabel("Raw Feature Value")
    plt.ylabel("Degree of Membership")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def print_fuzzy_rules(model, feature_names, label_names=None):
    """
    Translates the tensor rules into human-readable text.
    feature_names: List of physical feature names (e.g., ["KS_White", "KS_Black", "Center", "Material"])
    """
    if label_names is None:
        label_names = ["Low", "Med", "High"]

    model.eval()
    with torch.no_grad():
        print(model.is_free)
        free_rules = model.rule_indices[model.is_free].cpu().numpy()
        free_consequents = model._build_consequents()[model.is_free].cpu().numpy()
        # rules = model.rule_indices.cpu().numpy()
        # consequents = model._build_consequents().cpu().numpy()

    print(f"--- Extracted {len(free_rules)} Free Fuzzy Rules ---")

    # Sort rules by absolute consequent weight to see the most impactful ones first
    sorted_indices = np.argsort(np.abs(free_consequents))[::-1]

    for i in sorted_indices:
        rule = free_rules[i]
        weight = free_consequents[i]

        # Skip rules that have virtually no impact
        if abs(weight) < 0.01:
            continue

        antecedents = []
        for var_idx, label_idx in enumerate(rule):
            feature = feature_names[var_idx]
            label = label_names[label_idx]
            antecedents.append(f"{feature} is {label}")

        if_clause = " AND ".join(antecedents)

        # Format the weight
        impact = f"{weight:+.2f}"

        print(f"IF {if_clause} THEN Eval += {impact}")