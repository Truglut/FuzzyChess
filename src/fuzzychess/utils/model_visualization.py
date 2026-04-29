import numpy as np
import torch
import matplotlib.pyplot as plt
from fuzzychess.evaluation.evaluator import SymmetricEvaluator


def plot_membership_functions(
    model: SymmetricEvaluator,
    var_idx: int,  # Changed from feature_idx to clarify this is the var_config index
    feature_name: str,
    x_min: float,
    x_max: float,
) -> None:
    model.eval()

    x = torch.linspace(x_min, x_max, 500).unsqueeze(1)

    physical_idx = (model.var_indices == var_idx).nonzero(as_tuple=True)[0][0]

    if hasattr(model, "input_scales"):
        print("Scaling inputs...")
        x_scaled = x / model.input_scales[physical_idx]
    else:
        x_scaled = x

    with torch.no_grad():
        # Extract membership parameters for this grouped variable directly using var_idx
        # Get membership params
        center_sigma_sq, sigmoid_slope, sigmoid_center, centers = (
            model._get_membership_params_cached()
        )
        center_sigma_sq = center_sigma_sq[physical_idx]
        sigmoid_slope = sigmoid_slope[physical_idx]
        sigmoid_center = sigmoid_center[physical_idx]

        # Check for learnable centers safely
        if getattr(model, "has_learnable_centers", False):
            center = centers[physical_idx]
        else:
            center = 0.0

        # Evaluate parameters on the scaled coordinates
        mu_center = torch.exp(
            -0.5 * torch.divide((x_scaled - center) ** 2, center_sigma_sq)
        )
        mu_right = torch.sigmoid(sigmoid_slope * (x_scaled - sigmoid_center))
        mu_left = torch.sigmoid(-sigmoid_slope * (x_scaled + sigmoid_center))

    # Plot membership functions with corresponding labels
    x_np = x.squeeze().numpy()

    plt.figure(figsize=(8, 4))
    plt.plot(
        x_np, mu_left.squeeze().numpy(), label="Left / Negative / Low", color="red"
    )
    plt.plot(
        x_np, mu_center.squeeze().numpy(), label="Center / Zero / Medium", color="blue"
    )
    plt.plot(
        x_np, mu_right.squeeze().numpy(), label="Right / Positive / High", color="green"
    )

    plt.title(f"Learned Membership Functions: {feature_name}")
    plt.xlabel("Raw Feature Value")
    plt.ylabel("Degree of Membership")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def print_fuzzy_rules(model, feature_names, label_names=None):
    """
    Translates the tensor rules into human-readable text by reconstructing
    them from the evaluator's rule_matrix.
    """
    if label_names is None:
        label_names = ["Low", "Med", "High"]

    model.eval()
    with torch.no_grad():
        # Extraemos las matrices del modelo a la CPU
        rule_matrix = (
            model.rule_matrix.cpu().numpy()
        )  # Shape: (n_vars * n_labels, n_rules)
        is_free = model.is_free.cpu().numpy()
        consequents = model._build_consequents().cpu().numpy()

    n_vars = model.n_vars
    n_labels = model.n_labels
    n_rules = model.n_rules

    # 1. Reconstruir la lista de reglas desde la rule_matrix
    rules = []
    for r_idx in range(n_rules):
        rule = [-1] * n_vars  # Inicializamos todo con "Don't Care"
        for v_idx in range(n_vars):
            for label in range(n_labels):
                flat_idx = v_idx * n_labels + label
                # Si hay un 1.0 en esta posición, actualizamos la etiqueta para esta variable
                if rule_matrix[flat_idx, r_idx] > 0.5:
                    rule[v_idx] = label
                    break
        rules.append(rule)

    # 2. Filtrar solo las reglas "libres" (no espejo) y sus pesos
    free_rules = [rules[i] for i in range(n_rules) if is_free[i]]
    free_consequents = consequents[is_free]

    print(f"--- Extracted {len(free_rules)} Free Fuzzy Rules ---")

    # Ordenar reglas por el valor absoluto del peso para ver primero las más impactantes
    sorted_indices = np.argsort(np.abs(free_consequents))[::-1]

    for i in sorted_indices:
        rule = free_rules[i]
        weight = free_consequents[i]

        # Omitir reglas que no tienen impacto real (cercanas a 0)
        if abs(weight) < 0.01:
            continue

        antecedents = []
        for var_idx, label_idx in enumerate(rule):
            # Ignoramos los comodines (-1 / "Don't Care")
            if label_idx != -1:
                feature = feature_names[var_idx]
                label = label_names[label_idx]
                antecedents.append(f"{feature} is {label}")

        if_clause = " AND ".join(antecedents) if antecedents else "ALWAYS"

        # Formatear el peso
        impact = f"{weight:+.2f}"

        print(f"IF {if_clause} THEN Eval += {impact}")
