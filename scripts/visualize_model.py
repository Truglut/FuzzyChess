from src.tapered_bot.load_bot import load_bot
from src.model_visualization import plot_membership_functions, print_fuzzy_rules


MG_FEATURE_COLS = [
    "material_count",
    "center_control",
    "mobility_diff",
    "pawn_structure_diff",
    "king_safety_white",
    "king_safety_black",
    # "pawn_structure_white",
    # "pawn_structure_black",
    # "mobility_white",
    # "mobility_black",
]

MG_VAR_CONFIGS = [
    {"name": "material", "paired": False, "learnable_center": False},
    {"name": "center_control", "paired": False, "learnable_center": False},
    {"name": "mobility_diff", "paired": False, "learnable_center": False},
    {"name": "pawn_structure_diff", "paired": False, "learnable_center": False},
    {"name": "king_safety", "paired": True, "learnable_center": False},
    # {"name": "pawn_structure", "paired": True, "learnable_center": False},
    # {"name": "mobility", "paired": True, "learnable_center": False},
]

EG_FEATURE_COLS = [
    "material_count",
    "promotion_chances_diff",
    "mobility_diff",
    "pawn_structure_diff",
    # "king_distance_to_center_white",
    # "king_distance_to_center_black",
    # "promotion_chances_white",
    # "promotion_chances_black",
    # "pawn_structure_white",
    # "pawn_structure_black",
    # "mobility_white",
    # "mobility_black",
]

EG_VAR_CONFIGS = [
    {"name": "material", "paired": False, "learnable_center": False},
    {"name": "promotion_diff", "paired": False, "learnable_center": False},
    {"name": "mobility_diff", "paired": False, "learnable_center": False},
    {"name": "pawn_structure_diff", "paired": False, "learnable_center": False},
    # {"name": "king_distance", "paired": True, "learnable_center": False},
    # {"name": "promotion", "paired": True, "learnable_center": False},
    # {"name": "pawn_structure", "paired": True, "learnable_center": False},
    # {"name": "mobility", "paired": True, "learnable_center": False},
]

engine = load_bot(antecedent_length=2)

model = engine.evaluator

# Interpretability logic (Optional depending on your module)
try:
    print("\n--- Middle Game Rules ---")
    print_fuzzy_rules(model.mg_evaluator, MG_FEATURE_COLS)
    print("\n--- End Game Rules ---")
    print_fuzzy_rules(model.eg_evaluator, EG_FEATURE_COLS)

except Exception as e:
    print(
        "Could not print rules. Make sure your printing function handles asymmetric feature lists."
    )
    print({e})

# 6. Plotting Membership Functions
plot_graphs = (
    input("\nDo you want to plot the membership functions? (y/n): ").strip().lower()
)

if plot_graphs in ["y", "yes"]:
    print("\n--- Plotting Middle Game Membership Functions ---")
    for var_idx, config in enumerate(MG_VAR_CONFIGS):
        x_min, x_max = (
            (-10.0, 10.0) if config["name"] != "promotion_diff" else (-10.0, 10.0)
        )
        plot_membership_functions(
            model=model.mg_evaluator,
            var_idx=var_idx,
            feature_name=f"MG - {config['name']}",
            x_min=x_min,
            x_max=x_max,
        )

    print("\n--- Plotting End Game Membership Functions ---")
    for var_idx, config in enumerate(EG_VAR_CONFIGS):
        x_min, x_max = (
            (-10.0, 10.0) if config["name"] != "promotion_diff" else (-10.0, 10.0)
        )
        plot_membership_functions(
            model=model.eg_evaluator,
            var_idx=var_idx,
            feature_name=f"EG - {config['name']}",
            x_min=x_min,
            x_max=x_max,
        )
