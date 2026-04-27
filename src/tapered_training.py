import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, random_split

from src.evaluator import SymmetricEvaluator
from src.tapered_evaluator import TaperedEvaluator
from src.model_visualization import plot_membership_functions, print_fuzzy_rules
from .utils import build_heuristic_rules_and_consequents, get_scaling_factors

DATA_PATH = "data/positions/extended_features_10e5.csv"


# 1. Feature Definition (Splitting MG and EG)
# Order is critical: Material first, then unpaired (differential), then paired (white, black)
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

MG_FEATURE_VALUES = [
    12,  # material
    2,   # central control
    2,   # mobility diff
    2,   # pawn structure diff
    12,  # king safety
    # 2,   # pawn structure
    # 1,  # mobility
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

EG_FEATURE_VALUES = [
    12,  # material
    6,   # promotion diff
    2,    # mobility diff
    3,    # pawn structure diff
    # 2,  # king distance
    # 4,  # promotion
    # 2,  # pawn_structure
    # 2,  # mobility
]

PHASE_COL = "game_phase"
TARGET_COL = "Stockfish_Eval"

# 2. Data Loading & Preparation
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Extract tensors
X_mg = torch.tensor(df[MG_FEATURE_COLS].values, dtype=torch.float32)
X_eg = torch.tensor(df[EG_FEATURE_COLS].values, dtype=torch.float32)
phase = torch.tensor(df[PHASE_COL].values, dtype=torch.float32)

y = torch.tensor(df[TARGET_COL].values, dtype=torch.float32)
y = torch.clamp(y, min=-10.0, max=10.0)  # Clamp extreme engine evaluations

# Split dataset into train/val
val_ratio = 0.2
n_total = len(X_mg)
n_val = int(n_total * val_ratio)
n_train = n_total - n_val

# Random split using indices
indices = torch.randperm(n_total)
train_indices = indices[:n_train]
val_indices = indices[n_train:]

# Calculate scaling factors on training data only
mg_unique_stds, mg_expanded_stds = get_scaling_factors(
    X_mg[train_indices], MG_VAR_CONFIGS
)
eg_unique_stds, eg_expanded_stds = get_scaling_factors(
    X_eg[train_indices], EG_VAR_CONFIGS
)
y_std = y[train_indices].std() + 1e-8

# Scale all data using statistics from train
X_mg = X_mg / mg_expanded_stds
X_eg = X_eg / eg_expanded_stds
# y = y / y_std

# Build datasets and loaders
train_dataset = TensorDataset(
    X_mg[train_indices], X_eg[train_indices], phase[train_indices], y[train_indices]
)
val_dataset = TensorDataset(
    X_mg[val_indices], X_eg[val_indices], phase[val_indices], y[val_indices]
)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)


# 3. Model Setup
print("Initializing models...")
antecedent_length = 2

# Build rule initializers
mg_rules, mg_consequents = build_heuristic_rules_and_consequents(
    MG_VAR_CONFIGS, antecedent_length, MG_FEATURE_VALUES
)
eg_rules, eg_consequents = build_heuristic_rules_and_consequents(
    EG_VAR_CONFIGS, antecedent_length, EG_FEATURE_VALUES
)

mg_evaluator = SymmetricEvaluator(
    var_configs=MG_VAR_CONFIGS,
    manual_rules=mg_rules,
    manual_consequents=mg_consequents,
    antecedent_length=antecedent_length,
)

eg_evaluator = SymmetricEvaluator(
    var_configs=EG_VAR_CONFIGS,
    manual_rules=eg_rules,
    manual_consequents=eg_consequents,
    antecedent_length=antecedent_length,
)

model = TaperedEvaluator(mg_evaluator, eg_evaluator)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# 4. Training Loop
n_epochs = 100
training_losses = []
validation_losses = []
validation_corrs = []

print("Starting training...")
for epoch in range(n_epochs):
    model.train()
    train_loss = 0.0

    for X_mg_batch, X_eg_batch, phase_batch, y_batch in train_loader:
        optimizer.zero_grad()

        # Forward pass matching the new TaperedEvaluator signature
        preds = model(X_mg_batch, X_eg_batch, phase_batch)

        loss = loss_fn(preds.squeeze(), y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    training_losses.append(train_loss)

    # Validation
    model.eval()
    val_loss = 0.0

    # Initialize correlation metrics
    sum_x = 0.0
    sum_y = 0.0
    sum_x2 = 0.0
    sum_y2 = 0.0
    sum_xy = 0.0
    n = 0

    # Evaluate error
    with torch.no_grad():
        for X_mg_batch, X_eg_batch, phase_batch, y_batch in val_loader:
            preds = model(X_mg_batch, X_eg_batch, phase_batch)
            loss = loss_fn(preds.squeeze(), y_batch)
            val_loss += loss.item()

            # Calculate correlation
            sum_x += preds.sum().item()
            sum_y += y_batch.sum().item()
            sum_x2 += (preds ** 2).sum().item()
            sum_y2 += (y_batch ** 2).sum().item()
            sum_xy += (preds * y_batch).sum().item()
            n += preds.numel()

    val_loss /= len(val_loader)
    validation_losses.append(val_loss)

    # Pearson correlation
    numerator = sum_xy - (sum_x * sum_y / n)
    denominator = ((sum_x2 - sum_x**2 / n) * (sum_y2 - sum_y**2 / n)) ** 0.5
    val_corr = numerator / denominator
    validation_corrs.append(val_corr)

    print(
        f"Epoch {epoch + 1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}. | Val corr: {val_corr:.4f}"
    )

# # Absorb scaling factors into the models
# model.mg_evaluator.absorb_scaling_factors(mg_unique_stds)
# model.eg_evaluator.absorb_scaling_factors(eg_unique_stds)
model.mg_evaluator.set_input_scales(mg_unique_stds)
model.eg_evaluator.set_input_scales(eg_unique_stds)

# ---------------------------------------------------------------------------
# 5. Visualization & Saving
# ---------------------------------------------------------------------------
plt.plot(range(n_epochs), training_losses, label="Train Loss")
plt.plot(range(n_epochs), validation_losses, label="Validation Loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Tapered Evaluator Training")
plt.show()

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
        x_min, x_max = (-10.0, 10.0) if config["name"] != "promotion_diff" else (-20.0, 20.0)
        plot_membership_functions(
            model=model.mg_evaluator,
            var_idx=var_idx,
            feature_name=f"MG - {config['name']}",
            x_min=x_min,
            x_max=x_max,
        )

    print("\n--- Plotting End Game Membership Functions ---")
    for var_idx, config in enumerate(EG_VAR_CONFIGS):
        x_min, x_max = (-10.0, 10.0) if config["name"] != "promotion_diff" else (-20.0, 20.0)
        plot_membership_functions(
            model=model.eg_evaluator,
            var_idx=var_idx,
            feature_name=f"EG - {config['name']}",
            x_min=x_min,
            x_max=x_max,
        )

save = input("\nSave model? (y/n): ").strip().lower()
if save in ["y", "yes"]:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "mg_feature_cols": MG_FEATURE_COLS,
        "eg_feature_cols": EG_FEATURE_COLS,
        "mg_var_configs": MG_VAR_CONFIGS,
        "eg_var_configs": EG_VAR_CONFIGS,
    }
    model_path = "src/models/tapered_evaluator3.pth"
    torch.save(checkpoint, model_path)
    print(f"Model checkpoint successfully saved at {model_path}")
