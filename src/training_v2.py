import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from .evaluation_v2 import SymmetricEvaluator
import matplotlib.pyplot as plt
from .model_visualization import plot_membership_functions, print_fuzzy_rules

DATA_PATH = "data/positions/features_v2.csv"
FEATURE_COLS = [
    "king_safety_white",
    "king_safety_black",
    "center_control",
    "pawn_structure_white",
    "pawn_structure_black",
    "mobility_white",
    "mobility_black",
    "material_count",
]
VAR_CONFIGS = [
    {"name": "king_safety", "paired": True, "learnable_center": True},
    {"name": "center_control", "paired": False, "learnable_center": False},
    {"name": "pawn_structure", "paired": True, "learnable_center": True},
    {"name": "mobility", "paired": True, "learnable_center": True},
    {"name": "material_count", "paired": False, "learnable_center": False},
]
TARGET_COL = "Stockfish_Eval"

# Load data
df = pd.read_csv(DATA_PATH)

X = torch.tensor(df[FEATURE_COLS].values, dtype=torch.float32)
y = torch.tensor(df[TARGET_COL].values, dtype=torch.float32)
y = torch.clamp(y, min=-10, max=10)
# y_std = y.std()
# y = y / y_std

dataset = TensorDataset(X, y)

# Split into train and validation
val_ratio = 0.2
n_total = len(dataset)
n_val = int(n_total * val_ratio)
n_train = n_total - n_val

train_dataset, val_dataset = random_split(dataset, [n_train, n_val])

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

# Model specification
n_vars = len(FEATURE_COLS)
n_labels = 3
model = SymmetricEvaluator(
    var_configs=VAR_CONFIGS,
    antecedent_length=4
)

## Training specs
# Optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

# Number of epochs
n_epochs = 100

# Keep track of losses for graph
training_losses = []
validation_losses = []

print("Starting model trainig...")
for epoch in range(n_epochs):
    model.train()
    train_loss = 0.0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = loss_fn(preds, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    training_losses.append(train_loss)

    # Validation
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            preds = model(X_batch)
            loss = loss_fn(preds, y_batch)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    validation_losses.append(val_loss)

    print(f"Epoch {epoch + 1}. Train Loss: {train_loss:4f}. Val Loss: {val_loss}")

plt.plot(range(n_epochs), training_losses, label="Train Loss")
plt.plot(range(n_epochs), validation_losses, label="Validation Loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.show()

# Plot evaluation distributions
plt.hist((y).detach().cpu().numpy())
plt.show()

## Model evaluation for interpretability
for var_idx, var_config in enumerate(VAR_CONFIGS):
    var_name = var_config["name"]
    plot_membership_functions(model, var_idx, var_name, -10, 10)

print_fuzzy_rules(model, FEATURE_COLS)

# Ask user whether to save the model

save = input("Save model? (y/n):").strip().lower()

if save in ["y", "yes"]:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "feature_cols": FEATURE_COLS,
        "var_configs": VAR_CONFIGS,
        "n_labels": n_labels
    }

    model_path = "src/models/model_checkpointXXX.pth"
    torch.save(checkpoint, model_path)
    print(f"Model checkpoint saved as {model_path}")
