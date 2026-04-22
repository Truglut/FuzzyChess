import torch.nn as nn
import torch
from typing import Iterable
import itertools


class SymmetricEvaluator(nn.Module):
    def __init__(
        self,
        n_vars: int,
        n_labels: int | Iterable[int] = 3,
        var_types: Iterable[str] | None = None,
        X_mean: torch.Tensor | None = None,
        X_std: torch.Tensor | None = None,
    ):
        super().__init__()
        if isinstance(n_labels, int):
            n_labels = [n_labels] * n_vars

        if X_mean is None or X_std is None:
            raise ValueError("X_mean and X_std must be provided")

        self.register_buffer("X_mean", X_mean)
        self.register_buffer("X_std", X_std)

        membership_idx = [list(range(n)) for n in n_labels]

        # TODO: allow manual rule initialization
        rule_indices = list(itertools.product(*membership_idx))
        self.register_buffer("rule_indices", torch.tensor(rule_indices))

        # Initialize member function parameters
        self.log_center_sigma_sq_raw = nn.Parameter(
            torch.ones((n_vars,), dtype=torch.float32)
        )
        self.log_sigmoid_slope = nn.Parameter(
            torch.ones((n_vars,), dtype=torch.float32)
        )
        self.sigmoid_center_raw = nn.Parameter(
            torch.zeros((n_vars,), dtype=torch.float32)
        )

        # Rule consequent initialization
        n_rules = self.rule_indices.shape[0]

        # Store var_types for symmetry and membership handling
        if var_types is None:
            var_types = ["difference"] * n_vars
        self.var_types = var_types

        # Pair every rule with its mirror to enforce symmetry
        mirror_indices = self._compute_mirror_indices(n_labels, self.var_types)
        self.register_buffer("mirror_indices", mirror_indices)

        # Determine free rule consequents
        is_free = torch.arange(n_rules, device=mirror_indices.device) < mirror_indices
        self.register_buffer("is_free", is_free)
        n_free = is_free.sum().item()

        # TODO: allow consequent initialization with non-zero or non-random values
        # self.free_consequents = nn.Parameter(torch.zeros(n_free, dtype=torch.float32))
        self.free_consequents = nn.Parameter(
            0.01 * torch.randn(n_free, dtype=torch.float32)
        )

    def _compute_mirror_indices(
        self, n_labels: list[int], var_types: Iterable[str]
    ) -> torch.Tensor:
        """
        For each rule (row in rule_indices), find the index of its mirror rule
        """

        n_labels_tensor = torch.tensor(n_labels)

        # Compute mirrored rules according to variable types
        mirrored_rules = self.rule_indices.clone()
        k = 0
        while k < len(var_types):
            vtype = var_types[k]

            if vtype == "difference":
                # flip label index: i -> (L - 1 - i)
                mirrored_rules[:, k] = n_labels[k] - 1 - self.rule_indices[:, k]
                k += 1

            elif vtype == "paired_absolute":
                # swap paired variables (assumes k and k + 1 are the pair)
                mirrored_rules[:, k] = self.rule_indices[:, k + 1]
                mirrored_rules[:, k + 1] = self.rule_indices[:, k]
                k += 2

            else:
                raise ValueError(f"Unrecognised variable type: {vtype}")

        device = self.rule_indices.device
        # Flatten rule indices
        strides = torch.ones(len(n_labels), dtype=torch.long, device=device)
        for k in range(len(n_labels) - 2, -1, -1):
            strides[k] = n_labels[k + 1] * strides[k + 1]

        # Takes rule index and maps it to its flat index
        flat_original = (self.rule_indices * strides).sum(dim=1)

        # Takes rule index and maps it to the flat index of its mirror
        flat_mirrored = (mirrored_rules * strides).sum(dim=1)

        # Invert correspondence between flat indices
        inv = -torch.ones(flat_original.max() + 1, dtype=torch.long, device=device)
        inv[flat_original] = torch.arange(len(flat_original), device=device)
        # At this point, inv maps a rule flat index to a self.rule_indices index

        # Return inverse correspondence evaluated on the mirrored index
        return inv[flat_mirrored]

    def _build_consequents(self) -> torch.Tensor:
        """
        Reconstruct full consequent vector enforcing antisymmetry
        """
        n_rules = self.rule_indices.shape[0]
        consequents = torch.zeros(n_rules, device=self.free_consequents.device)

        # Set free consequents to their values
        consequents[self.is_free] = self.free_consequents

        # Set mirror consequents to the negative value of their free pair
        consequents[self.mirror_indices[self.is_free]] = -self.free_consequents

        # self-mirror rules stay 0 by default 0-initialization
        return consequents

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        # batch size: (n_samples, n_vars)

        batch = (batch - self.X_mean) / self.X_std

        n_samples = batch.shape[0]
        n_rules, n_vars = self.rule_indices.shape

        # Evaluate membership to central labels
        mu_center = torch.exp(
            -0.5
            * torch.divide(batch**2, torch.exp(self.log_center_sigma_sq_raw) + 1e-8)
        )  # should be (n_samples, n_vars)

        # Evaluate membership to extreme labels
        mu_right = torch.sigmoid(
            torch.exp(self.log_sigmoid_slope) * (batch - self.sigmoid_center_raw)
        )
        mu_left = torch.sigmoid(
            -torch.exp(self.log_sigmoid_slope) * (batch + self.sigmoid_center_raw)
        )

        memberships = torch.stack((mu_left, mu_center, mu_right), dim=2)

        # memberships: (n_samples, n_vars, n_labels)

        # TODO: membership padding in case different variables have different n_labels

        # Rule antecedent evaluation
        idx = self.rule_indices.T.unsqueeze(0).expand(n_samples, -1, -1)
        per_var_rule_ant = torch.gather(memberships, dim=2, index=idx)

        # Log of the firing strengths for each rule   shape: (n_samples, n_rules)
        log_firing_strength = torch.log(per_var_rule_ant + 1e-8).sum(dim=1)
        log_firing_strength -= log_firing_strength.max(dim=1, keepdim=True).values

        # Transform into weights through softmax
        weights = torch.softmax(log_firing_strength, dim=1)  # (n_rules, )

        consequents = self._build_consequents()  # (n_rules, )
        return (weights * consequents).sum(dim=1)
