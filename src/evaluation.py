import torch.nn as nn
import torch
from typing import Iterable
import itertools


class SymmetricEvaluator(nn.Module):
    def __init__(
        self,
        var_configs: Iterable[dict[str, bool]] | None = None,
        n_labels: int | Iterable[int] = 3,
        X_mean: torch.Tensor | None = None,
        X_std: torch.Tensor | None = None,
        n_vars: int | None = None,
    ):
        super().__init__()

        # Input validation
        if X_mean is None or X_std is None:
            raise ValueError("X_mean and X_std must be provided")

        if var_configs is None:
            if n_vars is None:
                raise ValueError("Must provide one of var_configs or n_vars")
            var_configs = [{"paired": False, "learnable_center": False}] * n_vars
        elif n_vars is not None:
            raise ValueError("Cannot provide both var_configs and n_vars")

        self.var_configs = list(var_configs)
        self.n_unique_features = len(self.var_configs)

        self.register_buffer("X_mean", X_mean)
        self.register_buffer("X_std", X_std)

        # Setup variables and parameter routing
        self._build_variable_mappings()

        # Setup membership function parameters
        self._init_membership_params()

        # Setup rules and consequents
        # TODO: allow manual initialization of rules and consequents
        self._init_rule_base(n_labels)

    def _build_variable_mappings(self):
        """
        Computes mappings for parameter sharing and learnable centers logic
        """
        # Iterate over feature types to calculate actual number of variables and indices
        var_indices = []
        center_indices = []
        center_multipliers = []
        learnable_count = 0

        for i, vconfig in enumerate(self.var_configs):
            is_learnable = vconfig.get("learnable_center", False)
            is_paired = vconfig.get("paired", False)

            # Parameter sharing logic
            var_indices.append(i)
            if is_paired:
                var_indices.append(i)

            # Center parameter logic
            if is_learnable:
                idx = learnable_count
                learnable_count += 1
                mult = 1.0
            else:
                idx = 0  # dummy index, gets zeroed out by multiplier
                mult = 0.0

            # Store index for center parameter and corresponding multiplier
            center_indices.append(idx)
            center_multipliers.append(mult)

            # Store index again with opposite multiplier for paired variables
            if is_paired:
                center_indices.append(idx)
                center_multipliers.append(-mult)

        # Register number of variables and variable mapping
        self.n_vars = len(var_indices)
        self.register_buffer("var_indices", torch.tensor(var_indices, dtype=torch.long))

        # Register center parameter logic if needed
        self.has_learnable_centers = learnable_count > 0
        if self.has_learnable_centers:
            self.learnable_count = learnable_count
            self.register_buffer(
                "center_indices", torch.tensor(center_indices, dtype=torch.long)
            )
            self.register_buffer(
                "center_multipliers",
                torch.tensor(center_multipliers, dtype=torch.float32),
            )

    def _init_membership_params(self):
        """
        Initializes the raw parameters for membership functions
        """
        # Initialize member function parameters
        self.log_center_sigma_sq_raw = nn.Parameter(
            torch.ones((self.n_unique_features,), dtype=torch.float32)
        )
        self.log_sigmoid_slope = nn.Parameter(
            torch.ones((self.n_unique_features,), dtype=torch.float32)
        )
        self.sigmoid_center_raw = nn.Parameter(
            torch.zeros((self.n_unique_features,), dtype=torch.float32)
        )
        if self.has_learnable_centers:
            self.learnable_centers = nn.Parameter(
                torch.zeros((self.learnable_count,), dtype=torch.float32)
            )

    def _init_rule_base(self, n_labels: int | Iterable[int]):
        """
        Sets up rule permutations, mirroring, and consequents
        """
        if isinstance(n_labels, int):
            n_labels = [n_labels] * self.n_vars

        membership_idx = [list(range(n)) for n in n_labels]
        rule_indices = list(itertools.product(*membership_idx))
        n_rules = len(rule_indices)
        self.register_buffer(
            "rule_indices_T", torch.tensor(rule_indices, dtype=torch.long).T
        )
        self.register_buffer(
            "rule_indices", torch.tensor(rule_indices, dtype=torch.long)
        )

        # Determine mirror pairings
        mirror_indices = self._compute_mirror_indices(n_labels)
        self.register_buffer("mirror_indices", mirror_indices)

        # Determine free rule consequents
        is_free = torch.arange(n_rules, device=mirror_indices.device) < mirror_indices
        self.register_buffer("is_free", is_free)

        n_free = is_free.sum().item()
        self.free_consequents = nn.Parameter(
            0.01 * torch.randn(n_free, dtype=torch.float32)
        )

    def _compute_mirror_indices(self, n_labels: list[int]) -> torch.Tensor:
        """
        For each rule (row in rule_indices), find the index of its mirror rule
        """

        # Compute mirrored rules according to variable types
        mirrored_rules = self.rule_indices.clone()
        k = 0
        for vconfig in self.var_configs:
            if vconfig.get("paired"):
                # swap paired variables (assumes k and k + 1 are the pair)
                mirrored_rules[:, k] = self.rule_indices[:, k + 1]
                mirrored_rules[:, k + 1] = self.rule_indices[:, k]
                k += 2

            else:
                # flip label index: i -> (L - 1 - i)
                mirrored_rules[:, k] = n_labels[k] - 1 - self.rule_indices[:, k]
                k += 1

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

    def _build_consequents_cached(self):
        if not hasattr(self, "_cached_consequents"):
            self._cached_consequents = None

        if self.training:
            self._cached_consequents = None  # invalidate during training

        if self._cached_consequents is None:
            self._cached_consequents = self._build_consequents()

        return self._cached_consequents

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        # batch size: (n_samples, n_vars)

        batch = (batch - self.X_mean) / self.X_std

        n_samples = batch.shape[0]

        # Get per-feature membership parameters
        center_sigma_sq = (
            torch.exp(self.log_center_sigma_sq_raw[self.var_indices]) + 1.0e-8
        )
        sigmoid_slope = torch.exp(self.log_sigmoid_slope[self.var_indices])
        sigmoid_center = self.sigmoid_center_raw[self.var_indices]

        # Evaluate membership to central labels
        if self.has_learnable_centers:
            centers = (
                self.learnable_centers[self.center_indices] * self.center_multipliers
            )
            mu_center = torch.exp(
                -0.5 * torch.divide((batch - centers) ** 2, center_sigma_sq)
            )  # should be (n_samples, n_vars)
        else:
            mu_center = torch.exp(
                -0.5 * torch.divide(batch**2, center_sigma_sq)
            )  # should be (n_samples, n_vars)

        # Evaluate membership to extreme labels
        mu_right = torch.sigmoid(sigmoid_slope * (batch - sigmoid_center))
        mu_left = torch.sigmoid(-sigmoid_slope * (batch + sigmoid_center))

        memberships = torch.stack((mu_left, mu_center, mu_right), dim=2)

        # memberships: (n_samples, n_vars, n_labels)

        # TODO: membership padding in case different variables have different n_labels

        # Rule antecedent evaluation
        idx = self.rule_indices_T.unsqueeze(0).expand(n_samples, -1, -1)
        per_var_rule_ant = torch.gather(memberships, dim=2, index=idx)

        # Log of the firing strengths for each rule   shape: (n_samples, n_rules)
        log_firing_strength = torch.log(per_var_rule_ant + 1e-8).sum(dim=1)

        # Transform into weights through softmax
        weights = torch.softmax(log_firing_strength, dim=1)  # (n_rules, )

        consequents = self._build_consequents_cached()  # (n_rules, )
        return (weights * consequents).sum(dim=1)
