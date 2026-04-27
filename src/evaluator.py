import torch.nn as nn
import torch
from typing import Iterable
import itertools


class SymmetricEvaluator(nn.Module):
    def __init__(
        self,
        var_configs: Iterable[dict[str, bool]] | None = None,
        manual_rules: list[list[int]] | None = None,
        manual_consequents: torch.Tensor | list[float] | None = None,
        antecedent_length: int | None = None,
    ):
        super().__init__()

        self.var_configs = list(var_configs)
        self.n_unique_features = len(self.var_configs)
        self.n_labels = 3

        # Initialize cache variables safely
        self._cached_consequents = None
        self._cached_membership_params = None

        # Setup variables and parameter routing
        self._build_variable_mappings()

        # Setup membership function parameters
        self._init_membership_params()

        # Setup rules and consequents
        self._init_rule_base(manual_rules, manual_consequents, antecedent_length)

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
            2 * torch.ones((self.n_unique_features,), dtype=torch.float32)
        )
        if self.has_learnable_centers:
            self.learnable_centers = nn.Parameter(
                torch.zeros((self.learnable_count,), dtype=torch.float32)
            )

    def _init_rule_base(
        self,
        manual_rules: list[list[int]],
        manual_consequents: torch.Tensor | list[float],
        antecedent_length: int,
    ):
        """
        Sets up rule permutations, mirroring, and consequents
        """
        if manual_rules is not None:
            rules = manual_rules
        else:
            rules = self._generate_rules(antecedent_length)

        self.n_rules = len(rules)

        # Build rule matrix for fast matmul inference
        # shape: (n_vars * n_labels, n_rules)
        rule_matrix = torch.zeros(
            (self.n_vars * self.n_labels, self.n_rules), dtype=torch.float32
        )
        for r_idx, rule in enumerate(rules):
            for v_idx, label in enumerate(rule):
                if label != -1:
                    flat_idx = v_idx * self.n_labels + label
                    rule_matrix[flat_idx, r_idx] = 1.0

        self.register_buffer("rule_matrix", rule_matrix)

        self._setup_mirror_pairings(rules)

        # Consequent initialization
        if manual_consequents is not None:
            if isinstance(manual_consequents, list):
                manual_consequents = torch.tensor(
                    manual_consequents, dtype=torch.float32
                )
            if len(manual_consequents) == self.n_rules:
                print("Initializing rule consequents using only free rule consequents")
                self.free_consequents = nn.Parameter(
                    manual_consequents.clone()[self.is_free]
                )
            elif len(manual_consequents) == self.n_free:
                self.free_consequents = nn.Parameter(manual_consequents.clone())
            else:
                raise ValueError(f"Unexpected manual consequent length")
        else:
            self.free_consequents = nn.Parameter(
                0.01 * torch.randn(self.n_free, dtype=torch.float32)
            )

    def _setup_mirror_pairings(self, rules: list[tuple[int, ...]]) -> int:
        """Determines mirror rule indices, masks free rules and returns free rule count"""
        rule_to_idx = {tuple(r): i for i, r in enumerate(rules)}
        mirror_indices = []
        is_free = []

        for i, rule in enumerate(rules):
            mirror_rule = self._get_mirror_rule(rule)
            if mirror_rule not in rule_to_idx:
                raise ValueError(
                    f"Rule set is not symmetric. Missing mirror for rule: {rule}"
                )

            m_idx = rule_to_idx[mirror_rule]
            mirror_indices.append(m_idx)
            is_free.append(i < m_idx)

        self.register_buffer(
            "mirror_indices", torch.tensor(mirror_indices, dtype=torch.long)
        )
        self.register_buffer("is_free", torch.tensor(is_free, dtype=torch.bool))
        self.n_free = self.is_free.sum().item()

    def _get_mirror_rule(self, rule: tuple[int, ...]) -> tuple[int, ...]:
        mirror = list(rule)
        k = 0

        for vconfig in self.var_configs:
            is_paired = vconfig.get("paired", False)
            if is_paired:
                # swap variables
                mirror[k], mirror[k + 1] = mirror[k + 1], mirror[k]
                k += 2
            else:
                # flip label if it not a "Don't Care"
                if mirror[k] != -1:
                    mirror[k] = (self.n_labels - 1) - mirror[k]
                k += 1

        return tuple(mirror)

    def _generate_rules(
        self, antecedent_length: int | None = None
    ) -> list[tuple[int, ...]]:
        """Generates rules automatically, optionally fixing antecedent length"""

        if antecedent_length is None:
            antecedent_length = self.n_vars

        rules = []
        # Generate combinations of subset size antecedent length
        for active_vars in itertools.combinations(
            range(self.n_vars), antecedent_length
        ):
            # For chosen active variables, generate all possible label combinations
            for labels in itertools.product(
                range(self.n_labels), repeat=antecedent_length
            ):
                rule = [
                    -1
                ] * self.n_vars  # -1 is a placeholder that indicates "Don't Care"
                for var_idx, label in zip(active_vars, labels):
                    rule[var_idx] = label
                rules.append(tuple(rule))

        return rules

    def _build_consequents(self) -> torch.Tensor:
        """
        Reconstruct full consequent vector enforcing antisymmetry
        """
        consequents = torch.zeros(self.n_rules, device=self.free_consequents.device)

        # Set free consequents to their values
        consequents[self.is_free] = self.free_consequents

        # Set mirror consequents to the negative value of their free pair
        consequents[self.mirror_indices[self.is_free]] = -self.free_consequents

        # self-mirror rules stay 0 by default 0-initialization
        return consequents

    def _build_consequents_cached(self):
        if self.training or self._cached_consequents is None:
            self._cached_consequents = self._build_consequents()

        return self._cached_consequents

    def _get_membership_params_cached(self):
        """Fetches membership function params, caching them during eval() mode"""

        if self.training or self._cached_membership_params is None:
            # Compute membership params
            center_sigma_sq = (
                torch.exp(self.log_center_sigma_sq_raw[self.var_indices]) + 1.0e-8
            )
            sigmoid_slope = torch.exp(self.log_sigmoid_slope[self.var_indices])
            sigmoid_center = self.sigmoid_center_raw[self.var_indices]

            centers = None
            if self.has_learnable_centers:
                centers = (
                    self.learnable_centers[self.center_indices]
                    * self.center_multipliers
                )

            params = (center_sigma_sq, sigmoid_slope, sigmoid_center, centers)

            # Cache if in eval mode
            if not self.training:
                self._cached_membership_params = params
            return params

        return self._cached_membership_params

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        # batch size: (n_samples, n_vars)
        if hasattr(self, "input_scales"):
            batch = batch / self.input_scales

        # Get membership params
        center_sigma_sq, sigmoid_slope, sigmoid_center, centers = (
            self._get_membership_params_cached()
        )

        # Evaluate center membership
        if self.has_learnable_centers:
            mu_center = torch.exp(
                -0.5 * torch.divide((batch - centers) ** 2, center_sigma_sq)
            )
        else:
            mu_center = torch.exp(-0.5 * torch.divide(batch**2, center_sigma_sq))

        # Evaluate extreme memberships
        mu_left = torch.sigmoid(-sigmoid_slope * (batch + sigmoid_center))
        mu_right = torch.sigmoid(sigmoid_slope * (batch - sigmoid_center))

        # Build membership stack
        memberships = torch.stack((mu_left, mu_center, mu_right), dim=2)

        ## Rule evaluation
        # Take logarithm of memberships.  shape: (batch_size, n_rules)
        log_membership = torch.log(memberships + 1.0e-8).view(batch.shape[0], -1)

        # Compute rule antecedents through matrix multiplication with rule matrix
        log_firing_strength = log_membership @ self.rule_matrix
        # log_firing_strength shape: (batch_size, n_rules)

        # Transform into weights with softmax   shape: (batch_size, n_rules)
        weights = torch.softmax(log_firing_strength, dim=1)

        # Get consequents
        consequents = self._build_consequents_cached()

        # Output shape: (batch_size, )
        return (weights * consequents).sum(dim=1)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self._cached_consequents = None
            self._cached_membership_params = None
        return self
    
    def set_input_scales(self, scales: torch.Tensor | list[float]):
        """
        Registers training scaling factors to rescale future raw data in inference.

        Parameters
        ----------
        scales : torch.Tensor | list[float]
            Per-feature scaling factors
        """
        if isinstance(scales, list):
            scales = torch.tensor(scales, dtype=torch.float32)
        # Expandir de n_unique_features a n_vars usando var_indices
        scales_expanded = scales[self.var_indices]  # (n_vars,)
        self.register_buffer("input_scales", scales_expanded)
