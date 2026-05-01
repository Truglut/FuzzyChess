import chess
import torch
from typing import Any

from fuzzychess.evaluation.features.extractors_cache import BoardCache
from fuzzychess.evaluation.engine import EvaluationEngine
from fuzzychess.explainability.explaining_searcher import ExplainingSearcher


class PositionExplainer:
    """
    Orchestrator class to explain the chess engine's evaluations and decisions.
    Combines detailed evaluation (forward_verbose) with explaining search.
    """

    def __init__(self, engine: EvaluationEngine, searcher: ExplainingSearcher):
        self.engine = engine
        self.searcher = searcher

    def _get_linguistic_label(self, feature_type: str, label_idx: int) -> str:
        """
        Maps the membership tensor index to friendly, understandable labels
        """
        if feature_type == "per_color":
            return {
                0: "Low",
                1: "Medium",
                2: "High",
            }.get(label_idx, "Unknown")
        elif feature_type == "diff":
            return {0: "Better for black", 1: "Equal", 2: "Better for white"}.get(
                label_idx, "Unknown"
            )
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")

    def _get_all_linguistic_labels(self, feature_type: str) -> list[str]:
        if feature_type == "per_color":
            return ["Low", "Medium", "High"]
        elif feature_type == "diff":
            return ["Better for black", "Equal", "Better for white"]
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")

    def analyze_static_features(
        self, board: chess.Board, phase: str = "MG"
    ) -> dict[str, Any]:
        # Get positional features
        raw_features = self.engine.extract_features_dict(board)
        if phase == "MG":
            fis_entries = self.engine.mg_fis_entries
            n_features = self.engine._n_features_mg
            evaluator = self.engine.evaluator.mg_evaluator
        else:
            fis_entries = self.engine.eg_fis_entries
            n_features = self.engine._n_features_eg
            evaluator = self.engine.evaluator.eg_evaluator
        features_dict = raw_features[phase]

        cache = BoardCache.from_board(board)

        # Build features for board evaluation
        feature_tensor = self.engine._extract_features(cache, fis_entries, n_features)
        evaluation_results = evaluator.forward_verbose(feature_tensor)

        # Extract results
        evaluation = evaluation_results["score"].item()
        memberships = evaluation_results["memberships"][0]  # [0] to extract from batch
        weights = evaluation_results["weights"][0]  # [0] to extract from batch
        consequents = evaluation_results["consequents"]
        rule_matrix = evaluation_results["rule_matrix"]

        # Feature analysis
        feature_analysis = dict()
        features_list = list(features_dict.keys())
        for i, key in enumerate(features_list):
            feature_name, feature_type = key
            raw_value = features_dict[key]

            mfs = memberships[i].tolist()
            dominant_idx = mfs.index(max(mfs))
            dominant_label = self._get_linguistic_label(feature_type, dominant_idx)

            feature_analysis[feature_name] = {
                "score": raw_value,
                "dominant_label": dominant_label,
                "memberships": dict(
                    zip(self._get_all_linguistic_labels(feature_type), mfs)
                ),
            }

        # Rule Analysis
        # Flatten weights and consequents to avoid errors
        weights_flat = evaluation_results["weights"].flatten()
        consequents_flat = evaluation_results["consequents"].flatten()

        contributions = (weights_flat * consequents_flat).tolist()
        weights_list = weights_flat.tolist()
        consequents_list = consequents_flat.tolist()

        rules_info = []
        for rule_idx in range(len(weights_list)):
            # Skip rules with weight 0
            if weights_list[rule_idx] < 1e-6:
                continue

            # Recover antecedent
            conditions = []
            col = rule_matrix[:, rule_idx]
            active_indices = torch.nonzero(col).squeeze(-1).tolist()

            # If active_indices is not list, make it list
            if not isinstance(active_indices, list):
                active_indices = [active_indices]

            for idx in active_indices:
                var_idx = idx // 3
                label_idx = idx % 3

                if var_idx < len(features_list):
                    name, f_type = features_list[var_idx]
                    label_name = self._get_linguistic_label(f_type, label_idx)
                    conditions.append(f"[{name} is {label_name}]")

            rule_text = " AND ".join(conditions)
            rules_info.append(
                {
                    "antecedent_text": rule_text,
                    "weight": weights_list[rule_idx],
                    "consequent": consequents_list[rule_idx],
                    "contribution": contributions[rule_idx],
                }
            )

        return {
            "evaluation": evaluation,
            "feature_analysis": feature_analysis,
            "rule_analysis": rules_info,
        }

    def _print_static_analysis(self, analysis_dict: dict[str, Any], phase: str = "MG"):
        if phase == "MG":
            phase_str = "Middle game"
        else:
            phase_str = "Endgame"

        print(f"--- {phase_str} analysis ---")
        print(f"Evaluation: {analysis_dict["evaluation"]}")
        print("\nFeature analysis:")

        for name, data in analysis_dict["feature_analysis"].items():
            print(f"\n{name}:")
            print(f"\t- Score:          {data["score"]:.2f}")
            print(f"\t- Dominant label: {data["dominant_label"]}")
            print(f"\t- Membership distribution:")
            for label, value in data["memberships"].items():
                print(f"\t\t- {label}: {value:.2f}")

        rules_info = analysis_dict["rule_analysis"]
        # Separate and sort top rules
        # Rules pushing eval up (>0.05 consequents) sorted by contribution
        top_white = sorted(
            [r for r in rules_info if r["consequent"] > 0.05],
            key=lambda x: x["contribution"],
            reverse=True,
        )[:3]

        # Rules pushing eval down (< -0.05 consequents) sorted by most negative contribution
        top_black = sorted(
            [r for r in rules_info if r["consequent"] < -0.05],
            key=lambda x: x["contribution"],
        )[:3]

        # Equalizing rules (consequents near 0) sorted by their weight (how much they fire)
        top_equal = sorted(
            [r for r in rules_info if abs(r["consequent"]) <= 0.05],
            key=lambda x: x["weight"],
            reverse=True,
        )[:3]

        def print_rule_group(title, rules, is_equal=False):
            print(f"\n{title}:")
            if not rules:
                print("\tNone active.")
            for i, r in enumerate(rules):
                # For equal rules, we highlight the weight (firing strength) instead of contribution
                impact_str = (
                    f"Weight: {r['weight']:.2f}"
                    if is_equal
                    else f"Contribution: {r['contribution']:+.2f}"
                )
                print(f"\t{i+1}. {r['antecedent_text']}")
                print(f"\t   -> {impact_str} (Consequent: {r['consequent']:+.2f})")

        print_rule_group("Top rules favoring white", top_white)
        print_rule_group("Top rules favoring black", top_black)
        print_rule_group("Top equalizing rules", top_equal, is_equal=True)

    def print_full_analysis(self, board: chess.Board):
        mg_analysis = self.analyze_static_features(board, "MG")
        eg_analysis = self.analyze_static_features(board, "EG")
        game_phase = self.engine._calculate_phase(board)
        mg_evaluation = mg_analysis["evaluation"]
        eg_evaluation = eg_analysis["evaluation"]
        global_evaluation = (
            game_phase * mg_evaluation + (1.0 - game_phase) * eg_evaluation
        )
        print(f"Board evaluation: {global_evaluation}.")
        print(f"Middle game evaluation: {mg_evaluation}.")
        print(f"End game evaluation: {eg_evaluation}.")
        print(f"Game phase: {game_phase}.", end=" ")
        if game_phase < 0.35:
            print("Endgame.")
        elif game_phase > 0.65:
            print("Opening/Middle game")
        else:
            print("Middle game to endgame transition.")

        print("\n")

        self._print_static_analysis(mg_analysis, phase="MG")
        print("\n")
        self._print_static_analysis(eg_analysis, phase="EG")
