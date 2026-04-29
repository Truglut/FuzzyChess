import torch.nn as nn
import torch



class TaperedEvaluator(nn.Module):
    def __init__(self, mg_evaluator: nn.Module, eg_evaluator: nn.Module):
        """
        Orchestrates tapered evaluation using a middlegame evaluator and
        and endgame evaluator.
        """
        super().__init__()

        self.mg_evaluator = mg_evaluator
        self.eg_evaluator = eg_evaluator

    def forward(
        self, x_mg: torch.Tensor, x_eg: torch.Tensor, phase: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the final tapered eval.
        """

        mg_score = self.mg_evaluator(x_mg)
        eg_score = self.eg_evaluator(x_eg)

        tapered_score = phase * mg_score + (1.0 - phase) * eg_score
        return tapered_score
