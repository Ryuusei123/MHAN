from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class GateRegularizationLoss(nn.Module):
    """Gate Regularization Loss.

    This loss ensures balance between the global and local gate weights.

    Args:
        loss_weight (float): Loss weight for gate regularization loss. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0):
        super(GateRegularizationLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, gate_global, gate_local, **kwargs):
        """
        Args:
            gate_global (Tensor): Global gate weights of shape (N, C, 1, 1).
            gate_local (Tensor): Local gate weights of shape (N, C, 1, 1).

        Returns:
            Tensor: The regularization loss value.
        """
        # Compute L1 loss between global and local gate weights
        loss = torch.mean(torch.abs(gate_global - gate_local))
        return self.loss_weight * loss
