"""
Custom loss functions for handling class imbalance in deep learning models.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection"
    https://arxiv.org/abs/1708.02002
    
    The focal loss is defined as:
    FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)
    
    where:
    - alpha_t is a weighting factor for the class
    - gamma is the focusing parameter (higher gamma → harder examples get more weight)
    - pt is the model's estimated probability for the class with ground truth label
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
        device: Optional[torch.device] = None
    ):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for positive class (0 < alpha < 1).
                   For class imbalance, typically set to inverse frequency.
                   Smaller alpha → weight positive class more.
            gamma: Focusing parameter (gamma >= 0).
                   Higher gamma → focus on hard negatives.
                   gamma=0 reduces to standard cross-entropy.
            reduction: Type of reduction to apply ('mean', 'sum', 'none')
            device: Device to compute on
        """
        super(FocalLoss, self).__init__()
        
        if not (0 < alpha < 1):
            logger.warning(f"Alpha should be in (0, 1), got {alpha}")
        if gamma < 0:
            raise ValueError(f"Gamma should be >= 0, got {gamma}")
        
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.device = device or torch.device("cpu")
        
        logger.info(
            f"Initializing Focal Loss with alpha={alpha}, gamma={gamma}, reduction={reduction}"
        )
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal Loss.
        
        Args:
            inputs: Logits or probabilities from model [batch_size, num_classes].
                   If binary classification, can be [batch_size, 1] or [batch_size].
            targets: Target labels [batch_size]
        
        Returns:
            Scalar loss value (or unreduced loss if reduction='none')
        """
        # Ensure inputs are on the correct device
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        # Handle 1D input for binary classification
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(1)
        
        # Get probabilities
        p = torch.sigmoid(inputs)
        
        # Compute cross entropy loss (without reduction)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float().unsqueeze(1), reduction='none')
        
        # Get the probability of the true class
        p_t = torch.where(targets.unsqueeze(1) == 1, p, 1 - p)
        
        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting for positive class
        alpha_weight = torch.where(targets.unsqueeze(1) == 1, self.alpha, 1 - self.alpha)
        
        # Focal loss: -alpha * (1-p_t)^gamma * log(p_t)
        focal_loss = alpha_weight * focal_weight * ce_loss
        
        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss.squeeze()
    
    def __repr__(self) -> str:
        return f"FocalLoss(alpha={self.alpha}, gamma={self.gamma}, reduction={self.reduction})"


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy Loss for handling class imbalance.
    
    Applies a weight to positive and negative classes:
    L = -[weight_pos * y * log(p) + weight_neg * (1-y) * log(1-p)]
    """
    
    def __init__(
        self,
        pos_weight: float = 1.0,
        reduction: str = "mean",
        device: Optional[torch.device] = None
    ):
        """
        Initialize Weighted BCE Loss.
        
        Args:
            pos_weight: Weight for positive class
            reduction: Type of reduction ('mean', 'sum', 'none')
            device: Device to compute on
        """
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
        self.device = device or torch.device("cpu")
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Weighted BCE Loss.
        
        Args:
            inputs: Logits [batch_size, 1] or [batch_size]
            targets: Target labels [batch_size]
        
        Returns:
            Scalar loss value (or unreduced loss if reduction='none')
        """
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(1)
        
        # Use pos_weight in PyTorch's BCEWithLogitsLoss
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self.pos_weight], device=self.device),
            reduction=self.reduction
        )
        
        return loss_fn(inputs, targets.float().unsqueeze(1))


class CombinedLoss(nn.Module):
    """
    Combination of Focal Loss and L2 regularization for better convergence.
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        l2_weight: float = 1e-5,
        reduction: str = "mean",
        device: Optional[torch.device] = None
    ):
        """
        Initialize Combined Loss.
        
        Args:
            alpha: Focal Loss alpha parameter
            gamma: Focal Loss gamma parameter
            l2_weight: Weight for L2 regularization
            reduction: Type of reduction
            device: Device to compute on
        """
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction, device=device)
        self.l2_weight = l2_weight
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, model: nn.Module = None) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            inputs: Model logits
            targets: Target labels
            model: Model for L2 regularization (optional)
        
        Returns:
            Combined loss value
        """
        focal = self.focal_loss(inputs, targets)
        
        # Add L2 regularization if model provided
        if model is not None and self.l2_weight > 0:
            l2_reg = torch.tensor(0.0, device=inputs.device)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            return focal + self.l2_weight * l2_reg
        
        return focal
