"""Multi-label losses with frequencies estimated from the model-fit split only.

ASL follows Ridnik et al.; DB follows the authors' COCO implementation,
including the divided initialization bias and negative-logit regularization.
The soft-F1 hybrid is a simple differentiable macro-F1 adaptation, rather than
a claim to reproduce every setting of the SigmoidF1 paper.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def asymmetric_loss(logits, targets, gamma_neg=2.0, gamma_pos=0.0, clip=0.0):
    probabilities = logits.float().sigmoid()
    negative = (probabilities - float(clip)).clamp_min(0)
    positive_term = targets * (1 - probabilities).pow(float(gamma_pos)) * F.logsigmoid(logits.float())
    # Without clipping, logsigmoid keeps the hard-negative gradient alive even
    # when sigmoid(z) rounds to 1. With clipping, 1-p+m is bounded below by m.
    negative_log = F.logsigmoid(-logits.float()) if not clip else torch.log1p(-negative)
    negative_term = (1 - targets) * negative.pow(float(gamma_neg)) * negative_log
    return -(positive_term + negative_term).mean()


def soft_macro_f1_loss(logits, targets, epsilon=1e-7):
    """Compute once over an actual head batch; omit labels absent in that batch.

    BCE/ASL still trains every negative. Omitting unsupported soft-F1 terms
    avoids a constant zero-gradient term diluting the meaningful class mean.
    Accumulating gradients of this function is not a larger-batch F1 surrogate.
    """
    probabilities = logits.float().sigmoid()
    support = targets.sum(0)
    present = support > 0
    if not bool(present.any()):
        return logits.sum() * 0.0
    true_positive = (probabilities * targets).sum(0)
    scores = (2 * true_positive + epsilon) / (probabilities.sum(0) + support + epsilon)
    return 1 - scores[present].mean()


def bernoulli_symmetric_kl(first, second):
    """Mean symmetric KL for independent Bernoulli labels (not label softmax)."""
    first, second = first.float(), second.float()
    p, q = first.sigmoid(), second.sigmoid()
    log_p, log_q = F.logsigmoid(first), F.logsigmoid(second)
    log_np, log_nq = F.logsigmoid(-first), F.logsigmoid(-second)
    pq = p * (log_p - log_q) + (1 - p) * (log_np - log_nq)
    qp = q * (log_q - log_p) + (1 - q) * (log_nq - log_np)
    return 0.5 * (pq + qp).mean()


class DistributionBalancedLoss(nn.Module):
    """DB rebalancing + negative tolerant regularization + focal weighting.

    Original COCO constants are used with ordinary shuffled document batches;
    this sampling choice is an explicit adaptation. An all-negative row uses
    the limiting rebalance weight alpha+1, avoiding division by zero.
    """
    def __init__(self, fit_targets, alpha=0.1, beta=10.0, mu=0.2,
                 negative_scale=2.0, initial_bias=0.05, gamma=2.0, balance=2.0):
        super().__init__()
        targets = torch.as_tensor(fit_targets, dtype=torch.float32)
        if targets.ndim != 2 or not len(targets) or not torch.isin(targets, torch.tensor([0., 1.])).all():
            raise ValueError("DB fit_targets must be a nonempty binary matrix")
        counts = targets.sum(0)
        if bool((counts <= 0).any()):
            raise ValueError("DB requires at least one positive per fit label")
        if negative_scale <= 0:
            raise ValueError("negative_scale must be positive")
        frequency = (counts / len(targets)).clamp(1e-6, 1 - 1e-6)
        self.register_buffer("inverse_frequency", counts.reciprocal())
        self.register_buffer("initial_bias", frequency.logit() * float(initial_bias) / float(negative_scale))
        self.alpha, self.beta, self.mu = float(alpha), float(beta), float(mu)
        self.negative_scale, self.gamma, self.balance = float(negative_scale), float(gamma), float(balance)

    def forward(self, logits, targets):
        denominator = (targets * self.inverse_frequency).sum(1, keepdim=True)
        ratio = self.inverse_frequency / denominator.clamp_min(1e-12)
        rebalance = torch.sigmoid(self.beta * (ratio - self.mu)) + self.alpha
        rebalance = torch.where(denominator > 0, rebalance, torch.full_like(rebalance, 1 + self.alpha))
        scale = targets + (1 - targets) * self.negative_scale
        normalized_logits = (logits.float() + self.initial_bias) * scale
        bce = F.binary_cross_entropy_with_logits(normalized_logits, targets, reduction="none")
        focal = (1 - torch.exp(-bce)).pow(self.gamma)
        weights = rebalance * (targets + (1 - targets) / self.negative_scale)
        return (self.balance * focal * weights * bce).mean()


class MultilabelLoss(nn.Module):
    def __init__(self, name, fit_targets, soft_f1_weight=0.2):
        super().__init__()
        targets = torch.as_tensor(fit_targets, dtype=torch.float32)
        counts = targets.sum(0)
        if bool((counts <= 0).any()):
            raise ValueError("Every fit label needs positive examples")
        self.name, self.soft_f1_weight = name, float(soft_f1_weight)
        weight = (((len(targets) - counts) / counts).clamp_min(1).sqrt()).clamp(1, 5)
        self.register_buffer("pos_weight", weight)
        self.db = DistributionBalancedLoss(targets) if name == "db" else None
        if name not in ("bce", "asl", "asl_soft_f1", "db"):
            raise ValueError(f"Unknown loss: {name}")

    def forward(self, logits, targets, head_phase=True):
        targets = targets.float()
        if self.name == "bce":
            return F.binary_cross_entropy_with_logits(logits.float(), targets, pos_weight=self.pos_weight)
        if self.name == "db":
            return self.db(logits, targets)
        loss = asymmetric_loss(logits, targets)
        if self.name == "asl_soft_f1" and head_phase:
            loss = loss + self.soft_f1_weight * soft_macro_f1_loss(logits, targets)
        return loss
