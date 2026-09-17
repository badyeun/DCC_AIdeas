import numpy as np
import pytest
import torch
from torch.nn import functional as F

from mission3_ver6.losses import (DistributionBalancedLoss, MultilabelLoss, asymmetric_loss,
                                 bernoulli_symmetric_kl, soft_macro_f1_loss)


def test_asl_matches_formula_and_hard_negative_gradient():
    logits = torch.tensor([[-2., 0., 2.], [100., -100., 3.]], requires_grad=True)
    targets = torch.tensor([[0., 1., 0.], [0., 1., 1.]])
    p = logits.sigmoid()
    expected = -(targets * F.logsigmoid(logits) + (1-targets) * p.square() * F.logsigmoid(-logits)).mean()
    actual = asymmetric_loss(logits, targets)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[1, 0] > 0 and logits.grad[1, 1] < 0
    assert logits.grad[0, 0].abs() < logits.grad[0, 2].abs()


def test_db_matches_authors_equations_including_all_negative_rows():
    fit = torch.tensor([[1., 0., 1.], [0., 1., 0.], [1., 1., 0.], [0., 0., 0.]])
    labels = torch.tensor([[1., 0., 1.], [0., 0., 0.]])
    logits = torch.tensor([[.2, -.1, .6], [1., -2., .5]], requires_grad=True)
    criterion = DistributionBalancedLoss(fit)
    frequency = fit.sum(0)
    inverse = 1 / frequency
    ratio = inverse / (labels[0] * inverse).sum()
    row = .1 + torch.sigmoid(10 * (ratio-.2))
    weights = torch.stack([row, torch.full((3,), 1.1)]) * (labels + (1-labels)/2)
    bias = -(len(fit)/frequency-1).log() * .05 / 2
    shifted = (logits+bias) * (labels+(1-labels)*2)
    bce = F.binary_cross_entropy_with_logits(shifted, labels, reduction="none")
    expected = (weights * (1-(-bce).exp()).square() * bce * 2).mean()
    actual = criterion(logits, labels)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(actual) and torch.isfinite(logits.grad).all()
    assert logits.grad[1].abs().sum() > 0
    # A label that is positive in every FIT row still has a finite bias.
    assert torch.isfinite(DistributionBalancedLoss(torch.ones(2, 3))(logits, labels))


def test_soft_f1_is_batch_classwise_and_absent_label_safe():
    logits = torch.tensor([[.1, .2, .3], [-1., .6, -.2]], requires_grad=True)
    labels = torch.tensor([[1., 0., 1.], [0., 0., 1.]])
    p = logits.sigmoid()
    expected = 1 - ((2*(p*labels).sum(0)+1e-7)/(p.sum(0)+labels.sum(0)+1e-7))[[0,2]].mean()
    actual = soft_macro_f1_loss(logits, labels)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert not logits.grad[:,1].any()
    zero = soft_macro_f1_loss(logits, torch.zeros_like(labels))
    assert zero == 0 and zero.requires_grad
    # Per-microbatch F1 is NOT a substitute for the whole head batch.
    separate = sum(soft_macro_f1_loss(x[None], y[None]) for x,y in zip(logits,labels))/2
    assert abs(float(separate.detach()-actual.detach())) > 1e-3


def test_bernoulli_rdrop_matches_distribution_kl_and_both_passes_get_gradient():
    first = torch.tensor([[2., -1., .5]], requires_grad=True)
    second = torch.tensor([[-.5, .4, 1.]], requires_grad=True)
    a,b = torch.distributions.Bernoulli(logits=first),torch.distributions.Bernoulli(logits=second)
    expected = .5*(torch.distributions.kl_divergence(a,b)+torch.distributions.kl_divergence(b,a)).mean()
    actual = bernoulli_symmetric_kl(first,second)
    torch.testing.assert_close(actual,expected)
    assert bernoulli_symmetric_kl(first,first).item() == 0
    actual.backward()
    assert first.grad.abs().sum() > 0 and second.grad.abs().sum() > 0


def test_hybrid_only_adds_softf1_in_actual_head_batch():
    labels = np.array([[1,0,1],[0,1,0],[1,1,0]], dtype=np.float32)
    logits = torch.randn(3,3,requires_grad=True)
    criterion = MultilabelLoss("asl_soft_f1", labels)
    targets = torch.tensor(labels)
    torch.testing.assert_close(criterion(logits,targets,head_phase=False),asymmetric_loss(logits,targets))
    torch.testing.assert_close(criterion(logits,targets),asymmetric_loss(logits,targets)+.2*soft_macro_f1_loss(logits,targets))
    # Pure ASL has no pos_weight multiplier.
    torch.testing.assert_close(MultilabelLoss("asl",labels)(logits,targets),asymmetric_loss(logits,targets))


@pytest.mark.parametrize("name", ["bce","asl","db","asl_soft_f1"])
def test_each_loss_has_finite_nonzero_gradients(name):
    logits = torch.randn(4,3,requires_grad=True)
    labels = torch.tensor([[1.,0,0],[0,1,0],[0,0,1],[0,0,0]])
    loss = MultilabelLoss(name,labels)(logits,labels)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0
