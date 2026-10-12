"""Minimal numerical and checkpoint-compatibility test; run after installation."""
import copy
import numpy as np
import numba
import torch
from torch.nn import functional as F
from ilrn_reference import StackedILRN
from ilrn_fused_cpu import enable_fused_cpu


def verify(context=32, batch=2):
    torch.manual_seed(20261011)
    torch.set_num_threads(2)
    numba.set_num_threads(2)
    original = StackedILRN()
    accelerated = enable_fused_cpu(copy.deepcopy(original))
    x = torch.randint(0, 1024, (batch, context))
    y = torch.randint(0, 1024, (batch, context))
    result = {}
    # Check identical parameter identifiers and shapes before gradient comparisons.
    assert original.state_dict().keys() == accelerated.state_dict().keys()
    for label, model in [('normal', original), ('fused', accelerated)]:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, 1024), y.reshape(-1))
        loss.backward()
        gradients = {k: v.grad.detach().clone()
                     for k, v in model.named_parameters()}
        result[label] = (logits.detach(), gradients, float(loss.detach()))
    a, ga, la = result['normal']
    b, gb, lb = result['fused']
    max_logit = (a-b).abs().max().item()
    max_grad = max((ga[k]-gb[k]).abs().max().item() for k in ga)
    agree = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    assert torch.allclose(a, b, atol=3e-4, rtol=3e-4)
    assert max_grad < 3e-4
    assert abs(la-lb) < 3e-4
    # Check one optimising step from equal initial weights.
    oa = torch.optim.AdamW(original.parameters(), lr=0.001, weight_decay=0.01)
    ob = torch.optim.AdamW(accelerated.parameters(), lr=0.001, weight_decay=0.01)
    oa.step(); ob.step()
    weight_diff = max((dict(original.named_parameters())[k] -
                       dict(accelerated.named_parameters())[k]).abs().max().item()
                      for k in ga)
    assert weight_diff < 3e-4, 'Large one-step optimiser drift'
    print(f'context={context} logits max diff={max_logit:.3g}; '
          f'grad max diff={max_grad:.3g}; top-1 agreement={agree:.2%}; '
          f'one-step parameter max diff={weight_diff:.3g}')
    return max_logit, max_grad, weight_diff


if __name__ == '__main__':
    verify(32, 2)
    verify(96, 2)
