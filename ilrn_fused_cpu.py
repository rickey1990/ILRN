"""CPU float32 opt-in fused ILRM recurrence, preserving reference parameters.

Requires torch, numpy, numba. Fuses the input-only projections and uses
Numba JIT for the recurrent forward/backward. The history remains FFT based.
The reference model's state_dict structure is unchanged by enable_fused_cpu.
"""
from types import MethodType
import numpy as np
import numba as nb
import torch
from torch.nn import functional as F


@nb.njit(parallel=True, cache=True)
def _forward(pre, ww, wr, wc):
    batch, steps, triple = pre.shape
    width = triple // 3
    hs = np.empty((batch, steps, width), np.float32)
    gates_w = np.empty_like(hs)
    gates_r = np.empty_like(hs)
    candidates = np.empty_like(hs)
    for b in nb.prange(batch):
        h = np.zeros(width, np.float32)
        for t in range(steps):
            for j in range(width):
                sw = pre[b, t, j]
                sr = pre[b, t, width + j]
                for k in range(width):
                    sw += ww[j, k] * h[k]
                    sr += wr[j, k] * h[k]
                gates_w[b, t, j] = 1.0 / (1.0 + np.exp(-sw))
                gates_r[b, t, j] = 1.0 / (1.0 + np.exp(-sr))
            for j in range(width):
                z = pre[b, t, 2 * width + j]
                for k in range(width):
                    z += wc[j, k] * (gates_r[b, t, k] * h[k])
                candidate = np.tanh(z)
                candidates[b, t, j] = candidate
                hs[b, t, j] = h[j] + gates_w[b, t, j] * (candidate - h[j])
            for j in range(width):
                h[j] = hs[b, t, j]
    return hs, gates_w, gates_r, candidates


@nb.njit(parallel=True, cache=True)
def _backward(grad, hs, gates_w, gates_r, candidates, ww, wr, wc):
    batch, steps, width = hs.shape
    gp = np.empty((batch, steps, 3 * width), np.float32)
    gww = np.zeros((batch, width, width), np.float32)
    gwr = np.zeros((batch, width, width), np.float32)
    gwc = np.zeros((batch, width, width), np.float32)
    for b in nb.prange(batch):
        following = np.zeros(width, np.float32)
        zeros = np.zeros(width, np.float32)
        for t in range(steps - 1, -1, -1):
            prev = zeros if t == 0 else hs[b, t - 1, :]
            gprev = np.empty(width, np.float32)
            d_c = np.empty(width, np.float32)

            for j in range(width):
                upstream = grad[b, t, j] + following[j]
                w = gates_w[b, t, j]
                candidate = candidates[b, t, j]
                d_c[j] = upstream * w * (1.0 - candidate * candidate)
                gp[b, t, j] = upstream * (candidate - prev[j]) * w * (1.0 - w)
                gprev[j] = upstream * (1.0 - w)
            dm = np.zeros(width, np.float32)
            for j in range(width):
                dc = d_c[j]
                gp[b, t, 2 * width + j] = dc
                for k in range(width):
                    dm[k] += wc[j, k] * dc
                    gwc[b, j, k] += dc * (gates_r[b, t, k] * prev[k])
            for k in range(width):
                r = gates_r[b, t, k]
                gprev[k] += dm[k] * r
                gp[b, t, width + k] = dm[k] * prev[k] * r * (1.0 - r)
            for j in range(width):
                dw = gp[b, t, j]
                dr = gp[b, t, width + j]
                for k in range(width):
                    gprev[k] += ww[j, k] * dw + wr[j, k] * dr
                    gww[b, j, k] += dw * prev[k]
                    gwr[b, j, k] += dr * prev[k]
            following = gprev
    return gp, gww, gwr, gwc


class _FusedRecurrence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pre, ww, wr, wc):
        if pre.device.type != 'cpu' or pre.dtype != torch.float32:
            raise ValueError('Fused recurrence requires CPU float32 inputs')
        arrays = [v.detach().contiguous().numpy() for v in (pre, ww, wr, wc)]
        hs, w, r, candidate = _forward(*arrays)
        output = torch.from_numpy(hs)
        ctx.save_for_backward(output, torch.from_numpy(w), torch.from_numpy(r),
                              torch.from_numpy(candidate), ww, wr, wc)
        return output

    @staticmethod
    def backward(ctx, grad):
        hs, w, r, candidate, ww, wr, wc = ctx.saved_tensors
        arrays = [v.detach().contiguous().numpy() for v in
                  (grad, hs, w, r, candidate, ww, wr, wc)]
        gp, gww, gwr, gwc = _backward(*arrays)
        return (torch.from_numpy(gp), torch.from_numpy(gww.sum(axis=0)),
                torch.from_numpy(gwr.sum(axis=0)), torch.from_numpy(gwc.sum(axis=0)))


def _fused_block_forward(self, x):
    if x.device.type != 'cpu' or x.dtype != torch.float32:
        raise ValueError('The prototype supports CPU float32 only')
    u = self.pre(x)
    p = self.history(u)
    d = self.width
    packed = torch.cat([

        torch.cat((weight[:, :d], weight[:, 2 * d:]), dim=1)
        for weight in (self.Ww, self.Wr, self.Wc)
    ], dim=0)
    bias = torch.cat((self.bw, self.br, self.bc), dim=0)
    pre = F.linear(torch.cat((u, p), dim=-1), packed, bias)
    state = _FusedRecurrence.apply(pre,
                                   self.Ww[:, d:2 * d],
                                   self.Wr[:, d:2 * d],
                                   self.Wc[:, d:2 * d])
    return self.post(x + state)


def enable_fused_cpu(model):
    """Patch only the forward implementations. All state_dict keys remain the same."""
    for block in model.blocks:
        block.forward = MethodType(_fused_block_forward, block)
    return model
