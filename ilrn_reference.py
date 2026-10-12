"""Four-layer ILRN-4-96 mathematical reference.
Not the original news training code.

Requires: torch. Input token tensor: [batch, context].
Outputs logits [batch, context, vocab].
State h is reset per window. No separate streaming inference API is included.
"""
import torch
from torch import nn
from torch.nn import functional as F


class ILRMBlock(nn.Module):
    """Inverse-lag history and the original access/write recurrence."""

    def __init__(self, width=96):
        super().__init__()
        self.width = width
        self.pre = nn.LayerNorm(width)
        self.post = nn.LayerNorm(width)
        # Input order of every gate weight: [u_t ; h_(t-1) ; P_t].
        self.Ww = nn.Parameter(torch.randn(width, 3 * width) * 0.035)
        self.Wr = nn.Parameter(torch.randn(width, 3 * width) * 0.035)
        self.Wc = nn.Parameter(torch.randn(width, 3 * width) * 0.035)
        self.bw = nn.Parameter(torch.zeros(width))
        self.br = nn.Parameter(torch.zeros(width))
        self.bc = nn.Parameter(torch.zeros(width))

    @staticmethod
    def history(u):
        """Causal harmonic 1/k summary. Current token has zero weight."""
        batch, steps, width = u.shape
        q = torch.zeros(steps, device=u.device, dtype=u.dtype)
        if steps > 1:
            lags = torch.arange(1, steps, device=u.device, dtype=u.dtype)
            q[1:] = lags.reciprocal()
        harmonics = torch.cumsum(q, dim=0).clamp_min(1.0)
        n = 1 << ((2 * steps - 2).bit_length())
        transformed = torch.fft.rfft(u.transpose(1, 2), n=n, dim=-1)
        kernel = torch.fft.rfft(q, n=n)
        result = torch.fft.irfft(transformed * kernel, n=n, dim=-1)
        return result[:, :, :steps].transpose(1, 2) / harmonics[None, :, None]

    def forward(self, x):
        u = self.pre(x)
        p = self.history(u)
        batch, steps, width = u.shape
        h = u.new_zeros((batch, width))
        sequence = []
        for t in range(steps):
            z = torch.cat((u[:, t], h, p[:, t]), dim=-1)
            w = torch.sigmoid(F.linear(z, self.Ww, self.bw))
            r = torch.sigmoid(F.linear(z, self.Wr, self.br))
            v = torch.cat((u[:, t], r * h, p[:, t]), dim=-1)
            candidate = torch.tanh(F.linear(v, self.Wc, self.bc))
            h = (1 - w) * h + w * candidate
            sequence.append(h)
        return self.post(x + torch.stack(sequence, dim=1))


class StackedILRN(nn.Module):

    """Four independent ILRM blocks, one embedding, and an untied output head."""

    def __init__(self, vocab=1024, width=96, layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab, width)
        self.blocks = nn.ModuleList([ILRMBlock(width) for _ in range(layers)])
        self.head = nn.Linear(width, vocab)

    def forward(self, tokens):
        x = self.embedding(tokens)
        for layer in self.blocks:
            x = layer(x)
        return self.head(x)


def demo_step(context=256, batch=6, seed=20261011):
    torch.manual_seed(seed)
    model = StackedILRN()
    tokens = torch.randint(0, 1024, (batch, context))
    labels = torch.randint(0, 1024, (batch, context))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    optimizer.zero_grad(set_to_none=True)
    logits = model(tokens)
    loss = F.cross_entropy(logits.reshape(-1, 1024), labels.reshape(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return model, float(loss.detach())


if __name__ == '__main__':
    torch.set_num_threads(2)
    model, loss = demo_step(context=32, batch=2)
    count = sum(p.numel() for p in model.parameters())
    print(f'parameters={count:,}; loss={loss:.4f}')
