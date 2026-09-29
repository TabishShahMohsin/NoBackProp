"""SPSA / weight perturbation: unbiased, high-variance zeroth-order gradient estimate (forward passes only)."""
import torch
from ..common import Algorithm, Net, mm, softmax, onehot


class SPSA(Algorithm):
    """g_hat = (1/k) sum_i [L(th + eps v_i) - L(th - eps v_i)] / (2 eps) * v_i,  v_i ~ N(0, I_d)."""
    name, label = "spsa", "Weight Perturbation (SPSA)"
    plausible = dict(weight_transport="no", local="no (global scalar loss)", needs_backward_pass="no")

    def __init__(self, sizes, seed, device, k=16, eps=1e-3, **hp):
        super().__init__(sizes, seed, device, **hp)
        self.net = Net(sizes, seed, device, act="relu", init="he")
        self.k, self.eps = int(k), eps
        self.dgen = torch.Generator(device="cpu").manual_seed(7000 + seed)

    def _loss(self, X, y):
        logits = self.net.logits(X)
        return torch.nn.functional.cross_entropy(logits, y).item()

    def grads(self, X, y):
        net = self.net
        flat = net.flat
        g = torch.zeros_like(flat)
        orig = flat.clone()
        for _ in range(self.k):
            v = torch.randn(flat.shape, generator=self.dgen).to(flat.device)
            flat.copy_(orig).add_(v, alpha=self.eps)
            lp = self._loss(X, y)
            flat.copy_(orig).add_(v, alpha=-self.eps)
            lm = self._loss(X, y)
            g.add_(v, alpha=(lp - lm) / (2 * self.eps))
        flat.copy_(orig)
        g /= self.k
        # split flat gradient into per-parameter views matching net.params
        out, off = [], 0
        for p in net.params:
            n = p.numel()
            out.append(g[off:off + n].view_as(p)); off += n
        return out
