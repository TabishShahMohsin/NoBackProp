"""Feedback Alignment (Lillicrap et al. 2016) and Direct Feedback Alignment (Nokland 2016)."""
import torch
from ..common import Algorithm, Net, mm, softmax, onehot, act_grad


class FA(Algorithm):
    """delta_l = (delta_{l+1} B_{l+1}) * f'(z_l) with FIXED random B (no weight transport)."""
    name, label = "fa", "Feedback Alignment"
    plausible = dict(weight_transport="no", local="partly", needs_backward_pass="yes")

    def __init__(self, sizes, seed, device, **hp):
        super().__init__(sizes, seed, device, **hp)
        self.net = Net(sizes, seed, device, act="relu", init="he")
        g = torch.Generator().manual_seed(5000 + seed)
        # B[l] has the shape of W[l].T, i.e. (out, in); only l>=1 is used
        self.B = [torch.randn(sizes[l + 1], sizes[l], generator=g).to(device) * (2.0 / sizes[l]) ** 0.5
                  for l in range(self.net.L)]

    def grads(self, X, y):
        net = self.net
        logits, acts, zs = net.forward(X)
        m = X.shape[0]
        delta = (softmax(logits) - onehot(y, net.sizes[-1])) / m
        out = [None] * (2 * net.L)
        for l in reversed(range(net.L)):
            out[2 * l] = mm(acts[l].T, delta)
            out[2 * l + 1] = delta.sum(0, keepdim=True)
            if l > 0:
                delta = mm(delta, self.B[l]) * act_grad(zs[l - 1], acts[l], net.act)
        return out


class DFA(Algorithm):
    """delta_l = (e B_l) * f'(z_l): the output error is sent DIRECTLY to every hidden layer."""
    name, label = "dfa", "Direct FA"
    plausible = dict(weight_transport="no", local="partly", needs_backward_pass="no (broadcast)")

    def __init__(self, sizes, seed, device, **hp):
        super().__init__(sizes, seed, device, **hp)
        self.net = Net(sizes, seed, device, act="relu", init="he")
        g = torch.Generator().manual_seed(6000 + seed)
        n_out = sizes[-1]
        self.B = [torch.randn(n_out, sizes[l + 1], generator=g).to(device) / n_out ** 0.5
                  for l in range(self.net.L - 1)]

    def grads(self, X, y):
        net = self.net
        logits, acts, zs = net.forward(X)
        m = X.shape[0]
        e = (softmax(logits) - onehot(y, net.sizes[-1])) / m
        out = [None] * (2 * net.L)
        out[2 * (net.L - 1)] = mm(acts[net.L - 1].T, e)
        out[2 * (net.L - 1) + 1] = e.sum(0, keepdim=True)
        for l in range(net.L - 1):
            delta = mm(e, self.B[l]) * act_grad(zs[l], acts[l + 1], net.act)
            out[2 * l] = mm(acts[l].T, delta)
            out[2 * l + 1] = delta.sum(0, keepdim=True)
        return out
