"""Difference Target Propagation (Lee et al. 2015).

Forward layers are trained with LOCAL regression losses ||h_l - t_l||^2 towards targets that are
propagated downward through learned approximate inverses g_l:
    t_{L-1} = h_{L-1} - alpha * dL/dh_{L-1}            (only the top hidden layer sees a true gradient)
    t_{l-1} = h_{l-1} + g_l(t_l) - g_l(h_l)             (difference correction)
The inverses g_l are trained by a noisy auto-encoder loss ||g_l(f_l(h+eps)) - (h+eps)||^2.
Hidden units are tanh (the original setting; the inverse maps live in (-1,1)).
"""
import torch
from ..common import Algorithm, Net, mm, softmax, onehot, act_grad


class DTP(Algorithm):
    name, label = "dtp", "Difference Target Prop"
    plausible = dict(weight_transport="no (learned inverses)", local="yes (after target propagation)",
                     needs_backward_pass="yes (feedback net)")

    def __init__(self, sizes, seed, device, alpha=0.1, sigma=0.1, fb_lr_mult=1.0, **hp):
        super().__init__(sizes, seed, device, **hp)
        self.net = Net(sizes, seed, device, act="tanh", init="glorot")
        self.alpha, self.sigma, self.fb_mult = alpha, sigma, fb_lr_mult
        g = torch.Generator().manual_seed(8000 + seed)
        L = self.net.L
        # feedback g_l : a[l+1] -> a[l]   for hidden->hidden weights l = 1..L-2
        self.V, self.c = {}, {}
        for l in range(1, L - 1):
            fi, fo = sizes[l + 1], sizes[l]
            self.V[l] = (torch.randn(fi, fo, generator=g) * (2.0 / (fi + fo)) ** 0.5).to(device)
            self.c[l] = torch.zeros(1, fo, device=device)
        self.fb_order = list(range(1, L - 1))
        self.ngen = torch.Generator(device="cpu").manual_seed(9000 + seed)

    def params(self):
        p = list(self.net.params)
        for l in self.fb_order:
            p += [self.V[l], self.c[l]]
        return p

    def lr_mult(self):
        return [1.0] * (2 * self.net.L) + [self.fb_mult] * (2 * len(self.fb_order))

    def _g(self, l, h):
        return torch.tanh(mm(h, self.V[l]) + self.c[l])

    def grads(self, X, y):
        net = self.net
        L, m = net.L, X.shape[0]
        logits, a, z = net.forward(X)
        p = softmax(logits)
        e = p - onehot(y, net.sizes[-1])
        out = [None] * (2 * L)
        # top layer: exact CE gradient w.r.t. its own weights (local)
        d = e / m
        out[2 * (L - 1)] = mm(a[L - 1].T, d)
        out[2 * (L - 1) + 1] = d.sum(0, keepdim=True)
        # targets
        t = {}
        t[L - 1] = a[L - 1] - self.alpha * mm(e, net.W[L - 1].T)
        for l in range(L - 2, 0, -1):          # t[l] for hidden layer index l (a[l])
            t[l] = a[l] + self._g(l, t[l + 1]) - self._g(l, a[l + 1])
        # local forward losses for hidden layers: layer l-1 produces a[l]
        for l in range(1, L):
            h = a[l]
            delta = (h - t[l]) * act_grad(z[l - 1], h, net.act) / m
            out[2 * (l - 1)] = mm(a[l - 1].T, delta)
            out[2 * (l - 1) + 1] = delta.sum(0, keepdim=True)
        # feedback (inverse) training
        fb = []
        for l in self.fb_order:
            noise = torch.randn(a[l].shape, generator=self.ngen).to(a[l].device) * self.sigma
            xin = a[l] + noise
            hout = torch.tanh(mm(xin, net.W[l]) + net.b[l])
            rec = torch.tanh(mm(hout, self.V[l]) + self.c[l])
            dpre = (rec - xin) * (1 - rec * rec) / m
            fb += [mm(hout.T, dpre), dpre.sum(0, keepdim=True)]
        return out + fb
