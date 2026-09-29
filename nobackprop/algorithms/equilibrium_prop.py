"""Equilibrium Propagation (Scellier & Bengio 2017), layered Hopfield-style network, hard-sigmoid units.

Energy  E(s) = 1/2 sum s^2 - sum_l rho(s_{l-1}) W_l rho(s_l) - sum_l b_l rho(s_l),   rho(s)=clip(s,0,1)
Dynamics (fixed-point form):  s_l <- rho( b_l + s_{l-1} W_l + s_{l+1} W_{l+1}^T  [+ beta (y - s_L) at the output] )
Free phase -> equilibrium s^0;  nudged phases with +/-beta -> s^{+beta}, s^{-beta}.
Weight update (centered / symmetric EP):   dW_l = 1/(2 beta) ( s_{l-1}^{+} s_l^{+} - s_{l-1}^{-} s_l^{-} )
which estimates -dC/dW,  C = 1/2 ||s_L - y||^2.   Symmetric W is used in both directions (by construction).
"""
import torch
from ..common import Algorithm, Net, mm, onehot


def rho(s):
    return s.clamp(0.0, 1.0)


class EqProp(Algorithm):
    name, label = "eqprop", "Equilibrium Prop"
    plausible = dict(weight_transport="symmetric weights by construction", local="yes (Hebbian, two phases)",
                     needs_backward_pass="no (relaxation)")

    def __init__(self, sizes, seed, device, beta=0.5, T_free=40, T_nudge=15, damp=1.0, init_scale=1.0, **hp):
        super().__init__(sizes, seed, device, **hp)
        self.net = Net(sizes, seed, device, act="relu", init="glorot", init_scale=init_scale)
        self.beta, self.T_free, self.T_nudge, self.damp = beta, int(T_free), int(T_nudge), damp

    # ---- dynamics
    def _step(self, x, s, W, b, y=None, beta=0.0):
        L = len(W)
        new = []
        for l in range(L):
            below = x if l == 0 else s[l - 1]
            pre = mm(below, W[l]) + b[l]
            if l < L - 1:
                pre = pre + mm(s[l + 1], W[l + 1].T)
            elif beta != 0.0:
                pre = pre + beta * (y - s[l])
            new.append(rho(pre) if self.damp == 1.0 else (1 - self.damp) * s[l] + self.damp * rho(pre))
        return new

    def _relax(self, x, s, T, W, b, y=None, beta=0.0):
        for _ in range(T):
            s = self._step(x, s, W, b, y, beta)
        return s

    def _init_state(self, x, W, b):
        s, a = [], x
        for l in range(len(W)):
            a = rho(mm(a, W[l]) + b[l])
            s.append(a)
        return s

    def free_phase(self, x, W=None, b=None, T=None):
        W = self.net.W if W is None else W
        b = self.net.b if b is None else b
        return self._relax(x, self._init_state(x, W, b), self.T_free if T is None else T, W, b)

    def logits(self, X):
        return self.free_phase(X)[-1]

    # ---- learning
    def grads(self, X, y):
        net = self.net
        W, b, L, m = net.W, net.b, net.L, X.shape[0]
        Y = onehot(y, net.sizes[-1])
        s0 = self.free_phase(X)
        sp = self._relax(X, [t.clone() for t in s0], self.T_nudge, W, b, Y, +self.beta)
        sm = self._relax(X, [t.clone() for t in s0], self.T_nudge, W, b, Y, -self.beta)
        out = []
        for l in range(L):
            bp_, bm_ = (X if l == 0 else sp[l - 1]), (X if l == 0 else sm[l - 1])
            dW = (mm(bp_.T, sp[l]) - mm(bm_.T, sm[l])) / (2 * self.beta * m)
            db = (sp[l] - sm[l]).sum(0, keepdim=True) / (2 * self.beta * m)
            out += [-dW, -db]        # minimisation convention: grad = -(EP ascent direction)
        return out

    def reference_grads(self, X, y):
        """Exact gradient of C = 1/2||s_L - y||^2 through the free-phase dynamics (BPTT on the
        unrolled relaxation). Equilibrium Propagation is a local estimator of THIS quantity."""
        net = self.net
        Y = onehot(y, net.sizes[-1])
        with torch.enable_grad():
            W = [w.detach().clone().requires_grad_(True) for w in net.W]
            b = [v.detach().clone().requires_grad_(True) for v in net.b]
            sL = self.free_phase(X, W, b)[-1]
            C = 0.5 * ((sL - Y) ** 2).sum(1).mean()
            grads = torch.autograd.grad(C, W + b)
        out = []
        for l in range(net.L):
            out += [grads[l], grads[net.L + l]]
        return out
