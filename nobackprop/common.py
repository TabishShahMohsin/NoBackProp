"""Shared building blocks: MAC counter, flat-buffer MLP, optimizers, base Algorithm class."""
import math
import numpy as np
import torch


# --------------------------------------------------------------------------
# MAC counter: every matmul in every algorithm goes through mm(), so compute
# cost is *measured* identically for all methods (multiply-accumulates).
# --------------------------------------------------------------------------
class _Counter:
    macs = 0
    on = True


COUNTER = _Counter()


def mm(a, b):
    if COUNTER.on:
        COUNTER.macs += a.shape[0] * a.shape[1] * b.shape[1]
    return a @ b


class no_count:
    """Context manager: temporarily disable MAC counting (evaluation / metrics)."""

    def __enter__(self):
        self.prev = COUNTER.on
        COUNTER.on = False

    def __exit__(self, *a):
        COUNTER.on = self.prev


# --------------------------------------------------------------------------
# Activations (manual derivatives expressed via the *output* where possible)
# --------------------------------------------------------------------------
def act_fwd(z, act):
    if act == "relu":
        return torch.relu(z)
    if act == "tanh":
        return torch.tanh(z)
    raise ValueError(act)


def act_grad(z, h, act):
    """d act(z)/dz given pre-activation z and output h."""
    if act == "relu":
        return (z > 0).to(z.dtype)
    if act == "tanh":
        return 1.0 - h * h
    raise ValueError(act)


def softmax(z):
    return torch.softmax(z, dim=1)


def onehot(y, n=10):
    out = torch.zeros(y.shape[0], n, device=y.device)
    out[torch.arange(y.shape[0], device=y.device), y] = 1.0
    return out


def ce_loss(logits, y):
    return torch.nn.functional.cross_entropy(logits, y).item()


# --------------------------------------------------------------------------
# MLP with all parameters living in ONE flat buffer (W and b are views).
# Convenient for SPSA (perturb the whole vector) and cosine metrics.
# --------------------------------------------------------------------------
class Net:
    def __init__(self, sizes, seed, device, act="relu", init="he", init_scale=1.0):
        self.sizes = list(sizes)
        self.L = len(sizes) - 1
        self.act = act
        g = torch.Generator(device="cpu").manual_seed(1000 + seed)
        d = sum(sizes[l] * sizes[l + 1] + sizes[l + 1] for l in range(self.L))
        self.flat = torch.zeros(d, device=device)
        self.W, self.b = [], []
        off = 0
        for l in range(self.L):
            fi, fo = sizes[l], sizes[l + 1]
            w = self.flat[off:off + fi * fo].view(fi, fo); off += fi * fo
            b = self.flat[off:off + fo].view(1, fo); off += fo
            if init == "he":
                std = math.sqrt(2.0 / fi)
            elif init == "glorot":
                std = math.sqrt(2.0 / (fi + fo))
            else:
                raise ValueError(init)
            w.copy_(torch.randn(fi, fo, generator=g) * std * init_scale)
            self.W.append(w)
            self.b.append(b)

    @property
    def params(self):
        out = []
        for w, b in zip(self.W, self.b):
            out += [w, b]
        return out

    @property
    def d(self):
        return self.flat.numel()

    def forward(self, X, W=None, b=None):
        """Returns (logits, acts, pre_acts). Hidden: act; output layer: linear (softmax applied outside)."""
        W = self.W if W is None else W
        b = self.b if b is None else b
        a = X
        acts, zs = [a], []
        for l in range(self.L):
            z = mm(a, W[l]) + b[l]
            zs.append(z)
            a = act_fwd(z, self.act) if l < self.L - 1 else z
            acts.append(a)
        return acts[-1], acts, zs

    def logits(self, X):
        return self.forward(X)[0]


def bp_grads(net, X, y):
    """Exact backprop gradient of mean cross-entropy w.r.t. [W0,b0,W1,b1,...]."""
    logits, acts, zs = net.forward(X)
    m = X.shape[0]
    delta = (softmax(logits) - onehot(y, net.sizes[-1])) / m
    gW, gb = [None] * net.L, [None] * net.L
    for l in reversed(range(net.L)):
        gW[l] = mm(acts[l].T, delta)
        gb[l] = delta.sum(0, keepdim=True)
        if l > 0:
            delta = mm(delta, net.W[l].T) * act_grad(zs[l - 1], acts[l], net.act)
    out = []
    for l in range(net.L):
        out += [gW[l], gb[l]]
    return out


# --------------------------------------------------------------------------
# Optimizers (manual, operate on lists of tensors + lists of gradients)
# --------------------------------------------------------------------------
class Adam:
    def __init__(self, params, lr, lr_mult=None, betas=(0.9, 0.999), eps=1e-8, wd=0.0):
        self.p, self.lr, self.b1, self.b2, self.eps, self.wd = params, lr, betas[0], betas[1], eps, wd
        self.mult = lr_mult or [1.0] * len(params)
        self.m = [torch.zeros_like(p) for p in params]
        self.v = [torch.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        c1 = 1 - self.b1 ** self.t
        c2 = 1 - self.b2 ** self.t
        for p, g, m, v, k in zip(self.p, grads, self.m, self.v, self.mult):
            if self.wd:
                g = g + self.wd * p
            m.mul_(self.b1).add_(g, alpha=1 - self.b1)
            v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            p.addcdiv_(m / c1, (v / c2).sqrt_().add_(self.eps), value=-self.lr * k)


class SGD:
    def __init__(self, params, lr, lr_mult=None, momentum=0.9, wd=0.0):
        self.p, self.lr, self.mom, self.wd = params, lr, momentum, wd
        self.mult = lr_mult or [1.0] * len(params)
        self.buf = [torch.zeros_like(p) for p in params]

    def step(self, grads):
        for p, g, buf, k in zip(self.p, grads, self.buf, self.mult):
            if self.wd:
                g = g + self.wd * p
            buf.mul_(self.mom).add_(g)
            p.add_(buf, alpha=-self.lr * k)


def make_optimizer(name, params, lr, lr_mult=None, **kw):
    return {"adam": Adam, "sgd": SGD}[name](params, lr, lr_mult=lr_mult, **kw)


# --------------------------------------------------------------------------
# Base class
# --------------------------------------------------------------------------
class Algorithm:
    """A learning rule. Subclasses implement grads() (an *update direction*, in the
    minimisation convention) and logits(). The trainer applies the optimizer."""

    name = "base"
    label = "Base"
    plausible = {}  # qualitative properties for the paper table

    def __init__(self, sizes, seed, device, **hp):
        self.sizes, self.seed, self.device, self.hp = list(sizes), seed, device, hp
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self.net = None  # MLP whose W/b are the "shared" parameters (for alignment metrics)

    # -- to override
    def params(self):
        return self.net.params

    def lr_mult(self):
        return None

    def grads(self, X, y):
        raise NotImplementedError

    def logits(self, X):
        return self.net.logits(X)

    def reference_grads(self, X, y):
        """Gradient this method is *trying* to estimate (default: exact BP on self.net)."""
        return bp_grads(self.net, X, y)

    def n_shared(self):
        """How many entries of params() correspond to net's [W0,b0,...] (for alignment)."""
        return 2 * self.net.L if self.net is not None else 0

    def predict(self, X, bs=5000):
        out = []
        with no_count():
            for i in range(0, X.shape[0], bs):
                out.append(self.logits(X[i:i + bs]).argmax(1))
        return torch.cat(out)
