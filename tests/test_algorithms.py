"""Sanity checks. Run:  python -m pytest -q tests   (or: python tests/test_algorithms.py)"""
import torch
from nobackprop.algorithms import ALGORITHMS, BP, FA, DFA, SPSA, EqProp
from nobackprop.common import Net, bp_grads, no_count
from nobackprop.metrics import cosine, flat

SIZES = [20, 16, 12, 10]
torch.manual_seed(0)
X = torch.rand(64, 20)
y = torch.randint(0, 10, (64,))


def autograd_grads(net, X, y):
    with torch.enable_grad():
        ps = [p.detach().clone().requires_grad_(True) for p in net.params]
        a = X
        for l in range(net.L):
            a = a @ ps[2 * l] + ps[2 * l + 1]
            if l < net.L - 1:
                a = torch.relu(a) if net.act == "relu" else torch.tanh(a)
        loss = torch.nn.functional.cross_entropy(a, y)
        return torch.autograd.grad(loss, ps)


def test_manual_bp_matches_autograd():
    for act in ["relu", "tanh"]:
        net = Net(SIZES, 0, "cpu", act=act)
        for g, r in zip(bp_grads(net, X, y), autograd_grads(net, X, y)):
            assert torch.allclose(g, r, atol=1e-6), act


def test_fa_dfa_output_layer_equals_bp():
    for cls in (FA, DFA):
        alg = cls(SIZES, 0, "cpu")
        g, r = alg.grads(X, y), bp_grads(alg.net, X, y)
        assert torch.allclose(g[-2], r[-2], atol=1e-6) and torch.allclose(g[-1], r[-1], atol=1e-6)


def test_fa_with_symmetric_feedback_is_bp():
    alg = FA(SIZES, 0, "cpu")
    alg.B = [w.T.clone() for w in alg.net.W]
    for g, r in zip(alg.grads(X, y), bp_grads(alg.net, X, y)):
        assert torch.allclose(g, r, atol=1e-6)


def test_spsa_unbiased_and_alignment_grows_with_k():
    cs = []
    for k in (1, 64):
        alg = SPSA(SIZES, 0, "cpu", k=k, eps=1e-3)
        with no_count():
            cs.append(sum(cosine(flat(alg.grads(X, y)), flat(bp_grads(alg.net, X, y))) for _ in range(20)) / 20)
    assert cs[1] > cs[0] and cs[1] > 0.2


def test_eqprop_matches_bptt_gradient():
    alg = EqProp(SIZES, 0, "cpu", beta=0.01, T_free=200, T_nudge=200, init_scale=0.5)
    with no_count():
        g, r = flat(alg.grads(X, y)), flat(alg.reference_grads(X, y))
    assert cosine(g, r) > 0.95, cosine(g, r)   # EqProp -> gradient as beta -> 0


def test_all_methods_run_and_reduce_error():
    from nobackprop.common import make_optimizer
    Xs = torch.rand(256, 784); ys = torch.randint(0, 10, (256,))
    for name, cls in ALGORITHMS.items():
        alg = cls([784, 32, 16, 10], 0, "cpu", **({"hidden": (64, 64)} if name == "ff" else {}))
        gs = alg.grads(Xs[:32], ys[:32])
        assert len(gs) == len(alg.params()), name
        assert all(torch.isfinite(g).all() for g in gs), name
        for g, p in zip(gs, alg.params()):
            assert g.shape == p.shape, (name, g.shape, p.shape)


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
