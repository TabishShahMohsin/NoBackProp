import torch
from .common import no_count


def flat(gs):
    return torch.cat([g.reshape(-1) for g in gs])


def cosine(a, b):
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def gradient_alignment(alg, X, y):
    """cos angle between the method's update and its reference gradient (exact BP for MLP methods,
    BPTT-through-the-relaxation for EqProp). Returns dict: cos_all, cos_l0, cos_l1, ... (weights only)."""
    n = alg.n_shared()
    if n == 0:
        return {}
    with no_count():
        g = alg.grads(X, y)[:n]
        r = alg.reference_grads(X, y)[:n]
    out = {"cos_all": cosine(flat(g), flat(r))}
    for l in range(n // 2):
        out[f"cos_l{l}"] = cosine(g[2 * l].reshape(-1), r[2 * l].reshape(-1))
    return out
