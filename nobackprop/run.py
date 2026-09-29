"""Training loop + experiment CLI.

  python -m nobackprop.run main      --datasets mnist fmnist --seeds 0 1 2 --epochs 20
  python -m nobackprop.run depth     ...
  python -m nobackprop.run datafrac  ...
  python -m nobackprop.run spsa_var  ...
"""
import argparse, json, math, os, time
import numpy as np
import pandas as pd
import torch

from .algorithms import ALGORITHMS
from .common import COUNTER, make_optimizer, no_count
from .data import load_dataset
from .metrics import gradient_alignment

HP_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "hparams.json")
DEFAULT_SIZES = [784, 256, 128, 10]


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_hparams(dataset="mnist"):
    base = os.path.join(os.path.dirname(HP_PATH), "hparams.json")
    hp = json.load(open(base)) if os.path.exists(base) else {}
    ds = os.path.join(os.path.dirname(HP_PATH), f"hparams_{dataset}.json")
    if os.path.exists(ds):
        hp.update(json.load(open(ds)))
    return hp


def accuracy(alg, X, y):
    return (alg.predict(X) == y).float().mean().item()


def train(method, data, seed=0, epochs=20, hp=None, sizes=None, batch_size=128, align=True,
          align_n=1000, eval_n=10000, verbose=True, max_seconds=None, eval_every=1):
    """Train one method. Returns a list of per-epoch dict rows (row 0 = before training)."""
    hp = dict(hp or {})
    device = data["Xtr"].device
    torch.manual_seed(seed)
    lr, opt_name, wd = hp.pop("lr", 1e-3), hp.pop("opt", "adam"), hp.pop("wd", 0.0)
    sched = hp.pop("sched", "cosine")
    mom = hp.pop("momentum", 0.9)
    clip = hp.pop("clip_norm", 0.0)          # global update-norm clipping (0 = off)
    bs = int(hp.pop("batch_size", batch_size))
    alg = ALGORITHMS[method](sizes or DEFAULT_SIZES, seed, device, **hp)
    opt = make_optimizer(opt_name, alg.params(), lr, lr_mult=alg.lr_mult(), wd=wd, **({"momentum": mom} if opt_name == "sgd" else {}))
    Xtr, ytr = data["Xtr"], data["ytr"]
    N = Xtr.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    d_params = sum(p.numel() for p in alg.params())
    rows, cum_macs, cum_time, step = [], 0, 0.0, 0
    total_steps = epochs * math.ceil(N / bs)
    # fixed diagnostic subsets
    sub = torch.randperm(N, generator=torch.Generator().manual_seed(123))[:min(eval_n, N)].to(device)
    ali = sub[:align_n]

    def log(ep, epoch_time, macs_epoch):
        nonlocal cum_macs
        row = dict(method=method, seed=seed, epoch=ep, n_params=d_params,
                   train_acc=accuracy(alg, Xtr[sub], ytr[sub]), val_acc=accuracy(alg, data["Xva"], data["yva"]),
                   test_acc=accuracy(alg, data["Xte"], data["yte"]), epoch_time_s=epoch_time,
                   cum_time_s=cum_time, cum_macs=cum_macs, macs_per_sample=(macs_epoch / max(N, 1)) if ep else 0)
        if align:
            row.update(gradient_alignment(alg, Xtr[ali], ytr[ali]))
        rows.append(row)
        if verbose:
            print(f"[{method:6s} s{seed}] ep {ep:3d} train {row['train_acc']:.4f} val {row['val_acc']:.4f} "
                  f"test {row['test_acc']:.4f} t={epoch_time:5.1f}s cos={row.get('cos_all', float('nan')):.3f}", flush=True)

    log(0, 0.0, 0)
    for ep in range(1, epochs + 1):
        t0 = time.time()
        macs0 = COUNTER.macs
        perm = torch.randperm(N, generator=g).to(device)
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            if sched == "cosine":                      # same cosine-to-zero LR schedule for every method
                opt.lr = lr * 0.5 * (1 + math.cos(math.pi * step / total_steps))
            step += 1
            grads = alg.grads(Xtr[idx], ytr[idx])
            if not all(torch.isfinite(x).all() for x in grads):
                if verbose:
                    print("non-finite gradient; aborting run")
                break
            if clip:
                gn = torch.sqrt(sum((x * x).sum() for x in grads[:alg.n_shared() or len(grads)]))
                if gn > clip:
                    grads = [x * (clip / gn) for x in grads]
            opt.step(grads)
        else:
            dt = time.time() - t0
            cum_time += dt
            cum_macs += COUNTER.macs - macs0
            if ep % eval_every == 0 or ep == epochs:
                log(ep, dt, COUNTER.macs - macs0)
            if max_seconds and cum_time > max_seconds:
                break
            continue
        break  # non-finite
    return rows


# ---------------------------------------------------------------------------------------------
def _hp(hps, method):
    return dict(hps.get(method, {}))


def exp_main(a):
    dev = get_device(); out = []
    for ds in a.datasets:
        data = load_dataset(ds, dev)
        hps = load_hparams(ds)
        for method in a.methods:
            for seed in a.seeds:
                hp = _hp(hps, method)
                for r in train(method, data, seed, a.epochs, hp, align=not a.no_align):
                    r["dataset"] = ds; out.append(r)
                pd.DataFrame(out).to_csv(a.out, index=False)


def exp_depth(a):
    """Accuracy vs depth (number of hidden layers) for methods that support arbitrary depth."""
    dev = get_device(); out = []
    for ds in a.datasets:
        data = load_dataset(ds, dev); hps = load_hparams(ds)
        for depth in a.depths:
            sizes = [784] + [a.width] * depth + [10]
            for method in a.methods:
                for seed in a.seeds:
                    hp = _hp(hps, method)
                    if method == "eqprop":       # deeper nets need longer relaxation
                        hp["T_free"] = int(hp.get("T_free", 40) * max(1, depth / 2))
                        hp["T_nudge"] = int(hp.get("T_nudge", 15) * max(1, depth / 2))
                    for r in train(method, data, seed, a.epochs, hp, sizes=sizes, align=True):
                        r.update(dataset=ds, depth=depth, width=a.width); out.append(r)
                    pd.DataFrame(out).to_csv(a.out, index=False)


def exp_datafrac(a):
    """Data efficiency: fixed number of gradient steps (= a.epochs epochs of the FULL set), varying train-set fraction."""
    dev = get_device(); out = []
    for ds in a.datasets:
        hps = load_hparams(ds)
        for frac in a.fracs:
            data = load_dataset(ds, dev, frac=frac)
            ep = max(1, int(round(a.epochs / frac)))
            for method in a.methods:
                for seed in a.seeds:
                    rows = train(method, data, seed, ep, _hp(hps, method), align=False, verbose=False, eval_every=max(1, ep // 10))
                    best = max(rows, key=lambda r: r["val_acc"])
                    out.append(dict(dataset=ds, method=method, seed=seed, frac=frac, n_train=data["Xtr"].shape[0],
                                    epochs=ep, final_test_acc=rows[-1]["test_acc"], best_val_test_acc=best["test_acc"]))
                    print(out[-1], flush=True)
                    pd.DataFrame(out).to_csv(a.out, index=False)


def exp_spsa_var(a):
    """cos(SPSA estimate, true gradient) vs number of directions k and parameter count d (at init)."""
    from .algorithms import SPSA
    from .common import bp_grads
    from .metrics import cosine, flat
    dev = get_device(); out = []
    data = load_dataset(a.datasets[0], dev)
    X, y = data["Xtr"][:512], data["ytr"][:512]
    for width in a.widths:
        sizes = [784, width, width, 10]
        for k in a.ks:
            for seed in a.seeds:
                alg = SPSA(sizes, seed, dev, k=k, eps=1e-3)
                with no_count():
                    gt = flat(bp_grads(alg.net, X, y)); ge = flat(alg.grads(X, y))
                out.append(dict(width=width, d=alg.net.d, k=k, seed=seed, cos=cosine(ge, gt),
                                rel_norm=(ge.norm() / gt.norm()).item()))
        print(width, flush=True)
    pd.DataFrame(out).to_csv(a.out, index=False)


def exp_eqprop_beta(a):
    """cos(EqProp update, BPTT gradient) vs nudging strength beta, for one-sided and centered (two-sided) estimates."""
    from .algorithms import EqProp
    from .metrics import cosine, flat
    dev = get_device(); out = []
    data = load_dataset(a.datasets[0], dev)
    X, y = data["Xtr"][:512], data["ytr"][:512]
    for seed in a.seeds:
        for beta in a.betas:
            alg = EqProp(DEFAULT_SIZES, seed, dev, beta=beta, T_free=150, T_nudge=150, init_scale=0.5)
            with no_count():
                ref = flat(alg.reference_grads(X, y))
                cen = flat(alg.grads(X, y))
                # one-sided: (s^beta - s^0)/beta
                Y = torch.nn.functional.one_hot(y, 10).float()
                s0 = alg.free_phase(X); sp = alg._relax(X, [t.clone() for t in s0], 150, alg.net.W, alg.net.b, Y, beta)
                one = []
                for l in range(alg.net.L):
                    lo = (X if l == 0 else sp[l - 1]); lo0 = (X if l == 0 else s0[l - 1])
                    one += [-((lo.T @ sp[l]) - (lo0.T @ s0[l])) / (beta * X.shape[0]), -(sp[l] - s0[l]).sum(0, keepdim=True) / (beta * X.shape[0])]
            out.append(dict(seed=seed, beta=beta, cos_centered=cosine(cen, ref), cos_onesided=cosine(flat(one), ref)))
    pd.DataFrame(out).to_csv(a.out, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["main", "depth", "datafrac", "spsa_var", "eqprop_beta"])
    ap.add_argument("--datasets", nargs="+", default=["mnist"])
    ap.add_argument("--methods", nargs="+", default=list(ALGORITHMS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--depths", nargs="+", type=int, default=[2, 3, 4, 6])
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--fracs", nargs="+", type=float, default=[0.02, 0.1, 0.5, 1.0])
    ap.add_argument("--widths", nargs="+", type=int, default=[16, 32, 64, 128])
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 4, 16, 64, 256])
    ap.add_argument("--betas", nargs="+", type=float, default=[1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01])
    ap.add_argument("--no-align", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.methods == ["all"]:
        a.methods = list(ALGORITHMS)
    a.out = a.out or f"results/{a.exp}.csv"
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    {"main": exp_main, "depth": exp_depth, "datafrac": exp_datafrac, "spsa_var": exp_spsa_var, "eqprop_beta": exp_eqprop_beta}[a.exp](a)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
