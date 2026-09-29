"""Small random search per method, selected on the VALIDATION set (never the test set).

  python -m nobackprop.tune --dataset mnist --trials 12 --epochs 5
writes configs/hparams_<dataset>.json  (picked up automatically by run.py)
"""
import argparse, json, math, os, random
import numpy as np
from .algorithms import ALGORITHMS
from .data import load_dataset
from .run import get_device, train

LOGU = lambda lo, hi: (lambda r: 10 ** r.uniform(math.log10(lo), math.log10(hi)))
CHOICE = lambda *c: (lambda r: r.choice(c))

SPACE = {
    "bp":     dict(lr=LOGU(3e-4, 1e-2)),
    "fa":     dict(lr=LOGU(3e-4, 1e-2)),
    "dfa":    dict(lr=LOGU(3e-4, 1e-2)),
    "dtp":    dict(lr=LOGU(3e-4, 1e-2), alpha=CHOICE(0.03, 0.1, 0.3, 1.0), sigma=CHOICE(0.05, 0.1, 0.2), fb_lr_mult=CHOICE(1.0, 3.0)),
    "ff":     dict(lr=LOGU(5e-4, 5e-3), theta=CHOICE(1.0, 2.0, 3.0), label_scale=CHOICE(1.0, 2.0)),
    "eqprop": dict(lr=LOGU(3e-4, 3e-3), beta=CHOICE(0.25, 0.5, 0.75), init_scale=CHOICE(0.35, 0.5, 0.7)),
    "spsa":   dict(lr=LOGU(1e-4, 2e-3), opt=CHOICE("sgd"), momentum=CHOICE(0.5, 0.9), eps=CHOICE(1e-2, 3e-2), clip_norm=CHOICE(0.0, 10.0, 30.0)),
}
FIXED = {"eqprop": dict(T_free=30, T_nudge=10), "spsa": dict(k=16)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--methods", nargs="+", default=list(ALGORITHMS))
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--subset", type=float, default=0.5, help="fraction of the train set used for tuning")
    ap.add_argument("--seed", type=int, default=100)
    a = ap.parse_args()
    dev = get_device()
    data = load_dataset(a.dataset, dev, frac=a.subset)
    data_spsa = load_dataset(a.dataset, dev, frac=0.25)   # SPSA: 20 epochs on a quarter of the data (it needs a long horizon to expose divergence)
    out_path = f"configs/hparams_{a.dataset}.json"
    best_all = json.load(open(out_path)) if os.path.exists(out_path) else {}
    log = []
    for m in a.methods:
        rng = random.Random(a.seed)
        best = (-1, None)
        n_trials = 8 if m == "spsa" else a.trials
        for t in range(n_trials):
            hp = {k: f(rng) for k, f in SPACE[m].items()}
            hp.update(FIXED.get(m, {}))
            rows = (train(m, data_spsa, a.seed, 20, dict(hp), align=False, verbose=False) if m == "spsa"
                    else train(m, data, a.seed, a.epochs, dict(hp), align=False, verbose=False))
            va = rows[-1]["val_acc"]
            print(f"{m:7s} trial {t:2d} val={va:.4f} {hp}", flush=True)
            log.append(dict(method=m, trial=t, val_acc=va, **{k: str(v) for k, v in hp.items()}))
            if va > best[0]:
                best = (va, hp)
        best_all[m] = best[1]
        print(f"==> {m}: best val {best[0]:.4f}  {best[1]}", flush=True)
        json.dump(best_all, open(out_path, "w"), indent=1)
    import pandas as pd
    csv = f"results/tuning_{a.dataset}.csv"
    new = pd.DataFrame(log)
    if os.path.exists(csv):
        old = pd.read_csv(csv)
        new = pd.concat([old[~old.method.isin(new.method.unique())], new])
    new.to_csv(csv, index=False)


if __name__ == "__main__":
    main()
