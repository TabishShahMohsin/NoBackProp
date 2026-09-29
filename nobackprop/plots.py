"""Make every figure + table from results/*.csv.   python -m nobackprop.plots"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, P = "results", "plots"
ORDER = ["bp", "fa", "dfa", "dtp", "ff", "eqprop", "spsa"]
LABEL = {"bp": "Backprop", "fa": "Feedback Alignment", "dfa": "Direct FA", "dtp": "Diff. Target Prop",
         "ff": "Forward-Forward", "eqprop": "Equilibrium Prop", "spsa": "SPSA (weight pert.)"}
COL = dict(zip(ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#e34948"]))
MRK = dict(zip(ORDER, ["o", "s", "^", "D", "v", "P", "X"]))
DSN = {"mnist": "MNIST", "fmnist": "Fashion-MNIST"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.axisbelow": True, "figure.dpi": 130, "savefig.dpi": 200, "legend.frameon": False})


def _load(name):
    p = os.path.join(R, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def _save(fig, name):
    os.makedirs(P, exist_ok=True)
    fig.savefig(os.path.join(P, name), bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def _methods(df):
    return [m for m in ORDER if m in set(df.method)]


def _legend(fig, ax, ncol=4):
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, -0.02))


def fig_main(df):
    dss = [d for d in DSN if d in set(df.dataset)]
    # --- learning curves (val acc) and acc vs MACs
    for key, xcol, xlabel, fname in [("val_acc", "epoch", "epoch", "fig_learning_curves.png"),
                                     ("val_acc", "cum_macs", "training compute (multiply-accumulates, per run)", "fig_acc_vs_compute.png")]:
        fig, axs = plt.subplots(1, len(dss), figsize=(5.2 * len(dss), 3.8), squeeze=False)
        for ax, ds in zip(axs[0], dss):
            for m in _methods(df):
                g = df[(df.dataset == ds) & (df.method == m) & (df.epoch > 0)].groupby(xcol)[key].agg(["mean", "std"]).reset_index()
                ax.plot(g[xcol], g["mean"], color=COL[m], marker=MRK[m], ms=3.5, lw=1.8, label=LABEL[m])
                ax.fill_between(g[xcol], g["mean"] - g["std"].fillna(0), g["mean"] + g["std"].fillna(0), color=COL[m], alpha=0.12, lw=0)
            ax.set_title(DSN[ds]); ax.set_xlabel(xlabel); ax.set_ylabel("validation accuracy")
            ax.set_ylim(0.85 if ds == "mnist" else 0.75, 1.0)   # runs below this (e.g. SPSA) fall off the axis; see table
            if xcol == "cum_macs":
                ax.set_xscale("log")
        _legend(fig, axs[0][0], 4)
        _save(fig, fname)
    # --- gradient alignment over training
    if "cos_all" in df:
        fig, axs = plt.subplots(1, len(dss), figsize=(5.2 * len(dss), 3.8), squeeze=False)
        for ax, ds in zip(axs[0], dss):
            for m in _methods(df):
                sub = df[(df.dataset == ds) & (df.method == m)].dropna(subset=["cos_all"])
                if m == "bp" or sub.empty:
                    continue
                g = sub.groupby("epoch")["cos_all"].agg(["mean", "std"]).reset_index()
                ax.plot(g.epoch, g["mean"], color=COL[m], marker=MRK[m], ms=3.5, lw=1.8, label=LABEL[m])
                ax.fill_between(g.epoch, g["mean"] - g["std"].fillna(0), g["mean"] + g["std"].fillna(0), color=COL[m], alpha=0.12, lw=0)
            ax.axhline(0, color="#898781", lw=0.8)
            ax.set_title(DSN[ds]); ax.set_xlabel("epoch"); ax.set_ylabel("cos( update, true gradient )")
            ax.set_ylim(-0.1, 1.02)
        _legend(fig, axs[0][0], 3)
        _save(fig, "fig_gradient_alignment.png")
        # per-layer alignment at the last epoch
        last = df[df.epoch == df.epoch.max()]
        lc = [c for c in df.columns if c.startswith("cos_l")]
        fig, axs = plt.subplots(1, len(dss), figsize=(5.2 * len(dss), 3.6), squeeze=False)
        for ax, ds in zip(axs[0], dss):
            ms = [m for m in _methods(df) if m not in ("bp", "ff")]
            w = 0.8 / len(ms)
            for i, m in enumerate(ms):
                v = last[(last.dataset == ds) & (last.method == m)][lc].mean().values
                ax.bar(np.arange(len(lc)) + i * w - 0.4 + w / 2, v, width=w * 0.9, color=COL[m], label=LABEL[m])
            ax.set_xticks(range(len(lc))); ax.set_xticklabels([f"layer {i + 1}" + (" (output)" if i == len(lc) - 1 else "") for i in range(len(lc))])
            ax.set_title(DSN[ds] + " - end of training"); ax.set_ylabel("cos( update, true gradient )"); ax.set_ylim(-0.1, 1.05)
        _legend(fig, axs[0][0], 3)
        _save(fig, "fig_alignment_per_layer.png")
    # --- cost
    fig, axs = plt.subplots(1, 2, figsize=(9.4, 3.6))
    ds = dss[0]
    ms = _methods(df)
    last = df[(df.dataset == ds) & (df.epoch > 0)]
    mac = last.groupby("method").macs_per_sample.mean().reindex(ms) / 1e6
    tim = last.groupby("method").epoch_time_s.mean().reindex(ms)
    for ax, v, yl in [(axs[0], mac, "training MACs per sample (millions)"), (axs[1], tim, "wall-clock per epoch (s)")]:
        ax.bar(range(len(ms)), v.values, color=[COL[m] for m in ms])
        ax.set_yscale("log"); ax.set_ylabel(yl)
        ax.set_xticks(range(len(ms))); ax.set_xticklabels([LABEL[m].split(" (")[0] for m in ms], rotation=35, ha="right")
        for i, val in enumerate(v.values):
            ax.text(i, val * 1.08, f"{val:.3g}", ha="center", fontsize=8, color=MUTED)
    _save(fig, "fig_compute_cost.png")


def fig_depth(df):
    dss = [d for d in DSN if d in set(df.dataset)]
    fig, axs = plt.subplots(1, 2 * len(dss), figsize=(5 * 2 * len(dss), 3.6), squeeze=False)
    for j, ds in enumerate(dss):
        last = df[(df.dataset == ds) & (df.epoch == df.epoch.max())]
        for m in _methods(df):
            g = last[last.method == m].groupby("depth")
            a = g.test_acc.agg(["mean", "std"]).reset_index()
            axs[0][2 * j].errorbar(a.depth, a["mean"], a["std"].fillna(0), color=COL[m], marker=MRK[m], ms=4, lw=1.8, capsize=2, label=LABEL[m])
            if m != "bp" and "cos_all" in last:
                c = g.cos_all.agg(["mean", "std"]).reset_index()
                axs[0][2 * j + 1].errorbar(c.depth, c["mean"], c["std"].fillna(0), color=COL[m], marker=MRK[m], ms=4, lw=1.8, capsize=2)
        axs[0][2 * j].set_ylim(0, 1.0); axs[0][2 * j].set_title(f"{DSN[ds]}: accuracy vs depth"); axs[0][2 * j].set_xlabel("hidden layers"); axs[0][2 * j].set_ylabel("test accuracy")
        axs[0][2 * j + 1].set_title(f"{DSN[ds]}: alignment vs depth"); axs[0][2 * j + 1].set_xlabel("hidden layers"); axs[0][2 * j + 1].set_ylabel("cos( update, true gradient )")
        axs[0][2 * j + 1].set_ylim(-0.1, 1.02)
    _legend(fig, axs[0][0], 5)
    _save(fig, "fig_depth.png")


def fig_datafrac(df):
    dss = [d for d in DSN if d in set(df.dataset)]
    fig, axs = plt.subplots(1, len(dss), figsize=(5.2 * len(dss), 3.7), squeeze=False)
    for ax, ds in zip(axs[0], dss):
        for m in _methods(df):
            a = df[(df.dataset == ds) & (df.method == m)].groupby("n_train").best_val_test_acc.agg(["mean", "std"]).reset_index()
            ax.errorbar(a.n_train, a["mean"], a["std"].fillna(0), color=COL[m], marker=MRK[m], ms=4, lw=1.8, capsize=2, label=LABEL[m])
        ax.set_xscale("log"); ax.set_title(DSN[ds]); ax.set_xlabel("training examples"); ax.set_ylabel("test accuracy")
    _legend(fig, axs[0][0], 4)
    _save(fig, "fig_data_efficiency.png")


def fig_spsa(df):
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    cols = ["#2a78d6", "#1baf7a", "#eda100", "#e34948"]
    for c, (w, g) in zip(cols, df.groupby("width")):
        a = g.groupby("k").cos.agg(["mean", "std"]).reset_index()
        d = int(g.d.iloc[0])
        ax.errorbar(a.k, a["mean"], a["std"].fillna(0), color=c, marker="o", ms=4, lw=1.8, capsize=2, label=f"d = {d:,}")
        kk = np.logspace(0, np.log10(a.k.max()), 50)
        ax.plot(kk, np.sqrt(kk / (d + kk)), color=c, lw=1, ls="--")
    ax.plot([], [], color="#52514e", ls="--", lw=1, label=r"theory $\sqrt{k/(d+k)}$ (isotropic)")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("perturbation directions k"); ax.set_ylabel("cos( SPSA estimate, true gradient )")
    ax.legend(fontsize=8)
    _save(fig, "fig_spsa_variance.png")


def fig_eqprop(df):
    fig, ax = plt.subplots(figsize=(5.2, 3.7))
    for col, c, lab in [("cos_onesided", "#eb6834", "one-sided nudge (+beta)"), ("cos_centered", "#4a3aa7", "centered nudge (+/-beta)")]:
        a = df.groupby("beta")[col].agg(["mean", "std"]).reset_index()
        ax.errorbar(a.beta, a["mean"], a["std"].fillna(0), color=c, marker="o", ms=4, lw=1.8, capsize=2, label=lab)
    ax.set_xscale("log"); ax.invert_xaxis(); ax.set_xlabel(r"nudging strength $\beta$  (smaller $\rightarrow$)")
    ax.set_ylabel("cos( EqProp update, BPTT gradient )"); ax.set_ylim(0, 1.02); ax.legend(fontsize=8)
    _save(fig, "fig_eqprop_beta.png")


def tables(df):
    dss = [d for d in DSN if d in set(df.dataset)]
    last = df[df.epoch == df.epoch.max()]
    rows = []
    for ds in dss:
        for m in _methods(df):
            g = df[(df.dataset == ds) & (df.method == m)]
            fin = g[g.epoch == g.epoch.max()]
            bestv = g.loc[g.groupby("seed").val_acc.idxmax()]
            tgt = 0.90
            reach = g[g.val_acc >= tgt].groupby("seed").epoch.min()
            rows.append(dict(dataset=DSN[ds], method=LABEL[m], params=int(fin.n_params.iloc[0]),
                             test_acc=100 * fin.test_acc.mean(), test_std=100 * fin.test_acc.std(),
                             train_acc=100 * fin.train_acc.mean(), gap=100 * (fin.train_acc - fin.test_acc).mean(),
                             epochs_to_90=reach.mean() if len(reach) == g.seed.nunique() else np.nan,
                             macs_per_sample_M=g[g.epoch > 0].macs_per_sample.mean() / 1e6,
                             sec_per_epoch=g[g.epoch > 0].epoch_time_s.mean(),
                             cos_final=fin.cos_all.mean() if "cos_all" in fin else np.nan,
                             n_seeds=g.seed.nunique()))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(R, "summary.csv"), index=False)
    with open(os.path.join(R, "summary.md"), "w") as f:
        f.write(t.round(3).to_markdown(index=False))
    with open(os.path.join(R, "summary.tex"), "w") as f:
        f.write("\\begin{tabular}{llrrrrrr}\\toprule\nDataset & Method & Test acc. (\\%) & Train--test gap & Epochs to 90\\% & MACs/sample (M) & s/epoch & $\\cos$ to grad.\\\\\\midrule\n")
        for _, r in t.iterrows():
            f.write(f"{r.dataset} & {r.method} & ${r.test_acc:.2f}\\pm{r.test_std:.2f}$ & {r.gap:.2f} & {r.epochs_to_90:.1f} & {r.macs_per_sample_M:.2f} & {r.sec_per_epoch:.1f} & {r.cos_final:.2f}\\\\\n")
        f.write("\\bottomrule\\end{tabular}\n")
    print(t.round(3).to_string())


def main():
    df = _load("main.csv")
    if df is not None:
        fig_main(df); tables(df)
    for name, fn in [("depth.csv", fig_depth), ("datafrac.csv", fig_datafrac), ("spsa_var.csv", fig_spsa), ("eqprop_beta.csv", fig_eqprop)]:
        d = _load(name)
        if d is not None:
            fn(d)


if __name__ == "__main__":
    main()
