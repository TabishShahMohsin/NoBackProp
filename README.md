# NoBackProp — backprop vs. learning rules that do not use it

A controlled benchmark of **seven** ways to train the *same* small MLP (784-256-128-10) on MNIST and Fashion-MNIST.
Every learning rule is written **by hand (no autograd)** in PyTorch, so each update can be pointed at an equation in the paper.

| key | method | idea | weight transport | backward pass |
|---|---|---|---|---|
| `bp` | Backpropagation | exact chain rule | yes | yes |
| `fa` | Feedback Alignment | fixed random `B` replaces `Wᵀ` | no | yes (random path) |
| `dfa` | Direct Feedback Alignment | output error broadcast straight to every layer | no | no (broadcast) |
| `dtp` | Difference Target Propagation | layer-local targets via learned inverses | no | feedback net |
| `ff` | Forward-Forward | layer-local “goodness” on pos./neg. data | no | no |
| `eqprop` | Equilibrium Propagation | two relaxation phases, Hebbian update | symmetric by construction | no (relaxation) |
| `spsa` | Weight perturbation (SPSA) | zeroth-order estimate from 2k forward passes | no | no |

## What is measured (all comparisons share data split, seeds, optimizer, LR schedule, batch size)

| # | experiment | command stage | figure |
|---|---|---|---|
| 1 | Accuracy, convergence speed, generalisation gap (7 methods x 2 datasets x 3 seeds) | `run main` | `fig_learning_curves`, `summary.md` |
| 2 | **Gradient alignment**: cos(update, true gradient) during training, per layer | `run main` | `fig_gradient_alignment`, `fig_alignment_per_layer` |
| 3 | **Depth scaling** (2-6 hidden layers) | `run depth` | `fig_depth` |
| 4 | **Data efficiency** (2 % - 100 % of the training set, fixed #steps) | `run datafrac` | `fig_data_efficiency` |
| 5 | **Compute cost**: measured multiply-accumulates per sample + wall-clock; accuracy-vs-compute | `run main` | `fig_compute_cost`, `fig_acc_vs_compute` |
| 6 | **SPSA variance**: alignment vs #directions `k` and #parameters `d` (theory: `sqrt(k/(d+k))`) | `run spsa_var` | `fig_spsa_variance` |
| 7 | **EqProp -> gradient as beta -> 0** (vs. BPTT through the relaxation) | `run eqprop_beta` | `fig_eqprop_beta` |

Hyper-parameters are chosen per method by a random search on the **validation** set (`nobackprop/tune.py`), never on test data.

## Results so far (MNIST, 3 seeds, 20 epochs; produced by the commands below, full tables in `results/summary.md`)

| Method | Test acc. (%) | train-test gap | MACs/sample (M) | cos(update, grad) at end (all / hidden layers) |
|---|---|---|---|---|
| Backprop | 98.47 ± 0.06 | 1.5 | 0.50 | 1.00 / 1.00 |
| Equilibrium Prop | 97.91 ± 0.14 | 1.9 | 14.1 | 0.97 / 0.97–0.98 |
| Feedback Alignment | 97.53 ± 0.02 | 2.0 | 0.50 | 0.96 / 0.37, 0.36 |
| Direct FA | 97.50 ± 0.08 | 2.2 | 0.47 | 0.84 / 0.50, 0.37 |
| Diff. Target Prop | 95.96 ± 0.07 | 1.6 | 0.64 | 0.66 / 0.26, 1.00* |
| Forward-Forward (3x500) | 94.35 ± 0.16 | 0.4 | 3.57 | n/a |
| SPSA (k=16) | 90.76 ± 0.12 | -0.6 | 7.51 | 0.006 |

\* DTP's top hidden layer receives a true gradient step by construction.
Depth (2 -> 6 hidden layers, MNIST test acc.): BP 98.3 -> 98.2, FA 97.1 -> 89.2, DFA 97.2 -> 92.3, DTP 95.4 -> 26.0, EqProp 98.0 -> 97.3 (4 layers), unstable at 6.
**Fashion-MNIST has not been run yet** (the sandbox used for development cannot download it): run `bash run_all.sh fmnist` on Colab/Kaggle.

## Quick start

```bash
git clone https://github.com/TabishShahMohsin/NoBackProp.git && cd NoBackProp
pip install -r requirements.txt            # torch, numpy, pandas, matplotlib, tabulate, pytest
python tests/test_algorithms.py            # ~10 s: manual BP == autograd, FA/DFA/SPSA/EqProp sanity checks
```

### Everything in one go (Colab / Kaggle GPU, ~1-2 h; laptop CPU ~5 h)

```bash
bash run_all.sh mnist fmnist               # tests -> tuning -> all experiments -> plots/ + results/summary.*
```

### Or stage by stage

```bash
# 0. (optional) re-tune hyper-parameters on the validation set  -> configs/hparams_<dataset>.json
python -m nobackprop.tune --dataset mnist  --trials 10 --epochs 5
python -m nobackprop.tune --dataset fmnist --trials 10 --epochs 5

# 1./2./5. main benchmark (also logs gradient alignment + compute cost every epoch)   -> results/main.csv
python -m nobackprop.run main --datasets mnist fmnist --seeds 0 1 2 --epochs 20

# 3. depth scaling                                                                  -> results/depth.csv
python -m nobackprop.run depth --datasets mnist fmnist --methods bp fa dfa dtp eqprop --seeds 0 1 --epochs 8 --depths 2 3 4 6

# 4. data efficiency                                                                -> results/datafrac.csv
python -m nobackprop.run datafrac --datasets mnist fmnist --seeds 0 1 --epochs 10 --fracs 0.02 0.1 0.5 1.0

# 6./7. estimator-quality studies                                                   -> results/spsa_var.csv, results/eqprop_beta.csv
python -m nobackprop.run spsa_var    --datasets mnist --seeds 0 1 2
python -m nobackprop.run eqprop_beta --datasets mnist --seeds 0 1 2

# figures (plots/*.png) + tables (results/summary.md / .csv / .tex)
python -m nobackprop.plots
```

Run a single method quickly: `python -m nobackprop.run main --datasets mnist --methods ff --seeds 0 --epochs 5 --out results/tmp.csv`

### Google Colab / Kaggle
Open `notebooks/colab_run_all.ipynb` in Colab (Runtime -> GPU), or on Kaggle: *New notebook -> Add data / GitHub -> GPU T4* and paste the same cells.
MNIST is fetched from the URL in the original README; Fashion-MNIST is fetched from the official Zalando repo mirror.
If your machine cannot download Fashion-MNIST, put the four `*-ubyte.gz` files into `data/fashion/`.

## Layout

```
nobackprop/
  common.py            flat-buffer MLP, exact BP gradient, Adam/SGD, MAC counter (all matmuls go through mm())
  data.py              MNIST / Fashion-MNIST, fixed 50k/10k/10k split, pixels in [0,1]
  algorithms/          backprop.py  feedback_alignment.py (FA, DFA)  target_prop.py  forward_forward.py
                       equilibrium_prop.py  weight_perturbation.py
  metrics.py           gradient alignment (cosine to the true gradient)
  run.py               training loop + experiment CLI      tune.py  random search on validation set
  plots.py             all figures and tables
configs/hparams*.json  tuned hyper-parameters              tests/  correctness checks
results/               CSVs, logs, summary tables         plots/  figures for the paper
src/                   the original NumPy backprop baseline (kept for reference)
```

## Fairness notes (state these in the paper)

* Same MLP 784-256-128-10, same init family, same optimizer (Adam or SGD+momentum chosen per method on validation), same cosine LR decay, batch 128, 20 epochs.
* **DTP** uses tanh hidden units (its inverse maps need bounded activations) and its top hidden layer receives a true gradient step.
* **Forward-Forward** has no output layer: it uses 3 x 500 hidden units, label embedded in the first 10 pixels, and classifies by summed goodness over the 10 candidate labels.
* **EqProp** uses hard-sigmoid units and a squared-error cost on the output state; predictions are read from the free-phase equilibrium.
* **SPSA** uses k = 16 antithetic directions per step (32 forward passes/step), so per-epoch it costs far more forward passes than the others -> see accuracy-vs-compute plot.
* Gradient alignment compares each rule to exact BP on the same weights (for EqProp: BPTT through the relaxation, the quantity EP estimates). FF has no comparable gradient and is excluded from that plot.

---

## Comparison framework: Backpropagation vs. Biologically Plausible Alternatives

This document outlines the axes along which each training algorithm (Backpropagation, Feedback Alignment, Target Propagation, Equilibrium Propagation, Forward-Forward, Hebbian/STDP, Weight Perturbation/SPSA) should be compared, both mathematically and experimentally.

## 1. Core Update Rules

| Method | Update Rule | Notes |
|---|---|---|
| Backpropagation | $\Delta W_l = -\eta\, \delta_l\, a_{l-1}^T$, $\delta_l = (W_{l+1}^T \delta_{l+1}) \odot f'(z_l)$ | Exact gradient via chain rule; requires weight transport |
| Feedback Alignment | $\delta_l = (B_l \delta_{l+1}) \odot f'(z_l)$ | $B_l$ fixed random matrix, no weight transport |
| Target Propagation | $\hat{h}_{l-1} = g_l(h_l, \hat{h}_l)$, $\Delta W_l \propto (\hat{h}_l - h_l)\, a_{l-1}^T$ | Propagates targets via approximate inverses |
| Equilibrium Propagation | $\Delta W_{ij} \propto \frac{1}{\beta}(s_i^\beta s_j^\beta - s_i^0 s_j^0)$ | Energy-based; converges to true gradient as $\beta \to 0$ |
| Forward-Forward | $\mathcal{L}_l = \log(1 + e^{-y(G_l - \theta)})$, $G_l = \sum_j y_{l,j}^2$ | Per-layer local objective, no backward pass |
| Hebbian / STDP | $\Delta w_{ij} = \eta\, x_i x_j$ (Hebb); $\Delta w_{ij} = \eta \sum W(\Delta t)$ (STDP) | Purely local, correlational |
| Weight Perturbation / SPSA | $\hat{g} = \frac{L(\theta+\epsilon v) - L(\theta)}{\epsilon} v$, $v \sim \mathcal{N}(0,I)$ | Unbiased, high-variance stochastic estimator |

## 2. Comparison Axes

- [ ] **Gradient alignment** — $\cos\angle(\hat{g}, g_{BP})$ between each method's update and the true BP gradient, tracked over training
- [ ] **Bias vs. variance** — is the estimator unbiased (EqProp as $\beta \to 0$, SPSA) or systematically biased (FA, FF)? How does variance scale with parameter count $d$?
- [ ] **Convergence rate** — iterations to reach a fixed loss threshold; for EqProp, also convergence of the inner relaxation dynamics to equilibrium (spectral radius of the Jacobian)
- [ ] **Computational complexity per update** — BP: $O(N)$ forward + $O(N)$ backward. SPSA: $O(N)$ forward, no backward, but 2+ passes needed. EqProp: $O(T \cdot N)$ for $T$ settling steps
- [ ] **Locality** — spatial locality (does the rule need info from non-adjacent layers?) and temporal locality (does it need the full forward trace stored?)
- [ ] **Weight transport requirement** — binary: does the method require forward and backward paths to share weights?
- [ ] **Scalability with depth** — does gradient alignment degrade as layer count increases? (known FA failure mode in deep/conv nets)
- [ ] **Fixed-point / stability conditions** — existence and uniqueness of equilibrium, Lyapunov argument for convergence of settling dynamics (EqProp and other energy-based methods)
- [ ] **Sample / statistical efficiency** — loss or accuracy vs. number of gradient/forward evaluations, not just vs. epochs
- [ ] **Generalization gap** — test-train accuracy gap under each method; some biologically plausible methods regularize implicitly
- [ ] **Energy / hardware cost** — FLOPs, memory bandwidth, and mapping to neuromorphic/analog hardware

## 3. Key Theoretical Results to Anchor the Paper

- **Equilibrium Propagation**: provably converges to the true BP gradient as $\beta \to 0$ (Scellier & Bengio)
- **Feedback Alignment**: gradient alignment is not guaranteed from initialization — $\cos\angle(\hat{g}, g_{BP})$ increases empirically over training (Lillicrap et al.)

This gives a natural organizing contrast: **provably convergent** (EqProp) vs. **empirically convergent** (FA) methods.

