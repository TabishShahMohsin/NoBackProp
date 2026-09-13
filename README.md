# NoBackProp

# Comparison Framework: Backpropagation vs. Biologically Plausible Alternatives

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

# Envisioned File Tree
```
NoBackProp/
├── environment.yml
├── README.md
├── nobackprop/                 # importable package — the ONE source of truth for algorithm code
│   ├── data.py                 # dataset loading, consistent across all experiments
│   ├── models.py                # shared MLP/CNN architectures
│   ├── algorithms/
│   │   ├── backprop.py
│   │   ├── feedback_alignment.py
│   │   ├── forward_forward.py
│   │   ├── target_propagation.py
│   │   ├── weight_perturbation.py
│   │   └── equilibrium_propagation.py
│   └── metrics.py               # accuracy, FLOPs, memory tracking — shared, so every algorithm reports comparably
├── configs/                     # one YAML per experiment: algorithm + hyperparams + seed + dataset
│   ├── ff_digits_baseline.yaml
│   └── wp_digits_16dir.yaml
├── experiments/
│   └── run.py                   # `python run.py --config configs/ff_digits_baseline.yaml`
├── results/
│   └── all_runs.csv             # one row per run: config hash, algorithm, seed, final_acc, flops, wall_time...
├── notebooks/                   # ONLY exploration + plotting from results/, never training code
│   ├── 01_explore_ff.ipynb
│   └── 02_compare_all_methods.ipynb
└── tests/
    └── test_algorithms.py       # sanity checks — this is what would've caught your TP double-backward bug immediately
```