#!/usr/bin/env bash
# Full pipeline. Usage:  bash run_all.sh [mnist fmnist]      (default: both datasets)
# Rough wall-clock on a free Colab/Kaggle GPU: ~1-2 h total. On a laptop CPU: ~4-6 h.
set -e
DS=${@:-mnist fmnist}
python -m pytest -q tests                                                            # sanity checks (seconds)
for d in $DS; do python -m nobackprop.tune --dataset $d --trials 10 --epochs 5; done  # hyper-params chosen on VALIDATION set
python -m nobackprop.run main     --datasets $DS --seeds 0 1 2 --epochs 20            # Exp 1 + 2 + 5: accuracy, alignment, cost
python -m nobackprop.run depth    --datasets $DS --methods bp fa dfa dtp eqprop --seeds 0 1 --epochs 8 --depths 2 3 4 6   # Exp 3
python -m nobackprop.run datafrac --datasets $DS --seeds 0 1 --epochs 10 --fracs 0.02 0.1 0.5 1.0                          # Exp 4
python -m nobackprop.run spsa_var --datasets mnist --seeds 0 1 2                                                            # Exp 6
python -m nobackprop.run eqprop_beta --datasets mnist --seeds 0 1 2                                                       # Exp 7
python -m nobackprop.plots                                                            # figures -> plots/, tables -> results/
