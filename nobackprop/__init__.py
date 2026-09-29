"""NoBackProp: backprop vs. non-backprop learning rules, all with manual (non-autograd) updates."""
import torch

# Every learning rule in this package computes its own update by hand.
# Autograd is only switched on locally (metrics.py / tests) to build reference gradients.
torch.set_grad_enabled(False)
