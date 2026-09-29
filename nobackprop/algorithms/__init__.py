from .backprop import BP
from .feedback_alignment import FA, DFA
from .target_prop import DTP
from .forward_forward import FF
from .equilibrium_prop import EqProp
from .weight_perturbation import SPSA

ALGORITHMS = {c.name: c for c in [BP, FA, DFA, DTP, FF, EqProp, SPSA]}
