from ..common import Algorithm, Net, bp_grads


class BP(Algorithm):
    name, label = "bp", "Backprop"
    plausible = dict(weight_transport="yes", local="no", needs_backward_pass="yes")

    def __init__(self, sizes, seed, device, **hp):
        super().__init__(sizes, seed, device, **hp)
        self.net = Net(sizes, seed, device, act="relu", init="he")

    def grads(self, X, y):
        return bp_grads(self.net, X, y)
