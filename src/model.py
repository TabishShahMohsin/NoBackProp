"""
Shared MLP used by every training method in the comparison.

Architecture is fixed across methods (784 -> 256 -> 128 -> 10, ReLU
hidden units, softmax output) so that BP, FA, EqProp, etc. are all
training the *same* network. Only the weight-update rule (backward_*)
differs between methods.

All math is plain NumPy, deliberately not autograd, so every gradient
here is one you can point to a specific line in the write-up's math
table and say "this is that equation."
"""
import numpy as np

LAYER_SIZES = [784, 256, 128, 10]


def relu(z):
    return np.maximum(0, z)


def relu_grad(z):
    return (z > 0).astype(z.dtype)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy_loss(probs, y_onehot):
    eps = 1e-9
    return -np.mean(np.sum(y_onehot * np.log(probs + eps), axis=1))


class MLP:
    """3-layer MLP: 784 -> 256 -> 128 -> 10."""

    def __init__(self, sizes=LAYER_SIZES, seed=0):
        rng = np.random.default_rng(seed)
        self.sizes = sizes
        self.num_layers = len(sizes) - 1  # number of weight matrices
        self.W = []
        self.b = []
        for l in range(self.num_layers):
            fan_in, fan_out = sizes[l], sizes[l + 1]
            # He initialization, appropriate for ReLU hidden units
            scale = np.sqrt(2.0 / fan_in)
            self.W.append(rng.normal(0, scale, size=(fan_in, fan_out)).astype(np.float32))
            self.b.append(np.zeros((1, fan_out), dtype=np.float32))

    def forward(self, X):
        """Forward pass. Returns (probs, cache) where cache holds every
        intermediate activation/pre-activation needed for the backward pass.
        """
        a = X
        activations = [a]      # a[0] = X, a[l] = output of layer l after nonlinearity
        pre_activations = []   # z[l] = pre-activation of layer l
        for l in range(self.num_layers):
            z = a @ self.W[l] + self.b[l]
            pre_activations.append(z)
            if l < self.num_layers - 1:
                a = relu(z)
            else:
                a = softmax(z)
            activations.append(a)
        cache = {"activations": activations, "pre_activations": pre_activations}
        return activations[-1], cache

    def backward_bp(self, cache, y_onehot):
        """Exact backpropagation gradient (the reference every other
        method in this project is measured against).

        delta_L = probs - y   (softmax + cross-entropy combined gradient)
        delta_l = (W_{l+1} @ delta_{l+1}) * relu'(z_l)   for hidden layers
        dW_l = a_{l-1}^T @ delta_l / batch_size
        db_l = mean(delta_l, axis=0)
        """
        activations = cache["activations"]
        pre_activations = cache["pre_activations"]
        m = y_onehot.shape[0]

        grads_W = [None] * self.num_layers
        grads_b = [None] * self.num_layers

        probs = activations[-1]
        delta = (probs - y_onehot) / m  # (batch, 10)

        for l in reversed(range(self.num_layers)):
            a_prev = activations[l]  # input to this layer
            grads_W[l] = a_prev.T @ delta
            grads_b[l] = delta.sum(axis=0, keepdims=True)
            if l > 0:
                z_prev = pre_activations[l - 1]
                delta = (delta @ self.W[l].T) * relu_grad(z_prev)

        return grads_W, grads_b

    def apply_grads(self, grads_W, grads_b, lr):
        for l in range(self.num_layers):
            self.W[l] -= lr * grads_W[l]
            self.b[l] -= lr * grads_b[l]

    def predict(self, X):
        probs, _ = self.forward(X)
        return np.argmax(probs, axis=1)

    def flat_grad_norm(self, grads_W, grads_b):
        """L2 norm of the full gradient vector, flattened across all
        layers. Used later for gradient-alignment comparisons against
        non-backprop methods (cos angle between this and e.g. FA's update).
        """
        total = 0.0
        for gw, gb in zip(grads_W, grads_b):
            total += np.sum(gw ** 2) + np.sum(gb ** 2)
        return np.sqrt(total)