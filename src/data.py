"""
Shared MNIST data loader.

Every training method (backprop, feedback alignment, equilibrium
propagation, ...) must use this exact loader so that comparisons
are apples-to-apples: same train/val/test split, same normalization,
same one-hot encoding, same batching logic.
"""
import gzip
import pickle
import numpy as np

DATA_PATH = "data/mnist_raw.pkl.gz"


def one_hot(y, num_classes=10):
    out = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def load_mnist(path=DATA_PATH):
    """Returns (X_train, y_train, X_val, y_val, X_test, y_test).

    X arrays are float32, shape (N, 784), values in [0, 1].
    y arrays are one-hot float32, shape (N, 10).
    """
    with gzip.open(path, "rb") as f:
        train, val, test = pickle.load(f, encoding="latin1")

    X_train, y_train = train[0].astype(np.float32), one_hot(train[1])
    X_val, y_val = val[0].astype(np.float32), one_hot(val[1])
    X_test, y_test = test[0].astype(np.float32), one_hot(test[1])
    return X_train, y_train, X_val, y_val, X_test, y_test


def iterate_minibatches(X, y, batch_size, rng, shuffle=True):
    """Yields (X_batch, y_batch) pairs for one epoch."""
    n = X.shape[0]
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    for start in range(0, n, batch_size):
        batch_idx = idx[start:start + batch_size]
        yield X[batch_idx], y[batch_idx]


if __name__ == "__main__":
    Xtr, ytr, Xval, yval, Xte, yte = load_mnist()
    print(f"train: {Xtr.shape}, {ytr.shape}")
    print(f"val:   {Xval.shape}, {yval.shape}")
    print(f"test:  {Xte.shape}, {yte.shape}")
    print(f"pixel range: [{Xtr.min()}, {Xtr.max()}]")