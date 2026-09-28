"""
Backprop baseline training run.

This is the reference every other method (FA, EqProp, ...) gets
compared against on: final test accuracy, convergence speed (epochs
to reach a given accuracy), wall-clock time per epoch, and gradient
norm trajectory.

Run from the bp-project/ directory:
    python3 src/train_backprop.py
"""
import time
import csv
import numpy as np

from data import load_mnist, iterate_minibatches
from model import MLP, cross_entropy_loss

SEED = 0
BATCH_SIZE = 64
LR = 0.1
EPOCHS = 20
RESULTS_CSV = "results/backprop_mnist.csv"


def evaluate(model, X, y_onehot):
    probs, _ = model.forward(X)
    loss = cross_entropy_loss(probs, y_onehot)
    preds = np.argmax(probs, axis=1)
    labels = np.argmax(y_onehot, axis=1)
    acc = np.mean(preds == labels)
    return loss, acc


def main():
    rng = np.random.default_rng(SEED)
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist()
    model = MLP(seed=SEED)

    rows = []
    print(f"{'epoch':>5} {'train_loss':>11} {'train_acc':>10} {'val_loss':>9} {'val_acc':>8} {'grad_norm':>10} {'time_s':>7}")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        epoch_grad_norms = []

        for X_batch, y_batch in iterate_minibatches(X_train, y_train, BATCH_SIZE, rng):
            probs, cache = model.forward(X_batch)
            grads_W, grads_b = model.backward_bp(cache, y_batch)
            epoch_grad_norms.append(model.flat_grad_norm(grads_W, grads_b))
            model.apply_grads(grads_W, grads_b, LR)

        elapsed = time.time() - t0
        train_loss, train_acc = evaluate(model, X_train, y_train)
        val_loss, val_acc = evaluate(model, X_val, y_val)
        mean_grad_norm = float(np.mean(epoch_grad_norms))

        print(f"{epoch:5d} {train_loss:11.4f} {train_acc:10.4f} {val_loss:9.4f} {val_acc:8.4f} {mean_grad_norm:10.4f} {elapsed:7.2f}")
        rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "mean_grad_norm": mean_grad_norm,
            "epoch_time_s": elapsed,
        })

    test_loss, test_acc = evaluate(model, X_test, y_test)
    print(f"\nFinal test accuracy: {test_acc:.4f}  (test loss: {test_loss:.4f})")

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + ["test_acc", "test_loss"])
        writer.writeheader()
        for i, row in enumerate(rows):
            row_out = dict(row)
            if i == len(rows) - 1:
                row_out["test_acc"] = test_acc
                row_out["test_loss"] = test_loss
            else:
                row_out["test_acc"] = ""
                row_out["test_loss"] = ""
            writer.writerow(row_out)

    print(f"Logged per-epoch metrics to {RESULTS_CSV}")


if __name__ == "__main__":
    main()