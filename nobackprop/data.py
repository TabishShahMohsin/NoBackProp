"""Dataset loading. Every method uses these exact tensors: 50k/10k/10k train/val/test, pixels in [0,1], shape (N,784).

MNIST         -> data/mnist_raw.pkl.gz (same file/URL as the original repo README)
Fashion-MNIST -> downloaded IDX files under data/fashion/ (or place the 4 *.gz files there by hand)
"""
import gzip, os, pickle, urllib.request
import numpy as np
import torch

MNIST_URL = "https://raw.githubusercontent.com/mnielsen/neural-networks-and-deep-learning/master/data/mnist.pkl.gz"
FASHION_URLS = ["https://github.com/zalando-research/fashion-mnist/raw/master/data/fashion/",
                "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"]
FASHION_FILES = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                 "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]


def _download(url, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    urllib.request.urlretrieve(url, dst)


def _load_mnist(root):
    path = os.path.join(root, "mnist_raw.pkl.gz")
    if not os.path.exists(path):
        _download(MNIST_URL, path)
    with gzip.open(path, "rb") as f:
        tr, va, te = pickle.load(f, encoding="latin1")
    return (tr[0].astype(np.float32), tr[1]), (va[0].astype(np.float32), va[1]), (te[0].astype(np.float32), te[1])


def _idx(path):
    with gzip.open(path, "rb") as f:
        raw = f.read()
    if raw[2] == 0x08 and raw[3] == 3:      # images
        n, r, c = (int.from_bytes(raw[4 + 4 * i:8 + 4 * i], "big") for i in range(3))
        return np.frombuffer(raw, np.uint8, offset=16).reshape(n, r * c)
    return np.frombuffer(raw, np.uint8, offset=8)


def _load_fashion(root):
    d = os.path.join(root, "fashion")
    for fn in FASHION_FILES:
        p = os.path.join(d, fn)
        if os.path.exists(p) and os.path.getsize(p) > 1000:
            continue
        for base in FASHION_URLS:
            try:
                _download(base + fn, p)
                if os.path.getsize(p) > 1000:
                    break
            except Exception:
                pass
        else:
            raise RuntimeError(f"Could not download {fn}. Put the four Fashion-MNIST *.gz files into {d}/ manually.")
    Xtr, ytr = _idx(os.path.join(d, FASHION_FILES[0])), _idx(os.path.join(d, FASHION_FILES[1]))
    Xte, yte = _idx(os.path.join(d, FASHION_FILES[2])), _idx(os.path.join(d, FASHION_FILES[3]))
    Xtr, Xte = Xtr.astype(np.float32) / 255.0, Xte.astype(np.float32) / 255.0
    perm = np.random.RandomState(0).permutation(60000)         # fixed split, identical for all runs
    tr, va = perm[:50000], perm[50000:]
    return (Xtr[tr], ytr[tr]), (Xtr[va], ytr[va]), (Xte, yte)


def load_dataset(name, device="cpu", root="data", frac=1.0):
    """Returns dict with Xtr,ytr,Xva,yva,Xte,yte (torch tensors on `device`). frac<1 subsamples the train set (first frac*50k)."""
    (a, b, c) = {"mnist": _load_mnist, "fmnist": _load_fashion}[name](root)
    n = int(round(frac * len(a[0])))
    t = lambda x, dt: torch.as_tensor(np.ascontiguousarray(x)).to(device=device, dtype=dt)
    return dict(Xtr=t(a[0][:n], torch.float32), ytr=t(a[1][:n], torch.long),
                Xva=t(b[0], torch.float32), yva=t(b[1], torch.long),
                Xte=t(c[0], torch.float32), yte=t(c[1], torch.long))


if __name__ == "__main__":
    for nm in ["mnist", "fmnist"]:
        try:
            d = load_dataset(nm)
            print(nm, {k: tuple(v.shape) for k, v in d.items()}, float(d["Xtr"].min()), float(d["Xtr"].max()))
        except Exception as e:
            print(nm, "FAILED:", e)
