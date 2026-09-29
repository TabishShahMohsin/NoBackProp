"""Forward-Forward (Hinton 2022). Every layer has its own local objective; there is no backward pass.
Positive data = image with the CORRECT label written into the first 10 pixels, negative = WRONG label.
goodness g = mean(h^2); loss = softplus(-(g-theta)) for positive, softplus(g-theta) for negative.
"""
import torch
from ..common import Algorithm, mm, no_count


class FF(Algorithm):
    name, label = "ff", "Forward-Forward"
    plausible = dict(weight_transport="no", local="yes (layer-local)", needs_backward_pass="no")

    def __init__(self, sizes, seed, device, hidden=(500, 500, 500), theta=2.0, label_scale=1.0, n_neg=1, **hp):
        super().__init__(sizes, seed, device, **hp)
        self.dims = [sizes[0]] + list(hidden)
        self.theta, self.label_scale, self.n_neg = theta, label_scale, int(n_neg)
        g = torch.Generator().manual_seed(1000 + seed)
        self.W, self.b = [], []
        for l in range(len(self.dims) - 1):
            fi, fo = self.dims[l], self.dims[l + 1]
            self.W.append((torch.randn(fi, fo, generator=g) * (2.0 / fi) ** 0.5).to(device))
            self.b.append(torch.zeros(1, fo, device=device))
        self.net = None
        self.ngen = torch.Generator(device="cpu").manual_seed(4000 + seed)

    def params(self):
        out = []
        for w, b in zip(self.W, self.b):
            out += [w, b]
        return out

    def n_shared(self):
        return 0

    @staticmethod
    def _norm(a):
        # length normalisation (removes the goodness of the previous layer), rescaled to unit mean-square
        return a / (a.pow(2).mean(1, keepdim=True).sqrt() + 1e-4)

    def embed(self, X, lab, n=10):
        Xe = X.clone()
        Xe[:, :n] = 0.0
        Xe[torch.arange(X.shape[0]), lab] = self.label_scale
        return Xe

    def grads(self, X, y):
        m, K = X.shape[0], self.n_neg
        # K negatives per sample: distinct random wrong labels (K=1: one random wrong label)
        if K == 1:
            offs = torch.randint(1, 10, (m, 1), generator=self.ngen).to(y.device)
        else:
            offs = (torch.rand(m, 9, generator=self.ngen).argsort(1)[:, :K] + 1).to(y.device)   # (m,K) distinct, in 1..9
        ap = self.embed(X, y)
        ans = [self.embed(X, (y + offs[:, j]) % 10) for j in range(K)]
        out = []
        for W, b in zip(self.W, self.b):
            xp = self._norm(ap)
            hp = torch.relu(mm(xp, W) + b)
            H = hp.shape[1]
            gp = (hp * hp).mean(1)
            dzp = (-torch.sigmoid(self.theta - gp))[:, None] * (2.0 / H) * hp / (2 * m)   # d softplus(-(g-th))
            gW, gb = mm(xp.T, dzp), dzp.sum(0, keepdim=True)
            new_an = []
            for an in ans:
                xn = self._norm(an)
                hn = torch.relu(mm(xn, W) + b)
                gn = (hn * hn).mean(1)
                dzn = torch.sigmoid(gn - self.theta)[:, None] * (2.0 / H) * hn / (2 * m * K)  # d softplus(g-th)
                gW = gW + mm(xn.T, dzn); gb = gb + dzn.sum(0, keepdim=True)
                new_an.append(hn)
            out += [gW, gb]
            ap, ans = hp, new_an            # detached: no gradient flows between layers
        return out

    def logits(self, X):
        """Goodness of layers 2..L summed, for each of the 10 candidate labels."""
        scores = []
        for lab in range(10):
            a = self.embed(X, torch.full((X.shape[0],), lab, device=X.device, dtype=torch.long))
            tot = torch.zeros(X.shape[0], device=X.device)
            for l, (W, b) in enumerate(zip(self.W, self.b)):
                a = torch.relu(mm(self._norm(a), W) + b)
                if l > 0:
                    tot += (a * a).mean(1)
            scores.append(tot)
        return torch.stack(scores, 1)
