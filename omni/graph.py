import torch
import torch.nn as nn


def small_world(n, k=2, p=0.2, seed=0):
    g = torch.Generator().manual_seed(seed)
    adj = torch.zeros(n, n)
    for i in range(n):
        for d in range(1, k + 1):
            adj[i, (i + d) % n] = 1.0
            adj[i, (i - d) % n] = 1.0
    for i in range(n):
        for j in range(n):
            if i != j and adj[i, j] == 1.0 and torch.rand(1, generator=g).item() < p:
                other = int(torch.randint(0, n, (1,), generator=g).item())
                if other != i:
                    adj[i, j] = 0.0
                    adj[i, other] = 1.0
    return adj


class MessagePassing(nn.Module):
    def __init__(self, n_experts, adj, rounds, beta):
        super().__init__()
        self.register_buffer("adj", adj)
        self.W = nn.Parameter(adj * 0.05)
        self.rounds = rounds
        self.beta = beta

    def forward(self, per, ifl):
        S, n, D = per.shape
        active = torch.zeros(S, n, dtype=torch.bool, device=per.device)
        active.scatter_(1, ifl, True)
        for _ in range(self.rounds):
            new = per.clone()
            for i in range(n):
                nbrs = (self.adj[i] > 0).nonzero(as_tuple=False).squeeze(-1)
                for j in nbrs.tolist():
                    mask = active[:, j]
                    if mask.any():
                        new[mask, i] = new[mask, i] + self.beta * self.W[i, j] * per[mask, j]
            per = new
        return per


class HebbianBank:
    def __init__(self, n_experts, lr, decay, strength=0.1):
        self.E = torch.zeros(n_experts, n_experts)
        self.n = n_experts
        self.lr = lr
        self.decay = decay
        self.strength = strength

    def update(self, ifl):
        if self.lr <= 0.0:
            return
        S, k = ifl.shape
        with torch.no_grad():
            self.E = self.E * self.decay
            for a in range(k):
                for b in range(k):
                    if a != b:
                        flat = ifl[:, a] * self.n + ifl[:, b]
                        counts = torch.bincount(flat, minlength=self.n * self.n).float().reshape(self.n, self.n)
                        self.E = self.E + self.lr * counts

    def prior(self):
        with torch.no_grad():
            scale = self.E.max()
            if scale <= 0.0:
                return torch.zeros(self.n)
            incoming = self.E.mean(0) / scale
            return self.strength * torch.tanh(incoming)
