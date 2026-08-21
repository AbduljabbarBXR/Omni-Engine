import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertBlock(nn.Module):
    def __init__(self, d_model, n_experts, top_k, mid_dim):
        super().__init__()
        self.router = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, mid_dim),
                    nn.GELU(),
                    nn.Linear(mid_dim, d_model),
                )
                for _ in range(n_experts)
            ]
        )
        self.n_experts = n_experts
        self.top_k = top_k
        for ex in self.experts:
            nn.init.zeros_(ex[2].weight)
            nn.init.zeros_(ex[2].bias)

    def forward(self, x, edge_prior=None, message_passing=None):
        B, T, D = x.shape
        logits = self.router(x)
        if edge_prior is not None:
            logits = logits + edge_prior
        probs = F.softmax(logits, dim=-1)
        topv, topi = torch.topk(probs, self.top_k, dim=-1)
        w = topv / (topv.sum(-1, keepdim=True) + 1e-8)

        S = B * T
        flat = x.reshape(S, D)
        wf = w.reshape(S, self.top_k)
        ifl = topi.reshape(S, self.top_k)

        per = torch.zeros(S, self.n_experts, D, device=x.device, dtype=x.dtype)
        for j in range(self.n_experts):
            mask = (ifl == j).any(dim=-1)
            if mask.any():
                per[mask, j] = self.experts[j](flat[mask])

        if message_passing is not None:
            per = message_passing(per, ifl)

        out = torch.zeros(S, D, device=x.device, dtype=x.dtype)
        for e in range(self.top_k):
            sel = ifl[:, e]
            wgt = wf[:, e].unsqueeze(-1)
            picked = torch.gather(
                per, 1, sel.unsqueeze(1).unsqueeze(1).expand(S, 1, D)
            ).squeeze(1)
            out = out + wgt * picked
        delta = out.reshape(B, T, D)

        f = F.one_hot(ifl, num_classes=self.n_experts).float().mean(dim=(0, 1))
        p = probs.mean(dim=(0, 1))
        aux = self.n_experts * (f * p).sum()
        usage = (f > 0).float().mean().item()
        return delta, aux, usage, ifl
