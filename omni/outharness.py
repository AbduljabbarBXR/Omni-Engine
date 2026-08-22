import torch
import torch.nn as nn
import torch.nn.functional as F


class OutputHarness(nn.Module):
    def __init__(self, top_k, n_experts, expert_top_k, mid_dim, delta_scale):
        super().__init__()
        self.top_k = top_k
        self.n_experts = n_experts
        self.expert_top_k = expert_top_k
        self.router = nn.Linear(top_k, n_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(top_k, mid_dim),
                    nn.GELU(),
                    nn.Linear(mid_dim, top_k),
                )
                for _ in range(n_experts)
            ]
        )
        self.delta_scale = delta_scale
        for ex in self.experts:
            nn.init.zeros_(ex[2].weight)
            nn.init.zeros_(ex[2].bias)

    def forward(self, logits):
        B, T, V = logits.shape
        topv, topi = torch.topk(logits, self.top_k, dim=-1)
        S = B * T
        flat = topv.reshape(S, self.top_k).float()
        router_logits = self.router(flat)
        probs = F.softmax(router_logits, dim=-1)
        topw, topn = torch.topk(probs, self.expert_top_k, dim=-1)
        w = topw / (topw.sum(-1, keepdim=True) + 1e-8)

        per = torch.zeros(S, self.n_experts, self.top_k, device=logits.device)
        for j in range(self.n_experts):
            mask = (topn == j).any(-1)
            if mask.any():
                per[mask, j] = self.experts[j](flat[mask])

        out = torch.zeros(S, self.top_k, device=logits.device)
        for e in range(self.expert_top_k):
            sel = topn[:, e]
            wgt = w[:, e].unsqueeze(-1)
            picked = torch.gather(
                per, 1, sel.unsqueeze(1).unsqueeze(1).expand(S, 1, self.top_k)
            ).squeeze(1)
            out = out + wgt * picked
        delta = out.reshape(B, T, self.top_k) * self.delta_scale

        new_topv = (topv.float() + delta).to(logits.dtype)
        corrected = logits.clone()
        corrected.scatter_(-1, topi, new_topv)

        f = F.one_hot(topn, num_classes=self.n_experts).float().mean(dim=(0, 1))
        p = probs.mean(dim=(0, 1))
        aux = self.n_experts * (f * p).sum()
        usage = (f > 0).float().mean().item()
        delta_sq = (delta ** 2).mean()
        return corrected, aux, usage, delta_sq
