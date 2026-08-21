import torch
import torch.nn as nn
import torch.nn.functional as F

from .experts import ExpertBlock
from .graph import HebbianBank, MessagePassing, small_world


class OmniModel(nn.Module):
    def __init__(self, base, cfg):
        super().__init__()
        self.base = base
        self.cfg = cfg
        d = base.config.hidden_size
        self.expert_layers = nn.ModuleList(
            [ExpertBlock(d, cfg.n_experts, cfg.top_k, cfg.mid_dim) for _ in range(cfg.n_expert_layers)]
        )
        n_layers = base.config.num_hidden_layers
        self.attach = list(range(n_layers - cfg.n_expert_layers, n_layers))
        self.message_passing = (
            nn.ModuleList(
                [
                    MessagePassing(
                        cfg.n_experts, small_world(cfg.n_experts), cfg.graph_rounds, cfg.beta
                    )
                    for _ in range(cfg.n_expert_layers)
                ]
            )
            if cfg.graph_rounds > 0
            else None
        )
        self.hebbian = (
            [
                HebbianBank(cfg.n_experts, cfg.hebbian_lr, cfg.hebbian_decay, cfg.hebbian_strength)
                for _ in range(cfg.n_expert_layers)
            ]
            if cfg.hebbian_lr > 0.0
            else None
        )

    def forward(self, input_ids, labels=None):
        out = self.base(input_ids, output_hidden_states=True)
        hidden = out.hidden_states
        final = hidden[-1]
        aux_total = torch.tensor(0.0, device=final.device)
        usages = []
        ifls = []
        for i, block in enumerate(self.expert_layers):
            h = hidden[self.attach[i] + 1].float()
            prior = None
            if self.hebbian is not None:
                prior = self.hebbian[i].prior().to(h.device)
            mp = self.message_passing[i] if self.message_passing is not None else None
            delta, aux, usage, ifl = block(h, edge_prior=prior, message_passing=mp)
            final = final + (delta * self.cfg.delta_scale).half()
            aux_total = aux_total + aux
            usages.append(usage)
            ifls.append(ifl)
        logits = self.base.get_output_embeddings()(final)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
            )
        return logits, loss, aux_total, usages, ifls
