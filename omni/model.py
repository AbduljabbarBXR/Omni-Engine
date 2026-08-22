import torch
import torch.nn as nn
import torch.nn.functional as F

from .experts import ExpertBlock
from .graph import HebbianBank, MessagePassing, small_world
from .outharness import OutputHarness


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
                        cfg.n_experts, small_world(cfg.n_experts), cfg.graph_rounds,
                        cfg.beta, cfg.edge_mode,
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
        self.output_harness = (
            OutputHarness(
                cfg.out_top_k, cfg.n_out_experts, cfg.out_expert_top_k,
                cfg.out_mid_dim, cfg.delta_scale,
            )
            if cfg.out_harness
            else None
        )

    def forward(self, input_ids, labels=None):
        out = self.base(input_ids, output_hidden_states=True)
        hidden = out.hidden_states
        final = hidden[-1]
        aux_total = torch.tensor(0.0, device=final.device)
        delta_sq = torch.tensor(0.0, device=final.device)
        usages = []
        ifls = []
        for i, block in enumerate(self.expert_layers):
            h = hidden[self.attach[i] + 1].float()
            prior = None
            if self.hebbian is not None:
                prior = self.hebbian[i].prior().to(h.device)
            mp = self.message_passing[i] if self.message_passing is not None else None
            delta, aux, usage, ifl = block(h, edge_prior=prior, message_passing=mp)
            scaled = delta * self.cfg.delta_scale
            final = final + scaled.half()
            delta_sq = delta_sq + (scaled ** 2).mean()
            aux_total = aux_total + aux
            usages.append(usage)
            ifls.append(ifl)
        logits = self.base.get_output_embeddings()(final)
        if self.output_harness is not None:
            logits, out_aux, out_usage, out_delta = self.output_harness(logits)
            aux_total = aux_total + out_aux
            delta_sq = delta_sq + out_delta
            usages.append(out_usage)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
            )
        return logits, loss, aux_total, delta_sq, usages, ifls
