import json
from dataclasses import asdict, dataclass


@dataclass
class Config:
    base_model: str = "EleutherAI/pythia-160m-deduped"
    n_expert_layers: int = 4
    n_experts: int = 8
    top_k: int = 2
    mid_dim: int = 256
    graph_rounds: int = 0
    beta: float = 0.5
    hebbian_lr: float = 0.0
    hebbian_decay: float = 0.999
    hebbian_strength: float = 0.1
    aux_coef: float = 0.01
    seed: int = 42
    seq_len: int = 256
    batch_size: int = 4
    lr: float = 3e-4
    grad_clip: float = 1.0

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def load(path):
        with open(path) as f:
            return Config(**json.load(f))
