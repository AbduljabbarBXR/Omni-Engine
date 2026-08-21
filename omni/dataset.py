import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, ids, seq_len, stride=None):
        self.seq_len = seq_len
        stride = stride or seq_len
        self.blocks = [
            ids[i : i + seq_len]
            for i in range(0, max(len(ids) - seq_len, 1), stride)
            if len(ids[i : i + seq_len]) == seq_len
        ]
        if not self.blocks:
            self.blocks = [torch.zeros(seq_len, dtype=torch.long)]

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, i):
        return self.blocks[i]
