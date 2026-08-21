import pickle
import sqlite3
from pathlib import Path

import numpy as np
import torch


class MemoryStore:
    def __init__(self, path):
        self.path = Path(path)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS expert_weights("
            "layer_id INT, expert_id INT, blob BLOB, PRIMARY KEY(layer_id, expert_id))"
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS memories("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, layer_id INT, cluster_id INT, "
            "embedding BLOB, text TEXT)"
        )
        self.conn.commit()

    def save_expert(self, layer_id, expert_id, state_dict):
        blob = pickle.dumps({k: v.cpu().numpy() for k, v in state_dict.items()})
        self.conn.execute(
            "INSERT OR REPLACE INTO expert_weights(layer_id, expert_id, blob) VALUES(?,?,?)",
            (layer_id, expert_id, sqlite3.Binary(blob)),
        )
        self.conn.commit()

    def load_expert(self, layer_id, expert_id):
        row = self.conn.execute(
            "SELECT blob FROM expert_weights WHERE layer_id=? AND expert_id=?",
            (layer_id, expert_id),
        ).fetchone()
        if row is None:
            return None
        return {k: torch.from_numpy(v) for k, v in pickle.loads(row[0]).items()}

    def expert_count(self):
        return self.conn.execute("SELECT COUNT(*) FROM expert_weights").fetchone()[0]

    def store_memory(self, embedding, text, layer_id=0, cluster_id=0):
        blob = np.ascontiguousarray(embedding.detach().cpu().float().numpy()).tobytes()
        self.conn.execute(
            "INSERT INTO memories(layer_id, cluster_id, embedding, text) VALUES(?,?,?,?)",
            (layer_id, cluster_id, sqlite3.Binary(blob), text),
        )
        self.conn.commit()

    def query_memories(self, embedding, k=5):
        rows = self.conn.execute(
            "SELECT id, layer_id, cluster_id, embedding, text FROM memories"
        ).fetchall()
        if not rows:
            return []
        q = np.ascontiguousarray(embedding.detach().cpu().float().numpy())
        embs = np.stack([np.frombuffer(r[3], dtype=np.float32) for r in rows])
        denom = np.linalg.norm(embs, axis=1) * np.linalg.norm(q) + 1e-8
        scores = embs @ q / denom
        idx = np.argsort(-scores)[:k]
        return [(int(rows[i][0]), float(scores[i]), rows[i][4]) for i in idx]

    def memory_count(self):
        return self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
