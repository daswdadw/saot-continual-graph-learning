import torch

from evaluation.linear_probe import evaluate_frozen_encoder


class IdentityEncoder(torch.nn.Module):
    def forward(self, x, edge_index=None, sparse=True):
        return x


def test_class_il_linear_probe_uses_frozen_embeddings():
    tasks = []
    for task_id, offset in enumerate((0.0, 4.0)):
        x = torch.tensor([[offset, 0.0], [offset + 0.1, 0.0], [offset + 2.0, 0.0], [offset + 2.1, 0.0]])
        tasks.append({
            "x": x,
            "edge_index": torch.tensor([[0, 1, 2], [1, 0, 3]]),
            "y": torch.tensor([task_id * 2, task_id * 2, task_id * 2 + 1, task_id * 2 + 1]),
            "classes": torch.tensor([task_id * 2, task_id * 2 + 1]),
            "train_mask": torch.tensor([True, False, True, False]),
            "test_mask": torch.tensor([False, True, False, True]),
        })
    scores = evaluate_frozen_encoder(IdentityEncoder(), tasks, 1, scenario="class-il", epochs=80, seed=7)
    assert set(scores) == {0, 1}
    assert min(scores.values()) > 0.9
