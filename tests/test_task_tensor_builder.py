import numpy as np

from data.build_task_tensor import build_task_payload


class Graph:
    name = "Toy-CL"
    feature_dim = 2
    num_nodes = 4
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    features = __import__("scipy.sparse").sparse.csr_matrix(np.eye(4, 2, dtype=np.float32))
    edge_index = np.array([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=np.int64)


def test_task_payload_induces_edges_and_masks():
    split = {
        "schema_version": 1,
        "dataset": "Toy-CL",
        "seed": 42,
        "tasks": [
            {
                "task_id": 0,
                "classes": [0, 1],
                "node_ids": [0, 1, 2, 3],
                "split_node_ids": {"train": [0, 2], "validation": [1], "test": [3]},
            }
        ],
    }
    payload = build_task_payload(split, Graph())
    task = payload["tasks"][0]
    assert tuple(task["x"].shape) == (4, 2)
    assert tuple(task["edge_index"].shape) == (2, 6)
    assert task["train_mask"].tolist() == [True, False, True, False]
    assert task["test_mask"].tolist() == [False, False, False, True]
