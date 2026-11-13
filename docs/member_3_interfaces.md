# 成员 3 模块接口契约

本文只描述成员 3 已实现代码与成员 1、4、5 的衔接方式，不包含报告和 PPT 内容。
所有路径相对于仓库根目录 `D:/MLFP`。

## 0. 统一约定

### 全局节点编号

- 所有 `node_id` 都是原始图中的全局、零起始、连续整数：`0 <= node_id < num_nodes`。
- 成员 1 生成的 `Data` 必须保留全局编号语义；不能在数据加载阶段把节点编号改成任务内编号。
- 邻居采样返回的 `Data` 会同时提供：
  - `n_id`: 局部节点位置到全局节点编号的映射；
  - `edge_index`: 只使用局部编号，范围是 `[0, len(n_id)-1]`；
  - `input_id`: 本批次 replay seed 的全局编号；
  - `seed_node_index`: `input_id` 在局部子图中的位置。

### PyG 图对象

成员 1 的数据管线和成员 4/5 的训练代码统一使用 `torch_geometric.data.Data`：

```python
Data(
    x=FloatTensor[num_nodes, num_features],
    edge_index=LongTensor[2, num_edges],
    y=LongTensor[num_nodes],              # 若当前任务有标签
    edge_weight=FloatTensor[num_edges],   # 可选；没有时可以省略
    num_nodes=num_nodes,
)
```

约束：

- `x` 和 `edge_index` 在同一 device；
- `edge_index` 使用 COO `[source, target]`；
- 若图按无向图存储，推荐显式存储两个方向；
- `edge_weight` 若存在，必须与 `edge_index` 第一维对应，长度为 `num_edges`；
- `num_nodes` 必须明确，不能依赖含孤立节点时的自动推断。

### 随机种子

- 任务划分、增强、Replay Buffer 和采样器都传入显式 `seed`；
- 相同输入和相同 seed 必须得到相同节点/边结果；
- 不要依赖全局随机状态来复现实验。

## 1. 给成员 1：数据管线需要提供什么

成员 1 的 `data/prepare_cgl.py` 最终应返回或缓存一个全局 `Data`，以及任务到节点的划分。

推荐任务索引格式：

```python
task_nodes: dict[int, Tensor]
# 例如：{0: tensor([0, 4, 9]), 1: tensor([2, 7, 11])}
```

如果写入 JSON，建议使用：

```json
{
  "num_nodes": 100,
  "num_tasks": 35,
  "task_to_classes": {"0": [0, 1], "1": [2, 3]},
  "task_to_node_ids": {"0": [0, 4, 9], "1": [2, 7, 11]},
  "seed": 42
}
```

成员 3 只需要 `task_to_node_ids` 和全局 `Data`，不需要成员 1 改动成员 3 的代码。

成员 1 需要向成员 3 确认：

1. `edge_index` 是否双向存储；
2. `x` 的特征维度 `input_dim`；
3. 任务节点 ID 是全局编号还是任务内编号；
4. 是否存在 `edge_weight`；
5. train/val/test 节点掩码的字段名。

## 2. 给成员 4：增强、Replay 和编码器调用

### 2.1 两个增强视图

```python
from data.augment import create_augmented_views

view_1, view_2 = create_augmented_views(
    data,
    edge_drop=0.2,
    feature_mask=0.2,
    seed=42,
)
```

返回值是两个独立的 PyG `Data`：

- `view.x`：增强后的节点特征；
- `view.edge_index`：增强后的局部/全局一致边索引；
- `view.edge_weight`：若输入有边权，会同步过滤；
- `view.augmentation_edge_keep_mask`：对应原边的布尔保留掩码；
- `view.augmentation_feature_mask`：被遮挡的特征位置；
- `view.augmentation_seed`：该视图使用的 seed。

### 2.2 Replay Buffer

```python
from data.replay_buffer import ReplayBuffer

buffer = ReplayBuffer(budget=200, seed=42)
buffer.add(task_id=task_id, node_ids=train_node_ids)
replay_nodes = buffer.sample(
    num_samples=64,
    balanced=True,
    replace=False,
    seed=42,
)
```

约束：

- `len(buffer) <= budget` 永远成立；
- buffer 存的是 `(task_id, global_node_id)`，不是特征张量；
- `budget=0` 表示关闭 replay；
- `sample` 返回 `LongTensor[batch_size]` 的全局节点 ID；
- 如需任务 ID，使用 `return_task_ids=True`。

### 2.3 Layer-wise Neighbor Sampling

```python
from data.neighbor_sampling import build_replay_loader

loader = build_replay_loader(
    data,
    seed_nodes=replay_nodes,
    num_neighbors=[15, 10],
    batch_size=64,
    direction="both",  # 双向存储的无向图也可使用 "in"
    seed=42,
)

for subgraph in loader:
    # subgraph.x: [num_sampled_nodes, input_dim]
    # subgraph.edge_index: 局部编号
    # subgraph.n_id: 全局编号映射
    z = encoder(subgraph.x, subgraph.edge_index)
```

### 2.4 GCN/GAT Encoder

```python
from models.gnn_encoder import GNNEncoder

encoder = GNNEncoder(
    input_dim=data.x.size(1),
    hidden_dim=256,
    output_dim=256,
    encoder_type="gcn",
    num_layers=2,
    dropout=0.0,
)

z = encoder(
    subgraph.x,
    subgraph.edge_index,
    edge_weight=getattr(subgraph, "edge_weight", None),
)
```

输出：

- `z.shape == [subgraph.num_nodes, output_dim]`；
- `z[i]` 对应 `subgraph.n_id[i]` 的全局节点表示；
- seed 节点表示位于 `z[subgraph.seed_node_index]`；
- 如果使用 GAT，将 `encoder_type="gat"`、`num_layers=3`。

成员 4 不应假设每个 batch 的节点数等于 seed 数；邻居采样后通常会更多。计算 replay loss 时请使用 `seed_node_index` 取回 seed 表示。

## 3. 给成员 5：统一评测需要如何读取输出

成员 5 的评测代码不需要了解 Replay Buffer 内部结构，只需使用统一的 encoder 输出：

```python
z = encoder(data.x, data.edge_index)
logits = classifier(z)
```

如果评测基于 replay 子图：

```python
for subgraph in replay_loader:
    z = encoder(subgraph.x, subgraph.edge_index)
    seed_z = z[subgraph.seed_node_index]
    seed_global_ids = subgraph.input_id
```

`seed_z[i]` 与 `seed_global_ids[i]` 一一对应。成员 5 应使用 `seed_global_ids` 将子图预测写回全局性能矩阵或任务指标，不能直接用子图的局部索引当作全局节点 ID。

## 4. 成员 3 当前可验收的代码命令

在仓库根目录运行：

```powershell
conda activate saot
python -m pytest -q --basetemp .pytest-tmp
python experiments/augmentation_ablation.py --config configs/local.yaml --seed 42
```

当前实验脚本默认使用 toy graph，用于验证成员 3 代码链路；接入成员 1 的 CoraFull-CL 后，再替换数据加载部分，不改变上述增强、buffer、采样和 encoder 接口。

## 5. 合作时必须先确认的 5 个问题

在成员 1/4/5 开始整合前，群里确认以下答案：

1. 全局 `num_nodes`、`x.shape[1]` 和 `edge_index` 是否确定；
2. 任务节点 ID 是否是全局零起始整数；
3. 无向图是否显式保存双向边；
4. replay 子图的 `seed_node_index` 是否用于取 seed 表示；
5. 三方使用的 `seed=42`、`num_neighbors=[15,10]`、`replay_budget=200` 是否一致。

