# 成员 1 数据与持续学习场景

## 工作目标

本工作包为 SAOT 项目提供统一、确定且可复查的数据入口。成员 1 负责把 CoraFull 和 OGB-Arxiv 的节点分类数据转化为连续到达的二分类任务序列，生成 Task-IL 与 Class-IL 所需索引，并用一个两层 GCN 的顺序训练实验验证数据划分、任务子图、评测矩阵和结果保存流程。后续成员应直接复用本工作包生成的 JSON，不应自行重新随机划分类别。

论文将持续图学习定义为依次到达的互不重叠任务。每个任务包含一个图、节点特征和拓扑结构。CoraFull-CL 有 70 个类别并划分为 35 个任务；Arxiv-CL 有 40 个类别并划分为 20 个任务。每个任务固定包含两个类别。论文实验使用不含跨任务边的任务子图，因此本实现只保留两个端点属于同一任务的边。标签仅用于构造任务、划分训练验证测试节点及 GCN 数据管线验证，不参与后续 SAOT 自监督表示学习。

## 数据来源与处理规则

CoraFull 原始文件采用 PyTorch Geometric `CoraFull` 数据类所指向的公开 `cora.npz`。该文件包含稀疏词袋特征、标签和引用邻接矩阵。本实现将邻接矩阵对称化，删除自环，再在每个任务内部提取诱导子图。Arxiv 使用 Open Graph Benchmark 发布的 `ogbn-arxiv` 压缩包，读取官方节点标签和有向引用边。原始数据保存在 `data/raw/`，受 `.gitignore` 保护，不纳入 Git 提交。

任务顺序由 `numpy.random.default_rng(seed).permutation` 生成。种子 42 下，类别顺序写入两份 JSON；每相邻两个类别组成一个任务。每个类别内部采用 60% 训练、20% 验证、20% 测试的确定性划分。Task-IL 为每个任务保存原始类别到局部标签 0、1 的映射；Class-IL 保存截至当前任务已经出现的全局类别集合。JSON 同时保存每个任务的全部节点、三个数据子集、类样本数、任务内边数及被排除的跨任务边数。

## 数据统计

| 数据集 | 节点数 | 原始有向边或邻接记录 | 特征维度 | 类别数 | 任务数 | 每任务节点数范围 | 每任务平均节点数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CoraFull-CL | 19,793 | 126,842 | 8,710 | 70 | 35 | 192–1,340 | 565.5 |
| Arxiv-CL | 169,343 | 1,166,243 | 128 | 40 | 20 | 626–32,279 | 8,467.2 |

处理后，CoraFull-CL 的任务内有向邻接记录合计为 72,204，Arxiv-CL 的任务内有向边合计为 775,291。其余边连接不同任务的节点，在论文所用的无跨任务边设置中被排除。任务规模图保存在 `results/figures/member_1/task_split_overview.png`，可以看到两套数据均存在明显的类别不平衡，Arxiv-CL 的最大任务包含 32,279 个节点，而最小任务仅有 626 个节点。

论文正文表 1 将 CoraFull-CL 的边数写为 130,622，附录数据说明写为 126,842。本实现直接按公开 `cora.npz` 解析并得到 126,842 条对称化且去除自环后的邻接记录，因此统计表、JSON 和日志统一采用 126,842。该选择保证结果与可下载原始文件一致，也避免手工修改边数以贴合论文正文中的另一处数字。

## Task IL 与 Class IL 接口

在 Task-IL 中，评测时已知任务编号，分类器只在当前任务的两个类别之间选择。JSON 中的 `task_il_label_map` 把原始类别编号映射为 0 和 1。在 Class-IL 中，评测时不提供任务编号，模型必须在当前任务及所有历史任务的类别中选择。JSON 中的 `class_il_seen_classes` 给出每个阶段允许参与预测的全局类别。两个场景共享完全相同的节点划分和任务顺序，因此性能差异来自评测难度，而不是数据不同。任务编号统一采用零起始整数，CoraFull-CL 为 0 至 34，Arxiv-CL 为 0 至 19。

每个任务 JSON 的核心字段如下：

```text
task_id
classes
task_il_label_map
class_il_seen_classes
node_ids
split_node_ids.train
split_node_ids.validation
split_node_ids.test
statistics
```

为与成员 3 的增强、Replay 模块及成员 5 的统一评测协议直接衔接，JSON 顶层同时提供 `task_to_classes`、`task_to_node_ids`、`task_to_train_ids`、`task_to_val_ids` 和 `task_to_test_ids`。所有节点编号均为原始图中的全局零起始编号。`load_graph(dataset, seed)` 返回包含 `x`、`edge_index`、`y` 和 `num_nodes` 的 `torch_geometric.data.Data`；CoraFull 的无向边显式保存两个方向，当前不提供 `edge_weight`。

固定种子 42 时，`corafull_seed42.json` 的 SHA-256 为 `a785fa0886628b055a4836e9f6f0d7d3b757ff7a82a0a609f32329f19c2204fc`，`arxiv_seed42.json` 的 SHA-256 为 `8776a04593f67c32e3da6c91a37c444f645239a21c271e6980a2e2ae576f4b85`。重复执行验收命令得到相同哈希。

## GCN 顺序训练验证

独立实验使用 NumPy 和 SciPy 实现的两层 GCN，对 CoraFull-CL 的 35 个任务依次训练。隐藏维度为 32，每个任务训练 20 个 epoch，学习率为 0.2，随机种子为 42。该实验的目的仅是验证任务子图、特征索引、局部与全局类别空间、性能矩阵以及结果目录能完整工作；它不是 SAOT 的自监督主结果，也不应用于和论文表 2、表 3 的方法数值比较。

| 场景 | 最终 AP | 最终 AF | 解释 |
| --- | ---: | ---: | --- |
| Class-IL | 1.85% | -8.95% | 类别空间随任务扩大，且模型只用当前任务数据更新，旧类性能明显下降 |
| Task-IL | 56.02% | 0.01% | 推理时只区分当前任务的两个类别，难度显著低于 Class-IL |

Class-IL 的低 AP 与负 AF 说明数据管线确实产生了论文所描述的持续学习困难。Task-IL 接近二分类基线以上的结果表明任务局部标签和任务子图可以正常训练与评测。原始配置、指标、性能矩阵、日志和 checkpoint 分别保存在两个标准结果目录：

```text
results/gcn_sequential/CoraFull-CL/class-il/seed_42/
results/gcn_sequential/CoraFull-CL/task-il/seed_42/
```

## 运行与验收

只生成 CoraFull-CL 划分并执行项目规定的验收：

```bash
python data/prepare_cgl.py --dataset CoraFull-CL --seed 42
```

生成两套数据划分并复现实验：

```bash
python data/prepare_cgl.py --dataset all --seed 42 --run-gcn-validation \
  --validation-epochs 20 --hidden-dim 32 --learning-rate 0.2
```

验收时应检查控制台输出的数据集规模、任务数量和 SHA-256，并确认两次运行的 JSON 哈希相同。还应检查 `data/data_statistics.csv`、任务规模图、两个结果目录及 `logs/member_1/gcn_sequential_seed42/train.log`。集成训练时，其他成员只需读取任务 JSON 中的节点索引，不能重新排列类别或恢复跨任务边。

## 交付结论

成员 1 已提供 CoraFull-CL 与 Arxiv-CL 的固定任务划分、Task-IL 与 Class-IL 索引、逐任务数据统计、任务规模图以及 GCN 顺序训练验证结果。数据资产满足两个类别一个任务、种子固定、无跨任务边和标准结果目录四项约束，可直接供 SAOT、基线和消融实验复用。

