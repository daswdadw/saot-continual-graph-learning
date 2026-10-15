# SAOT 六人分工、文件交付与验收规范

论文：**SAOT: Self-Supervised Continual Graph Learning with Structure-Aware Optimal Transport**  
会议：ICML 2026  
论文页面：https://icml.cc/virtual/2026/poster/66335

## 一、强制交付规则

每个人必须提交四类成果：

1. **代码或数据资产**：必须位于本文件指定路径；
2. **独立实验**：至少一组，不得只复用别人的结果；
3. **原始产物**：配置、日志、CSV 或 checkpoint，按统一目录保存；
4. **文字与汇报**：报告章节、PPT 页面和 5 分钟讲解稿。

“只做 PPT”“只查资料”“只帮忙跑实验”不算独立任务。

成员编号为临时编号。组内确定姓名后，将 `成员 1` 至 `成员 6` 替换为真实姓名。

## 二、统一文件和结果规范

### 代码路径

所有路径均相对于仓库根目录 `saot-continual-graph-learning/`。

```text
configs/example.yaml
configs/local.yaml                         # 本地配置，不提交

data/prepare_cgl.py                       # 成员 1
data/task_splits/*.json                   # 成员 1
data/augment.py                           # 成员 3
data/replay_buffer.py                     # 成员 3

models/gnn_encoder.py                      # 成员 3
models/saot.py                             # 成员 4
losses/ot_solver.py                        # 成员 2
losses/transport_alignment.py              # 成员 4
losses/structural_distillation.py          # 成员 4

baselines/gcn_sequential.py               # 成员 5
baselines/gae_or_dgi.py                   # 成员 5
baselines/replay_baseline.py              # 成员 5
evaluation/metrics.py                     # 成员 5

experiments/toy_ot_alignment.py          # 成员 2
experiments/augmentation_ablation.py      # 成员 3
experiments/train_sequential.py           # 成员 4
experiments/main_baselines.py             # 成员 5
experiments/ablation.py                   # 成员 6
experiments/adaptive_beta.py              # 成员 6
```

> 注意：上面 `experiments/` 是排版错误，实际目录必须使用 `experiments/`。所有成员提交前请检查路径。

### 统一结果路径

```text
results/<model>/<dataset>/<scenario>/seed_<seed>/
├── config.yaml
├── metrics.json
├── performance_matrix.csv
├── train.log
└── checkpoint.pt
```

图表保存到：

```text
results/figures/member_<N>/
```

报告章节保存到：

```text
docs/report/member_<N>_<topic>.md
```

汇报材料保存到：

```text
docs/slides/member_<N>_<topic>.md
```

原始日志保存到：

```text
logs/member_<N>/<experiment_name>/
```

### 统一命令约定

每个脚本必须支持：

```bash
python <script> --config configs/local.yaml --seed 42
```

README 或脚本顶部必须说明：输入、输出、运行命令和预计显存。

## 三、成员 1：数据与持续学习场景

### 负责文件

```text
data/prepare_cgl.py
data/task_splits/corafull_seed42.json
data/task_splits/arxiv_seed42.json
results/figures/member_1/task_split_overview.png
docs/report/member_1_data_and_problem.md
docs/slides/member_1_data_and_problem.md
```

### 具体任务

- 处理 CoraFull-CL，之后扩展 Arxiv-CL；
- 将类别划分为连续任务，每个任务含两个类别；
- 实现 Task-IL 和 Class-IL 的任务索引；
- 固定任务顺序、随机种子和数据统计；
- 用 GCN 顺序训练验证数据管线。

### 独立实验

运行固定种子的 GCN 顺序训练，输出任务数量、类别数量、节点/边统计和每个任务的样本数量。

### 必须提交

- 数据处理脚本；
- 两份任务划分 JSON；
- `data_statistics.csv`；
- `task_split_overview.png`；
- GCN 验证日志和结果；
- 报告约 2 页；
- PPT 2-3 页。

### 验收命令

```bash
python data/prepare_cgl.py --dataset CoraFull-CL --seed 42
```

验收：输出任务统计，且相同 seed 生成相同 JSON 哈希。

## 四、成员 2：OT、GW、FGW 理论与算法

### 负责文件

```text
losses/ot_solver.py
experiments/toy_ot_alignment.py
results/figures/member_2/toy_transport_plan_*.png
docs/report/member_2_optimal_transport.md
docs/slides/member_2_optimal_transport.md
```

### 具体任务

- 解释 Wasserstein、Gromov-Wasserstein、Fused Gromov-Wasserstein；
- 计算节点特征代价矩阵和边结构代价；
- 实现特征/结构权重 `sigma`；
- 输出图空间 `pi_G` 和表示空间 `pi_Z`；
- 用两个 5-10 节点的小图验证匹配。

### 独立实验

固定小图，分别使用 `sigma=0.2/0.5/0.8`，比较 transport plan 热力图和结构匹配代价。

### 必须提交

- OT 求解器；
- 小图实验脚本；
- 原始 transport plan 矩阵；
- 至少 3 张热力图；
- sigma 实验 CSV 和日志；
- 报告约 2-3 页；
- PPT 3-4 页。

### 验收命令

```bash
python experiments/toy_ot_alignment.py --config configs/local.yaml --seed 42
```

验收：输出 `pi_G.npy`、`pi_Z.npy` 和至少一张热力图；改变 sigma 后结果发生变化。

## 五、成员 3：GNN、图增强与 Replay Buffer

### 负责文件

```text
data/augment.py
data/replay_buffer.py
models/gnn_encoder.py
experiments/augmentation_ablation.py
results/figures/member_3/augmentation_*.png
docs/report/member_3_encoder_and_replay.md
docs/slides/member_3_encoder_and_replay.md
```

### 具体任务

- 实现 2 层 GCN；
- 预留 Reddit-CL 的 3 层 GAT 配置；
- 实现随机删边和节点特征遮挡；
- 实现 layer-wise neighbor sampling；
- 实现固定 budget replay buffer；
- 比较增强强度和 buffer 大小。

### 独立实验

比较 `edge_drop={0.0,0.2,0.5}` 或 `buffer_budget={0,200,800}` 对训练速度和 AP/AF 的影响。

### 必须提交

- 编码器、增强和 buffer 代码；
- 增强前后图统计；
- 至少 3 个配置的结果；
- 图表、日志和 CSV；
- 报告约 2 页；
- PPT 2-3 页。

### 验收命令

```bash
python experiments/augmentation_ablation.py --config configs/local.yaml --seed 42
```

验收：同一 seed 可生成两种合法增强视图；buffer 不超过 budget；脚本能独立输出实验结果。

## 六、成员 4：SAOT 核心损失与跨任务蒸馏

### 负责文件

```text
models/saot.py
losses/transport_alignment.py
losses/structural_distillation.py
experiments/train_sequential.py
results/figures/member_4/loss_curves.png
docs/report/member_4_saot_method.md
docs/slides/member_4_saot_method.md
```

### 具体任务

- 实现 `L_mat`：对齐 `pi_G` 与 `pi_Z`；
- 实现 `L_str`：保持结构代价；
- 实现 `L_gta = L_mat + alpha * L_str`；
- 保存上一个任务结束时的 teacher encoder；
- 实现 `L_skd`；
- 实现完整 sequential training loop。

### 独立实验

在 CoraFull-CL 上比较 SAOT 和 `w/o Cross-task Distillation`，分别记录 `L_mat`、`L_str`、`L_skd`、总损失、AP 和 AF。

### 必须提交

- SAOT 模型和三个损失模块；
- 训练脚本；
- checkpoint；
- loss 曲线；
- 完整版与无蒸馏版结果；
- 报告约 3 页；
- PPT 4-5 页。

### 验收命令

```bash
python experiments/train_sequential.py --config configs/local.yaml --model saot --seed 42
```

验收：日志中分别出现四项损失；去掉 `L_skd` 后结果发生变化；teacher 能按任务保存和加载。

## 七、成员 5：基线与统一评测

### 负责文件

```text
baselines/gcn_sequential.py
baselines/gae_or_dgi.py
baselines/replay_baseline.py
evaluation/metrics.py
experiments/main_baselines.py
results/tables/main_table.csv
results/figures/member_5/main_ap_af.png
docs/report/member_5_baselines_and_metrics.md
docs/slides/member_5_baselines_and_metrics.md
```

### 具体任务

- 实现 GCN 顺序训练；
- 实现 GAE 或 DGI；
- 实现一个 replay 或 embedding-distillation 基线；
- 统一训练轮数、学习率、维度和 seeds；
- 实现 AP、AF 和性能矩阵；
- 完成 Class-IL 和 Task-IL 评估。

### 独立实验

至少运行 3 个基线和 SAOT，使用同一数据划分和至少 3 个随机种子，生成主结果表。

### 必须提交

- 三个基线模块；
- 统一评测脚本；
- `main_table.csv`；
- 性能矩阵；
- AP/AF 图；
- 报告约 3 页；
- PPT 3-4 页。

### 验收命令

```bash
python experiments/main_baselines.py --config configs/local.yaml --dataset CoraFull-CL --seeds 42 43 44
```

验收：任意模型都能输出相同格式的 AP、AF 和 `performance_matrix.csv`。

## 八、成员 6：消融、扩展与可视化

### 负责文件

```text
experiments/ablation.py
experiments/adaptive_beta.py
results/figures/member_6/ablation_*.png
results/figures/member_6/embedding_umap.png
results/tables/member_6_results.csv
docs/report/member_6_ablation_and_extension.md
docs/slides/member_6_ablation_and_extension.md
```

### 具体任务

- 复现 `w/o Optimal Transport`；
- 复现 `w/o Cross-task Distillation`；
- 用 Cosine 和 MSE 替换 plan-level distillation；
- 测试 alpha、beta、任务数量和 buffer budget；
- 绘制 t-SNE/UMAP 节点表示；
- 实现任务变化感知的 adaptive beta。

### 独立实验

比较固定 beta 与自适应 beta：任务结构变化大时提高 beta，变化小时降低 beta，并报告 AP、AF、训练时间。

### 必须提交

- 消融脚本；
- adaptive beta 脚本；
- 至少 4 张消融图；
- 1 张表示空间图；
- 结果 CSV；
- 至少 3 个失败案例；
- 报告约 3 页；
- PPT 4-5 页。

### 验收命令

```bash
python experiments/ablation.py --config configs/local.yaml --seed 42
python experiments/adaptive_beta.py --config configs/local.yaml --seed 42
```

验收：每个消融只移除一个组件；adaptive beta 有明确公式、配置和对照结果。

## 九、Git 提交规范

每个人只提交自己工作包的文件，并使用以下前缀：

```text
feat(member-1): add CoraFull task preparation
feat(member-2): add FGW toy alignment
feat(member-3): add graph augmentation and replay buffer
feat(member-4): add SAOT losses and teacher distillation
feat(member-5): add baselines and AP AF metrics
feat(member-6): add ablations and adaptive beta
exp(member-N): add independent experiment results
docs(member-N): add report section and slides
```

提交前必须运行：

```bash
git status
git diff --check
git add <自己的文件>
git commit -m "<符合上面格式的提交信息>"
```

不要提交：

- 原始数据集；
- 大于 50MB 的 checkpoint；
- `__pycache__/`；
- `*.log` 大文件；
- AutoDL 临时目录；
- 个人密码、Token 或云平台配置。

## 十、阶段验收

### 第 1 周：独立最小成果

- 成员 1：数据加载和任务统计；
- 成员 2：小图 transport plan；
- 成员 3：增强视图和 buffer；
- 成员 4：三个损失输出数值；
- 成员 5：AP/AF 可由性能矩阵计算；
- 成员 6：CSV 可生成图表。

### 第 2 周：独立正式结果

- 成员 1：CoraFull-CL 数据管线验证；
- 成员 2：三个 sigma 的 OT 实验；
- 成员 3：增强或 buffer 实验；
- 成员 4：SAOT 与无蒸馏版本；
- 成员 5：三个基线和统一评测；
- 成员 6：四种消融和 adaptive beta 初版。

### 第 3 周：最终成果包

每个人必须把以下文件放入仓库：

```text
代码或数据/
results/ 原始结果/
docs/report/ 报告章节/
docs/slides/ 汇报页面/
README 或脚本注释中的运行命令
```

最终合并前，六个人逐项检查自己的文件路径、运行命令、日志、CSV、图表和文字章节。

