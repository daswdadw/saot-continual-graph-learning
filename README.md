# SAOT: Self-Supervised Continual Graph Learning

本仓库用于复现和扩展 ICML 2026 论文 **SAOT: Self-Supervised Continual Graph Learning with Structure-Aware Optimal Transport**。

## 论文

- [论文 PDF](docs/paper/ICML2026_66335_SAOT.pdf)
- [ICML 官方页面](https://icml.cc/virtual/2026/poster/66335)
- [OpenReview](https://openreview.net/forum?id=4bAPXdeTbb)
- [arXiv](https://arxiv.org/abs/2607.00377)

## 六人协作规则

所有成员必须阅读 [六人分工与实际成果](docs/SAOT_六人分工与实际成果.md)。该文件明确写出每个人负责的文件路径、必须提交的代码/数据、独立实验、报告章节、PPT 页面和验收命令。

成员编号暂用成员 1-6，组内确定姓名后直接替换文档中的编号即可。

## 目录

```text
configs/       共享实验配置
 data/         数据下载、处理、任务划分、增强和 replay buffer
models/        GNN 编码器和 SAOT 模型
losses/        OT 对齐损失和结构蒸馏损失
baselines/     GCN、GAE/DGI 等对比方法
evaluation/    AP、AF、性能矩阵和评测脚本
experiments/   主实验、消融、参数和扩展实验
results/       只保留小型示例；正式结果默认被 .gitignore 忽略
docs/paper/    论文 PDF
docs/report/   报告章节
docs/slides/   汇报页面或导出文件
logs/          训练日志，默认不提交大文件
```

## 环境建议

- Python 3.10+
- PyTorch
- PyTorch Geometric
- POT 或 GeomLoss（OT 求解）
- NumPy、SciPy、scikit-learn、matplotlib、seaborn、PyYAML
- AutoDL 单卡 24GB 起步；先跑 CoraFull-CL，再扩展 Arxiv-CL

安装基础依赖：

```bash
pip install -r requirements.txt
```

## 共享配置

从模板开始：

```bash
copy configs\example.yaml configs\local.yaml
```

所有实验都必须保存：

```text
results/<model>/<dataset>/<scenario>/seed_<seed>/
├── config.yaml
├── metrics.json
├── performance_matrix.csv
├── train.log
└── checkpoint.pt
```

## 提交规则

每次提交只处理自己的工作包，提交信息使用：

```text
feat(member-N): <deliverable>
exp(member-N): <experiment>
docs(member-N): <report or slides>
```

不要提交数据集、缓存、checkpoint 或大型日志。详细忽略规则见 `.gitignore`。

## 当前状态

仓库已完成基础初始化。六位成员可以从各自的工作包开始，不需要等待其他成员完成第一版；接口和文件交付要求以分工 Markdown 为准。
