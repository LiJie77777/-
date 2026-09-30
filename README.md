# UOVS 研究：Earth2Ocean / UOVSBench 调研与实验方案

- 主文档：[`docs/uovs_research_plan.md`](docs/uovs_research_plan.md)。包括新核查到的事实、交接包审阅、文献核对、线 B / 线 C 实验方案、共享基础设施和按周计划。
- 脚本（`scripts/`）：

| 脚本 | 用途 |
|---|---|
| `audit_official_package.py` | 审计官方代码包：类表来源、图像列表重叠、类名格式、MLLM JSON 覆盖与准确率 |
| `protocol_eval.py` | 离线多协议评测（P0 官方 / P1 背景计分 / P2 类门控），支持 bootstrap CI 和 Kendall τ |
| `dedup_and_mask_consistency.py` | 在服务器上审计数据：掩码取值普查、MD5/dHash 去重、AquaOV255↔USIS16K 掩码一致性 |
| `oracle_fill_reference.py` | 在任意数据集（含 PASCAL VOC）上算整图 oracle 填充的多协议分数 |
| `prevalidate_pseudolabel_prototypes.py` | 在 probe dump 上离线做跨图伪标签原型的前置验证（B3 的继续/放弃判据） |

运行测试：`python scripts/tests/test_protocol_eval.py`（依赖 numpy）。
