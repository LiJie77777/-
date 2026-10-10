# UOVS 研究：Earth2Ocean / UOVSBench 调研与实验方案

- 主文档：[`docs/uovs_research_plan.md`](docs/uovs_research_plan.md)。包括新核查到的事实、交接包审阅、文献核对、线 B / 线 C 实验方案、共享基础设施和按周计划。
- 几何与多源融合调研：[`docs/geometry_fusion_research.md`](docs/geometry_fusion_research.md)。讨论怎样真正利用深度/几何特征、怎样做多源特征融合。v2 的证据基础限定为 2022–2026 年分割领域与垂直领域顶会（逐篇标注 venue，非顶会文献只作附录线索），包括正负证据分类、方案卡片（含预设的通过/放弃标准）、排除"通用特征更强"的对照设计。
- 每日调研 agent 已迁到独立仓库 [LiJie77777/paper-research-agent](https://github.com/LiJie77777/paper-research-agent)（代码、设计文档、整合 skill）。几何融合调研报告此后也在那里的 `reports/` 下更新。
- 脚本（`scripts/`）：

| 脚本 | 用途 |
|---|---|
| `audit_official_package.py` | 审计官方代码包：类表来源、图像列表重叠、类名格式、MLLM JSON 覆盖与准确率 |
| `protocol_eval.py` | 离线多协议评测（P0 官方 / P1 背景计分 / P2 类门控），支持 bootstrap CI 和 Kendall τ |
| `dedup_and_mask_consistency.py` | 在服务器上审计数据：掩码取值普查、MD5/dHash 去重、AquaOV255↔USIS16K 掩码一致性 |
| `oracle_fill_reference.py` | 在任意数据集（含 PASCAL VOC）上算整图 oracle 填充的多协议分数 |
| `prevalidate_pseudolabel_prototypes.py` | 在 probe dump 上离线做跨图伪标签原型的前置验证（B3 的继续/放弃判据） |
| `dump_depth_cache.py` | 用 E2O 自带的 Depth Anything V2 缓存整图相对逆深度（需要 torch + GPU） |
| `prevalidate_depth_figure_ground.py` | 几何卡 G1/G2/G3 的离线预检：前景/背景深度序、Otsu 前景 IoU、深度环对比、像素对 AUROC、边界 F，带对照和预设判据（只依赖 numpy） |
| `e2o_patches/geo_affinity.py` | 卡 G2 的 3D 感知亲和：numpy 参考实现 + torch 移植 + 非几何平滑图对照 |

运行测试（依赖 numpy、Pillow）：`python scripts/tests/test_protocol_eval.py`、`python scripts/tests/test_prevalidate_depth.py`、`python scripts/tests/test_geo_affinity.py`。
