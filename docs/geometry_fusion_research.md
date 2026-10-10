# 怎样真正利用几何/深度特征，以及怎样做多源特征融合：基于近年顶会的调研与候选方案（v2）

> 日期：2026-10-10。读者：导师、执行同学、跑实验的同学。分支 `claude/busy-bohr-kkw7nd`。
> **后续更新不在这里。** 本报告已迁到 [paper-research-agent/reports/geometry_fusion_research.md](https://github.com/LiJie77777/paper-research-agent/blob/main/reports/geometry_fusion_research.md)，每日调研结果整合进那一份；这里保留 v2 快照。
> **v2 改动**：按你的要求，证据基础改为"2022–2026 年分割领域顶会和垂直领域顶会"（纳入标准见 §1）。v1 中的 arXiv 预印本和普通期刊论文移到附录 C，只作为线索，不作为结论依据。§1.2 逐篇列出 v1 引用的重新分级。
> 与请求 1 报告 [`docs/uovs_research_plan.md`](uovs_research_plan.md) 配套使用。脚本：`scripts/dump_depth_cache.py`、`scripts/prevalidate_depth_figure_ground.py`、`scripts/e2o_patches/geo_affinity.py`，测试在 `scripts/tests/`。
> 标注约定：
> - 【文献事实】读过原文（全文问答）确认的内容；
> - 【本项目事实】你们提供的数字，或我在官方代码里逐行核对过的内容；
> - 【推断】我的推断，每条附依据；
> - 【未核实】只读过摘要或检索摘录、凭记忆，或来自二手转述。
>
> 每篇文献后的方括号是**已核实的 venue**，例如 [CVPR'25]。核实方式见 §1.1。

## 0 一页结论

**核心判断（与 v1 相同，但证据全部换成顶会）**
- E2O [CVPR'26] 对 DepthAnything 的用法，等价于换了一个 DINOv2 家族的代理特征。深度图、深度不连续、3D 距离、物理衰减这些几何信息都没用上。
- 在官方协议 P0 下，即使几何把分组做到完美，ViT-B 也只涨 +1.3。
- 所以"真正利用几何"需要同时满足两点：
  - 用显式几何量；
  - 在能反映几何作用的指标上评测，并且排除"只是通用特征更强"这种解释。

1. **导师的批评成立，代码可以证明。**
   - 【本项目事实】`get_geo_feats()` 已经跑了 DPT 深度头，但返回的 `geo_map` 在 `forward_feature()` 里被 `_` 丢弃（`Earth2Ocean.py` L289）。实际只用了 block-11 的 token 自相似。
   - 【文献事实】DAv2 编码器用 DINOv2 初始化，并加了特征对齐损失向冻结的 DINOv2 看齐（Depth Anything [CVPR'24] §3.3，容忍阈 α=0.85；Depth Anything V2 [NeurIPS'24] §5.2 沿用）。
   - 【文献事实】DINOv2-B 的冻结特征深度可读性已经很高（NYU 探针 δ1 93.4）。CLIP-B/16 只有 52.5，深度监督训练的 MiDaS-L 只有 78.7（Probe3D [CVPR'24]）。
   - 【推断】因此 A4−A3 ≈ 0 是意料之中的。
2. **几何主要帮"分出物体"，几乎不帮"认出是什么"。** 在三类顶会证据上表现一致：
   - DFormerv2 [CVPR'25]：LUSS 前景 wF 0.818→0.868，分类只从 83.1 到 83.4；
   - FiT3D [ECCV'24]：3D 感知微调让分割 +1.2~2.6，分类 80.02→80.00，不变；
   - MARIS [CVPR'26]（与 E2O 同组）：几何模块让已见类 +5.8 mAP，开放词汇未见类只 +1.2。
   - 【本项目事实】本项目的瓶颈在识别：文本头 33% vs 视觉质心 85%；完美分组只 +1.3/+2.5/+2.1。
   - 【推断】只改亲和或分组的几何方案，P0 预期 0~+0.6（ViT-B）。
3. **伪深度直接融合常常无益，必须有专门设计。** 这一条有多篇顶会证据：
   - PopNet [ICCV'23] 表 7：现有 RGB-D 方法换用估计深度后常常不如纯 RGB（SPNet 在 NC4K 上 Fm .846→.828）；
   - MultiMAE [ECCV'22]：没在深度上预训练过的 MAE，加伪深度后在 NYU 上 −1.0；
   - Vireo [NeurIPS'25]：直接拼接深度特征只有 +0.1~0.3，有一项 −0.1；
   - OVCamo [ECCV'24]：把融合模块换成简单相加，深度/边缘的增益基本消失；
   - DepthSAM [CVPR'26]：明确指出直接用单目深度会造成"任务错配"【未核实（仅摘要）】。
4. **骨干越强，深度的边际增益越小。**
   - 【文献事实】PDDM [AAAI'25] 表 2：DA 伪深度给 EMSANet +3.3，给 CMNeXt、DFormer、PDDM 只有 +0.75~0.8。
   - 【推断】E2O 的代理已经是 DINOv2 系，处在"强骨干"端。
5. **水下：相对远近可用，度量距离不可用。**
   - 【文献事实】DAv2-B 在 DA-2K 水下子集上远近判断准确率 100%（[NeurIPS'24] 表 14）。
   - 【文献事实】在陆地数据上训练的有监督深度模型，在 Sea-thru/SQUID 上 δ1 只有 0.01–0.22（Atlantis [CVPR'24]）。
   - 所以优先用远近顺序，不用度量距离。
6. **顶会空白。** 检索了 CVF/ECVA/OpenReview/NeurIPS 等官方站点，没有找到"2D 免训练开放词汇分割 + 单目深度"的顶会工作。
   - 最接近的有三篇，都需要训练：CutS3D [ICCV'25]（无监督实例分割，要训检测器）、Vireo [NeurIPS'25]（训练式开放词汇域泛化分割）、MARIS [CVPR'26]（训练式）。
   - 这既是机会，也意味着没有可直接照搬的先例。
7. **建议（卡片见 §5，优先级见 §7）：**
   - **D0 零成本诊断 + G4 多层/注意力 key**：只需改配置或小改动。
   - **G1 深度前景/背景通道**（以 PopNet 的"凸起"先验为依据）：先离线预检，再接入；主指标为 P2/P1b。
   - **G2 3D 感知亲和**：用 DFormerv2 的 β^G 形式，或 CutS3D 的"深度高频锐化"；必须带四个对照。
   - **G3 深度引导边界精修**：用 NAF [CVPR'26] 或 SegEarth-OV [CVPR'25] 式图像引导上采样作强对照。
   - **多源融合**：CorrCLIP [ICCV'25]、Trident [ICCV'25]、CASS [CVPR'25] 在自然图像上增益明显（CorrCLIP 加到 ProxyCLIP 上 +1.4~5.7，Trident 相对 ProxyCLIP +3.5~4.2，CASS 的 KK 比特征 +2.9），其中 CorrCLIP 在 SUIM 上达到 58.0。但本项目在 P0 下的分组余量只有 +1.3，所以重点放在 SUIM/DUT 等多类场景和识别端。
8. **排除"通用特征更强"的解释**（详见 §6.4）：
   - 同构 DINOv2 对照；
   - 打乱深度和模糊深度；
   - 非几何平滑图；
   - 仅空间项（DFormerv2 显示空间先验单独就有 +1.8）；
   - FiT3D 式的"特征复制"对照；
   - 事先写定的异质性预测。
   - 注意：深度质量剂量—反应不能作为证伪依据。CutS3D [ICCV'25] 和 DFormer [ICLR'24] 都显示，深度质量对结果影响很小。

---

## 1 纳入标准，以及 v1 证据的重新分级

### 1.1 纳入标准
- **年份**：2022–2026。早于 2022 的只作背景，单独标注。
- **通用顶会**：CVPR、ICCV、ECCV、NeurIPS、ICLR、ICML、AAAI、IJCAI、ACM MM。ECCV 在 CCF 名单里是 B 类，但在视觉领域公认是三大会之一，所以纳入并注明。
- **垂直顶会**：
  - 医学：MICCAI；
  - 机器人：ICRA、IROS；
  - 遥感和水下没有公认的专属顶会，采用发表在上述通用顶会上的遥感/水下工作。
- **特殊赛道**：CVPR 2026 的 Findings 赛道（CVF 路径为 `CVPR2026F`）单独标为 [CVPR'26 Findings]，不等同于主会。
- **排除**：只在 arXiv 发布的预印本、Workshop 论文、非顶级期刊。TPAMI/IJCV 这类顶刊本次没有作为主证据。
- **venue 核实方式**（每篇至少一种）：
  - 官方论文集 URL：openaccess.thecvf.com、ecva.net、openreview.net、proceedings.neurips.cc、proceedings.mlr.press、ojs.aaai.org、dl.acm.org、ijcai.org；
  - 官方会议海报页（cvpr.thecvf.com/virtual、neurips.cc/virtual）；
  - PDF 首页的会议声明，例如 "Published as a conference paper at ICLR 2024"；
  - 官方 GitHub README 的会议标注；
  - Consensus（Semantic Scholar/OpenAlex）的 venue 字段。
  - dblp、Semantic Scholar API 和 CVF 直连在本环境被代理拒绝，所以没有用。

### 1.2 v1 引用的重新分级

| v1 引用 | v1 中用于支持 | 核实后的 venue | v2 处理 |
|---|---|---|---|
| DFormerv2 | 几何先验注意力、LUSS 分类/分割对比 | CVPR'25 | 保留 |
| DepthG | 深度相关损失，深度质量有影响 | CVPR'24 | 保留 |
| Probe3D（El Banani 等） | DINOv2 已含深度 | CVPR'24 | 保留 |
| Depth Anything / V2 | 编码器来源与特征对齐、水下远近判断 | CVPR'24 / NeurIPS'24 | 保留 |
| NAF | 零样本上采样 | CVPR'26（Highlight） | 保留 |
| RADSeg | RADIO 代理与自相关注意力 | **CVPR'26 Findings** | 降为辅助证据 |
| LPOSS、ProxyCLIP（请求 1） | 深度像素特征只 +0.1；GMG 的前身 | CVPR'25、ECCV'24 | 保留 |
| Cetinkaya 等（检测） | 估计深度对检测无益 | Image and Vision Computing（期刊） | 移到附录 C |
| Xiang 等（COD 深度贡献） | 伪深度直接融合有害 | 只在 arXiv | 移到附录 C；改用 PopNet [ICCV'23] 表 7 |
| GeoDistill（When Depth Hurts） | 增益来自分支容量 | 只在 arXiv | 移到附录 C；改用 FiT3D [ECCV'24] 特征复制对照、DFormerv2 空间先验 |
| GeoCueFormer | 深度注入所有 stage 会掉点 | 只在 arXiv | 移到附录 C；改用 Vireo、OVCamo |
| 深度环 OOD | 免训练深度判前景 | 只在 arXiv | 移到附录 C；改用 PopNet "凸起"先验 |
| DepSeg（See in Depth） | 深度提示 SAM | 只在 arXiv | 移到附录 C；改用 CutS3D [ICCV'25]、CorrCLIP [ICCV'25] |
| Cai & Metzler（水下度量深度） | 水下深度退化 | 只在 arXiv | 移到附录 C；改用 Atlantis [CVPR'24] |
| Beneath the Surface | 增强对检测无益 | 期刊预印本 | 移到附录 C；顶会上没有对应证据，G6 改为"无依据" |
| PEARL | 免训练 OVSS 里的 "geometry" 含义 | 只在 arXiv（检索未见接收信息） | 移到附录 C |
| TPA、GeoSeg-OV、LoCoFuse | 决策层融合、遥感用 DAv2 | arXiv / 未知 | 移到附录 C |

**一句话总结**：v1 精读的 16 篇里，确认是主会顶会的只有 6 篇（另有 1 篇 Findings）。v2 精读的顶会主会论文共 22 篇（另有 1 篇 Findings），另有约 24 篇只在官方站点核实了 venue、读过摘要。

---

## 2 问题界定

### 2.1 E2O 到底用了 DepthAnything 的什么

| 信息 | E2O 是否使用 | 依据【本项目事实，官方代码逐行核对】 |
|---|---|---|
| block-11 patch token（24×24，余弦自相似后替换 CLIP 末层注意力） | 用 | `Earth2Ocean.py` L259–268；`open_clip/transformer.py` L618–662 |
| block 2/5/8 中间层特征 | 没用；但 `custom_attn` 已支持传入列表，对各层注意力取平均（L635 起） | `base_config.py` `geofeats_idx='3'` |
| 注意力的 Q/K（而非输出特征） | 没用 | 同上 |
| DPT 头输出的相对逆深度 `geo_map` | **算了但丢弃** | `Earth2Ocean.py` L260 返回，L289 用 `_` 接收 |
| DPT 多尺度解码特征 `path_1..4` | 没用 | `depth_anything_v2/dpt.py` L141–146 |
| 深度不连续、3D 邻近、法向、物理量、尺度 | 都没用 | — |

另外两点影响后续实现【本项目事实】：
- **深度按窗口归一。** DA 在每个 336×336 滑窗上单独跑，深度只在窗口内做了仿射归一，跨窗口的尺度和偏移不一致。
- **模型没有背景输出。** `prob_thd` 把低置信像素置为 0，而 0 是一个真实类别。要加背景，必须用新标签 C，这与 `protocol_eval.py` 的约定一致。MMSeg v1.1.1 的 `IoUMetric.intersect_and_union` 用 `torch.histc(min=0, max=C−1)` 统计【本项目事实，源码核对】，按 torch 文档会忽略标签 C，所以 P0 下这些像素算作漏检，语义正确【推断，依据 torch 文档；建议用一张图实测】。

### 2.2 为什么 A4≈A3 在意料之中
- 【文献事实】Depth Anything [CVPR'24]：
  - 编码器用 DINOv2 初始化；
  - 在无标签数据上加 `1−cos(f, f′)` 特征对齐损失，f′ 来自冻结的 DINOv2；
  - 余弦超过 α=0.85 的像素不计入损失；
  - 下游微调到 ADE20K：DINOv2 58.8，DA 59.4。
- 【文献事实】Depth Anything V2 [NeurIPS'24]：学生模型基于 DINOv2 S/B/L/G，"沿用 V1，加特征对齐损失以保留预训练 DINOv2 的语义"。
- 【文献事实】Probe3D [CVPR'24]，冻结特征 + 多尺度探针：

| 模型（冻结特征） | NYU 深度 δ1 | NAVI 深度 δ1 | NYU 法向 11.25° |
|---|---|---|---|
| DINOv2 ViT-B/14 | 93.43 | 68.84 | 62.01 |
| DINO ViT-B/16 | 80.44 | 56.62 | 49.11 |
| CLIP ViT-B/16（LAION） | 52.46 | 24.33 | 30.11 |
| MiDaS ViT-L/16（深度监督） | 78.65 | 58.53 | 49.58 |

- 【推断】从中可得三点：
  - CLIP 几乎不含几何；
  - "换上 DINOv2 系代理"这一步本身就引入了隐式几何；
  - 深度监督编码器的深度可读性不一定高于 DINOv2，所以"用深度模型的编码器特征"不等于"用几何"。

### 2.3 "真正利用几何"的可操作定义
1. **显式几何量。** 使用 DINOv2 特征中无法直接调用的信息，例如远近顺序、深度不连续、3D 距离、法向、物理量。
2. **因果可归因。** 把深度换成错配深度（同一数据集另一张图的深度）后，增益应显著消失；模糊深度、非几何平滑图、仅空间项都不能复现这一增益。
3. **超过同构对照。** 凡是用 DA 特征的方案，都要超过同一流程下的 DINOv2-B/14；凡是集成多个源的方案，都要超过"同等数量的非几何源"。后者参照 FiT3D [ECCV'24] 的特征复制对照。
4. **机制与指标对应，并有事先写定的异质性预测。** 例如深度对比度高的图增益更大。不把"深度质量剂量—反应"当证伪依据，原因见 §4.5 R4。

### 2.4 余量账本

| 环节 | 现有余量【本项目事实】 | 可能的几何作用 | 判断 |
|---|---|---|---|
| P0 分组 | GT-Uniform oracle +1.3/+2.5/+2.1（B/L/H） | 3D 亲和、多层、多源 | 上限小 |
| P0 识别 | 文本头 33.17% vs 视觉质心 85.51% | 几何对语义帮助小（R1） | 几乎不可期 |
| 前景/背景（P1/P2） | 没有背景输出，所有配置 P2 前景 IoU 相同 | 远近顺序、"凸起"先验 | 空间最大，需要预检 |
| 边界 | 尚未度量 | 深度不连续 | 待验证；有负证据（LPOSS [CVPR'25] +0.1） |
| 多类场景（SUIM/DUT） | P0 整图填充 oracle 约 46 | 区域/范围约束（CorrCLIP 在 SUIM 上 58.0） | 中等 |

【本项目事实】几何特征会加剧同科物种之间的混淆。这与 R1 一致，所以不要把几何特征塞进分类器。

---

## 3 检索策略记录（v2）

### 3.1 工具与数据库
- **发现与 venue 核实**：Web 检索，限定在官方论文集域名内（每次检索都用 `allowed_domains` 过滤）：
  - openaccess.thecvf.com（CVPR/ICCV）、ecva.net（ECCV）；
  - openreview.net、proceedings.neurips.cc（ICLR/NeurIPS）；
  - proceedings.mlr.press（ICML）、ojs.aaai.org（AAAI）、dl.acm.org（ACM MM）、ijcai.org；
  - papers.miccai.org / link.springer.com（MICCAI）、ieeexplore.ieee.org（ICRA/IROS）。
- **辅助核实**：Consensus（Semantic Scholar/OpenAlex 的 venue 字段）；官方 GitHub README（经 raw.githubusercontent.com 读取）。
- **全文问答（精读）**：alphaXiv。
- **限制**：dblp、Semantic Scholar API、CVF 直连在本环境被代理拒绝；Elicit 没有 API 权限。
- **规模与时间**：约 22 次限定域名检索，2026-10-10 进行。

### 3.2 关键词（按主题）
- 深度/几何与 2D 分割：monocular depth / pseudo depth semantic segmentation、geometry prior attention、depth pre-training for segmentation、3D-aware fine-tuning
- 无监督/零样本：depth-guided unsupervised semantic / instance segmentation、training-free object discovery depth
- 显著性/伪装（垂直）：source-free depth object pop-out、depth-aided camouflaged、open-vocabulary camouflaged、depth unreliable RGB-D SOD
- 开放词汇与多源融合：training-free OVSS DINO SAM CLIP、agglomerative RADIO、open-vocabulary segmentation depth、depth and language domain generalization
- 上采样：feature upsampling VFM（LoftUp、JAFAR、AnyUp、FeatUp、NAF）
- 水下（垂直）：underwater segmentation SAM、marine open-vocabulary、underwater depth estimation
- 医学/遥感/机器人（垂直）：depth surgical segmentation MICCAI、remote sensing training-free OVSS、underwater segmentation ICRA/IROS

### 3.3 各领域阅读量（只计顶会）

| 领域 | 精读（全文） | 只核实 venue + 摘要【未核实细节】 |
|---|---|---|
| 深度/几何与 2D 语义分割 | DFormer [ICLR'24]、DFormerv2 [CVPR'25]、MultiMAE [ECCV'22]、PDDM [AAAI'25]、FiT3D [ECCV'24] | 单目深度预训练的可行性 [ECCV'24]、DiPFormer [IROS'25] |
| 无监督/零样本分割与深度 | DepthG [CVPR'24]、CutS3D [ICCV'25] | MoonSeg3R [CVPR'26 Findings] |
| 显著性/伪装（垂直） | PopNet [ICCV'23]、OVCamo [ECCV'24] | DaCOD [ACM MM'23]、DepthSAM [CVPR'26]、免训练 OV-COS [CVPR'26] |
| 免训练 OVSS 与多源融合 | ProxyCLIP [ECCV'24]、LPOSS [CVPR'25]、CASS [CVPR'25]、Trident [ICCV'25]、CorrCLIP [ICCV'25]、RADSeg [CVPR'26 Findings] | AM-RADIO [CVPR'24]、RADIOv2.5 [CVPR'25]、CLIPer [ICCV'25]、FreeCP [ICCV'25]、The Power of Prior [CVPR'26]、SigLino [CVPR'26]、OV-Stitcher [CVPR'26 Findings] |
| 开放词汇 + 深度 + 域偏移 | Vireo [NeurIPS'25] | — |
| 特征上采样 | NAF [CVPR'26]；FeatUp/JAFAR/AnyUp 的数字由 NAF 表格同表转引 | FeatUp [ICLR'24]、LoftUp [ICCV'25]、JAFAR [NeurIPS'25]、AnyUp [ICLR'26]、Upsample Anything [CVPR'26]、UPLiFT [CVPR'26] |
| VFM 的 3D 感知与 DA 本身 | Probe3D [CVPR'24]、Depth Anything [CVPR'24]、Depth Anything V2 [NeurIPS'24] | Lexicon3D [NeurIPS'24] |
| 水下（垂直） | E2O [CVPR'26]（代码与论文）、MARIS [CVPR'26]、Atlantis [CVPR'24] | USIS-SAM [ICML'24]、MAS-SAM [IJCAI'24]、CoralSCOP [CVPR'24]、BiPA [CVPR'26] |
| 遥感（垂直） | SegEarth-OV [CVPR'25] | — |
| 医学（垂直） | — | MICCAI 2024/2025 只检到深度估计论文，没有"深度引导分割"的论文 |
| 合计 | 22 篇主会 + 1 篇 Findings | 约 24 篇 |

### 3.4 局限
- 只检索了英文顶会，中文顶刊和 TPAMI/IJCV 等顶刊没有纳入主证据。
- CVPR 2026、NeurIPS 2025 的论文较新，部分只能读到 arXiv 版本，与正式版可能有出入。
- 负结果少发表，负面证据应视为下限。
- 每个垂直领域的顶会论文数量有限，见 §4.6"空白"。

---

## 4 分类体系与证据（只含顶会）

### 4.1 维度
- **用了什么几何信息**：
  - I0 深度模型编码器特征（隐式）；
  - I1 远近顺序/前景"凸起"；
  - I2 深度不连续；
  - I3 3D 邻近（深度差 + 空间距离）；
  - I5 物理量；
  - I6 度量尺度。
- **注入位置**：输入融合 / 注意力与亲和 / 后期精修 / 区域与提示 / 只作训练信号。
- **设定**：免训练 / 轻训练 / 训练式。

### 4.2 总表

| 类 | 几何 × 位置 × 设定 | 代表工作 | 报告效果【文献事实】 | 条件 / 局限 |
|---|---|---|---|---|
| T0 | I0 × 亲和 × 免训练 | E2O GMG [CVPR'26]【本项目事实】 | A4−A3 = +0.09/+0.38/+0.61；DUT 全为负 | 与 DINOv2 同源 |
| T1 | I0 × 门控融合 × 训练式 | MARIS GPEM [CVPR'26]：冻结的 DA 多尺度特征，α=σ(W[F_V‖F_G])，F_VG=MLP(F_V+α⊙F_G) | 大骨干：已见类 54.29→60.05；**未见类 50.99→52.19**；总体 52.17→54.99。门控比 MLP/相加高 0.7–1.0。跨域时 vitb（46.18）> vitl（43.70） | 需要标注训练；**没有"非深度编码器"对照** |
| T2 | I0+I2 × 编码器逐层查询 × 训练式 | Vireo [NeurIPS'25]：冻结的 DAv2 + DINOv2/EVA02，GeoText Query | 在 REIN 上：**直接拼接深度特征 +0.1~0.3（雾天 −0.1）**；Prompt-DA 时好时坏；DAv2 +0.5~0.9；DA+AO +1.4~1.7；GeoText Query +1.6~3.4 | 训练式域泛化；没有非深度对照 |
| T3 | I1 × "凸起"先验 × 训练式 | PopNet [ICCV'23]：物体位于接触面之前；DPT 估计深度 + 学习接触面 | NC4K：RGB .825 → 加估计深度 .839 → PopNet .861（Fm）；约 76% 的图获益 | 物体被深度网络"骗过"（高度伪装）时失败 |
| T4 | I3 × 注意力偏置 × 训练式 | DFormerv2 [CVPR'25]：(Softmax(QKᵀ)⊙β^G)V | NYU/SUN：深度先验 +2.6/+1.7；**仅空间先验 +1.8/+1.3**；两者合用 +4.5/+3.9。LUSS：分类 83.1→83.4，前景 wF 0.818→0.868 | RGB-D 室内；空间项本身贡献很大 |
| T5 | I2/I3 × 亲和锐化 + 3D 切割 × 生成伪标签免训练、检测器需训练 | CutS3D [ICCV'25]：Spatial Importance ΔD=abs(G_σ*D−D) 对语义亲和逐元素做指数锐化；在点云上做 MinCut | COCO AP_mask：CutLER 9.4 → +SIS 9.6 → +LocalCut 9.8；整体 +0.7~1.2。深度源：ZoeDepth 9.1、KBR 9.1、Marigold 8.9、MiDaS-S 8.7 | 相邻同类、3D 边界不明显时失败 |
| T6 | I3 × 训练信号 × 训练式 | DepthG [CVPR'24] | COCO-Stuff ViT-S/8 24.5→25.6（加 3D-LHP 为 26.7）；ViT-B/8 28.2→29.0；Cityscapes 21.0→23.1 | 深度质量有影响；推理时不用深度 |
| T7 | I0/I1 × 预训练 × 训练式 | DFormer [ICLR'24]：ImageNet + AdaBins 估计深度做 RGB-D 预训练 | 比 RGB 预训练 +2.3（DFormer-B）；CMX +1.4。**AdaBins 换成 Omnidata 结果相同**（55.6） | 微调阶段用传感器深度 |
| T8 | I1 × 伪深度输入 × 训练式 | MultiMAE [ECCV'22]；PDDM [AAAI'25] | MultiMAE 加伪深度：ADE +0.6、Hypersim +0.9、NYU +1.6（真实深度 NYU +4.0）；**MAE 加伪深度 NYU −1.0**。PDDM 用 DA 伪深度：EMSANet +3.3、CMX +3.5、CMNeXt +0.8、DFormer +0.8、PDDM +0.75；三张伪深度直接相加，CMX 52.43→51.19 | 模型越强增益越小；融合方式决定正负 |
| T9 | I1/I2 × 只作训练辅助 × 训练式 | OVCamo / OVCoser [ECCV'24]：估计深度和边缘作辅助监督 | cIoU 0.397 → 加深度 0.422 → 加边缘 0.432 → 两者 0.436；**换成相加融合只有 0.402** | 推理时不用深度 |
| T10 | 隐式 3D × 特征微调 × 训练式 | FiT3D [ECCV'24] | DINOv2-S 线性探针：NYU +2.0、ADE +1.6、VOC +1.2；**特征复制对照只 +0.1**；分类不变 | 需要多视图 3D 数据 |
| T11 | I2 × 后期精修 × 免训练 | LPOSS [CVPR'25] | 深度作为像素特征 +0.1（42.2 vs 42.1） | 直接负证据 |
| T12 | 图像引导 × 上采样 × 零样本 | NAF [CVPR'26]；SegEarth-OV 的 SimFeatUp [CVPR'25] | NAF：VOC 线性探针相对最近邻 +5.58，开放词汇 +0.44。SegEarth-OV：ClearCLIP + 上采样 + 去全局偏置在遥感上 +8.1~+15.5；**原版 FeatUp 在 2/3 数据集上反而下降** | 都没用深度；上采样器的选择决定正负 |
| T13 | 多源 × 范围/亲和 × 免训练 | CorrCLIP [ICCV'25]、Trident [ICCV'25]、CASS [CVPR'25] | CorrCLIP 用 SAM 区域限定注意力范围，加到 ProxyCLIP 上 +1.4~5.7；ViT-B 平均 51.0；**SUIM（MESS）58.0**。Trident 相对 ProxyCLIP +3.5（B）/+4.2（H），但吞吐 118.8→10.0 张/秒。CASS 用 VFM 注意力 KK 比用特征 +2.9 | 自然图像为主；不含深度 |
| T14 | 多源聚合骨干 × 免训练 | AM-RADIO [CVPR'24]、RADIOv2.5 [CVPR'25]、RADSeg [CVPR'26 Findings] | RADSeg：ProxyRADIO 45.44/46.05 vs ProxyCLIP 40.87 | 属于更强的通用特征 |
| T15 | I6 × 深度估计域适配 × 训练式 | Atlantis [CVPR'24] | 陆地模型在 D3/D5 上 δ1 0.09–0.22，在 SQUID 上 0.01–0.20；用 Atlantis 训练后 0.48–0.68 | 水下度量深度难 |

### 4.3 正面证据要点
- **深度能把物体从背景中"凸起"。** PopNet [ICCV'23] 给出了最直接的先验："物体位于接触面之前"。约 76% 的图因深度获益。
- **深度不连续可以免训练地注入亲和。** CutS3D [ICCV'25] 用深度高频图锐化语义亲和，并在 3D 中切分实例。
- **几何先验进入注意力有效**，但空间项本身贡献很大（DFormerv2 [CVPR'25]）。
- **冻结的 DAv2 在开放词汇 + 域偏移场景中有小幅稳定增益**：Vireo [NeurIPS'25] +0.5~1.7。门控融合的 DA 多尺度特征对已见类帮助大（MARIS [CVPR'26]）。
- **免训练多源（SAM/DINO/CLIP）在自然图像上增益大**，而且在水下 SUIM 上有效（CorrCLIP [ICCV'25]）。

### 4.4 负面与限制性证据
- **直接融合估计深度常常无益或有害。**
  - PopNet [ICCV'23] 表 7：DASNet 加估计深度后 COD10K Fm .643→.642、NC4K .747→.743；SPNet 在 NC4K 上 .846→.828。作者原话："只有少数 RGB-D 方法能从这种深度中获益。"
  - MultiMAE [ECCV'22]：MAE 加伪深度，NYU −1.0。
  - Vireo [NeurIPS'25]：直接拼接 ≈ 0。
  - OVCamo [ECCV'24]：相加融合会丢掉大部分增益。
  - PDDM [AAAI'25]：多张伪深度直接相加会掉点。
  - DepthSAM [CVPR'26]：直接用单目深度会"任务错配"【未核实（仅摘要）】。
- **几何对识别帮助很小。**
  - FiT3D [ECCV'24]：分类不变。
  - MARIS [CVPR'26]：未见类只 +1.2。
  - DFormerv2 [CVPR'25]：分类 +0.3。
  - OVCamo [ECCV'24]：即使用 GT 掩码池化，CLIP 分类准确率也只有 0.704。这与本项目"识别是瓶颈"相同。
- **深度作像素特征做后期精修，只 +0.1**（LPOSS [CVPR'25]）。
- **水下度量深度很差**（Atlantis [CVPR'24]）。

### 4.5 归纳出的规律（v2，依据全部是顶会）
- **R1 深度主要改善定位，很少改善识别。** 依据：DFormerv2、FiT3D、MARIS、OVCamo。→ 主指标应选 P1/P2、边界、多类场景。
- **R2 直接融合估计深度常常无益，有效的做法必须有专门设计。** 专门设计指可靠性/门控、浅层查询或训练辅助、结构先验（"凸起"、3D 切割）。依据：PopNet、MultiMAE、Vireo、OVCamo、PDDM、DepthSAM。→ 免训练方案也要内置"什么时候信任深度"的门。
- **R3 骨干越强，深度的边际增益越小。** 依据：PDDM 表 2、Probe3D。
- **R4 对深度质量的依赖弱于预期（与 v1 的结论不同）。** CutS3D 中各深度源相差不到 0.4 AP；DFormer 中 AdaBins 与 Omnidata 结果相同；但 DepthG 显示质量有影响，Vireo 中 Prompt-DA 比 DAv2 差。→ 不能把"深度质量剂量—反应"当证伪依据，主要靠打乱深度和平滑图对照。
- **R5 容量、空间先验、通用特征这三类混淆很常见。** DFormerv2 的空间先验单独就有 +1.8；FiT3D 用特征复制对照排除了维度效应，是好的做法；MARIS 和 Vireo 都没有"非深度 VFM"对照，所以它们的增益不能完全归因于几何。
- **R6 水下：相对远近可用，度量距离不可用。** 依据：DAv2 [NeurIPS'24] 的 DA-2K 水下子集；Atlantis [CVPR'24]。
- **R7 免训练的多源范围/亲和融合在自然图像上增益大。** 但本项目 P0 的分组余量只有 +1.3。【推断】可能获益的是 SUIM/DUT 这类多类场景。依据：CorrCLIP、Trident、CASS；本项目的 oracle。
- **R8 训练式几何融合的增益，主要落在已见类和定位上；对开放词汇未见类很小。** 依据：MARIS（+5.8 vs +1.2）；Vireo（它对未见类的领先幅度 2–3 点，小于对已见类的 7–10 点）。

### 4.6 各垂直领域的顶会空白
- **水下**：顶会上用到深度的只有 MARIS [CVPR'26]（训练式）。USIS-SAM、MAS-SAM、CoralSCOP、BiPA 从摘要看都没有用深度【未核实】。**免训练 + 深度：空白。**
- **医学（MICCAI 2024/25）**：只检到深度估计论文，没有"深度引导分割"的论文。
- **遥感**：SegEarth-OV [CVPR'25] 是免训练的，但没用高度/深度；nDSM 融合在本次纳入的顶会范围内没有检到。
- **驾驶/机器人**：DiPFormer [IROS'25] 用传感器深度做注意力偏移，摘要报告道路检测 +7.5%、分割 +4.9%/+1.5%【未核实（仅摘要）】；Vireo [NeurIPS'25] 用冻结的 DAv2 做域泛化。

---

## 5 候选方案卡片

成本按你们的条件估算：5×RTX 3090；ViT-B 跑一遍六个数据集约 75 min【本项目事实】。图片数：AquaOV255 20722、USIS16K 16150、USIS10K val+test 3189、MAS3K 2909、SUIM 1487、DUT 1486，共约 4.6 万张【本项目事实】。

### D0 诊断：A4 现有的增益是否"几何的"
- **机制**：
  - 逐图计算 Δ_i = A4_i − A3_i（用逐图像素准确率或 P2 指标）；
  - 与 G1 预检脚本给出的逐图 `ring_contrast` 和 `auroc_depth` 做 Spearman 相关；
  - 同时跑 DA 多层 `geofeats_idx="0,1,2,3"`，与 DINOv2 同层多层对比。
- **文献依据**：Depth Anything [CVPR'24]、Depth Anything V2 [NeurIPS'24]、Probe3D [CVPR'24]（§2.2）。
- **设定**：免训练，只做分析。
- **成本**：深度缓存约 1 GPU·h，分析 1 人日，多层 DA 一个配置 75 min。
- **判据**：ρ 的 95% CI 不含 0，且 ≥4/6 个数据集同号 → "A4 的残余增益有几何成分"；否则如实写"没有可归因于几何的证据"。
- **预期**：ρ≈0【推断】。
- **风险**：低。

### G1 深度 figure-ground / 背景通道（首推）
- **机制**：对整图算一次 DAv2 相对逆深度，判定前景（近）与背景（远），输出背景标签 C。按把握递进有三种变体：
  - (a) Otsu 阈值；
  - (b) PopNet 式的"凸起"判定：以模型自己预测的连通区域为单位，区域内视差减去外环视差，再除以 MAD，小于 δ（定为 0）的区域判为背景；
  - (c) 文本"背景"提示竞争，作为非几何对照。
- **几何**：I1 远近顺序 + 局部深度对比。
- **顶会依据**：
  - PopNet [ICCV'23]："物体位于接触面之前"的先验，NC4K Fm .825→.861，约 76% 的图获益。但它是训练式的：接触面是学出来的，不是免训练。
  - DFormerv2 [CVPR'25]：前景 wF +0.05。
  - DAv2 [NeurIPS'24]：水下远近判断可靠。
- **从训练式到免训练的转化**：
  - PopNet 用标注学接触面；我们用无标签规则替代，即外环对比 / Otsu。【推断】这一步在顶会上没有先例，是本项目的潜在贡献点，也是主要风险。
- **迁移理由**：SUIM、USIS10K 的背景主要是水体（类表中没有 waterbody，海床是一个类）【本项目事实】。
- **迁移障碍**：
  - DUT 四类都是底栖动物，与海床几乎共面【本项目事实】；
  - MAS3K 有大量伪装物种【本项目事实，类表】，对应 PopNet 的失败情形；
  - P0 不计背景，加背景在 P0 上只会扣分。
- **设定**：免训练。
- **实现**：
  1. `dump_depth_cache.py` 缓存深度；
  2. 在 `postprocess_result` 里加 `seg_pred[bg_mask]=num_classes`；
  3. 用 `RunAccumulator` 同时得到 P0/P1/P2；
  4. 如果有逐图预测，可以离线组合，不必重跑。
- **成本**：缓存约 1 GPU·h；预检不到 1 h（CPU，每个数据集 500 张）；接入 0.5 人日。
- **前置验证**：`prevalidate_depth_figure_ground.py`。
  - **通过**（`G1_pass`）：在 ≥4/6 个数据集上同时满足
    - AUROC_depth ≥ 0.70；
    - 比中心先验和打乱深度对照中最好的那个高 ≥ 0.05，差值 CI 下界 > 0；
    - Otsu 前景 IoU 比全前景高，差值 CI 下界 > 0.05。
  - **放弃**：在 ≥3/6 个数据集上 AUROC < 0.60，或不优于对照。
- **预期**：
  - P2 前景 IoU：**无依据**。预检的 `iou_otsu_depth − iou_allfg` 就是离线估计值。
  - P0：0 到负。
  - 分数据集排序的事先预测【推断】：SUIM、USIS10K > AquaOV255、USIS16K > MAS3K、DUT。依据是背景构成和 PopNet 的失败模式。
- **风险**：各数据集的背景编码不一致；浑浊远景深度噪声大。结论只能在 P1/P2 框架下陈述。

### G2 3D 感知亲和
- **机制**：两种变体。
  - **G2a（DFormerv2 形式，β 固定、不学习）**：在 S̃ 上减去 `pen_ij = s·(λ_z·abs(z_i−z_j)/med + λ_s·L1(p_i−p_j)/med)`。
    - soft 模式等价于 A⊙exp(−pen) 再行归一化（已数值验证）。
    - z 用窗口内中位数/MAD 归一，对 DA 的尺度和偏移不变；s 是 S̃ 正值的中位数。
  - **G2b（CutS3D 形式）**：先算深度高频图 ΔD = abs(G_σ*D − D)，归一到 [β,1]（β=0.45，取 CutS3D 原文默认值），再对语义亲和做逐元素指数锐化。
    - 【未核实】原文公式中 ΔD 的下标 i,j 如何对应到成对亲和，需要看官方代码确认。确认之前建议用 max(ΔD_i, ΔD_j)。
- **几何**：I3 3D 邻近（G2a）；I2 深度不连续（G2b）。
- **顶会依据**：
  - DFormerv2 [CVPR'25]：深度 +2.6/+1.7，空间 +1.8/+1.3（训练式）；
  - CutS3D [ICCV'25]：免训练的锐化步骤，AP +0.2；
  - NAF [CVPR'26] 表 7：空间编码很重要；
  - 负证据：LPOSS [CVPR'25] +0.1。
- **设定**：免训练。主配置事先写定为 G2a soft、λ_z=0.25、λ_s=0；剂量档 {0.125, 0.25, 0.5, 1} 全部报告。G2b 的 β 取原文默认。
- **实现**：`scripts/e2o_patches/geo_affinity.py`（numpy 参考实现已测试；torch 移植未实跑）。G2b 需要另加约 20 行代码。
- **必做对照**：
  - 仅空间项；
  - 打乱深度；
  - DINOv2 第一主成分平滑图（`pca1_map_torch`）；
  - 模糊深度。
- **成本**：1–1.5 人日；约 10 个配置，约 12 GPU·h，5 卡并行约 2.5 h。
- **前置验证**：预检脚本的 A1 部分，即控制空间距离后，深度是否仍能区分同区域/异区域的像素对。
  - **通过**：`pair_auroc_depth_given_spatial` ≥ 0.60，且 CI 下界 > 0.55。
- **预期**：P0 0~+0.5（ViT-B）【推断，依据 oracle 上限 +1.3，以及 CutS3D 锐化步骤只有 +0.2 的量级】。SUIM/DUT：无依据。
- **风险**：仅空间项就复现了大部分增益（R5）。那样结论只能写成"局部性先验有效"。

### G3 深度引导的边界精修/上采样
- **机制**：把 24×24 logits 的双线性上采样（`forward_feature` L313/315）换成联合双边上采样（JBU），值域核用 [RGB, 深度] 或只用深度。强 RGB 对照用 NAF，以及 SegEarth-OV 式的 JBU。
- **几何**：I2 深度不连续。
- **顶会依据**：
  - NAF [CVPR'26]；
  - SegEarth-OV [CVPR'25]：垂直领域免训练 OVSS 中，图像引导上采样 + 去全局偏置 +8.1~+15.5；但**原版 FeatUp 反而下降**。
  - 负证据：LPOSS [CVPR'25]。
- **设定**：免训练（NAF 和 SimFeatUp 需要一次性的无标签预训练，作为"轻训练"另行标注）。
- **前置验证**：预检脚本的 B1 部分。
  - **通过**：bF_depth − bF_lum ≥ 0.02，CI 下界 > 0，在 ≥3 个数据集上成立。
- **预期**：P0 ≈0；边界 F：无依据。
- **风险**：增益其实来自"任何边缘感知平滑"，所以 RGB 对照必须做。

### G4 DA 多层特征、注意力 key 与 DPT 特征
- **机制**：三种做法。
  - (a) block 2/5/8/11 多层注意力取平均（现成功能）；
  - (b) **借鉴 CASS [CVPR'25]**：用 DA 末层的注意力 key 做 KK 自相似，替代输出特征的余弦相似。CASS 中 KK 比用特征 F 高 +2.9；
  - (c) DPT 的 `path_1..4` 特征。
- **同构对照**：DINOv2-B/14 用完全相同的做法。
- **顶会依据**：Probe3D [CVPR'24]（3D 信息分布在多层）；CASS [CVPR'25]（特征类型消融）。
- **设定**：免训练。
- **成本**：(a) 零代码；(b) 在 `get_geo_feats` 中取最后一个 block 的 key，0.5 人日；每个配置 75 min。
- **放弃标准**：DA 版本不优于 DINOv2 同构版本（差值 CI 含 0）。
- **预期**：P0 −0.5~+0.5【推断】。(b) 可能有正增益，但这属于"通用特征用法"上的改进，与几何无关，必须同时报告 DINOv2 的 KK 版本。

### G5 区域先验：SAM/SAM2 范围约束与深度提示
- **机制**：
  - 借鉴 CorrCLIP [ICCV'25]：用 SAM 区域掩码限定注意力范围（scope reconstruction）；
  - 可选：借鉴 CutS3D [ICCV'25]，在区域内再按 3D 点云 MinCut 切分；
  - 与 G1 结合，在区域级判断前景/背景。
- **顶会依据**：
  - CorrCLIP：加到 ProxyCLIP 上 +1.4~5.7；SUIM 上 58.0；
  - Trident [ICCV'25]：相对 ProxyCLIP +3.5，但吞吐降到 1/12；
  - CutS3D：LocalCut +0.2。
- **设定**：免训练。
- **成本**：SAM2 自动掩码约 13 GPU·h（4.6 万张），5 卡约 3 h；3–5 人日。
- **前置验证**：每个数据集 500 张，比较 SAM2 区域与 GT 的类无关 best-overlap mIoU；同时比较 CorrCLIP 式范围约束在 SUIM/DUT 的 GT-Uniform oracle 下能带来多少余量。
- **预期**：P0 六集平均 0~+1（ViT-B）；SUIM/DUT 单集可能更大【推断，依据 CorrCLIP 在 SUIM 上的结果和本项目的 oracle】。
- **风险**：工作量大；范围约束是"多源融合"，不算几何，只有深度提示和 3D 切割那部分才算几何。

### G6 物理复原用于识别
- **机制**：用深度和成像模型估计后向散射与衰减并复原，再送入 CLIP；或者原图与复原图集成后，用无标签准则选择。
- **顶会依据**：Atlantis [CVPR'24] 只展示了基于深度的 Sea-thru 复原效果（定性），**没有任何顶会证据表明复原能提升识别或分割**。
- **预期**：**无依据**。
- **前置验证**：用区域分类探针比较原图、灰度世界白平衡、深度复原三种输入。
  - **通过**：深度复原比原图高 ≥1.0，比灰度世界高 ≥0.5，CI > 0。
- **优先级**：低。

### G7 尺寸先验（最低优先级）
- 需要度量深度和相机内参，但水下度量深度很差（Atlantis [CVPR'24]）。没有顶会依据。只建议做 GT 区域上的 oracle 上限分析。

### M1 亲和层多源融合（不宣称是几何）
- **机制**：
  - (a) 多个源的注意力取平均（`custom_attn` 列表模式）；
  - (b) 共识融合（专家乘积）；
  - (c) 一致性门控；
  - (d) 用 VFM 注意力 key 构图（CASS）；
  - (e) RADIOv2.5-B 作代理（[CVPR'25]；RADSeg [CVPR'26 Findings]）。
- **对照**：**同等数量的源**。例如 A3 + A3′（DINOv2 的两种尺度或两组层）对比 A3 + A4，用来排除"集成本身"的效应，参照 FiT3D 的特征复制对照。
- **预期**：P0 0~+0.8（ViT-B）【推断】。

### M2 决策层多源融合（识别端）
- 依据 R1/R7，多源融合真正有余量的地方是识别端：文本 + 视觉原型 + MLLM，见请求 1 的 B3。
- 顶会可参考的有：CASS [CVPR'25] 的"物体存在先验"文本嵌入调整；The Power of Prior [CVPR'26] 用 LLaVA 先验【未核实（仅摘要）】。
- 【推断】不要把几何特征加进原型。

### T1 训练式门控融合（单列设定，与免训练分开报告）
- **机制**：MARIS [CVPR'26] 式门控，或 Vireo [NeurIPS'25] 式 GeoText Query。
- **轻训练变体**：在测试集以外的无标签水下图上，用 SAM2 掩码作伪标签，训练"两个 token 是否属于同一掩码"的小 MLP，输入 [S_DINOv2, S_DA, abs(Δz), L1(Δp)]。
- **必须补的对照**：MARIS 和 Vireo 都没做的"非深度编码器替换"对照。
- **预期**：无依据；P0 上限仍为 +1.3。
- **风险**：数据泄漏，需先完成 C1 去重。

---

## 6 评测设计

### 6.1 为什么 P0 看不到几何
- 【本项目事实】P0 下的整图填充 oracle：AquaOV255 99.66、MAS3K 99.97、USIS16K 99.45。
- 【本项目事实】完美分组只 +1.3/+2.5/+2.1。
- 所以每张卡都事先指定"机制对应的主指标"，P0 作为次要指标照常报告。

### 6.2 各卡的主指标（预注册）

| 卡 | 主指标 | 次要指标 | 范围 |
|---|---|---|---|
| D0 | Spearman ρ(逐图 Δ, ring_contrast) | 三分位均值 | 六集，分集报告 |
| G1 | P2 类无关前景 IoU | P1b、P1 背景 IoU、P0（代价） | 六集，附分集排序预测 |
| G2、G4、M1 | P0 六集平均 | SUIM/DUT 的 P0、P1b、边界 F | 六集 |
| G3 | 边界 F（2 px） | P0 | SUIM、DUT、含 ≥2 个 GT 类的图（按 GT 事先定好子集） |
| G5 | SUIM/DUT 的 P0 | 六集平均、P2 | 六集 |
| G6 | 区域分类文本头准确率 | P0 | AquaOV255 |

### 6.3 同源规则
- 所有比较都来自同一次推理，计算相对直接基线（A4 或 A3）的差值。
- 骨干（LAION OpenCLIP B/16）、文本侧（UW + CSA(4o)）、类表、滑窗（336/112）、深度缓存版本（DAv2-B，518 输入）都保持一致。
- ViT-B 过了判据，才扩到 L/H。

### 6.4 怎样排除"增益来自更强的通用视觉特征，而不是几何"

| 替代解释 | 对照 | 顶会先例 | 适用卡 |
|---|---|---|---|
| DA 特征只是 DINOv2 的变体 | 同构 DINOv2-B/14（同层、同网格、同流程） | — | D0、G4、M1 |
| 局部性/空间先验 | 仅空间项（λ_z=0）；中心先验 | DFormerv2 [CVPR'25]：空间先验单独 +1.8 | G1、G2 |
| 任何平滑图都行 | DINOv2 第一主成分图 | — | G1、G2 |
| 与图像内容无关的正则化 | 打乱深度（同数据集下一张图） | — | G1、G2、G3 |
| 增加了特征维度或源的数量 | 特征/源复制对照（A3 + A3′ vs A3 + A4） | FiT3D [ECCV'24]：复制特征只 +0.1 | G4、M1、T1 |
| 任何边缘感知平滑都行 | RGB 引导的 JBU；NAF | SegEarth-OV [CVPR'25] 中不同上采样器有正有负 | G3 |
| 深度质量剂量—反应 | 只作为描述性报告，**不作为证伪依据** | CutS3D [ICCV'25]、DFormer [ICLR'24]：质量影响小 | 全部 |

**事先写定的异质性预测**：按逐图 `ring_contrast` 分三档，几何方案的增益应在高对比档更大，交互项 CI 不含 0。

### 6.5 统计
- 以图像为单位做配对 bootstrap（`protocol_eval.py --bootstrap`），六集平均也做 bootstrap。
- 先完成 C1 去重。
- 每张卡一个主指标，跨卡用 Holm 校正。
- 最小有意义效应（SESOI，约定）：P0 +0.5；P2 前景 IoU +3；P1b +1；边界 F +2。

### 6.6 不在测试集上调参
1. 文献默认值或无标签规则，每张卡只有一个主配置。
2. 外部验证集，只用于类无关的几何超参：SUIM 官方 TEST、USIS10K train。两者都不在基准里【本项目事实：cfg/imglist】，需要先确认与测试集无重复。
3. 留一数据集交叉拟合。
4. 剂量档全部报告，不挑最好的一档。

---

## 7 优先级、关键未知与需要提供的信息

### 7.1 优先级

| 顺序 | 内容 | 成本 | 决策点 |
|---|---|---|---|
| 1 | 深度缓存、背景码普查、C1 去重 | 约 1 GPU·h + 0.5 人日 | — |
| 2 | D0 + G4(a)(b)（改配置/小改动）及 DINOv2 同构版本 | 约 5 GPU·h + 1.5 人日 | 现有增益是否几何；KK 是否有效 |
| 3 | G1 预检 → G1 接入 | 半天 + 0.5 人日 | `G1_pass` / `G1_abandon` |
| 4 | G2a/G2b + 四个对照 | 1.5 人日 + 约 12 GPU·h | 深度版 − 仅空间版 > 0？ |
| 5 | G3（先过 B1 预检） | 1–2 人日 | `G3_pass` |
| 6 | G5（重点看 SUIM/DUT）、M1 | 3–5 人日 | 只作工程增益 |
| 7 | G6、G7、T1 | 视情况 | — |

**两周最小执行包**：
- **第 1 周**：深度缓存 → 三项预检 → D0 → G4 (a)(b)。
- **第 2 周**：只推进预检通过的卡，跑主配置和全部对照，填同源表。

### 7.2 关键未知
1. 水下深度质量（没有 GT，只能用预检间接估计）。
2. 各数据集"背景"的定义和编码。
3. MAS3K/DUT 中底栖/伪装样本的比例。
4. 研究分支里 A3 是否能取中间层和注意力 key。
5. torch 移植与 numpy 参考是否一致。
6. 评审是否接受 P1/P2 作为主证据（取决于请求 1 线 C 的结论）。
7. CutS3D 锐化公式中成对下标的含义。

### 7.3 需要你们提供
1. 导师期望的结论类型：P0 涨分，还是"几何作用可归因"。
2. 服务器能否访问 HF/GitHub，以下载 SAM2、RADIOv2.5、NAF、DAv2-S 的权重；以及这些权重的许可。
3. 逐图统计的文件格式。
4. A3 实现的位置。
5. 背景码普查结果（`dedup_and_mask_consistency.py --census_only` 的输出）。
6. 是否已保存 A4 的逐图预测。
7. 能否使用 SUIM TEST 和 USIS10K train 作外部验证集。
8. GT-Uniform oracle 的确切定义。
9. **与 MARIS [CVPR'26] 的关系**：它和 E2O 是同一组作者，已经在训练式设定下用了 DA 多尺度门控融合。导师的意见是否参照了 MARIS 的做法？

---

## 附录 A 顶会引用（按 venue）

精读【文献事实】：
- **CVPR'26**：
  - Bingyu Li 等，*Exploring the Underwater World Segmentation without Extra Training*（Earth2Ocean），arXiv 2511.07923
  - Bingyu Li 等，*MARIS: Marine Open-Vocabulary Instance Segmentation*，arXiv 2510.15398
  - Chambon 等，*NAF: Zero-Shot Feature Upsampling via Neighborhood Attention Filtering*，arXiv 2511.18452（Highlight）
- **CVPR'26 Findings**：Alama 等，*RADSeg*，arXiv 2511.19704
- **NeurIPS'25**：Siyu Chen 等，*Leveraging Depth and Language for Open-Vocabulary Domain-Generalized Semantic Segmentation*（Vireo），arXiv 2506.09881
- **ICCV'25**：
  - Sick 等，*CutS3D: Cutting Semantics in 3D for 2D Unsupervised Instance Segmentation*，arXiv 2411.16319
  - Zhang、Liu、Tang，*CorrCLIP: Reconstructing Patch Correlations in CLIP for Open-Vocabulary Semantic Segmentation*，arXiv 2411.10086
  - Shi、Dong、Xu，*Harnessing Vision Foundation Models for High-Performance, Training-Free Open Vocabulary Segmentation*（Trident），arXiv 2411.09219
- **CVPR'25**：
  - Yin、Cao、Cheng、Hou，*DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation*，arXiv 2504.04701
  - Kim 等，*Distilling Spectral Graph for Object-Context Aware Open-Vocabulary Semantic Segmentation*（CASS），arXiv 2411.17150
  - Kaiyu Li 等，*SegEarth-OV: Towards Training-Free Open-Vocabulary Segmentation for Remote Sensing Images*，arXiv 2410.01768
  - Stojnić 等，*LPOSS*，arXiv 2503.19777（请求 1 精读）
- **AAAI'25**：Xu、Liu、Wu、Liu，*PDDM: Pseudo Depth Diffusion Model for RGB-PD Semantic Segmentation Based in Complex Indoor Scenes*，arXiv 2503.18393
- **NeurIPS'24**：Lihe Yang 等，*Depth Anything V2*，arXiv 2406.09414
- **ICLR'24**：Yin 等，*DFormer: Rethinking RGBD Representation Learning for Semantic Segmentation*，arXiv 2309.09668
- **ECCV'24**：
  - Yue 等，*Improving 2D Feature Representations by 3D-Aware Fine-Tuning*（FiT3D），arXiv 2407.20229
  - Pang 等，*Open-Vocabulary Camouflaged Object Segmentation*（OVCamo），arXiv 2311.11241
  - Lan 等，*ProxyCLIP*，arXiv 2408.04883（请求 1 做过代码级核对）
- **CVPR'24**：
  - Lihe Yang 等，*Depth Anything*，arXiv 2401.10891
  - El Banani 等，*Probing the 3D Awareness of Visual Foundation Models*，arXiv 2404.08636
  - Sick 等，*Unsupervised Semantic Segmentation Through Depth-Guided Feature Correlation and Sampling*（DepthG），arXiv 2309.12378
  - Zhang、You、Li、Fu，*Atlantis: Enabling Underwater Depth Estimation with Stable Diffusion*，arXiv 2312.12471
- **ICCV'23**：Zongwei Wu 等，*Source-free Depth for Object Pop-out*（PopNet），arXiv 2212.05370
- **ECCV'22**：Bachmann 等，*MultiMAE: Multi-modal Multi-task Masked Autoencoders*，arXiv 2204.01678

只核实了 venue、读过摘要【未核实（细节）】：
- **CVPR'26**：DepthSAM（*Beyond Appearance: Camouflaged Object Detection via Geometric Structure*）、Ren 等的免训练 OV-COS、The Power of Prior、BiPA、SigLino、UPLiFT、Upsample Anything
- **CVPR'26 Findings**：MoonSeg3R、OV-Stitcher
- **ICLR'26**：AnyUp
- **NeurIPS'25**：JAFAR
- **ICCV'25**：LoftUp、CLIPer、FreeCP
- **IROS'25**：DiPFormer（*Depth Matters: Exploring Deep Interactions of RGB-D for Semantic Segmentation in Traffic Scenes*）
- **CVPR'25**：RADIOv2.5
- **ECCV'24**：Lao 等，*On the Viability of Monocular Depth Pre-training for Semantic Segmentation*（摘要：比 ImageNet 预训练平均 +5.8 mIoU，训练式）
- **NeurIPS'24**：Lexicon3D
- **ICML'24**：USIS-SAM
- **IJCAI'24**：MAS-SAM
- **CVPR'24**：AM-RADIO、CoralSCOP
- **ICLR'24**：FeatUp
- **ACM MM'23**：DaCOD（*Depth-aided Camouflaged Object Detection*）

## 附录 B 本次新增代码

| 文件 | 作用 | 验证情况 |
|---|---|---|
| `scripts/dump_depth_cache.py` | 用 E2O 自带的 DA 缓存整图相对逆深度 | 需要 torch + GPU，本环境无法运行 |
| `scripts/prevalidate_depth_figure_ground.py` | G1/G2/G3 离线预检（F1–F3、A1、B1），带对照和预设判据 | numpy-only；合成数据测试通过 |
| `scripts/e2o_patches/geo_affinity.py` | G2a 3D 感知亲和（numpy 参考 + torch 移植）+ PCA 平滑图对照 | numpy 参考 5 项测试通过；torch 版未实跑；G2b（CutS3D 式）尚未实现 |

## 附录 C 非顶会线索（只作参考，不作为结论依据）

| 文献 | venue | 它提示了什么 | 与顶会证据是否一致 |
|---|---|---|---|
| Cetinkaya、Kalkan、Akbas，*Does Depth Estimation Help Object Detection?*，arXiv 2204.06512 | Image and Vision Computing（期刊） | 估计深度对 VOC/COCO 检测无益；"检测器越强，深度收益越小" | 一致（PDDM 的趋势） |
| Xiang 等，*Exploring Depth Contribution for Camouflaged Object Detection*，arXiv 2106.13217 | 只在 arXiv | MiDaS 深度直接融合不如纯 RGB | 一致（PopNet 表 7） |
| Wang 等，*When Depth Hurts*（GeoDistill），arXiv 2609.03378 | 只在 arXiv | 增益主要来自分支容量 | 方向一致（FiT3D 对照思路） |
| Wu 等，*GeoCueFormer*，arXiv 2609.18069 | 只在 arXiv | SUIM 上深度只适合浅层弱注入 | 一致（Vireo、OVCamo） |
| Varghese、Hueger、Maag，*Combining Foundation Model Confidence and Monocular Depth for Training-Free Out-of-Distribution Segmentation*，arXiv 2609.22896 | 只在 arXiv | 免训练的"分割块—外环"深度差 | 与 PopNet 的先验相容；是 G1 免训练化的直接参考 |
| Yang 等，*See in Depth*（DepSeg），arXiv 2512.05529 | 只在 arXiv | 深度提示 SAM2 | 与 CutS3D、CorrCLIP 相容 |
| Cai、Metzler，arXiv 2507.02148 | 只在 arXiv | DAv2 在水下的度量 δ1 只有 0.2–0.64 | 一致（Atlantis） |
| Awad 等，*Beneath the Surface*，arXiv 2411.14626 | 期刊预印本 | 增强对检测在数据集层面无益 | 顶会上没有对应证据 |
| Pei 等，*PEARL*，arXiv 2603.21528 | 只在 arXiv | OVSS 中的 "geometry" 多指嵌入空间几何 | 不适用 |
| TPA（arXiv 2608.08290）、GeoSeg-OV（arXiv 2608.10426）、LoCoFuse（未知） | 预印本/未知 | 决策层融合；遥感用 DAv2-B 作结构先验 | 待它们被顶会接收后再纳入 |
