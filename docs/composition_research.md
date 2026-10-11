# training-free 分割中"组件怎样组合"：发散调研、候选组合与前置验证（v1）

> 日期：2026-10-11。分支 `claude/jolly-lovelace-mbszku`。
> 与 [`uovs_research_plan.md`](uovs_research_plan.md)（下称"计划书"）和 [`geometry_fusion_research.md`](geometry_fusion_research.md)（下称"几何报告"）配套。几何/深度的讨论和协议论证已在那两份里，这里不重复，只在需要时引用。
>
> **标注约定**
> - 【文献事实】读过原文页面（PDF 抽取文本）确认的内容；
> - 【文献·二手】只读过 alphaXiv 基于全文生成的报告，或检索引擎对官方页面的摘录；数字要以原文为准；
> - 【未核实】只读过摘要，或凭记忆；
> - 【本项目事实】你们在本次请求中给出的数字和事实；
> - 【推断】我的推断，每条写依据；没有依据就写"无依据"。
>
> **venue 标注**：`[CVPR'24]` 在官方站点或 PDF 首页核实；`[CVPR'24]*` 通过其他论文的参考文献或检索摘录间接核实；`[arXiv]` 只在 arXiv；`[?]` 未核实。

---

## 0 一页结论

1. **余量决定组合该放在哪里。**
   - 【本项目事实】在 P0 下，最后一层聚合的完美分组只多 +1.3（SUIM +4.85）；识别是主要瓶颈（GT 区域上文本头 33.17%，视觉质心 85.51%）；模型没有背景输出，所以一计背景分数就崩（34.33 → 14.97 / 16.29 / 11.91）。
   - 【推断】因此 P0 上真正能动的只有识别（S4、S4p、S7）。视觉侧要成为创新点，只有两条路：一是输出"背景/前景"，在 P1/P2 上体现；二是"用视觉通路做识别"，例如区域级二次识别、生成式视觉原型。继续改聚合方式不是好方向。
2. **识别收益和背景判定是互补的，不能分开评。**
   - 【本项目事实】MLLM 先验在 P0 上 +14.9~23.2，但 ViT-L 在 P1b 上不涨。
   - 【推断】机制是：没有背景输出时，背景像素整片被判成某个前景类。识别越准，背景像素就越集中地落进"正确的那个类"，这个类的假阳性随之暴涨，IoU 被压到大约"前景面积占比"。所以识别侧的增益在 P1b/P2 上会被背景吞掉，只有和背景通道组合才能显现。这可以作为论文主线：视觉侧给背景，文本侧给识别，修正后的协议同时检验两者。
3. **文献里的增益集中在四种组合算子上**【文献事实】：
   - (a) **以区域为单元组织推理**：SCI-CLIP [arXiv] 的 scope reconstruction 让平均 mIoU 从 45.8 升到 49.5；CorrCLIP [ICCV'25]* 也属此类；
   - (b) **视觉参照/原型记忆**：SCI-CLIP 的 reference memory 再 +3.6，FreeDA [CVPR'24]*、OVDiff [ECCV'24]* 也属此类；
   - (c) **图像级"哪些类存在"的裁剪**：FreeCP [ICCV'25] 只把词表限制为 GT 类，SCLIP 在 PC 上从 30.4 升到 45.3、ADE 从 16.1 升到 32.8；ActiveSAM [arXiv] 用 SAM 3 的 presence 分数先裁剪词表；
   - (d) **显式背景建模**：OVDiff 去掉背景原型，VOC 掉 11.2；CLIP-DIY [WACV'24]* 去掉物体性，VOC 从 59.9 掉到 24.1；CaR [CVPR'24]* 去掉递归过滤（T=1），只有 15.2，完整版 67.6。
   - **注意**：(c)(d) 里的大多数方法用了逐数据集调的阈值（CaR、FreeCP、SCI-CLIP、FreeDA、SegEarth-OV3）。UOVSBench 没有验证集，不能照搬，必须换成固定值、无标签规则或开发集上定的值（§3.4）。
4. **结构上就能输出"背景/不存在"的家族**：先出掩码再分类（CaR、InstructSAM）、生成原型（OVDiff 的背景原型）、显著性/物体性引导（CLIP-DIY）、文本提示分割与检测（SAM 3 的 presence 头）、MLLM + SAM（ProMaC、The Power of Prior），以及负标签式的 OOD 打分（NegLabel 的思路）。E2O 所在的"改 CLIP 注意力"和"外部代理注意力"两族本身做不到，只能靠"背景名 + 阈值"（§3.2）。
5. **建议优先用实验检验的 6 个组合**（§5，每个都有几小时内能做完的前置验证和事先写定的判据）：

   | 编号 | 组合 | 侧别 | 主协议 |
   |---|---|---|---|
   | C1 | 复用 E2O 已算的代理亲和图做前景/背景判定（背景种子 + 边界连通度 + 谱二分），再和类证据、单图背景原型耦合 | **视觉侧主候选** | P1b / P2 |
   | C2 | 区域级二次识别：用前景掩码做视觉提示（模糊背景），经 CLIP 全局通路分类，再和稠密 logits 在区域内融合 | 视觉通路做识别 | P0 / P2 |
   | C3 | 逐图稀疏类先验的 EM 估计；MLLM 列表只作 Dirichlet 软先验，强度由多个 MLLM 的一致性决定 | **文本/识别侧主候选** | P0 / P2 / P1b |
   | C4 | 名称接口：无标签地从原名/拆词/常用名/学名/同义词中选名，再加分类学层级解码 | 文本侧 | P0 / P2 |
   | C5 | 文本 → 扩散模型生成图 → 视觉原型（含背景原型），弥补 33 vs 85 的差距 | 识别（视觉原型） | P0 / P2 / P1 |
   | C6 | 负标签（WordNet 挖掘）+ 逐图背景名（取自 MLLM 的 caption）做"不属于任何类"的判定 | 文本侧背景 | P1 / P2 |

6. **先做低成本诊断**（§6，12 项）：top-k 召回、GT 掩码裁剪分类、生成原型小探针、混淆结构、存在性 AUROC、物体性 vs 中心先验、"主导类填充"、P1/P2 下的阶段交换 oracle 等。多数可以用现有缓存在 CPU 上算完。
7. **不建议作为主创新**：
   - 再改末层聚合或范围约束：P0 余量只有 +1.3，SCI-CLIP、CorrCLIP 已经占了这个位置；
   - 通用后处理（PEARL 的 TLP、DSLO、LPOSS）：在自然图像上有 +2~5 的现成增益，但没有新意，适合作为统一后处理；
   - 固定的水下增强：QMSR [arXiv] 里 9 种固定增强在开放词汇检索上都不如原图；
   - 利用近重复帧做跨图一致性：和线 C 对"基准重复"的批评自相矛盾。
8. **公平对照的总原则**（§3.3）：组件对等、信息对等、计算对等、超参来源对等，并且要做"拆掉耦合、保留组件"的消融。
9. **对预期要诚实**：这些组合几乎都没有水下的直接证据。有依据的只有上界（GT 类集合 oracle、GT 掩码 oracle）和自然图像上的效应量。§5 逐条给出范围和依据，没有依据的明说。

---

## 1 检索策略与覆盖

### 1.1 工具与数据库
- **发现与精读**：alphaXiv 的语义检索（`discover_papers`），全文 PDF 页面问答，以及全文摘要报告。
- **venue 核实**：Web 检索，主要命中 openaccess.thecvf.com、cvpr/iccv/wacv/neurips/iclr 的 virtual 海报页、ojs.aaai.org、ecva.net、proceedings.iclr.cc、mlanthology。另外也从已读论文的参考文献里交叉核对。
- **继承**：前两份报告已读约 45 篇（ProxyCLIP、CorrCLIP、Trident、CASS、LPOSS、TPA、PTC、FLOSS、MLMP、DAF、SegEarth-OV3、BioCLIP 2 等）。这里直接引用，不再重读。
- **没用的**：Consensus 和 Scholar Gateway 偏同行评议期刊，对 2025–2026 的 CS 预印本覆盖不如 alphaXiv；GitHub 代码本次没有访问。
- **时间**：2026-10-11，一天内完成，约 15 次语义检索、约 25 次 Web 检索。

### 1.2 关键词（按家族）
- training-free OVSS 总体：training-free open-vocabulary semantic segmentation、CLIP DINO SAM、segment-centric、sliding window context、background、absent classes
- 先出掩码再分类：CLIP as RNN、mask proposals visual prompting、lazy visual grounding、class purification
- 生成/扩散：OVDiff、FreeDA、diffusion prototypes background prototypes、diffusion self-attention random walk
- MLLM + 分割：GenSAM、ProMaC、InstructSAM、LLaVA prior segmentation、localization heads
- 上下文/少样本分割：in-context segmentation DINOv3、background subspace、Matcher/PerSAM/GF-SAM、no time to train
- 无监督物体发现/显著性：TokenCut、Deep Spectral Methods、FOUND、LOST、boundary connectivity
- OOD/负标签：NegLabel、MCM、GL-MCM、negative prompts
- TTA/直推：TransCLIP、StatA、EM-Dirichlet、ZLaP、Frolic、ZERO、MTA、label shift EM
- 识别侧/物种名：scientific names species recognition、neglected tails REAL、CHiLS、FuDD、LMM post-hoc correction、fish nomenclature
- 水下：UOVSBench、Earth2Ocean、underwater SAM zero-shot、underwater image enhancement open-vocabulary retrieval
- CLIP 表征修复：test-time registers、feature purification

### 1.3 各领域阅读量（本次新增）

| 领域 | 全文页面精读 | 只读摘要报告/检索摘录 | 只看到标题和摘要 |
|---|---|---|---|
| 通用 training-free OVSS（CLIP 内部、代理、区域、后处理） | SCI-CLIP、NERVE、PEARL、GLA-CLIP、FreeCP、DSLO、CaR、CLIP-DIY（8） | ActiveSAM、LaVG、TagCLIP、PnP-OVSS、ReME、SegEarth-OV 去全局偏置（6） | DouC、VIP、OV-Stitcher、DINOde、OVS-DINO、EOVSAM、Standalone-DINOv3-RS、SatOV、Text-to-seed、MAVISEG（约 10） |
| 生成/扩散原型 | OVDiff、FreeDA（2） | — | FA-Seg、Seg4Diff、iSeg、DiffSegmenter |
| MLLM + 分割 | ProMaC、InstructSAM（2） | The Power of Prior、Localization Heads（2） | GenSAM、Seg-Agent、GeoSeg、Discover-Segment-Select、GFR-SAM |
| 上下文/少样本分割 | REBASE（1） | INSID3、No-time-to-train（2） | Matcher、PerSAM、GF-SAM |
| 无监督物体发现/显著性/经典图方法 | — | Deep Spectral Methods、boundary connectivity（2） | TokenCut、FOUND、LOST、PANC、CuVLER、DADO |
| OOD/负标签 | NegLabel（1） | — | GL-MCM、MCM、CLIPN、NegRefine、ANTS |
| TTA/直推 | StatA（1） | EM-Dirichlet（1） | TransCLIP、ZLaP、Frolic、ZERO、MTA、GATE、在线 EM |
| 识别侧/物种名称 | REAL、学名→常用名、IchthyoNoma、POC（4） | — | CHiLS、FuDD、Taxonomy-aware eval、Neural Catalog、DeepTaxon |
| 弱监督（CAM 系） | — | CLIP-ES（1） | — |
| 视觉提示 | — | FGVP（1） | Red Circle |
| 水下 | QMSR（1） | E2O 会议出处（1） | MARIS、UW-VOS、MAS-SAM、Dual-SAM |
| CLIP 表征修复 | — | test-time registers（1） | SC-CLIP、SFP |
| 合计 | 约 21 篇 | 约 19 篇 | 约 40 篇 |

另外，前两份报告已精读约 45 篇，主要是代理注意力/多源融合、几何、TTA、BioCLIP 2、SAM 3 系。

### 1.4 局限
- 2026 年的新工作大多是预印本，venue 也多为海报页或补充材料核实，正式版可能有出入。
- alphaXiv 的"摘要报告"是机器生成的，凡标【文献·二手】的数字都要回原文核对。
- 没检索中文期刊；负结果少发表，负面证据应视为下限。
- "检索范围内未见先例"只代表这一天的检索，不能当作新意的证明。投稿前要再做一次针对性检索（§5 每张卡里列了要查的关键词）。

---

## 2 分类体系

### 2.1 扩展后的阶段表

在你们的 S0–S7 之外，新增 S2r（区域单元）、S4p（图像级类存在/先验）、S8（跨图/直推）、S9（闭环校验）。理由是文献里有大量增益正好来自这几个位置，而原表没有单列。

| 阶段 | 内容 | 文献例子 | 本项目已测 |
|---|---|---|---|
| S0 输入/视图 | 多尺度裁剪、放大二次推理、增强视图、视觉提示（模糊背景、红圈） | CLIP-DIY 多尺度；FGVP；CaR；QMSR（水下增强） | 未测 |
| S1 CLIP 中间层 | 多层注意力/特征融合、异常 token 修复、test-time registers | GLA-CLIP 多层融合；SCI-CLIP 的 spatial 分支；test-time registers；SFP | 未测 |
| S2 末层聚合 | 代理注意力、KK 注意力、跨窗 KV、Procrustes 对齐 | ProxyCLIP、CASS、GLA-CLIP、PEARL | 已测，上限小 |
| **S2r 区域单元** | 超像素、谱分割、SAM 掩码作为推理单元或注意力范围 | CorrCLIP、SCI-CLIP、FreeDA、LaVG | 未测（几何报告 G5 规划了 SAM2） |
| S3 分类特征空间 | 去全局偏置、模态间隙中心化、在 DINO 空间里比原型 | SegEarth-OV、PTC、CASS 的文本调整 | 全局映射失败 |
| S4 分类器/原型 | 文本原型、视觉原型（生成、检索、伪标签）、记忆库 | OVDiff、FreeDA、ReME、SCI-CLIP、TPA | MLLM 先验已测 |
| **S4p 图像级类存在/先验** | 词表裁剪、presence 门控、逐图类先验 | FreeCP、ActiveSAM、SegEarth-OV3、DSLO 的早拒绝、EM-Dirichlet | MLLM 偏置/硬约束已测 |
| S5 后处理 | CRF/PAMR、随机游走、标签传播、联合双边上采样、文本感知拉普拉斯 | CaR、NERVE、LPOSS、PEARL、DSLO | 只测了 JBU |
| S6 背景判定 | 背景名 + 阈值、背景原型、显著性/物体性、presence、计数指派 | CLIP-ES、CaR、OVDiff、CLIP-DIY、ActiveSAM、InstructSAM | 只测了深度 |
| S7 文本侧 | 名称、模板、描述、负标签、层级 | REAL、FLOSS、FuDD、CHiLS、NegLabel | 已测多种 |
| **S8 跨图/直推** | 测试集原型库、直推 EM、视频帧一致性 | TPA、TransCLIP、StatA、EM-Dirichlet | B3 规划中 |
| **S9 闭环校验** | 分割 → 验证 → 再分割，多模型互证 | CaR、ProMaC、GenSAM、POC | 未测 |

### 2.2 组合算子

"组合"不只是"在某个阶段加一个组件"，更关键的是两个阶段之间用什么算子连接。下面 9 种算子覆盖了本次读到的几乎所有 training-free 组合：

| 算子 | 含义 | 例子 | 超参能否无标签设定 |
|---|---|---|---|
| O1 替换 | 用外部量替换内部量 | ProxyCLIP/E2O 用 VFM 自相似替换末层注意力 | 多数可沿用默认值 |
| O2 约束/门控 | 用一个阶段的输出限制另一个阶段的范围 | CorrCLIP/SCI-CLIP 用区域掩码限定注意力；FreeCP、ActiveSAM 裁剪词表；CLIP-DIY 用物体性门控类分数 | 阈值多为逐数据集调 |
| O3 传播 | 在图上扩散标签或证据 | LPOSS、NERVE（随机游走）、PEARL-TLP、CLIP-ES-CAA | 步数、温度多为固定值 |
| O4 融合 | logit/概率层的加权或乘积 | FreeDA 局部 + 全局（β）；TPA；SCI-CLIP 的稠密 + 记忆 | 权重常逐数据集调 |
| O5 指派/优化 | 带约束的匹配 | InstructSAM 用计数约束的二值整数规划；FGVP 部件检测用匈牙利匹配 | 约束来自 MLLM，不需阈值 |
| O6 迭代/闭环校验 | 分割 → 验证 → 再分割 | CaR 递归过滤；ProMaC 提示-掩码循环；GenSAM | 迭代次数固定；过滤阈值逐数据集 |
| O7 检索/记忆 | 从外部库取视觉参照 | ReCo、FreeDA、ReME、SCI-CLIP | K 值较稳健 |
| O8 生成 | 合成参照或负样本 | OVDiff、FreeDA（扩散模型）；ProMaC（修补出对比图） | 生成数量较稳健 |
| O9 直推/群体估计 | 利用多个样本或像素的统计 | TPA、TransCLIP、EM-Dirichlet、StatA | 部分方法对"有效类数"敏感 |

### 2.3 一个观察：组合大多发生在哪两个阶段之间【推断】

- 2024–2026 年的 training-free OVSS 新工作，组合主要集中在 **S2↔S2r**（区域约束注意力：CorrCLIP、SCI-CLIP、GLA-CLIP）和 **S4↔S8**（记忆/原型：FreeDA、ReME、SCI-CLIP、TPA）。
- **S6↔S4p**（背景判定和"有哪些类"的联合估计）很少被当作一个整体设计，而这正是本项目协议问题所在的位置（第 0 节第 2 条）。
- 另外，**S6→S0→S4**（"先找物体，再用物体去提示识别"）在 CaR、CLIP-DIY、FGVP 里各出现了一半，但没有在代理注意力式稠密分割器上和识别先验闭环组合过。
- 依据：§3.1 总表里 15 个家族的组合位置统计；"很少"是基于本次检索范围的判断。

---

## 3 方向二：其他框架家族怎样组合

### 3.1 总表

| 家族 | 代表工作（venue） | 组合机制 | 报告效果 | 适用条件/局限 | 能否输出背景 | 能迁移到方向一的机制 |
|---|---|---|---|---|---|---|
| F1 改 CLIP 内部注意力/特征 | MaskCLIP [ECCV'22]*、SCLIP [ECCV'24]*、ClearCLIP [ECCV'24]*、NACLIP [WACV'25]*、ResCLIP [CVPR'25]*、SFP [ICCV'25]*、PEARL [CVPR'26]、test-time registers [NeurIPS'25] | O1：QQ/KK/邻域注意力、去残差和 FFN、Procrustes 对齐、异常 token 修复 | PEARL：只用对齐（PA）从 13.8 到 40.6，再加 TLP 到 43.2（8 集平均，B/16）【文献事实】；test-time registers 让 OpenCLIP 的注意力式零样本分割 mIoU 从 34.7 到 40.0【文献·二手】 | 只改 CLIP 本身；E2O 已替换末层注意力，同位置的改动会冲突【推断】 | 否（需背景名 + 阈值） | 异常 token 修复（S1）；PEARL 的 TLP 作后处理（S5） |
| F2 外部视觉模型代理亲和 | ProxyCLIP [ECCV'24]、CASS [CVPR'25]*、GLA-CLIP [CVPR'26]、RADSeg [CVPR'26 Findings]、NERVE [WACV'26]（用 SD 自注意力） | O1 + O3：VFM 自相似替换注意力；跨窗 KV；熵加权的多头融合 + 随机游走 | GLA-CLIP 加在 ProxyCLIP 上 +0.6（同设定）；NERVE 熵加权比单头 +2.5（VOC）【文献事实】 | 本项目的 E2O 就在这一族；P0 余量已测得很小 | 否 | 熵加权选头/选源（可用于多个代理源的融合）；跨窗 KV（大图） |
| F3 区域单元 / 先出掩码再分类 | CorrCLIP [ICCV'25]*、SCI-CLIP [arXiv]、LaVG [ECCV'24]、CaR [CVPR'24]*、FreeDA [CVPR'24]*、CLIP-DIY [WACV'24]*、InstructSAM [NeurIPS'25] | O2 + O4 + O6：区域限定注意力；区域池化后分类；视觉提示后用全局通路分类；递归过滤 | SCI-CLIP 的 scope reconstruction +3.7（平均）；CaR 递归 15.2→67.6（VOC），分类器换 L/14 在 VOC 上从 54.1 到 67.6；CLIP-DIY 多尺度 +3.9（VOC）、+5.1（COCO）【文献事实】 | 依赖掩码质量；CaR 的 η/θ/λ 逐数据集设定；SCI-CLIP 的融合权重逐数据集设定 | **CaR、CLIP-DIY、InstructSAM 能；CorrCLIP 不能** | 区域 = 推理单元；视觉提示的全局通路二次识别（C2）；递归过滤（G3） |
| F4 文本提示直接分割/检测 | SAM 3 [ICLR'26]、SegEarth-OV3 [arXiv]、ActiveSAM [arXiv]、EOVSAM [arXiv]；OWLv2 / Grounding DINO + SAM [?] | presence 头把识别与定位解耦；先用低分辨率 presence 裁剪词表，再全分辨率解码；margin 校准背景 | ActiveSAM 8 集平均 65.3，比 SegEarth-OV3 高 1.4，比 SAM 3 原样高 7.8；去掉类裁剪精度不变；MABC 让 VOC21 +0.9【文献·二手】 | 基础模型更强（需公平对照）；SAM 3 在 254 个细粒度鱼名上的识别能力无证据 | **能**（presence、阈值） | "presence → 词表裁剪"可迁移到 CLIP 的 S4p；"SAM 3 管在哪、CLIP 管是什么"（C7） |
| F5 扩散特征/生成参照 | OVDiff [ECCV'24]*、FreeDA [CVPR'24]*、NERVE、DSLO [arXiv]、iSeg [arXiv] | O8 + O7：每类生成支持集，抽前景/背景/部件原型；或用 SD 自注意力作亲和 | OVDiff：去掉背景原型 VOC 64.4→53.2，去掉类别预过滤 →54.4；SD+CLIP+DINO 融合 66.4。FreeDA：CLIP 全局 + DINOv2 局部，VOC 85.6（B）对比纯 DINOv2 78.4【文献事实】 | 生成成本（OVDiff 每类约 110 秒，2080Ti）；生成图与目标域有差距；稀有概念可能生成失败（REAL 的观察） | **OVDiff 能**（背景原型） | 生成式视觉原型与背景原型（C5）；"类别预过滤"即 S4p |
| F6 MLLM 定位 + 分割 | The Power of Prior [CVPR'26]、Localization Heads [CVPR'25]、ProMaC [NeurIPS'24]、GenSAM [?]、InstructSAM | MLLM 回答"有哪些类"，中间层激活给出初始定位，SAM 精修；或 MLLM 与 SAM 循环、用修补出的对比图消除幻觉 | The Power of Prior 在 VOC、COCO-Object 的"物体类"上达到当时最好，不需要显式背景名【文献·二手】；ProMaC 的迭代次数 1→4，COD10K 的 Fβ 从 0.626 到 0.714【文献事实】 | 在线 MLLM 成本高；本项目只有离线 JSON（可本地部署开源 MLLM） | **能** | MLLM 给出环境名（ProMaC 的 A_back）→ 逐图背景名（C6）；MLLM 的类存在判断作先验（C3） |
| F7 原型/缓存/检索 | ReCo [NeurIPS'22]*、FreeDA、ReME [ICCV'25]、SCI-CLIP RM、TPA [arXiv]、PTC [arXiv] | O7 + O4：离线或测试时建视觉参照库，和文本头融合 | SCI-CLIP RM 平均 +3.6；ReME 声称超过所有 training-free 基线，且参照集优于 GT 片段-文本数据【文献·二手】；TPA 在 ProxyCLIP 上 +2.0（见计划书） | 参照来源：生成图、带标签图、测试集伪标签，各自的设定不同 | 依参照而定 | 参照库来源换成生成图或伪标签（C5、D3） |
| F8 上下文/单样本分割 | PerSAM [?]、Matcher [?]、GF-SAM [?]、INSID3 [CVPR'26]、REBASE [arXiv]、No-time-to-train [arXiv] | 参照图 → DINO 对应 → SAM 提示；背景子空间投影；位置去偏 | REBASE 的背景子空间投影在 PACO-Part 上 +4.4，在 ISIC 上 +3.9；替换 INSID3 的位置去偏后在自然图像上反而下降【文献事实】 | 需要带掩码的参照（本项目没有，除非用生成图或伪标签） | 能（相对参照） | **单图背景子空间投影**（A6）；伪参照驱动的 SAM 提示 |
| F9 无监督物体发现/显著性 | LOST [?]、TokenCut [?]、FOUND [CVPR'23]*、Deep Spectral Methods [CVPR'22]、boundary connectivity [CVPR'14] | 自监督特征上的图割/谱分解；背景种子；边界先验 | CLIP-DIY 中：免训练的 FOUND-bkg 版本 48.4，自训练的 FOUND 59.9，不用物体性 24.1（VOC，含背景类）【文献事实】 | 偏向物体居中的图；多物体、伪装场景会漏 | **能**（类无关前景） | 背景种子、边界连通度、谱二分（C1） |
| F10 弱监督 CAM 系 | CLIP-ES [CVPR'23]*、TagCLIP [AAAI'24]、PnP-OVSS [CVPR'24] | softmax-GradCAM + 背景类集合竞争；类感知的注意力亲和细化；局部到全局的多标签识别 | CLIP-ES：softmax 让初始 CAM 从 49.4 到 53.3，加背景集合再 +9.2（VOC）【文献·二手】 | 原设定需要图像级标签；改用 MLLM 列表或类先验后可以免训练 | 能（背景集合） | 背景类集合与前景类在 softmax 中竞争（A7/A8） |
| F11 TTA/直推 | TDA [CVPR'24]、MTA [?]、ZERO [?]、TransCLIP [?]、EM-Dirichlet [CVPR'24]、StatA [CVPR'25]、MLMP [NeurIPS'25] | O9：用批或流的统计调整类模型或先验；EM-Dirichlet 惩罚类数 | EM-Dirichlet：每批 75 张时 ImageNet 零样本约 +20【文献·二手】；StatA 指出 EM-Dirichlet 在"少有效类"设定下好，类一多就掉，TransCLIP 正好相反【文献事实】；MLMP 在 SUIM 持续设定下塌缩（见计划书） | 群体统计的偏置要和实际"有效类数"匹配 | 否 | **把"像素群体"当作批，做逐图的类先验 EM**（C3） |
| F12 OOD/负标签 | MCM [?]、GL-MCM [?]、NegLabel [ICLR'24] | 扩展标签空间：挖掘大量远离 ID 类的负标签，用 sum-softmax 打分 | NegLabel：ImageNet-1k 上 AUROC 94.21，FPR95 25.40（图像级）【文献事实】 | 图像级；稠密特征噪声更大 | **能**（"不属于任何类"） | 像素级负标签背景（C6） |
| F13 识别侧专门化 | REAL [CVPR'24]*、学名→常用名 [EMNLP'23]*、CHiLS [?]、FuDD [?]、BioCLIP 2 [NeurIPS'25]*、POC [arXiv] | 改名称接口；层级标签；差分描述；领域模型；MLLM 在 top-k 上重排 | REAL-Prompt 8 集平均 62.6→66.0；学名→常用名在 iNat 的 13 个鱼类物种上 6.15→32.30；POC 用 Qwen2.5-VL-7B 重排 +6.4，但只给 top-5 名称、不给示例图时 −9.5【文献事实】 | REAL 需要 LAION 的 caption 计数；POC 需要示例图 | 否 | 名称选择 + 层级（C4）；"生成图当示例图"的 POC 变体（D2） |
| F14 经典图方法与后处理 | DenseCRF、PAMR、GrabCut、超像素（SLIC、Felzenszwalb）、JBU、LPOSS [CVPR'25]、PEARL-TLP、DSLO | O3：在 RGB/特征图上传播 | PEARL 的 TLP 插到 SCLIP/NACLIP/SFP 上 +1.9~4.0；DSLO 插到 CASS/SC-CLIP 上 +2~5【文献事实】 | 本项目 JBU：边界 F 翻倍，P0 只 +0.13【本项目事实】 | 否 | 统一后处理（所有对比方法都加，避免混杂） |
| F15 水下/遥感/医学垂直流水线 | E2O [CVPR'26]、QMSR [arXiv]、SegEarth-OV [CVPR'25]、InstructSAM、MedCLIP-SAM [?] | 领域模板；领域 VFM 代理；增强路由；去全局偏置 | QMSR：9 种固定增强的 NDCG@10 都低于原图（最好 0.4578 vs 原图 0.4655），均匀融合 0.4666；用 GT 选专家的 oracle 0.7498【文献事实】 | QMSR 的路由是训练出来的；oracle 用了 GT | 依流水线 | 水下增强只能"按需选择"，不能固定施加（H1） |

### 3.2 结构上能输出"背景/不存在"的机制

| 机制 | 代表 | 是否需要阈值 | 阈值来源 | 迁移到本项目要注意 |
|---|---|---|---|---|
| 背景名与前景类在 softmax 里竞争 | CLIP-ES、CaR 的背景查询、ProxyCLIP 的 VOC21 配置 | 有时还要概率阈值 | ProxyCLIP 用 0.2/0.25/0.15（逐数据集） | SUIM 的 "Sand/sea-floor"、USIS10K 的 "sea-floor" 本身是类，背景名不能和它们冲突（计划书 C4 已列） |
| 背景原型 | OVDiff（来自生成图）、REBASE（来自参照图的背景子空间） | 不需要（最近邻） | — | 本项目可以从本图的高置信背景 patch 取原型（A5） |
| 类无关物体性/显著性 | CLIP-DIY + FOUND、TokenCut | 乘性融合不需要；二值化需要 | FOUND 自训练；免训练版本较弱 | 要超过中心先验和深度（深度已被中心先验复现） |
| presence 头 | SAM 3、ActiveSAM、SegEarth-OV3 | 需要 | SegEarth-OV3 逐数据集调（见计划书 §3.2C）；ActiveSAM 用分位数自适应 + 全局指数 | 强模型，要公平对照 |
| margin/熵校准 | ActiveSAM MABC、PACL（熵 >1.5 判背景）【文献事实】 | 需要 | 固定常数 | 可作为基线 |
| 计数约束指派 | InstructSAM | 不需要 | 计数来自 MLLM | 需要类无关的区域候选；本项目 JSON 有物体列表，没有计数 |
| 递归过滤（区域证据不足就剔除类） | CaR | 需要 θ | 逐数据集 | 用开发集定，或用固定分位数 |
| 负标签 | NegLabel | 分组平均后打分；判定需阈值 | — | 负标签集合要剔除水下物体名 |

### 3.3 引入新组件时怎样设计公平对照

**总原则**
1. **组件对等**：方法用到某个组件 X，基线也配上 X，并且用 X 最常见、最简单的方式。目的是证明增益来自"组合方式"，而不是"有了 X"。
2. **信息对等**：MLLM 输出同源（同一份 JSON，文件名先换成哈希）；类名文件、模板、滑窗设置都相同。
3. **计算对等**：报告前向次数和延迟。如果方法多用了一次 CLIP 前向，基线也给一次等价的额外前向，例如多尺度或翻转 TTA。
4. **超参来源对等**：所有方法的超参都来自文献默认值、无标签规则或开发集。不允许只给自己的方法在测试集上调。
5. **"拆掉耦合、保留组件"的消融**：保留所有组件，但让它们各自独立工作，比如物体性只做后处理门控，先验只做独立加偏置。这一项最能说明"组合方式"本身的贡献。

**按组件列出的对照**

| 新组件 | 给 E2O 配上的最简用法（基线） | 证明组合方式有效的对照 | 备注 |
|---|---|---|---|
| SAM2 / SAM 3（类无关掩码或通用概念） | E2O + SAM 区域内多数投票（CaR/Trident 式）；E2O + SAM 3（"fish/animal"）做前景门控 | 同样的 SAM 掩码，换成我们的耦合（例如掩码用于视觉提示 + 先验）；把 SAM 换成超像素或谱分割（弱掩码源）看机制是否还成立 | SAM 3 许可证要确认 |
| 扩散模型（生成参照） | 生成图 → 原型 → 最近原型分类，直接替换文本头（OVDiff 式） | 同一批生成图在我们的融合/先验框架下 vs 简单替换；等成本的文本侧对照（同等 token 预算的模板/描述） | 生成图与测试图做 pHash 去重检查（防止撞上网络图片） |
| BioCLIP 2 | 把 E2O 的 CLIP 整体换成 BioCLIP 2（同管线） | 在 BioCLIP 2 骨干上再加我们的组合；CLIP 骨干上同样加，比较增益是否一致 | 计划书 B4 已列；训练数据可能和网络爬取的 AquaOV255 重叠 |
| 本地开源 MLLM（如 Qwen2.5-VL-7B） | 把它的物体列表当硬约束（和本项目已测的 4o 硬约束同式） | 同一份列表放进软先验（C3）或重排框架 | 要先把文件名换成哈希（计划书 N6） |
| DINOv2/DINOv3 作新代理 | E2O 换代理源（计划书 B5 的同构对照） | 新代理源 + 我们的背景通道 vs 旧代理源 + 背景通道 | 几何报告 §6.4 已有完整的同构对照表 |
| WordNet / LLM 文本资源 | 固定背景名列表（计划书 C4） | NegLabel 负标签、caption 名词各自单独，再组合 | 负标签集合事先固定 |
| FOUND 等显著性模型 | CLIP-DIY 的融合规则 + FOUND | 我们复用 E2O 代理图的物体性（不加模型）vs 外加 FOUND | 证明"复用图"不比外加模型差 |
| 水下增强模型 | 固定施加一种增强 | 无标签的逐区域选择 vs 固定增强 vs 均匀平均 | QMSR 已表明固定增强不如原图 |

### 3.4 横向观察：哪些家族的超参能在 UOVSBench 上合法使用

- **逐数据集调参**：CaR（η、θ、λ 三个阈值，VOC/COCO/Context 各一组）、FreeCP（T_rp、T_ap 按数据集复杂度设）、SCI-CLIP（λ_d、λ_m 逐数据集）、FreeDA（β 和超像素参数逐数据集）、SegEarth-OV3（背景/置信阈值）。这些都是【文献事实】，来自它们的实现细节或表格。
- **固定一组超参**：OVDiff（全实验同一组）、PEARL（只有 Cityscapes 的网格大小不同）、GLA-CLIP（默认设定不调，另报调参的最好值）、NegLabel、StatA（α=1 全实验固定）。
- **含义**【推断】：UOVSBench 没有验证集，迁移时应该优先选固定超参或有无标签规则的组件。逐数据集阈值类的组件要么用 USIS10K train（去重后）开发集来定，要么改成分位数式的自适应规则（如 ActiveSAM 用 presence 分数的 β 分位数），并事先登记。

---

## 4 方向一：组合思路清单（不筛选）

下面列出 44 条，按主阶段分组。"侧别"一栏：V = 视觉侧，T = 文本/识别侧，V→T = 用视觉通路做识别。"协议"一栏写这条思路主要在哪个协议下才能体现效果。

### 4.1 A 组：背景/前景（S6）

| 编号 | 涉及阶段与组件 | 机制 | 依据 | 主要数据 / 协议 | 侧别 |
|---|---|---|---|---|---|
| A1 | S2 代理图 → S6 | **背景种子**：在 E2O 已算的 DA/DINOv2 token 自相似里，找出"被关注最少"的 patch 作背景种子，与它相似的 patch 判为背景。不加新模型 | FOUND [CVPR'23]* 的思路；CLIP-DIY 中免训练的 FOUND-bkg 版本在 VOC 上 48.4，不用物体性只有 24.1 | 单目标三集、USIS10K；P1b/P2 | V |
| A2 | S2 代理图 → S6 | **边界连通度**：在 token 图上算每个区域/patch 和图像边界的测地连通程度，连通越强越像背景。比中心先验更能适应偏心物体和贴边的水体 | boundary connectivity [CVPR'14]（显著性检测的经典先验）【文献·二手】 | 单目标集、SUIM；P2 | V |
| A3 | S2 → S2r → S6 | **谱二分**：对代理亲和做归一化割，取 Fiedler 向量二分，再用 A1/A2 决定哪一侧是前景 | TokenCut [?]、Deep Spectral Methods [CVPR'22]、LaVG [ECCV'24]（迭代 NCut 覆盖全图） | 单目标集；P2 | V |
| A4 | S6 × S4 | **类证据 × 物体性门控**：前景类 logit 乘物体性 Θ，背景类乘 (1−Θ)，再 softmax | CLIP-DIY [WACV'24]* 式 (3)(4) | 全部；P1b/P2 | V |
| A5 | S6 → S4 | **单图背景原型**：用 A1–A3 判出的高置信背景 patch，取其 CLIP 特征均值作为本图的"背景类"原型，加入分类器 | OVDiff 背景原型（类级，去掉后 VOC −11.2）；REBASE 的背景子空间 | 全部；P1/P2 | V |
| A6 | S6 → S3 | **背景子空间投影**：对本图背景 patch 的特征做低秩 SVD，把所有 patch 特征投影到其正交补后再和文本比。目的是去掉水体色偏对识别的污染 | REBASE [arXiv]（ISIC +3.9，PACO-Part +4.4，单样本设定）；SegEarth-OV 的"去全局偏置"是它的秩-1 特例 | 单目标集；P0/P2 | V（作用于识别） |
| A7 | S7 → S6 | **负标签背景**：从 WordNet 挖掘离 254 个类最远的约 1 万个负标签，像素级 sum-softmax 打分，作为"不属于任何类"的概率 | NegLabel [ICLR'24]（图像级 AUROC 94.21） | 全部；P1/P2 | T |
| A8 | S7（MLLM caption）→ S6 | **逐图背景名**：把 4o caption 里不在类表中的名词（sand、blue water、rocks…）当本图的背景类 | ProMaC [NeurIPS'24] 让 MLLM 同时输出前景名和环境名；CaR 的背景查询 | 全部；P1/P2 | T |
| A9 | S2r + S7 + S4 → S6 | **计数约束指派**：MLLM 物体列表（如需计数，用本地 MLLM 补）+ 区域候选 + CLIP 相似度，做二值整数规划指派，未被指派的区域就是背景 | InstructSAM [NeurIPS'25]（无需置信阈值） | SUIM、USIS10K；P1a/P1b | V+T |
| A10 | S6（强模型） | **SAM 3 通用概念前景**："fish / marine animal / coral / diver" 等概念给出类无关前景，E2O 只负责识别 | SAM 3 [ICLR'26]、SegEarth-OV3、ActiveSAM；水下无证据 | 全部；P2 | V（强模型） |
| A11 | S4 → S6 | **margin 校准背景**：top-1 与 top-2 的差小、物体性也低的像素判为背景 | ActiveSAM MABC（VOC21 +0.9）【文献·二手】 | 全部；P1 | V+T |
| A12 | S6 迭代 | **特征空间 GrabCut**：用 A1–A4 初始化前景/背景，在 DINO 特征上拟合两个 GMM，交替更新若干次 | GrabCut（SIGGRAPH'04）【未核实，凭记忆】 | 单目标集；P2 | V |

### 4.2 B 组：区域单元与"用视觉通路做识别"（S2r / S0 / S4）

| 编号 | 涉及阶段与组件 | 机制 | 依据 | 主要数据 / 协议 | 侧别 |
|---|---|---|---|---|---|
| B1 | S6 → S0 → S4 → S5 | **区域二次识别**：第一遍得前景掩码；把背景模糊（或 1.2 倍框裁剪、或画红椭圆）；送 CLIP 图像编码器走全局通路分类；再和稠密 logits 在区域内融合 | CaR、FGVP [NeurIPS'23]（背景模糊最好）、FreeCP 的歧义消解（裁剪歧义区 + 描述）、TagCLIP（局部到全局再识别）；**反证**：IchthyoNoma 中 BioCLIP2 紧裁剪 −13.8、白色掩码 −8.4 | 单目标三集；P0/P2 | V→T |
| B2 | S6 → S0 | **放大二次推理**：对前景外接框以更高分辨率重跑一次，再贴回原位置 | CLIP-DIY 多尺度（VOC +3.9，COCO +5.1） | USIS10K、SUIM 的小目标；P0/P1 | V |
| B3 | S2r → S5 | **区域内投票**：在谱分割/超像素/SAM2 区域内做多数投票或区域 argmax | FreeDA 超像素；CorrCLIP；本项目 oracle 上限 +1.3（SUIM +4.85） | SUIM、DUT；P0/P1 | V（新意低） |
| B4 | S6 → S2 | **前景/背景两区域限定的代理注意力**：前景 token 只聚合前景，背景只聚合背景，防止背景的 value 污染前景特征 | CorrCLIP 的范围约束（这里只用两个区域）；SCI-CLIP | 单目标集；P0 有上限，P2 | V |
| B5 | S6 + S4 | **前景内主导类**：在单目标图里把整个前景统一成一个类（区域投票的极端形式） | 本项目"整图填充"事实；不配 P2 报告就是钻协议漏洞 | 单目标集；P0/P2 | T |
| B6 | S0 + S4 | 全局-局部 logit 融合 | LoCoFuse、GLA-CLIP 已基本占据 | — | V（新意低） |

### 4.3 C 组：识别先验与类存在（S4p）

| 编号 | 涉及阶段与组件 | 机制 | 依据 | 主要数据 / 协议 | 侧别 |
|---|---|---|---|---|---|
| C-a | S4 → S4p | **逐图稀疏类先验 EM**：把一张图的像素后验当成一个"批"，用标签漂移 EM 估计本图的类比例 π，加稀疏先验让不存在的类的 π 趋于 0，再用 π 重新加权像素后验 | 标签漂移 EM（Saerens 等，Neural Computation 2002）【未核实，凭记忆】；EM-Dirichlet [CVPR'24]（惩罚类数）；StatA [CVPR'25]（少有效类时的偏置）；FreeCP 的 GT 词表 oracle | 全部；P0/P2/P1b | T |
| C-b | S7（MLLM）→ S4p | **MLLM 列表作 Dirichlet 软先验**：被提名的类获得更大的先验伪计数；强度由 4o、Qwen7B、Qwen3B 的一致票数决定 | 本项目：硬约束在 P0 上 +14.9~23.2，但 4o 首对象错 27%，Qwen7B 错 59%；POC 表明只给名称的重排会伤（−9.5） | 全部；P0/P2 | T |
| C-c | S4 ↔ S7 | **CLIP 校验 MLLM**：MLLM 提名的类只有落在 CLIP（前景区域）的 top-k 里才被采纳 | 计划书 B3 的种子方案 (b)；POC 的反向用法 | 单目标集；P0/P2 | T |
| C-d | S4p 融合 | **presence 分数融合**：图像级 CLIP、像素最大概率、面积占比、MLLM 提名四路信号合成每类的存在分数，用于门控 | CASS 的物体存在先验；ActiveSAM、SegEarth-OV3 的 presence；DSLO 的早拒绝 | 全部；P1/P2 | T |
| C-e | S4 → S0 → S7 | **歧义组局部消解**：两类的激活空间重叠高时，裁剪重叠区域，用细粒度描述或层级判别 | FreeCP 的 AP 阶段 [ICCV'25] | 多类集、细粒度；P0 | T |

### 4.4 D 组：原型、记忆、生成（S4 / S8）

| 编号 | 涉及阶段与组件 | 机制 | 依据 | 主要数据 / 协议 | 侧别 |
|---|---|---|---|---|---|
| D1 | S7 → 扩散生成 → S4/S6 | **生成式视觉原型 + 背景原型**：每类生成若干张"水下 {名称}"图，在前景内池化得前景原型，背景处池化得背景原型 | OVDiff、FreeDA、SuS-X [?]；本项目视觉质心 85 vs 文本头 33 | 全部；P0/P2/P1 | T（视觉原型） |
| D2 | D1 → 本地 MLLM | **生成图当示例图的 MLLM 重排**：POC 的零样本版，用生成图代替真实示例 | POC（有示例图 +6.4，无示例图 −9.5） | 单目标集；P0/P2 | T（成本高） |
| D3 | S8 + C-a | **跨图伪标签原型（TPA/B3）+ 逐图先验**：库提供视觉证据，EM 提供逐图类比例 | TPA；计划书 B3 | 全部；P0/P2 | T（直推） |
| D4 | S2r → S4 | **段级参考记忆**：参考段来自生成图或伪标签 | SCI-CLIP RM（+3.6） | 全部；P0 | T |
| D5 | 外部无标签库 → S4 | **检索 + 共分割**：在 USIS10K train 等无标签水下图上用 CLIP 文本检索，再共分割得类原型 | ReCo [NeurIPS'22]* | 7 个粗类的数据集；P0 | T（类别覆盖有限） |

### 4.5 E 组：文本接口（S7）

| 编号 | 涉及阶段与组件 | 机制 | 依据 | 主要数据 / 协议 | 侧别 |
|---|---|---|---|---|---|
| E1 | S7 | **名称变体的无标签选择**：候选包括原名、拆词名、常用英文名、学名、同义词；用 FLOSS 式熵准则逐类选"名称"（而不是选模板） | FLOSS [ICCV'25]；REAL [CVPR'24]*；学名→常用名 [EMNLP'23]*；IchthyoNoma；本项目拆词 +10 | AquaOV255、MAS3K、USIS16K；P0/P2 | T |
| E2 | S7 + S4 | **分类学层级解码**：p(种) = p(大类) · p(种 \| 大类)，或在区域级先判大类再判种 | CHiLS [?]、Taxonomy-aware eval [?]、BioCLIP 的层级文本 | 单目标集；P0/P2 | T |
| E3 | S7 | **差分描述**：只针对文本空间里最容易混淆的类对，生成"区分性"描述 | FuDD [?]；**反证**：本项目手工描述没有收益 | AquaOV255；P0 | T |
| E4 | S7 | 类专家模板（计划书已列，低优先） | FLOSS | 人造物类；P0 | T |
| E5 | S7（caption 属性）→ S4 | 逐图属性词（颜色、条纹）注入原型 | E2O 的 CSA；ITACLIP [?] | 全部；P0 | T（新意低） |

### 4.6 F 组：CLIP 内部与特征空间（S1 / S3）

| 编号 | 涉及阶段与组件 | 机制 | 依据 | 主要数据 / 协议 | 侧别 |
|---|---|---|---|---|---|
| F1 | S1 | **test-time registers**：把高范数异常激活从 patch 移到一个未训练的附加 token | Jiang 等 [NeurIPS'25]（OpenCLIP 注意力分割 34.7→40.0）【文献·二手】 | 全部；P0 小 | V |
| F2 | S1 | **代理注意力注入最后 k 层**（k = 2、3） | GLA-CLIP 多层融合 +0.9；SCI-CLIP 去掉 spatial 分支（α=0）后 4 集平均 −1.5 | SUIM、DUT；P0 | V |
| F3 | S3 | **去全局偏置**：patch 特征减去 λ 倍的 CLS 特征 | SegEarth-OV [CVPR'25]（和上采样合计提升，没有单独消融【文献·二手】）；水下色偏可能放大全局偏置【推断】 | 全部；P0 | V/T |
| F4 | S3 | **无标签模态中心化**：减去图像 patch 均值和文本均值。和你们失败的"已见类拟合映射"不同，这里不拟合任何东西 | Mind the gap [?] | 全部；P0 | T（预期小） |
| F5 | S2 | PEARL 的 Procrustes 对齐 | PEARL；但 E2O 已用代理注意力替换末层，二者作用位置冲突【推断】 | — | V（低） |

### 4.7 G 组：后处理与闭环（S5 / S9）

| 编号 | 涉及阶段与组件 | 机制 | 依据 | 主要数据 / 协议 | 侧别 |
|---|---|---|---|---|---|
| G1 | S5 | 文本感知拉普拉斯传播（插件） | PEARL：+1.9~4.0（通用数据） | 全部；P0 | V（新意低） |
| G2 | S5 | 分布差异/最优传输读出（插件） | DSLO：+2~5（通用数据） | 全部；P0 | V（新意低） |
| G3 | S9 | **CaR 式递归过滤**：区域二次识别分数低于 θ 的类剔除后重算，直到类集不变 | CaR（T=1 时 15.2 → 收敛后 67.6） | SUIM、USIS10K；P1/P0 | V+T |
| G4 | S9 | ProMaC 式 MLLM-SAM 循环 | ProMaC（需在线 MLLM + 修补模型） | MAS3K 伪装；P2 | V+T（成本高） |
| G5 | S5 | DenseCRF / PAMR 统一后处理 | CaR、TCL | 全部 | 作统一设置，不作贡献 |

### 4.8 H 组：输入视图（S0）

| 编号 | 涉及阶段与组件 | 机制 | 依据 | 主要数据 / 协议 | 侧别 |
|---|---|---|---|---|---|
| H1 | S0 → S4 | **多视图 + 无标签选择**：原图、灰度世界白平衡、CLAHE、一种公开水下复原模型，按置信度或熵逐区域选视图（ZERO/MTA 式） | QMSR：固定增强不如原图，用 GT 逐候选选择的 oracle 大幅更高；ZERO [?] | 单目标集；P0 | V→T（先验低） |
| H2 | S0 | 翻转、多尺度 TTA 平均 | 通用做法 | 全部 | 作计算对等的对照 |
| H3 | S8 | 近重复簇内的跨图一致性 | 视频分割的标签传播 | AquaOV255、USIS16K | **不推荐**作主方法（与线 C 的论点冲突） |

### 4.9 清单的几点说明【推断】
- P0 上真正有空间的是 B1、B5、C 组、D 组、E 组（识别）。A 组只有在 P1/P2 下才有意义。
- 按依据强度排序（自然图像上已有效应量，且机制能迁移）：A4 ≈ C-a（类比 FreeCP） > D1 > B1 > A1/A3 > E1 > A7 > 其余。
- 按新意排序（检索范围内没找到直接先例）：A1/A2 复用代理图做背景、C-a/C-b 逐图像素 EM 先验 + 多 MLLM 一致性、A8 caption 背景名、E1 用熵准则选名称、A6 单图背景子空间投影。

---

## 5 筛选建议：最值得用实验检验的 6 个组合

每张卡包括：组合内容、最接近的已有工作与区别（新意评估）、低成本的前置验证与事先写定的判据、预期效果与依据、计算成本与风险、公平对照。

### C1 复用代理亲和图的前景/背景判定（视觉侧主候选）

**组合**
1. 输入：E2O 前向中已经算出的 DA（或 DINOv2）block-11 token 及其自相似矩阵 S（不增加模型）。
2. 三种无标签物体性，各自独立计算：
   - **背景种子**（A1）：种子 = 与全图平均相似度最低的 patch，或 [CLS] 注意力最低的 patch。背景度 = 与种子的相似度。
   - **边界连通度**（A2）：在 4 邻接 token 图上，边权 w_pq = 1 − cos(f_p, f_q)，求测地距离 d(p, q)。
     - 面积 A(p) = Σ_q exp(−d(p,q)² / 2σ²)；
     - 边界长度 L(p) = Σ_{q∈边界} exp(−d(p,q)² / 2σ²)；
     - 背景度 = 1 − exp(−(L/√A)² / 2σ_b²)。
     - σ、σ_b 按原论文默认值的思路取，需要换算到特征空间【未核实具体默认值】。
   - **谱二分**（A3）：对 S 的正值部分做归一化割，取 Fiedler 向量的符号，哪一侧和边界连通度更高就判为背景。
3. 无标签融合三者（几何平均，或"多数票 + 置信度"，事先登记一种）得到物体性 Θ。
4. 和识别耦合：
   - A4：前景类 logit × Θ，背景类 × (1−Θ)；
   - A5：取 Θ 最低的 20% patch 的 CLIP 特征均值，作为本图背景原型，加入分类器；
   - 输出背景标签 C（与 `protocol_eval.py` 的约定一致）。

**最接近的工作与区别**
- CLIP-DIY [WACV'24]*：用外部的 FOUND 模型给物体性，与 CLIP 多尺度分数做乘性融合。区别：我们复用 E2O 已算的代理图，不加模型；物体性与"单图背景原型"双向耦合，不只是门控。
- OVDiff [ECCV'24]*：背景原型是类级的，来自生成图。我们是逐图的，来自测试图本身。
- TokenCut [?] / FOUND [CVPR'23]* / Deep Spectral [CVPR'22]：类无关，没有和识别耦合。
- REBASE [arXiv]：从带掩码的参照图取背景子空间，用于单样本分割。
- 本项目的深度前景/背景判定：已失败（AUROC 0.79~0.95，但几乎被中心先验和打乱深度复现）。
- **新意评估**：中等。在检索范围内没有看到"复用代理注意力的亲和图同时做背景判定和单图背景原型"的先例【推断】。投稿前要用 "proxy attention background" "objectness reuse affinity open-vocabulary" 再查一次。

**前置验证**（约半天；CPU 为主，GPU 只用来提 token，约 0.5 GPU·h）
- 每个数据集随机取 500 张（固定随机种子）。
- 物体性来源：中心先验、深度（已缓存）、A1、A2、A3、三者融合、DINOv2 [CLS] 注意力。
- 指标：
  - 像素级 AUROC（GT 前景 vs GT 背景）；
  - Otsu 阈值 IoU；
  - 固定阈值 IoU（阈值只在 USIS10K train 去重开发集上定）。
- **通过**：在 ≥4/6 个数据集上同时满足
  - AUROC 比"中心先验"和"深度"中较好者高 ≥0.05，差值的 bootstrap CI 下界 > 0；
  - Otsu IoU 比"全前景"高 ≥0.10。
- **放弃**：在 ≥3/6 个数据集上不超过中心先验。这和几何报告 G1 的判据同构，便于比较。

**预期效果**
- P2 的前景 IoU 相对"全图前景"一定会大幅提升，因为基线就是前景面积占比，太弱了。所以要和中心先验比，相对中心先验的增量**无依据**。
- 外部参照：CLIP-DIY 在 VOC 上，物体性让 mIoU（含背景类）从 24.1 升到 59.9；其中免训练的 FOUND-bkg 版本是 48.4。
- P2 的增量可以在集成前就估出来：P2 ≈ 逐图的 [主类识别正确] × [前景 IoU] 的平均。所以 ΔP2 ≈ 识别率 × Δ前景 IoU（在识别与前景质量近似独立时）。前置验证的 IoU 一出，就能预测 P2。

**成本与风险**
- 成本：DINOv2/DA token 在 4.6 万张图上提一遍约 0.5 GPU·h；算法 1–2 人日；接入后 ViT-B 全量 75 分钟。
- 风险：
  - 被中心先验复现（深度就是这个下场）；
  - MAS3K 的伪装物种、DUT 的底栖动物与海床共面（几何报告 G1 已指出）；
  - SUIM、USIS10K 的 seabed 是类，不能当背景，背景要按数据集定义（计划书 C2 的协议定义已考虑）。

**公平对照**
- E2O + 中心先验门控；
- E2O + FOUND（CLIP-DIY 的融合规则）；
- E2O + SAM 3（"animal"）门控，作为强模型的参照；
- 消融：只做 A4、不做 A5；只用 A1/A2/A3 中的一种；把代理图换成 CLIP 自身的 key 自相似（检验"代理图质量"的作用）。

### C2 区域级二次识别：视觉提示 → CLIP 全局通路 → 区域内融合

**组合**
1. S6：前景掩码（来自 C1；或 SAM，作为换掩码源的对照）。
2. S0：视觉提示。主设定事先登记为"背景模糊"（FGVP 的 Blur Reverse Mask）；另报 1.2 倍外接框裁剪（InstructSAM 的做法）和红椭圆。
3. S4：同一个 CLIP（B/L/H）走图像级通路，用同样修复过的类名和 UW 模板分类，得到区域级分数 g(c)。
4. S5：区域 logits = 区域内稠密 logits 的均值 + w · g(c)。w 用无标签规则：按两路 top-1 margin 的比值，或两路熵的倒数加权。
5. 可选：把 g 作为 C3 的观测输入，参与类先验估计。

**最接近的工作与区别**
- CaR [CVPR'24]*：掩码来自 CAM，视觉提示后过滤类，递归直到收敛；阈值逐数据集。
- FGVP [NeurIPS'23]：用于指代表达理解，不是语义分割。
- FreeCP [ICCV'25]：只在歧义区域裁剪，用 LLM 描述判别。
- TagCLIP [AAAI'24]：局部到全局的再识别，用于多标签分类。
- InstructSAM [NeurIPS'25]：1.2 倍裁剪 + CLIP + 计数约束指派。
- 区别：作用在代理注意力式稠密分割器的输出上；提示用的是本项目自己的前景掩码；目的是用全局通路纠正稠密头的识别（33% 瓶颈）；在 P2 下评估。
- **新意评估**：中低。机制都有先例，新的是组合和场景。

**前置验证**（GPU 1–2 h）
- 用 GT 掩码（oracle），在 AquaOV255、MAS3K、USIS16K 各取 1000 张，做裁剪分类（B/L/H 三个骨干）。
- 对比：同一批 GT 区域上的稠密池化文本头（用修复后的类名，你们测到约 40%）。
- **通过**：
  - 主设定（背景模糊）的裁剪分类比稠密头高 ≥5；
  - 并且两路的错误互补：两路中至少一路正确的比例，比单路最好者高 ≥8（说明融合有空间）。
- **放弃**：主设定 ≤ +2，且互补性 < +5。

**预期效果**
- 水下无依据。
- 方向相反的证据：
  - IchthyoNoma [arXiv] 中，BioCLIP2 做鱼类识别时，紧裁剪 −13.8，白色掩码 −8.4，模糊背景 −0.5~−2.9（背景有用）；
  - FGVP 中，COCO 的物体-标签匹配，裁剪（70.9）好于背景模糊（67.8）；
  - E2O 论文 Fig.5 中，CLIP-L 的整图分类只有 34.14%，和稠密头相当【本项目事实，来自交接包】。
- 所以即使有收益，更可能来自"多一路互补判别"，而不是"全局通路更强"。我给不出数值。

**成本与风险**
- 成本：每张图多 1 次 CLIP 前向，六集 ViT-B 约多 20–40 分钟【推断】。
- 风险：伪装物种的掩码差；单目标集本来就以物体为中心，裁剪带来的信息增益小。

**公平对照**
- E2O + 整图 CLIP 分类融合（不用掩码，前向次数相同）；
- E2O + GT 掩码裁剪分类（上限）；
- E2O + SAM 掩码裁剪分类（换掩码源）。

### C3 逐图稀疏类先验 EM + MLLM 软先验 + 多 MLLM 一致性（文本/识别侧主候选）

**机制**
- 设一张图的像素（或 patch）后验 P[i, c] = softmax(T · logits)，T 是温度，用尺度匹配规则定，同计划书 B3 的 `auto_lambda`。
- MLLM 提名票数 v_c ∈ {0, 1, 2, 3}（4o、Qwen7B、Qwen3B 各一票）。
- 做 MAP-EM：

```python
# P: (N, C) 像素后验；votes: (C,) 被几个 MLLM 提名
# gamma: 最小面积比例（稀疏先验，"一个类至少要解释多少比例的像素才算存在"）
# eta: 每张票折合的伪计数比例
def em_prior(P, votes, gamma=0.01, eta=0.05, iters=30, warm=10):
    N, C = P.shape
    pi = np.full(C, 1.0 / C)
    for t in range(iters):
        Q = P * pi                      # 训练先验视为均匀时，p(c|x)·π/u ∝ p·π
        Q /= Q.sum(1, keepdims=True)
        g = gamma if t >= warm else 0.0 # 先做 warm 轮普通 EM（最大似然），再加稀疏项
        n = Q.sum(0) + N * (eta * votes - g)       # Dirichlet MAP 的伪计数形式
        if (n > 0).any():
            pi = np.maximum(n, 0)
        else:                           # 全部被剪掉时，只保留证据最多的一类
            pi = np.zeros(C); pi[np.argmax(Q.sum(0))] = 1.0
        pi = pi / pi.sum()
    return Q, pi                        # Q 取 argmax 得到分割；pi 是图像级类存在估计
```

- **为什么要先做普通 EM**：如果一开始就加稀疏项，后验较平时所有类的伪计数都会是负的，先验被整体剪光。我在合成数据上试过（254 类、576 个 patch、目标类与两个近似类混淆）：直接加稀疏项会全部剪光；先做 10 轮普通 EM 再加稀疏项，图中只剩 2–3 个类，目标类的像素数从 40 增加到 256（MLLM 正确时到 320）。
- **这只是合成数据上的健全性检查，不是证据。** 同一个检查也暴露了风险：当 3 个 MLLM 都投给错误的类时，错误类能把正确类整个压掉。所以 η 要小，"MLLM 错误子集上不变差"这条判据必须保留。

- 没有 MLLM 时（votes 全 0）退化为纯稀疏先验 EM，这一档要单独报告。
- γ 有物理含义（一个类在图中至少占多少面积），而且与类无关，可以在 USIS10K train 开发集上定，或者事先固定为 1%。
- η 事先固定。所有其它档位在附录全部报告。

**最接近的工作与区别**
- 标签漂移 EM（Saerens 等 2002）【未核实】：整个测试集估一个先验，不是逐图。
- EM-Dirichlet [CVPR'24]：批级直推，惩罚类数，用于分类。
- StatA [CVPR'25]：指出这类方法只在"有效类少"时好；而本项目一张图里的有效类正好很少。
- FreeCP [ICCV'25]：类裁剪基于 CAM 细化前后的一致性，阈值逐数据集。
- ActiveSAM / SegEarth-OV3：presence 裁剪，依赖 SAM 3。
- DSLO [arXiv]：早拒绝只是取 argmax 的唯一类集合。
- 本项目已测：硬/软偏置先验。
- 区别：
  - 在一张图的像素群体上做 MAP-EM，把"图里有哪些类"和"每个像素是什么"联合估计；
  - MLLM 只作先验浓度，不是硬约束；
  - 先验强度由多个 MLLM 的一致性给出。
- **新意评估**：中。检索范围内没有看到 OVSS 中逐图像素级 EM 类先验的先例【推断】。要用 "label shift EM segmentation per-image prior" "transductive pixel EM open-vocabulary" 再查一次。

**前置验证**（需要一次带 logits 的子集推理，约 1 GPU·h，加 CPU 分钟级）
- 每集 1000 张，ViT-B，保存 1/8 分辨率的逐类 logits（fp16）。
  - 存储估算：AquaOV255 1000 张 × 254 类 × 约 4096 像素 × 2 B ≈ 2 GB。
- 离线跑 EM，同时用 `protocol_eval.py` 算 P0/P1b/P2。
- **通过**：
  - P0 和 P2 的主类识别率在 ≥4/6 个数据集上同时提升；
  - 六集平均 P0 ≥ +1.0，P2 识别率 ≥ +2，CI 不含 0；
  - 在"4o 首对象错误"的子集上不比"无先验"差（≥ −1）。
- **放弃**：P2 识别率不涨，或在 MLLM 错误子集上明显变差。

**预期效果**
- 有上界依据：
  - "GT 类集合"oracle：FreeCP 在自然图像上 +4~+16；在本项目的单目标集上，这个 oracle 约等于整图填充；
  - 你们的硬约束在 P0 上 +14.9~23.2。
- 【推断】在 P0 上，EM 版应当在 4o 正确时低于硬约束，在 4o 错误时高于硬约束。纯 EM（不用 MLLM）的增益**无依据**。
- 和 C1 组合后，P1b 才可能出现增益（第 0 节第 2 条）。

**成本与风险**
- 成本：子集推理 1 h；EM 在 CPU 上几分钟；全量每个配置约 75 分钟（需要在推理代码里加 logits 输出）。
- 风险：
  - 多类集（SUIM 92% 多目标）里的小类可能被稀疏先验吃掉，γ 要小；
  - EM 对后验的尖锐程度很敏感：E2O 的余弦乘 40 后，后验可能接近 one-hot（EM 几乎不动），也可能很平（稀疏项一加就全剪光），取决于实际余弦的分布。所以温度必须用尺度匹配规则来定，不能照搬 40。

**公平对照**
- 与本项目的硬/软偏置先验：同 MLLM、同 JSON；
- 与 FreeCP 式的冗余类剔除：阈值来源相同；
- 消融：去掉一致性（只用 4o）；去掉稀疏项（γ=0）；去掉 MLLM（votes=0）。

### C4 名称接口的无标签选择 + 分类学层级解码

**组合**
1. 每类的候选名集合 N_c：原名、拆词名、常用英文名（FishBase/WoRMS 等公开名录）、学名、同义词（本地 LLM 生成，再按 REAL 的做法用 CLIP 文本编码器过滤掉与其它类混淆的同义词）。
2. 无标签选择（FLOSS 式，事先登记一种准则）：对每个类，选使"被预测为该类的测试像素/图像的平均预测熵最低"的名称。另报"与其它类文本间距最大"的准则。
3. 层级：为 254 类建立"大类（鱼、无脊椎、人造物、植物、哺乳类…）→ 科 → 种"的层级，来源是 `category_split` 加公开分类学。解码时 p(种) = p(大类) · p(种 | 大类)；或者在区域级先判大类、再判种。

**最接近的工作与区别**
- REAL-Prompt [CVPR'24]*：选 LAION caption 中出现频率最高的同义词（需要计数资源）。8 个数据集平均 62.6 → 66.0（ViT-B/32），Flowers +8.6。
- 学名→常用名 [EMNLP'23]*：iNat 的 13 个鱼类物种上，学名 6.15，常用名 32.30。
- FLOSS [ICCV'25]：用熵选模板。
- CHiLS [?]：用子类帮粗类分类，方向相反。
- IchthyoNoma [arXiv]：同一组 BioCLIP2 图像嵌入，只换名称写法，准确率在 35.77% 到 72.36% 之间变化；学名换一个同义写法又差几个点。
- 本项目：拆词名让区域分类从 30.20 升到 40.26。
- 区别：用无标签准则逐类选"名称"，并与层级解码组合。
- **新意评估**：低到中。部件都有，但水下物种的名称接口没人系统研究过；也可以写成线 C 审计的一部分。

**前置验证**（CPU 1–2 h，用 GT 区域 dump）
- 计算每种命名方案、每种选择准则的区域分类准确率，以及层级解码的准确率；基线是拆词名（40.26）。
- **通过**：
  - 任一事先登记的方案在 AquaOV255 上 ≥ +3，且 MAS3K、USIS16K 不降；
  - 层级解码 ≥ +2，或者把"跨大类错误"减少 ≥30%。
- **注意**：这里用 GT 区域来**评估**选择准则，但准则本身不得看标签。

**预期效果**
- 方向有依据：自然物种上学名与常用名差 2–5 倍。但 AquaOV255 的类名可能本来就多是常用名【推断，需核对类表】，增益范围**无依据**。

**成本与风险**
- 成本：名称表 1–2 人日（LLM 辅助 + 人工核对）；评估分钟级。
- 风险：
  - 名称选择是在测试集上做的，必须无标签并事先登记；
  - FishBase 的名称与 AquaOV255 的类定义可能不一致（例如属级类 vs 种级名）。

### C5 文本 → 生成图像 → 视觉原型（含背景原型）

**组合**
1. 每类用 SD（2.1 或 XL，公开权重）生成 N=16 张 "an underwater photo of a {name}"。
2. 用同一条 E2O 管线（或 DINOv2）提特征：在交叉注意力图或 C1 物体性的前景内池化得前景原型；背景处池化得背景原型。
3. 推理：logit = 文本头 + λ · 原型头，λ 用计划书 B3 的尺度匹配规则；所有类的背景原型并成"背景类"。

**最接近的工作与区别**
- OVDiff [ECCV'24]*：N=32；SD + CLIP + DINO 特征融合；有背景原型和类别预过滤。
- FreeDA [CVPR'24]*：用 COCO caption 驱动生成，建离线原型库；推理时超像素 + CLIP 全局。
- SuS-X [?]：生成或检索支持集。
- ReME [ICCV'25]：强调参照集的质量。
- 区别：面向水下的细粒度长尾物种；与代理注意力的稠密头融合；可以与 C3 的先验组合；背景原型自带水体上下文。
- **新意评估**：中低。

**前置验证**（GPU 2–3 h）
- 固定选 40 类（例如 AquaOV255 的 ID 0–39，事先登记），每类 16 张。
- 在 GT 区域 dump 上做 40 选 1 分类：原型头 vs 文本头 vs 两者融合。
- **通过**：原型头或融合比文本头高 ≥5（40 选 1），并且对"生成失败"的类（用 CLIP 自检：生成图和本类文本的相似度排名）不崩。
- **放弃**：< +2。

**预期效果**
- 水下无依据。
- 自然图像上，OVDiff 在 VOC（含背景）上 64.4（只用 SD 特征），同一张表里 TCL 为 51.2【文献事实】。
- 但 REAL [CVPR'24]* 指出 SD/DALL·E 对稀有概念常常生成错误；稀有鱼种正是这种情况。

**成本与风险**
- 成本：254 类 × 16 张 ≈ 4,000 张，在 3090 上约 2–3 GPU·h（SD2.1，512 分辨率，25 步）【推断】；SDXL 更慢。
- 风险：生成图与真实水下图的域差；稀有物种生成错误；SD 的许可条款。

**公平对照**
- 同一批生成图只做"最近原型分类"vs 我们的融合；
- 等成本的文本侧对照：更多模板或描述，token 预算相同。

### C6 负标签 + 逐图背景名（文本侧背景）

**组合**
1. NegMining：从 WordNet 的名词和形容词里，选与 254 个类文本距离（取 5% 分位）最远的 M 个作为负标签（M 事先固定，例如 1 万）。剔除水下物体名，如 coral、rock、sand，因为它们在某些数据集里是类或者是前景。
2. 像素级 NegLabel 分数：S = Σ_ID exp / (Σ_ID exp + Σ_neg exp)，分组平均。
3. 逐图背景名：4o caption 中不在类表里的名词，过滤掉与类名文本相似度高的，作为本图的背景类。
4. 与 C1 的物体性做乘性融合（A4）。

**最接近的工作与区别**
- NegLabel [ICLR'24]：图像级 OOD。
- CLIP-ES [CVPR'23]*：固定的背景类集合。
- CaR [CVPR'24]*：背景查询。
- ProMaC [NeurIPS'24]：MLLM 给出环境名。
- 计划书 C4：固定背景名 + ProxyCLIP 阈值。
- 区别：做到像素级；背景名来自逐图 MLLM caption；与物体性耦合。
- **新意评估**：中。

**前置验证**（GPU < 1 h）
- 在 E2O 的 patch 特征（子集）上算 NegLabel 分数，计算 GT 前景/背景的像素 AUROC。
- 对比：固定背景名列表（计划书 C4）、NegLabel、caption 名词，以及三者组合。
- **通过**：AUROC ≥ 0.70，且比固定背景名列表高 ≥0.03，在 ≥4/6 个数据集上成立。

**预期效果**
- 无依据。NegLabel 的图像级 AUROC 是 94.21，但稠密特征的噪声大得多。

**风险**
- 负标签集合混入水下物体名，与类表冲突；
- SUIM、USIS10K 的 seabed 是类。

### C7（可选，强模型路线）"SAM 3 管在哪，E2O/CLIP 管是什么"

- **组合**：SAM 3 用通用概念（fish、marine animal、coral、diver、robot…）给出类无关前景和实例；E2O + C3/C4 只负责识别；区域内用 C2 的方式融合。
- **定位**：这是审稿人很可能会问的"简单强基线"（计划书 B6）。适合做上界参照，不适合当创新点，因为 SegEarth-OV3 和 ActiveSAM 已有先例。
- **公平对照**：所有 C1–C6 的结果，都要再报一次"把 C1 的物体性换成 SAM 3 前景"的版本，用来说明组合机制在强前景源下是否仍然成立。
- **前置验证**：500 张/数据集，SAM 3 通用概念前景的 IoU 和 AUROC，与 C1 的前置验证同表比较。

### 5.8 建议的组合主线与消融【推断】

- **主线**：C1（背景通道，视觉侧）+ C3（逐图类先验，识别侧），C2、C4、C5 作为识别模块可插拔，C6 作为背景通道的文本侧补充。
- **主表**：P0、P1b、P2 同列报告，再按"识别率 × 前景 IoU"分解 P2。
- **核心消融**：
  1. 只有 C1：P2 的前景 IoU 升，识别率不变；
  2. 只有 C3：P0 和识别率升，P1b 几乎不变（重现你们 ViT-L 的现象）；
  3. C1 + C3：P1b、P2 同时升，并且超过 1 和 2 之和或接近之和。这正是"组合方式"的证据；
  4. "拆耦合"版本：C1 只做后处理门控，C3 只做独立偏置。
- **备选**：如果 C1 的前置验证没过（被中心先验复现），视觉侧退到 C2（视觉通路识别）或 C7 的强模型路线，并如实报告 C1 的负结果。

### 5.9 预期效果汇总

| 组合 | 主协议 | 有依据的上界 | 自然图像上的效应量 | 水下证据 | 我能给的范围 |
|---|---|---|---|---|---|
| C1 | P2 前景 IoU、P1b | 完美背景 oracle（你们在算） | CLIP-DIY 物体性 24.1→59.9（VOC，含背景） | 深度判前景：AUROC 0.79~0.95 但被中心先验复现【本项目事实】 | 相对中心先验：无依据 |
| C2 | P0、P2 识别率 | GT 掩码裁剪分类（D2） | CaR 中分类器从 B/16 换 L/14：VOC 54.1→67.6；FGVP 在 RefCOCO 上 +3~4.6 | IchthyoNoma：裁剪/掩码对 BioCLIP2 有害 | 无依据 |
| C3 | P0、P2 识别率；与 C1 组合后 P1b | GT 类集合 oracle | FreeCP 冗余类剔除：SCLIP 平均 +3.9 | 本项目 MLLM 硬约束 P0 +14.9~23.2 | P0 低于硬约束（4o 对时）、高于硬约束（4o 错时）【推断】 |
| C4 | P0、P2 识别率 | 视觉质心 85.51（远高于任何名称方案） | REAL +3.4（平均）；学名→常用名 2~5 倍 | 本项目拆词 +10.06（区域分类） | 无依据 |
| C5 | P0、P2；背景原型 → P1 | 视觉质心 85.51 | OVDiff VOC 64.4 vs TCL 51.2 | 无 | 无依据 |
| C6 | P1、P2 | 完美背景 oracle | NegLabel 图像级 AUROC 94.21 | 无 | 无依据 |

---

## 6 建议补测的 oracle 与诊断

按"成本低、对决策影响大"排序。

| 编号 | 诊断 | 怎么算（用什么现有资源） | 用来决定 | 成本 |
|---|---|---|---|---|
| D1 | **top-k 召回** | GT 区域 dump 上文本头的 top-1/3/5/10/20 准确率；以及 4o/Qwen 候选与 CLIP top-k 的交集精度 | C3（先验有多大空间）、C-c、D2（重排空间） | CPU 几分钟 |
| D2 | **GT 掩码裁剪分类 oracle** | 见 C2 的前置验证 | C2 去留 | GPU 1–2 h |
| D3 | **生成原型小探针** | 见 C5 的前置验证 | C5 去留 | GPU 2–3 h |
| D4 | **混淆结构** | 区域分类的错误中，"同科/同大类"错误占多少（需要分类学表） | C4 的层级解码、E3 差分描述是否值得 | CPU，加 1 人日建表 |
| D5 | **类存在 AUROC** | 每张图、每个类，各种存在信号（像素最大概率、面积占比、图像级 CLIP、MLLM 提名、EM 后的 π）对"该类是否在图中"的 AUROC | S4p 的设计（C3、C-d） | 需要 logits 子集（与 C3 共用） |
| D6 | **物体性 vs 中心先验** | 见 C1 的前置验证 | C1 去留 | 半天 |
| D7 | **主导类诊断** | 用**现有的 E2O 逐图预测图**：单目标集上主导类的图像级准确率 vs 像素准确率；"主导类整图填充"和"主导类 × 前景掩码"在 P0/P1b/P2 下的分数 | 衡量 P0 中"碎片化"的代价，也是 B5 的上界；在论文里说明 P0 的退化（线 C 证据） | CPU 几分钟 |
| D8 | **P1 错误分解** | 在 SUIM、USIS10K 上，按 GT 类型（水体、海床、未标注物体）统计背景假阳性；在单目标集上统计背景面积分布 | C1 和 C6 的重点放在哪 | CPU |
| D9 | **区域单元质量** | 谱分割、超像素、SAM2 的分区相对 GT 的 best-overlap IoU（几何报告 G5 规划了 SAM2，这里补免训练的分区） | B3、B4、C2 用哪种区域 | 0.5 天 |
| D10 | **名称接口审计** | 见 C4 的前置验证 | C4 去留；也是线 C 的审计材料 | CPU |
| D11 | **近重复簇内的一致率** | 同一簇内预测类是否一致 | 量化模型不稳定性；支持 P3 去重 | CPU |
| D12 | **P1b/P2 下的阶段交换 oracle** | (i) GT 背景 + E2O 识别；(ii) GT 主类 + 全图前景；(iii) GT 主类 + C1 物体性前景；(iv) GT 背景 + GT 主类 | 把 P2 拆成"识别 × 前景"两个乘子，直接告诉你视觉侧和文本侧各值多少分 | CPU（与你们正在算的分阶段 oracle 合并） |

**补充说明**
- **D7 必须同时报告 P2**，否则"主导类填充"就是在钻 P0 的漏洞。它的价值在于量化"像素级碎片化"让 E2O 在 P0 上损失了多少，这对线 C 是很强的证据。
- **D12 的 (ii)** 就是"识别全对、前景全错（全图）"时的 P2，等于逐图前景面积占比的平均，可以直接写成公式：P2_(ii) = mean_i( |GT_FG_i| / |valid_i| )。

---

## 7 关键未知问题与需要你们补充的信息

1. **分阶段 oracle 的结果**，尤其是 P1b/P2 下的"完美背景"和"对现有分割重新命名"。它们直接决定 C1 和 C3 哪个优先。
2. **GT 区域 dump 的格式**：特征是哪个空间（E2O 的 V 输出，还是也有 DINOv2）？有没有区域外接框坐标（C2 要用）？有没有背景区域（C6 要用）？
3. **能否保存 logits**（不只是 argmax），以及服务器磁盘余量（C3、D5 要用）。
4. **导师对设定的态度**：额外模型（SD、SAM 3、BioCLIP 2、本地 MLLM）和直推设定（D3）能否接受，这决定 C5、C7、D3 能不能进主表。
5. **P1 中各数据集"背景"的定义**：普查结果，特别是 SUIM/USIS10K 的 seabed、DUT 的背景码。
6. **AquaOV255 的类名**：是否已经是常用名？是否有学名？`category_split.json` 的层级内容是什么（C4 要用）？
7. **LoCoFuse 的"文本对比校准"具体做法**：它是主要竞争者，C3/C4 要和它区分开。
8. **本地 MLLM 的版本和 GPU 预算**：Qwen2.5-VL-7B，还是更新的模型？（C-b 的一致票、D2 都要用）
9. **论文主协议的决策**：P0 还是 P1b/P2。这决定 C1 能否作为主创新。
10. **服务器的外网访问**：能否下载 SD、SAM 3、FOUND 的权重和 WordNet 词表。
11. **开发集**：USIS10K train 去重后的规模和类别分布，用来定 γ、阈值等类无关超参。

---

## 附录 A 本报告新增引用

"读法"一列：全文 = 读过 PDF 页面；二手 = alphaXiv 摘要报告或检索摘录；题录 = 只看了标题和摘要。

| 文献 | 作者（第一作者等） | 年 | venue | arXiv | 读法 | 本报告用于 |
|---|---|---|---|---|---|---|
| CLIP as RNN (CaR) | Sun, Li, Torr, Gu, Li | 2024 | CVPR'24* | 2312.07661 | 全文 | F3、A/G 组、C2 |
| Diffusion Models for Open-Vocabulary Segmentation (OVDiff) | Karazija, Laina, Vedaldi, Rupprecht | 2024 | ECCV'24* | 2306.09316 | 全文 | F5、A5、C5 |
| Training-Free OVS with Offline Diffusion-Augmented Prototype Generation (FreeDA) | Barsellotti 等 | 2024 | CVPR'24* | 2404.06542 | 全文 | F5、F7、C5 |
| SCI-CLIP: Segment-Centric Inference with Reference Memory | Zamini, Shukla | 2026 | arXiv | 2608.05627 | 全文 | F3、F7 |
| ActiveSAM | Tien, Shen | 2026 | arXiv | 2606.16996 | 二手 | F4、A11 |
| NERVE | Mahatha, Dolz, Desrosiers | 2025 | WACV'26 | 2511.08248 | 全文 | F2、F5 |
| PEARL: Geometry Aligns Semantics for Training-Free OVSS | Pei 等 | 2026 | CVPR'26 | 2603.21528 | 全文 | F1、G1 |
| Looking Beyond the Window: GLA-CLIP | Lee 等 | 2026 | CVPR'26 | 2603.23030 | 全文 | F2、F2 组（S1 多层融合） |
| REBASE: Reference-Background Subspace Elimination | Mantha Sai Gopal 等 | 2026 | arXiv | 2607.09082 | 全文 | F8、A6 |
| Training-Free Class Purification (FreeCP) | Chen 等 | 2025 | ICCV'25 | 2508.00557 | 全文 | S4p、C3、C-e |
| Direct Segmentation without Logits Optimization (DSLO) | Li 等 | 2026 | arXiv | 2604.07723 | 全文 | G2、S4p 的早拒绝 |
| CLIP-DIY | Wysoczańska 等 | 2024 | WACV'24* | 2309.14289 | 全文 | F9、A4、C1 |
| InstructSAM | Zheng 等 | 2025 | NeurIPS'25 | 2505.15818 | 全文 | F3、A9 |
| ProMaC: Leveraging Hallucinations… | Hu, Lin, Yan, Gong | 2024 | NeurIPS'24 | 2408.15205 | 全文 | F6、A8 |
| NegLabel: Negative Label Guided OOD Detection | Jiang 等 | 2024 | ICLR'24 | 2403.20078 | 全文 | F12、A7、C6 |
| Realistic Test-Time Adaptation of VLMs (StatA) | Zanella, Fuchs, De Vleeschouwer, Ben Ayed | 2025 | CVPR'25 | 2501.03729 | 全文 | F11、C3 |
| Transductive Zero-Shot and Few-Shot CLIP (EM-Dirichlet) | Martin 等 | 2024 | CVPR'24 | 2405.18437 | 二手 | F11、C3 |
| The Neglected Tails in VLMs (REAL) | Parashar 等 | 2024 | CVPR'24* | 2401.12425 | 全文 | F13、E1、C4 |
| Prompting Scientific Names for Zero-Shot Species Recognition | Parashar, Lin, Li, Kong | 2023 | EMNLP'23* | 2310.09929 | 全文 | F13、C4 |
| IchthyoNoma | Nazim-E-Alam 等 | 2026 | arXiv | 2609.03985 | 全文 | F13、C2/C4 的反证与证据 |
| Visual Species Recognition with LMMs as Post-Hoc Correctors (POC) | Liu, Basu, Caverlee, Kong | 2025 | arXiv | 2512.15748 | 全文 | F13、C-b、D2 |
| QMSR | Zhang 等 | 2026 | arXiv | 2609.18752 | 全文 | F15、H1 |
| CLIP-ES (CLIP is Also an Efficient Segmenter) | Lin 等 | 2023 | CVPR'23* | 2212.09506 | 二手 | F10 |
| Fine-Grained Visual Prompting (FGVP) | Yang 等 | 2023 | NeurIPS'23 | 2306.04356 | 二手 | B1、C2 |
| TagCLIP（局部到全局多标签） | Lin 等 | 2024 | AAAI'24 | 2312.12828 | 二手 | F10、C2 |
| The Power of Prior (LLaVA) | Zhang 等 | 2026 | CVPR'26 | — | 二手 | F6 |
| Your LVLM Only Needs A Few Attention Heads For Visual Grounding | Kang 等 | 2025 | CVPR'25 | 2503.06287 | 二手 | F6 |
| In Defense of Lazy Visual Grounding (LaVG) | Kang, Cho | 2024 | ECCV'24 | 2408.04961 | 二手 | F3、A3 |
| ReME | Xuan, Deng, Ma | 2025 | ICCV'25 | 2506.21233 | 二手 | F7 |
| INSID3 | Cuttano 等 | 2026 | CVPR'26 | 2603.28480 | 二手 | F8 |
| No time to train! | Espinosa 等 | 2025 | [?] | 2507.02798 | 二手 | F8 |
| Vision Transformers Don't Need Trained Registers | Jiang, Dravid, Efros, Gandelsman | 2025 | NeurIPS'25 | 2506.08010 | 二手 | F1 |
| SegEarth-OV（去全局偏置部分） | Li 等 | 2025 | CVPR'25 | 2410.01768 | 二手 | F3（S3） |
| SAM 3: Segment Anything with Concepts | Carion 等 | 2025 | ICLR'26 | 2511.16719 | 二手 | F4、A10、C7 |
| Deep Spectral Methods | Melas-Kyriazi 等 | 2022 | CVPR'22 | 2205.07839 | 二手 | F9、A3 |
| Saliency Optimization from Robust Background Detection | Zhu, Liang, Wei, Sun | 2014 | CVPR'14 | — | 二手 | F9、A2 |
| Unsupervised Object Localization: Observing the Background (FOUND) | Siméoni 等 | 2023 | CVPR'23* | 2212.07834（凭记忆） | 题录（经 CLIP-DIY 转引） | F9、A1 |
| TokenCut | Wang 等 | 2022 | [?] | 2202.11539 | 题录 | F9 |
| Emergent OVSS from Off-the-shelf VLMs (PnP-OVSS) | Luo 等 | 2024 | CVPR'24 | 2311.17095 | 题录 | F10 |
| Matcher / PerSAM / GF-SAM | — | 2023–2024 | [?] | 2305.13310 / 2305.03048 / — | 题录 | F8 |
| GenSAM（单个通用提示做伪装分割） | Hu 等 | 2023 | [?] | 2312.07374 | 题录 | F6 |
| CHiLS | Novack 等 | 2023 | [?] | 2302.02551 | 题录 | E2 |
| FuDD（Follow-Up Differential Descriptions） | Esfandiarpoor, Bach | 2024 | [?] | 2311.07593 | 题录 | E3 |
| Taxonomy-Aware Evaluation of VLMs | — | 2025 | [?] | 2504.05457 | 题录 | E2 |
| Boosting VLMs with Transduction (TransCLIP) | Zanella 等 | 2024 | [?] | 2406.01837 | 题录 | F11 |
| Label Propagation for Zero-shot Classification (ZLaP) | Stojnić 等 | 2024 | [?] | 2404.04072 | 题录 | F11 |
| Label-Free Prompt Distribution Learning and Bias Correcting (Frolic) | — | 2024 | [?] | 2410.19294 | 题录 | F11 |
| GL-MCM | Miyai 等 | 2023 | [?] | 2304.04521 | 题录 | F12 |
| Earth2Ocean（会议出处核实） | Li 等 | 2025 | CVPR'26 | 2511.07923 | 二手 | 背景 |

已在计划书或几何报告中引用、这里继续使用的：ProxyCLIP、CorrCLIP、Trident、CASS、LPOSS、FLOSS、TPA、PTC、MLMP、DAF、SegEarth-OV3、BioCLIP 2、MARIS、What Holds Back OVS 等，venue 和读法以那两份报告为准。

**凭记忆、未核实的经典文献**：Saerens 等，标签漂移 EM（Neural Computation, 2002）；GrabCut（SIGGRAPH 2004）；DenseCRF（NeurIPS 2011）；PAMR（CVPR 2020）；SLIC（TPAMI 2012）；Felzenszwalb 图分割（IJCV 2004）；联合双边上采样 JBU（SIGGRAPH 2007）；Mind the Gap（NeurIPS 2022）；MCM（NeurIPS 2022）；SuS-X（ICCV 2023）；ZERO（NeurIPS 2024）；MTA（CVPR 2024）。

## 附录 B 与前两份报告的去重说明

- **几何/深度**：全部归几何报告，本报告不再讨论。C1 的判据与几何报告 G1 同构，方便直接比较"深度物体性"和"代理图物体性"。
- **计划书已有、本报告做了扩展或替代的条目**：
  - B1 名称修复 → C4 扩展为"无标签选名 + 层级"；
  - B2 MLLM 先验 → C3 替代为"逐图 EM + 软先验 + 多 MLLM 一致性"，并保留 B2 作对照；
  - B3 跨图原型（TPA）→ C5 提供无标签数据源的替代（生成图），D3 是两者的组合；
  - B4 BioCLIP 2、B6 SAM 3 → 归入 §3.3 的强模型公平对照，以及 C7；
  - C4 背景名基线 → C6 扩展为"负标签 + 逐图背景名"，C1 补上视觉侧的背景通道。
- **全新的条目**：C1、C2，以及第 0 节第 2 条"识别收益与背景互补"的机制推断。这条推断可以直接用 D12 的阶段交换 oracle 检验。
