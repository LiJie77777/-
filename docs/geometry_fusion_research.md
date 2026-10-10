# 怎样真正利用几何/深度特征，以及怎样做多源特征融合：调研与候选方案

> 日期：2026-10-10。读者：导师、执行同学、跑实验的同学。分支 `claude/busy-bohr-kkw7nd`。
> 与请求 1 报告 [`docs/uovs_research_plan.md`](uovs_research_plan.md) 配套使用。本次新增脚本：`scripts/dump_depth_cache.py`、`scripts/prevalidate_depth_figure_ground.py`、`scripts/e2o_patches/geo_affinity.py`，测试在 `scripts/tests/`。
> 标注约定：【文献事实】= 读过原文（全文问答）确认的内容；【本项目事实】= 你们提供的数字，或我在官方代码里能逐行核对的内容；【推断】= 我的推断，每条都附依据；【未核实】= 只看过检索摘要、凭记忆或二手转述，没读原文。

## 0 一页结论

**核心判断。**
- E2O 现在对 DepthAnything 的用法，相当于换了一个 DINOv2 家族的代理特征。深度本身完全没用到：深度图、深度不连续、3D 距离、物理衰减都不在流程里。
- 官方协议 P0 下，即使几何把分组做到完美，ViT-B 也只能涨 +1.3。
- 所以"真正利用几何"要同时满足两点：一是用显式几何量；二是在能反映几何作用的指标上评测，并且加对照，排除"只是特征更强"的解释。

1. **导师的批评成立，代码可以证明。**
   - 【本项目事实】`get_geo_feats()` 已经跑了 DPT 深度头。返回的深度图 `geo_map` 在 `forward_feature()` 里被 `_` 丢掉了（`Earth2Ocean.py` L289）。
   - 实际只用了 block-11 的 patch token：`geofeats_idx='3'`，即 [2,5,8,11] 中的最后一层。用它算余弦自相似，作为 ProxyCLIP 式注意力。
   - 【文献事实】DAv2 编码器用 DINOv2 初始化，并用特征对齐损失向冻结的 DINOv2 看齐（V1 的容忍阈 α=0.85）[DA1, DA2]。
   - 【文献事实】DINOv2-B/14 本身的深度可读性已经很强：NYU 深度探针 δ1 为 93.4；OpenCLIP-B/16 只有 52.5；用深度监督训练的 MiDaS ViT-L 反而只有 78.7 [ElB]。
   - 【推断】因此 A4−A3 只有 +0.09/+0.38/+0.61 并不意外。DA 的 block-11 特征能多带来的"几何"，DINOv2 已经有了。
2. **P0 下，几何的增益上限很小。**
   - 【本项目事实】完美分组（GT-Uniform oracle）只带来 +1.3/+2.5/+2.1（B/L/H）。区域分类上，文本头 33.17%，视觉质心 85.51%。瓶颈在识别，不在分组。
   - 【文献事实】几何主要帮助"把物体分出来"，对"识别是什么"帮助很小。DFormerv2 在 LUSS 上，RGB→RGB-D 的分类只从 83.1 到 83.4，前景分割 wF 却从 0.818 到 0.868 [DF2]。用估计深度做检测，VOC、COCO 都没有收益，而且"检测器越强，深度收益越小"[Cet]。
   - 【推断】所以只改亲和或分组的几何方案（包括多源亲和融合、RADIO 代理），ViT-B 在 P0 六集平均上预期只有 0~+0.6。这不能作为"利用了几何"的主要证据。
3. **几何真正能起作用的三个位置**（按把握从高到低）：
   - **(a) 前景/背景（figure-ground）。** 【本项目事实】模型没有背景输出，所以所有配置的 P2 类无关前景 IoU 完全相同。这是现有系统里几何贡献为零的指标，也正是深度天然能补的缺口。
     - 【文献事实】在水下，相对深度（远近顺序）比度量深度可靠得多：DAv2-B 在 DA-2K 水下子集的远近判断准确率是 100% [DA2]；而度量深度的 δ1 在 FLSea/SQUID 上只有 0.2–0.64 [CM]。
     - 【推断】预计按数据集分化：背景是开阔水体的 SUIM、USIS10K 有利；DUT 的四类（海参、海胆、扇贝、海星）都贴在海床上，与背景基本共面，可能不利。
   - **(b) 边界和多物体场景（SUIM、DUT）。** 用深度不连续引导上采样或精修。但有直接负证据：LPOSS 把深度作为像素特征，只 +0.1 [LPOSS]。所以要先离线检验"深度边缘是否比 RGB 边缘更贴 GT 边界"。
   - **(c) 亲和里加显式 3D 距离。** 采用 DFormerv2 的 A⊙β^G 形式，但衰减固定、不训练。
     - 【文献事实】这在 RGB-D 室内场景有证据：+2.6/+1.7。但空间先验单独就有 +1.8/+1.3，而且是训练得到的 [DF2]。
     - 本项目的上限受第 2 点约束，所以作为机理研究。
4. **文献里正负证据合起来的规律**（§3.5）：
   - 伪深度直接融合，常常没有收益甚至有害：COD [XC]；检测 [Cet]；GeoCueFormer 在所有 stage 注入深度时 82.23→80.06 [GCF]。
   - 有效的做法都带三类限制之一：可靠性门控、只在浅层或边界处注入、只作训练信号 [GD, GCF, DG]。
   - 增益常和"多了容量或通用特征"混在一起。GeoDistill 里，只加几何分支、不加 DA 监督，就已拿到大部分收益：SIP MAE .046→.037，加 DA 监督后才到 .033 [GD]。这正是我们要排除的"更强通用特征"解释。
   - 本次检索没找到用单目深度做 2D 免训练 OVSS 的工作。PEARL 标题里的"geometry"指嵌入空间几何，不是 3D [PEARL]。这是空白，也是风险（§2.4）。
5. **建议的做法**（卡片见 §4，优先级见 §6）：
   - **第一步：零成本诊断 D0。** 看 A4−A3 的逐图差异是否随深度对比度增大。多层 DA（`geofeats_idx="0,1,2,3"`）只要改配置。
   - **第二步：卡 G1，深度 figure-ground/背景通道。**
     1. 先用 `dump_depth_cache.py` 缓存深度。
     2. 再跑 `prevalidate_depth_figure_ground.py`（不需要推理，半天以内），按预先写定的标准决定继续或放弃。
     3. 通过后再接入模型。主指标为 P2 前景 IoU 和 P1b。
   - **第三步：卡 G2，3D 感知亲和。** 参考实现已给出并通过测试。必须带四个对照：仅空间项、打乱深度、DINOv2 PCA 平滑图、模糊深度。
   - **第四步：卡 G3，深度引导的边界精修。** 先通过离线边界对齐检验再做。
   - **多源融合要分清楚。** 亲和层融合（A1–A4 集成、RADIO 代理）只算工程增益，不宣称是"几何"。真正值得做的多源融合在识别端：文本 + 视觉原型 + MLLM（请求 1 的 B3/TPA），因为瓶颈在识别。
6. **怎样排除"其实是通用特征更强"。**
   - 凡是用 DA 特征的方案，都同时跑 DINOv2-B/14 的同构版本。
   - 凡是用深度图的方案，都同时跑四个对照：打乱深度、模糊深度、非几何平滑图、仅空间项。
   - 事先写定两类检验：
     - 剂量—反应：λ 分档；深度质量分档，DAv2-S→B。
     - 异质性：深度对比度高的图，增益应更大。
   - 主配置和最小有意义效应（SESOI）都事先写定，不在测试集上挑。
7. **需要你们补充的信息**（§6.3）。最关键的三项：
   - 导师要的结论是"P0 涨分"，还是"几何作用可归因"？
   - 服务器能否访问 HF/GitHub，用来下载 SAM2、RADIO、NAF 的权重？
   - 研究分支里逐图统计和 A3 的接口是什么样？

---

## 1 问题界定

### 1.1 E2O 到底用了 DepthAnything 的什么

| DepthAnything 能提供的信息 | E2O 是否使用 | 依据【本项目事实，官方代码逐行核对】 |
|---|---|---|
| block-11 patch token（24×24，算余弦自相似后替换 CLIP 末层注意力） | 用 | `Earth2Ocean.py` L259–268 `get_geo_feats`；`open_clip/transformer.py` L618–662 `custom_attn` |
| block 2/5/8 的中间层特征 | 没用。`custom_attn` 已支持传入列表（L635 起，对各层注意力取平均），改 `geofeats_idx` 就能用 | `base_config.py` `geofeats_idx='3'` |
| DPT 头输出的相对逆深度 `geo_map` | **算了但丢弃** | `Earth2Ocean.py` L260 返回，L289 用 `_` 接收 |
| DPT 解码器的多尺度特征（`path_1`…`path_4`） | 没用 | `depth_anything_v2/dpt.py` L141–146 |
| 深度不连续/边缘 | 没用 | — |
| 3D 邻近（深度差、反投影坐标） | 没用 | — |
| 表面法向/平面 | 没用 | — |
| 物理量（随距离的衰减、后向散射） | 没用 | — |
| 物体物理尺度 | 没用 | — |

另外两处会影响后续方案的实现细节【本项目事实】：
- **DA 按窗口运行。** 它在每个 336×336 滑窗上单独跑，因此深度只在窗口内做了仿射归一，跨窗口的尺度和偏移不一致。用深度的方案必须二选一：要么每个窗口内用中位数/MAD 归一，要么对整图单独算一次深度（`dump_depth_cache.py` 就是这样做的）。
- **没有背景标签。** `postprocess_result` 的 `prob_thd` 把低置信像素置为 0（L413），而 0 是一个真实类别，不是背景。要加背景输出，必须用新标签 C（即类别数），这与 `protocol_eval.py` 的约定一致。MMSeg v1.1.1 的 `IoUMetric.intersect_and_union` 用 `torch.histc(min=0, max=C−1)` 统计【本项目事实，源码核对】；按 torch 文档，超出范围的值会被忽略，所以 P0 下这些像素算作所属 GT 类的漏检，语义正确【推断，依据 torch 文档；建议用一张图实测确认】。

### 1.2 为什么 A4≈A3 在意料之中

- 【文献事实】Depth Anything V1：
  - 编码器用 DINOv2 预训练权重初始化 [DA1 §3.1]。
  - 在无标签数据上加了特征对齐损失 `1 − cos(f, f′)`，其中 f′ 来自冻结的 DINOv2；余弦超过容忍阈 α=0.85 的像素不计入 [DA1 §3.3, §4.1]。
  - 目的是"保留 DINOv2 的语义先验"。下游微调到 ADE20K，DINOv2 是 58.8，DA 是 59.4 [DA1 表 11]。
- 【文献事实】Depth Anything V2：学生模型基于 DINOv2 S/B/L/G，在伪标签真实图像上"沿用 V1，加特征对齐损失以保留预训练 DINOv2 的语义"[DA2 §5.2]。
- 【文献事实】El Banani 等用冻结特征加多尺度探针，测各模型的深度可读性 [ElB 附表 2、3]：

| 模型（冻结特征） | NYU 深度 δ1 | NAVI 深度 δ1 | NYU 法向 11.25° 召回 |
|---|---|---|---|
| DINOv2 ViT-B/14 | 93.43 | 68.84 | 62.01 |
| DINO ViT-B/16 | 80.44 | 56.62 | 49.11 |
| CLIP ViT-B/16（LAION） | 52.46 | 24.33 | 30.11 |
| MiDaS ViT-L/16（深度监督） | 78.65 | 58.53 | 49.58 |
| SAM ViT-B/16 | 75.60 | 49.28 | 46.37 |

- 【推断】由此可得三点：
  1. CLIP 几乎不含几何；DINOv2 家族已经隐式含有很强的几何。所以"换上 DINOv2 系代理"这一步本身就引入了隐式几何。A1→A3 的 +3.2 里有多少来自几何、多少来自语义，现有实验分不开。
  2. 深度监督的编码器，深度可读性不一定比 DINOv2 好（MiDaS < DINOv2）。所以"用深度模型的编码器特征"不等于"用几何"。
  3. DAv2 的特征被刻意拉向 DINOv2，A4 与 A3 的差就只剩很小的残差。DUT 上为负、去重后 ViT-B 约为 0，都与此一致。

### 1.3 "真正利用几何"的可操作定义

满足以下四条，才称"真正利用了几何"：

1. **显式几何量。** 方案使用的信息在 DINOv2 特征里不能直接调用，例如深度值和远近顺序、深度不连续、3D 距离、法向、与距离相关的物理量。只把深度模型的编码器当作通用特征的，不算。
2. **因果可归因。** 把深度换成错配深度（同一数据集另一张图的深度）后，增益显著消失；换成模糊深度、非几何平滑图（同网格的 DINOv2 第一主成分）、或只保留空间距离项，增益都不能被复现。
3. **超过同构对照。** 凡用 DA 特征，增益必须超过同样流程下的 DINOv2-B/14（A3 同构版本）。
4. **机制与指标对应，且有剂量—反应。** 增益出现在该机制应该影响的指标上（figure-ground→P2/P1b，边界→边界 F，3D 亲和→多物体/多类场景）；并且随几何强度或质量单调变化（λ 分档；DAv2-S→B 的深度质量；逐图深度对比度）。

### 1.4 余量账本：几何在本任务中还能在哪里起作用

| 环节 | 现有余量【本项目事实】 | 可能的几何作用 | 判断 |
|---|---|---|---|
| P0 分组（亲和） | GT-Uniform oracle +1.3/+2.5/+2.1（B/L/H，qwen7b） | 3D 亲和、多层特征、多源亲和 | 上限小【推断，依据即 oracle】 |
| P0 识别 | 文本头 33.17%，视觉质心 85.51%，线性探针 86.85%（AquaOV255，ViT-B） | 几何对语义帮助小；尺寸先验理论上可能 | 几乎不可期【推断，依据 DF2、Cet，以及本项目"几何特征加剧同科混淆"】 |
| 前景/背景（P1/P2） | 无背景输出；所有配置的 P2 前景 IoU 相同；P1a A4 只有 14.97 | 远近顺序、深度环对比 | 空间最大，但上限要靠预检估计 |
| 边界 | 尚未度量（需要加边界 F） | 深度不连续 | 待验证；有负证据 [LPOSS] |
| 多类场景（SUIM、DUT） | P0 整图填充 oracle 约 46 | 分组 + 边界 | 中等 |

另有一条【本项目事实】：几何上"不可靠"的类别组并没有得到更多增益，几何特征反而加剧了同科物种之间的混淆。这与 [DF2, Cet] 一致：几何不是识别线索，几何特征不应再塞进分类器。

---

## 2 检索策略记录

### 2.1 数据库与工具
- **alphaXiv**：
  - `discover_papers` 做语义检索，覆盖 arXiv，共约 21 次检索。
  - `answer_pdf_queries` 做全文问答，用于"精读"。
- **Hugging Face Hub 元数据**：只用来确认权重能否获取。
- **本环境的网络限制**：github.com、arxiv.org、huggingface.co、download.pytorch.org 直连都被代理拒绝（403）。官方代码在请求 1 中经 raw.githubusercontent.com 取得。
- **没用的库**：Google Scholar、Semantic Scholar、IEEE Xplore、CNKI。所以只发期刊、不上 arXiv 的工作，以及中文文献，覆盖不足。
- **检索时间**：2026-10-09 至 10-10。

### 2.2 关键词（按主题分组）
- 深度辅助的分割：`pseudo depth semantic segmentation`、`RGB-D geometry attention`、`depth prior segmentation monocular`、`Depth Anything segmentation prior`
- 无监督/免训练：`unsupervised semantic segmentation depth`、`training-free segmentation monocular depth SAM prompts`、`training-free out-of-distribution segmentation depth`
- 显著性与伪装：`camouflaged object detection depth contribution`、`RGB-D salient object detection depth-free / unreliable depth`
- 检测：`does depth estimation help object detection`
- 开放词汇：`training-free open-vocabulary semantic segmentation`、`open-vocabulary segmentation depth`、`depth-aware CLIP`
- 多源/聚合模型：`agglomerative vision foundation model RADIO segmentation`、`proxy attention ProxyCLIP fusion`
- 上采样：`zero-shot feature upsampling VFM-agnostic`、`joint bilateral upsampling features`
- 3D 感知：`3D awareness visual foundation models probing`、`geometry semantic correspondence 3D prior`
- 水下：`underwater monocular depth benchmark`、`underwater image enhancement downstream detection`、`underwater semantic segmentation geometry depth`
- 注意力中的空间/几何衰减：`spatial decay attention Manhattan`、`geometry self-attention RGBD`

### 2.3 覆盖的领域和阅读量

| 领域 | 精读（全文问答） | 摘要级【未核实】 |
|---|---|---|
| RGB-D/伪深度语义分割（通用 + 水下） | 2：DFormerv2、GeoCueFormer | 5：AMMNet、GAMUS、RSGPNet、RMT、Learning Spatial Decay |
| 无监督/免训练分割与深度 | 3：DepthG、DepSeg、深度环 OOD | 5：BED-SAM2、DepthPolyp、DGA-Net、GeomPrompt、SPDA-SAM |
| 显著性/伪装与深度 | 2：Xiang 等（COD）、GeoDistill | 4：PopNet、Depth Awakens、PDDM、Saliency-Depth conditioning |
| 检测与深度 | 1：Cetinkaya 等 | 1：Exploiting depth from single monocular images |
| 免训练 OVSS 与多源融合 | 本轮 2：RADSeg、PEARL；复用请求 1：ProxyCLIP、LPOSS、TPA、GeoSeg-OV（部分） | 4：SENSE、AM-RADIO、RADIOv2.5、CASS（经 PEARL 表 1 转引） |
| 特征上采样 | 1：NAF（其表格含 FeatUp、JAFAR、AnyUp、LoftUp 的同表数字） | 5：FeatUp、JAFAR、LoftUp、AnyUp、RaysUp |
| 视觉基础模型的 3D 感知与 DA 本身 | 3：El Banani 等、DA V1、DA V2 | 4：Lexicon3D、Geometry Matters、SeeSE3、DINOcular |
| 水下深度/成像物理/增强 | 2：Cai & Metzler、Beneath the Surface | 6：下游任务导向 UIE、WaterFlow、UniV2D、DY-LUT、Weather-Conditioned DA、Wat3R |
| 合计 | 本轮 16 篇 + 复用约 4 篇 | 约 34 篇 |

### 2.4 检索的局限
- **"geometry"一词多义。** OVSS 文献里的"geometry"多指嵌入空间几何（如 PEARL 的 Procrustes 对齐），不是 3D。用"geometry"检索，召回会偏。
- **2D 免训练 OVSS + 单目深度：检索无果。** 两次专门检索，一次没有结果，另一次只返回 3D/机器人方向的工作。这可能是真空白，也可能是检索漏召回。投稿前建议用 Semantic Scholar 再查一遍。
- **多为未评审的新稿。** 2026 年的 arXiv 文献大多未经同行评审（如 GeoCueFormer、GeoDistill、深度环 OOD），数字只能作为方向性参考。
- **负结果少发表。** 负面证据天然偏少，本报告收集到的负面证据应视为下限。

---

## 3 分类体系与证据

### 3.1 三个维度
- **用了什么几何信息**：
  - I0 深度模型编码器特征（隐式）
  - I1 深度值/远近顺序
  - I2 深度不连续
  - I3 3D 邻近（深度差 + 空间距离）
  - I4 法向/平面
  - I5 物理量
  - I6 度量尺度
- **注入位置**：
  - 输入（前融合、复原）
  - 中层（注意力/亲和）
  - 后期（logits 精修、上采样）
  - 区域/提示（SAM）
  - 只作训练信号（损失、蒸馏）
- **设定**：
  - 免训练
  - 轻训练（无人工标注，或只学少量参数）
  - 训练式

### 3.2 总表

| 类 | 几何信息 × 位置 × 设定 | 代表工作 | 原领域报告的效果 | 适用条件 / 失败条件 |
|---|---|---|---|---|
| T0 | I0 × 中层亲和 × 免训练 | E2O GMG（A4）【本项目事实】；GeoSeg-OV 在遥感中用 DAv2-B 作结构先验（请求 1 部分阅读） | A4−A3 = +0.09/+0.38/+0.61；DUT 上全为负 | 与 DINOv2 同源时，几乎等于换代理 |
| T1 | I0 × 中层门控 × 训练式 | MARIS：α=σ(W[F_V‖F_G])，多尺度 DA【未核实，数字来自交接材料】 | +2.8 mAP；跨域时 vitb 优于 vitl | 需要标注训练 |
| T2 | I1 × 区域判别 × 免训练 | 深度环 OOD：SAM2 + DAv2-L，比较分割块与其外环道路的视差差，δ_min=0.001 [DR]【文献事实】 | 在 SegmentMeIfYouCan 上有竞争力，主要用来压制与路面共面的误检 | 物体与周围要有深度差；共面物体会失败 |
| T3 | I1/I2 × 中层 × 训练式 | DFormerv2 LUSS：RGB vs RGB+D [DF2]【文献事实】 | 分类 83.1→83.4；前景 wF 0.818→0.868，MAE 0.054→0.048 | 深度主要帮"分出物体"，几乎不帮语义 |
| T4 | I3 × 注意力偏置 × 训练式 | DFormerv2：(Softmax(QKᵀ)⊙β^G)V，G 融合深度差与曼哈顿距离 [DF2]【文献事实】 | NYU/SUN：深度先验 +2.6/+1.7；空间先验 +1.8/+1.3；两者合用 +4.5/+3.9 | 用的是 RGB-D 室内数据；空间项单独就贡献很大 |
| T5 | I3 × 训练信号 × 训练式 | DepthG：深度—特征相关损失 + 3D 最远点采样 [DG]【文献事实】 | COCO-Stuff ViT-S/8 24.5→25.6（加 3D-LHP 为 26.7）；ViT-B/8 28.2→29.0；Cityscapes 21.0→23.1 | 深度质量要求高（ZoeDepth 最好，KBR 最差）；推理时不用深度 |
| T6 | I1 × 提示 × 免训练（但要带标签的模板库） | DepSeg：DAv2 深度→k-means/Canny/Otsu/分水岭→SAM2 点提示→DINOv3 模板匹配 [DS]【文献事实】 | CholecSeg8k：SAM2 自动掩码 + 模板 31.6 → 深度提示 35.9 | 器官与器械之间有明显深度层次 |
| T7 | I2 × 浅层门控 × 训练式 | GeoCueFormer：DAv2-S 离线深度生成空间门，只调制 stage 1–2 的小波高频 [GCF]【文献事实】 | SUIM：小波 80.98 → 小波+深度 81.58；完整模型 82.23；**深度注入全部 stage 时降到 80.06** | 深度只能弱、浅层地注入 |
| T8 | I1–I2 × 训练信号 × 训练式 | GeoDistill：DAv2-S 只作训练期教师，带可靠性门 [GD]【文献事实】 | SIP MAE：RGB .046 / 原始深度 .036 / **只加几何分支不加教师 .037** / 完整 .033 | 增益大部分来自分支容量，教师只额外带来约 10% 的 MAE 下降 |
| T9 | I2 × 后期精修 × 免训练 | LPOSS：深度作为像素特征参与标签传播 [LPOSS]（请求 1 已读）【文献事实】 | 42.2 vs 42.1（+0.1） | 直接负证据 |
| T10 | 图像引导 × 上采样 × 零样本（预训练一次） | NAF：只用图像引导的邻域注意力上采样，与 VFM 无关 [NAF]【文献事实】 | VOC 线性探针，相对最近邻平均 +5.58（双线性 +3.34，AnyUp +4.09）；开放词汇 +0.44（相对此前最佳） | 没用深度；适合作为强 RGB 对照 |
| T11 | I5 × 输入复原 × 免训练/训练 | Beneath the Surface：9 种水下增强 + YOLO-NAS [BtS]【文献事实】 | 数据集层面全部不升反降（RUOD 0.62 → 0.55–0.62；CUPDD 0.38 → 0.22–0.37）；逐图 oracle 挑选能升（RUOD 0.68→0.77） | 只对低质量图有益；挑选需要无标签准则 |
| T12 | 多源 × 中层亲和 × 免训练 | RADSeg：RADIOv3 作代理 + 自相关注意力 SCRA/SCGA + RADIO-SAM [RAD]【文献事实】 | 中分辨率平均：ProxyRADIO 45.44/46.05 vs ProxyCLIP 40.87；SCRA +2.0，SCGA +1.4，R-SAM +2.0 | 自然图像，不计"background"标签；属于更强通用特征 |
| T13 | 多源 × 决策层 × 免训练（直推式） | TPA：无标签部署图像上的 DINOv2 原型 + 后融合（请求 1 已读） | ProxyCLIP +2.0，Trident +1.9 | 针对识别瓶颈 |
| T14 | I0 × 嵌入空间"几何" × 免训练 | PEARL：末层 Procrustes 对齐 + 文本感知拉普拉斯传播 [PEARL]【文献事实】 | 8 集平均 43.2；只加 TLP 就让 SCLIP/NACLIP/SFP 提升 +1.9~+4.0 | 与 3D 无关；TLP 的边界保持靠图像梯度 |

### 3.3 正面证据要点
- **深度擅长分割物体，不擅长识别语义。** DFormerv2 的前景 wF 提升 +0.05，分类只 +0.3 [DF2]。DepthG 用深度后，无监督分割 mIoU 提升 +0.8~+2.2 [DG]。
- **深度能修补"共面"误检。** 深度环 OOD 用分割块与外环的深度差，压制共面误检 [DR]。这和我们的前景/背景问题结构相同。
- **相对深度在水下可靠，度量深度不可靠。** DAv2 在 DA-2K 水下子集的远近判断准确率：S 99.2、B 100、L 99.2 [DA2 表 14]。
- **免训练的亲和/区域改造在自然图像上确实有效**（RADSeg 的 SCRA +2.0、R-SAM +2.0）[RAD]，**但在本项目受分组 oracle 上限约束。**

### 3.4 负面证据要点
- **COD：** 用 MiDaS 生成的深度，直接做早/中/晚融合，都不如纯 RGB（COD10K Fβ：0.696 vs 0.677/0.694/0.681）。伪装物体占满视野，或自身深度跨度大时，深度有害；需要辅助深度分支加置信度损失才有收益 [XC]。
- **检测：** 估计深度在 VOC（74.9→74.2–74.5）和 COCO（26.9→26.4）上无益，只在 SUN RGB-D 上有益。原文说"检测器越强，深度收益越小"；估计深度还会增加漏检 [Cet]。
- **水下深度质量差：** DAv2 S/B/L 在 FLSea-Canyon 上 δ1 只有 0.446/0.570/0.636，在 SQUID 上 0.205–0.290。浑浊、低纹理、远距离时进一步变差 [CM]。
- **LPOSS 的深度像素特征只 +0.1** [LPOSS]。**GeoCueFormer 在全部 stage 注入深度掉 2.2 点** [GCF]。
- **增强/复原对识别在数据集层面没有帮助** [BtS]。

### 3.5 归纳出的规律（每条附依据）
- **R1 深度主要改善定位，几乎不改善识别。** 【推断】依据 [DF2] LUSS、[Cet]、本项目"几何特征加剧同科混淆"。→ 主要评测指标应选 P1/P2、边界、多类场景。
- **R2 伪深度直接融合常常无益甚至有害；有效的做法都是可靠性门控、浅层/局部注入、只作训练信号三者之一。** 【推断】依据 [XC, Cet, GCF, GD, DG]。→ 免训练方案也要内置"何时信任深度"的门（例如只在深度对比明显时启用）。
- **R3 骨干越强，深度的边际收益越小。** 【文献事实】[Cet] 的原话。【推断】DINOv2 已隐式编码深度 [ElB]，所以在 DINOv2 系代理之上叠加 DA 特征，收益自然趋近于零。
- **R4 深度质量决定上限。** 【文献事实】[DG] 中深度模型越好，增益越大；[CM] 显示水下深度退化。→ 剂量—反应可以用深度质量分档来做（DAv2-S vs B；若能拿到水下专用深度模型，再加一档）。
- **R5 "几何"收益常被容量/空间先验/通用特征混淆。** 【文献事实】[GD] 里只加分支就拿到大部分增益；[DF2] 里空间先验单独 +1.8；[NAF] 表 7 里光是加位置编码，就从 57.07 升到 58.03/60.41。→ 本项目必须做"仅空间项"和"同构 DINOv2"对照。
- **R6 水下"相对远近"可用，"度量距离"不可用。** 【推断】依据 [DA2] 的 DA-2K 水下准确率对比 [CM] 的度量 δ1。→ 优先用远近顺序（G1、G2），慎用度量（G7）。
- **R7 多源融合在亲和层收益有界，在决策层（识别端）才有空间。** 【推断】依据：本项目分组 oracle +1.3、识别差距 33%→85%；TPA 在决策层有 +2.0。

---

## 4 候选方案卡片

所有成本按你们的条件估算：5×RTX 3090；ViT-B 跑一遍六个数据集约 75 min【本项目事实】。六个数据集的图片数【本项目事实，官方 imglist】：AquaOV255 20722、USIS16K 16150、USIS10K val+test 3189、MAS3K 2909、SUIM train_val 1487、DUT 1486，共约 4.6 万张。

### D0 诊断：A4 现有的那点增益是不是"几何的"
- **机制**：不提新方法。把 A3、A4 的逐图结果配对，求 Δ_i = A4_i − A3_i（逐图像素准确率或 P2 指标；单物体图的逐图 IoU 容易饱和），再与 G1 预检脚本给出的逐图几何统计做相关：`ring_contrast`、`auroc_depth`。同时跑多层版本：DA 用 `geofeats_idx="0,1,2,3"`，并与 DINOv2 同层多层对比。
- **用到的几何**：只用于分析。
- **文献依据**：[ElB]、[DA1, DA2]，见 §1.2。
- **设定**：免训练，纯分析。
- **实现**：先用 `dump_depth_cache.py` 缓存深度，再跑 `prevalidate_depth_figure_ground.py --out`，按图片 id 与研究分支的逐图统计 join，算 Spearman ρ（bootstrap CI）和三分位分组均值。多层 DA 只需改配置。
- **算力/工作量**：缓存深度约 1 GPU·h【推断，依据 DAv2-B 在 518 输入下单张数十毫秒】；分析 1 人日；多层 DA 一组 75 min。
- **预设判据**：
  - 若 ρ(Δ_i, ring_contrast) 的 95% CI 不含 0，且在 ≥4/6 个数据集上同号，判为"A4 的残余增益有几何成分"。
  - 否则在论文里如实写："现有 GMG 增益没有可归因于几何的证据"。
- **预期**：ρ≈0【推断，依据：A4−A3 很小，DUT 为负，去重后 ViT-B≈0】。
- **风险**：低。这一步本身就回答了导师"有没有真正用到"的问题。

### G1 深度 figure-ground / 背景通道（首推）
- **核心机制**：对整图算一次 DAv2 相对逆深度，判定前景（近）和背景（远），输出背景标签 C。按把握递进，有三个变体：
  - (a) 逐图 Otsu 阈值，阈值以上为前景；
  - (b) 区域级深度环对比：以模型自己预测的连通区域为单位（或用 SAM2 区域，见 G5），若区域内与外环的视差差 / MAD < δ，判为背景（δ 事先定为 0）；
  - (c) 与文本"背景"提示竞争，例如 "open water"、"sea floor"。**这是非几何对照**，也可能与几何互补。
- **用到的几何**：I1 远近顺序 + 局部深度对比（显式几何）。
- **文献依据与原领域效果**：
  - [DR] 免训练，用分割块对外环的深度差压制共面误检；
  - [DF2] 前景 wF +0.05（训练式）；
  - [GD] 蒸馏后推理时不用深度；
  - 水下相对深度可靠 [DA2 表 14]。
- **迁移理由**：水下单物体数据集以特写为主，前景通常比背景水体近【推断，待预检】。
  - SUIM 的类表里没有 waterbody，USIS10K 有 sea-floor 类、没有水体类【本项目事实，官方 cls 文件】。可见两者的"背景"主要是水体，而水体通常是最远的区域。
- **迁移障碍**：
  - DUT 四类都是底栖动物（海参、海胆、扇贝、海星）【本项目事实】，与海床几乎共面。
  - MAS3K 含大量伪装/底栖物种（Flounder、FrogFish、ScorpionFish、StarFish、SeaCucumber 等）【本项目事实，类表】，正是 [XC] 所说深度有害的情形。
  - P0 不计背景，加背景输出在 P0 上只会扣分：被误判为背景的前景像素变成漏检。
- **设定**：免训练（Otsu、δ 固定，没有任何学习参数）。
- **实现**：
  1. `scripts/dump_depth_cache.py` 缓存整图深度；
  2. 在 `postprocess_result` 里加 `seg_pred[bg_mask] = num_classes`；
  3. 在 `RunAccumulator` 里同时得到 P0/P1a/P1b/P2；
  4. 如果已经存有逐图预测图，可以离线把预测图和深度掩码组合起来，**不必重跑推理**。
- **算力/工作量**：缓存约 1 GPU·h；预检每个数据集抽 500 张，CPU 上不到 1 h；接入 0.5 人日；一个配置 75 min（离线组合则为 0）。
- **低成本前置验证**：`python scripts/prevalidate_depth_figure_ground.py --mask_dir … --depth_dir … --bg_codes … --void_codes … --max_images 500`。
  - 输出：前景比背景更近的图像占比、深度 AUROC、中心先验与打乱深度两种对照的 AUROC、Otsu 前景 IoU 与"全前景"IoU 的对比、深度环对比。
  - "全前景"IoU 就是 E2O 现在的 P2 前景 IoU。
  - **通过**（脚本按此打印 `G1_pass`）：在 ≥4/6 个数据集上同时满足
    - AUROC_depth ≥ 0.70；
    - 比最好的对照高 ≥ 0.05，且差值的 CI 下界 > 0；
    - Otsu 前景 IoU 比全前景高，差值的 CI 下界 > 0.05。
  - **放弃**：在 ≥3/6 个数据集上 AUROC < 0.60，或不优于对照。
- **预期区间**：
  - P2 前景 IoU：**无文献依据**（水下没有 figure-ground 的定量文献）。预检直接给出离线估计：`iou_otsu_depth − iou_allfg` 就是上线后 P2 前景 IoU 增量的近似值。
  - P0：0 到负，代价约等于"被误判为背景的前景像素占比"。
  - 分数据集的事先预测【推断】：SUIM、USIS10K 的增益大于 AquaOV255、USIS16K，后两者又大于 MAS3K、DUT。依据是背景构成（开阔水体 vs 海床）与 [XC]。
- **风险**：
  - 各数据集的背景编码和"背景"语义不一致：AquaOV255 用 16 位编码 65535，其它数据集要普查（`dedup_and_mask_consistency.py --census_only`）；
  - 浑浊远景的深度噪声大 [CM]；
  - P0 必然扣分，所以结论只能放在 P1/P2（线 C）框架下陈述。

### G2 3D 感知亲和（免训练版 DFormerv2 几何先验）
- **核心机制**：在 GMG 的 S̃ = γ(S − β·mean S) 上减去几何罚项：
  - 罚项为 `pen_ij = s·(λ_z·|z_i−z_j|/med|Δz| + λ_s·‖p_i−p_j‖₁/med‖Δp‖₁)`。
  - z 是窗口内按中位数/MAD 归一的视差，对 DA 未知的尺度和偏移不变；s 是 S̃ 正值的中位数，是无标签的尺度。
  - **soft 模式**：先按原规则截断，再对保留的项减罚。数学上等价于 A⊙exp(−pen) 后再行归一化，也就是 DFormerv2 的 A⊙β^G，只是 β 固定、不学习。这个等价性已在 `scripts/tests/test_geo_affinity.py` 中数值验证。
  - **hard 模式**：先减罚再截断，可以切断跨深度层的连接。
- **用到的几何**：I3 3D 邻近，即深度差加 2D 距离。
- **文献依据**：
  - [DF2]：深度 +2.6/+1.7，空间 +1.8/+1.3，训练式，RGB-D 室内；
  - [NAF] 表 7：空间编码很重要；
  - [DG]：深度—特征相关；
  - 负证据：[LPOSS] +0.1。
- **迁移理由**：鱼群和多物体分布在不同深度层；前景与背景水体的深度不同，可以抑制跨层聚合。
- **迁移障碍**：P0 上限 +1.3（B）；深度按窗口仿射归一（已用中位数/MAD 处理）；浑浊区深度噪声大。
- **设定**：免训练。
  - **主配置事先写定**：soft，λ_z=0.25，λ_s=0。依据是无标签规则：中位罚项 = 典型正 margin 的 1/4。
  - **剂量档**：λ_z ∈ {0.125, 0.25, 0.5, 1}，全部报告，不从中挑最好的。
- **实现**：`scripts/e2o_patches/geo_affinity.py`。numpy 参考实现已测试；torch 移植是逐行对应的，**本环境装不了 torch，没有实跑**。需要改三处，见文件头注释：
  1. 在 `forward_feature` 里保留 `geo_map` 并池化到 24×24；
  2. 让 `encode_image` 和 `VisionTransformer.forward` 透传参数；
  3. 在 `custom_attn` 里替换 −inf 那一行。
- **对照（必做）**：
  - 仅空间项（λ_z=0，λ_s=0.25）；
  - 打乱深度（用同数据集下一张图的深度）；
  - 非几何平滑图（`pca1_map_torch`，即 DINOv2 token 的第一主成分）；
  - 模糊深度（σ=3 个 token），用来区分"边缘"与"低频远近顺序"。
- **算力/工作量**：编码 1 人日；主配置 + 4 个对照 + 3 个剂量档 = 8 个配置，约 10 GPU·h，5 卡并行约 2 h。
- **低成本前置验证**：同一个预检脚本的 A1 部分。在 GT 标签下随机抽像素对，正样本为同一区域，算 −|Δz| 区分同/异区域对的 AUROC；并在空间距离三分位内分别算（即控制空间距离后，深度是否仍有区分力）。
  - **通过**（脚本打印 `G2_pass`）：`pair_auroc_depth_given_spatial` ≥ 0.60，且 CI 下界 > 0.55。
  - **放弃**：在 ≥3/6 个数据集上 CI 含 0.5。
  - 注意这只是必要条件。真正的检验是加进模型后，打乱深度等对照能否复现增益。
- **预期区间**：
  - P0 六集平均 0~+0.5（ViT-B）【推断】。依据：上限 +1.3；DFormerv2 的 +1.7~2.6 是在传感器深度、可学习衰减的条件下得到的。
  - SUIM/DUT 和 P1b：无依据。
- **风险**：最可能的结果是"仅空间项就复现了大部分增益"（R5）。那样结论应写成"局部性先验有效"，不能写成"几何有效"。

### G3 深度引导的边界精修/上采样
- **核心机制**：两种做法。
  - 把 `forward_feature` 里 24×24 logits 的双线性上采样（L313/315）换成联合双边上采样（JBU），值域核用 [RGB, 深度] 或只用深度。
  - 或者在 `forward_slide` 拼完窗口后，对整图 logits 做 1–3 步 PAMR 式精修，亲和由 RGB+深度定义。
  - 强 RGB 对照：NAF 零样本上采样器（只用图像引导）。
- **用到的几何**：I2 深度不连续。
- **文献依据**：
  - [NAF]：图像引导上采样在 VOC 线性探针上 +5.58，开放词汇 +0.44；双线性本身就是强基线；
  - [PEARL] 的 TLP（图像梯度保边）给其它方法带来 +1.9~4.0；
  - 负证据：[LPOSS] 深度 +0.1；
  - [GCF]：深度只适合浅层注入。
- **迁移理由**：水下散射让 RGB 边缘变弱，深度边缘也许能补上。
- **迁移障碍**：DAv2 在细边缘处可能过平滑；有直接负证据；C=254 时整图 logits 的显存较大，需在 1/2 分辨率处理。
- **设定**：免训练。σ 取经典默认：空间 σ 等于网格步长，值域 σ 按 MAD 归一后取 1。
- **实现**：1–2 人日。
- **对照（必做）**：只用 RGB 的 JBU/PAMR；NAF（需要权重，见 §6.3）。
- **低成本前置验证**：预检脚本的 B1 部分（加 `--img_dir`），在等边缘密度下比较深度梯度与亮度梯度的边界 F。
  - **通过**（`G3_pass`）：bF_depth − bF_lum ≥ 0.02 且 CI 下界 > 0，在 ≥3 个数据集上成立，尤其是 SUIM/DUT。
  - **放弃**：不通过就只保留 RGB 版本（那就不再是几何方案）。
- **预期区间**：P0 ≈0【推断，依据：边界像素占比小、分组上限小】；边界 F 无依据。
- **风险**：增益其实来自"任何边缘感知平滑"，所以 RGB 对照必须做。

### G4 DA 多层特征与 DPT 解码器特征
- **核心机制**：
  - 用 block 2/5/8/11 四层特征的注意力平均（现成功能）；
  - 或把 DPT 头的 `path_1..path_4`（更偏深度的专用特征）池化到 24×24 作代理。
- **用到的几何**：I0+，比末层更接近深度的中间表示。
- **文献依据**：[ElB] 指出 3D 信息分布在多层（所以他们用多尺度 DPT 式探针）；Lexicon3D【未核实】。
- **迁移理由/障碍**：中层特征更局部，可能有利于边界；但低层语义弱，可能按纹理或深度平面分组而不是按物体分组，导致掉点。DPT 特征若只编码深度，就相当于 G2 的隐式版本。
- **设定**：免训练。
- **实现**：多层只需 `geofeats_idx="0,1,2,3"`，零代码；DPT 特征需在 `DPTHead.forward` 加 hook，0.5 人日。**同构对照**：DINOv2-B/14 同层 [2,5,8,11] 的多层版本（需确认研究分支是否支持）。
- **算力**：每个配置 75 min。
- **前置验证**：不需要，直接跑，这是成本最低的一项。
- **放弃标准**：多层 DA 与多层 DINOv2 的差值 CI 含 0，则停止整个"DA 特征"方向。
- **预期**：P0 −0.5~+0.5【推断，依据较弱：上限 +1.3；ProxyCLIP 类方法都用末层】。
- **风险**：低。

### G5 区域先验：SAM2 与深度提示
- **核心机制**：
  - 用 SAM2 的自动掩码，或用深度聚类/Otsu 种子点作提示的 SAM2 掩码（DepSeg 式），定义区域，再在区域内平均 logits（CorrCLIP、RADSeg R-SAM 式）。
  - 结合 G1，在区域级判断前景/背景。
- **用到的几何**：I1（远近顺序/深度聚类生成提示）+ 区域内深度一致性。
- **文献依据**：[DS] 31.6→35.9（带标签模板）；[RAD] R-SAM +2.0；BED-SAM2、DGA-Net、DepthPolyp【未核实】。
- **迁移理由/障碍**：SAM2 在水下的掩码质量未知【未核实】；P0 上限仍是 +1.3；需要 SAM2 权重。
- **设定**：免训练。
- **算力**：SAM2 自动掩码每张约 1 s 量级【推断】，4.6 万张约 13 GPU·h，5 卡约 3 h。工作量 3–5 人日，对执行同学偏大。
- **前置验证**：每个数据集抽 500 张，比较 SAM2 网格提示与深度种子提示的类无关 best-overlap mIoU。**通过**：深度提示 ≥ 网格提示 +3 点。
- **预期**：P0 0~+1（ViT-B）【推断，依据 oracle 上限】；与 G1 叠加后的 P2：无依据。
- **风险**：与 G1 有部分重叠，所以排在 G1 出结果之后。

### G6 物理复原用于识别
- **核心机制**：
  - 用深度和水下成像模型 I = J·e^(−βz) + B∞(1 − e^(−βz)) [CM] 估计后向散射与衰减，复原出 J 后送进 CLIP；DA 仍用原图。
  - 或者把原图和复原图的 logits 集成，再用无标签准则逐图选择（例如取文本最大置信度更高的版本）。
- **用到的几何**：I5，与距离相关的物理量。
- **文献依据**：[BtS] 数据集层面为负，逐图 oracle 选择为正；[CM] 给出成像模型，并指出水下深度退化。
- **迁移理由**：识别是瓶颈，色偏可能干扰 CLIP 的颜色/物种线索【推断】。
- **迁移障碍**：
  - 只有相对深度，β 和 B∞ 的估计不稳；
  - 文献结论偏负；
  - 检测器是训练出来的，零样本 CLIP 未必同理。
- **设定**：免训练。
- **算力/工作量**：实现 2–3 人日；预检约 1 GPU·h。
- **前置验证**：用现有区域分类探针流程（AquaOV255，ViT-B，GT 区域）比较三种输入：原图、不用深度的灰度世界白平衡、深度复原。
  - **通过**：深度复原的文本头区域准确率比原图高 ≥1.0，比灰度世界高 ≥0.5，且 CI > 0。
  - **放弃**：不满足上述条件。如果只有灰度世界有效，说明这不是几何的作用。
- **预期**：无依据，以负证据为主。
- **风险**：工程量中等。

### G7 尺寸先验（最低优先级）
- **机制**：物理尺寸 ≈ 像素尺寸 × 距离 / 焦距。把它与 LLM 给出的类别典型尺寸比较，作为候选类先验。这是**唯一一个直接作用于识别瓶颈**的几何方向。
- **障碍**：
  - 需要度量深度和相机内参。水下度量深度很差：DAv2 δ1 0.2–0.64，UniDepth 0.52–0.94 [CM]。
  - 焦距未知。
- **设定**：免训练 + LLM。
- **前置验证**：先做 oracle 分析：GT 区域 + GT 级尺寸 + 不同深度源。如果在 GT 区域上加尺寸先验，区域分类都提升不了 ≥1 点，就放弃。
- **预期**：无依据。**风险**：高。

### M1 亲和层多源融合（A1–A4 集成、RADIO 代理）
- **机制**：
  - (a) 注意力平均：`custom_attn` 的列表模式已实现，但要先把各源特征重采样到同一 24×24 网格（DINOv1-B/8 在 336 下是 42×42，CLIP 是 21×21）；
  - (b) 共识：把各源 S̃ 相加后再 softmax，相当于专家乘积；
  - (c) 一致性门控：只保留在 ≥k 个源中都为正的边；
  - (d) 用 RADIOv2.5-B 作代理。
- **用到的几何**：没有，**不宣称是几何**。
- **文献依据**：
  - [RAD]：在自然图像上，ProxyRADIO 比 ProxyCLIP 高 4.6–5.2；
  - LoCoFuse 的 LoCoMix 只有 +0.2~1.0【未核实，数字来自交接材料】；
  - CASS 的代理从 DINOv2 换成 DINOv3：40.1→42.2（[PEARL] 表 1 转引）。
- **设定**：免训练。
- **算力**：(a)–(c) 各 75 min；RADIO 需要权重，许可协议【未核实】。
- **预期**：P0 0~+0.8（ViT-B）【推断，依据：上限 +1.3；A2–A4 已经很接近】。
- **风险**：RADIO 本身蒸馏了 DINOv2、SAM 等多个模型，属于"更强通用特征"的正例。它适合作为工程上限对照，不适合作为几何证据。

### M2 决策层多源融合（文本 + 视觉原型 + MLLM）
- 见请求 1 报告的 B3 与 TPA，这里不重复。理由是 R7：瓶颈在识别（33% vs 85%）。
- 【推断】不要把深度或几何特征加进原型。依据是本项目"几何特征加剧同科混淆"。

### T1 训练式门控融合（单列设定，与免训练分开报告）
- **机制**：两种思路。
  - MARIS 式 α=σ(W[F_V‖F_G])，或 GeoDistill 式可靠性门。
  - **轻训练变体（不用人工标注）**：训练一个两层 MLP，输入 token 对的 [S_DINOv2, S_DA, |Δz|, ‖Δp‖]，预测"是否属于同一个 SAM2 掩码"，用输出替换 GMG 的相似度。训练图像必须来自测试集以外（如 USIS10K train），并且先去重。
- **与免训练设定的关系**：它不再是 training-free，因为用了外部图像和 SAM2 伪标签。必须和最好的免训练方案（G2）同表对照，并在设定栏标明"轻训练"。
- **预期**：无依据；P0 上限仍是 +1.3。
- **风险**：数据泄漏（必须先做 C1 去重）；工作量 1–2 周，不适合当前执行同学。

---

## 5 评测设计

### 5.1 为什么 P0 看不到几何
- 【本项目事实】P0 下整图填充 oracle：AquaOV255 99.66、MAS3K 99.97、USIS16K 99.45。在单物体数据集上，只要识别对，定位几乎不影响分数。
- 【本项目事实】完美分组也只有 +1.3/+2.5/+2.1。G1 这种前景/背景方法在 P0 上只会扣分。
- **因此每张卡都事先指定"机制对应的主指标"，P0 作为次要指标照常报告，用来展示代价。**

### 5.2 各卡的主指标（预注册）

| 卡 | 主指标 | 次要指标 | 主要数据范围 |
|---|---|---|---|
| D0 | Spearman ρ(逐图 Δ, ring_contrast) | 三分位均值 | 六集，分集报告 |
| G1 | P2 类无关前景 IoU（六集平均） | P1b 前景 mIoU、P1 背景 IoU、P0（代价） | 六集，分集报告；事先预测 SUIM/USIS10K > MAS3K/DUT |
| G2、G4、M1 | P0 六集平均 | SUIM/DUT 的 P0、P1b、边界 F | 六集 |
| G3 | 边界 F（2 px 容差） | P0 | SUIM、DUT，以及其它数据集中含 ≥2 个 GT 类的图（在看任何结果之前，按 GT 定好子集） |
| G6 | 区域分类文本头准确率（探针） | P0 | AquaOV255 |

边界 F 需要加进评测：直接复用预检脚本里的 `semantic_boundary` 和 `boundary_f`，作用在预测图上即可。

### 5.3 同源规则
- **比较的是什么**：所有比较都在同一次推理中得到 P0/P1b/P2，比较对象是该卡相对其**直接基线**（A4 或 A3）的差值。
- **必须一致的设置**：骨干（LAION OpenCLIP B/16）、文本侧（UW + CSA(4o)）、类表（官方 cls）、滑窗（336/112），以及深度缓存版本（DAv2-B，518 输入）。
- **扩到 L/H 的条件**：只有 ViT-B 主结果过了判据，才扩到 L/H。

### 5.4 怎样排除"增益来自更强的通用视觉特征，而不是几何"

| 威胁（替代解释） | 对照 | 适用卡 | 判定 |
|---|---|---|---|
| DA 特征只是 DINOv2 的轻微变体 | 同构 DINOv2-B/14：同层、同网格、同流程 | D0、G4、M1 | 须 DA − DINOv2 > 0（CI） |
| 增益来自局部性/空间先验 | 仅空间项（λ_z=0）；中心先验 | G1、G2 | 须 深度版 − 仅空间版 > 0 |
| 任何平滑图都行 | DINOv2 第一主成分图（同网格、同归一化） | G1、G2 | 须 深度 > PCA 图 |
| 与具体图像内容无关的"正则化效应" | 打乱深度：同数据集下一张图的深度 | G1、G2、G3 | 打乱后增益应消失 |
| 起作用的是低频远近顺序，不是边缘 | 模糊深度（σ=3 token） | G2、G3 | 用来区分机制，不作为否决 |
| 任何边缘感知平滑都行 | 只用 RGB 引导的 JBU/PAMR；NAF | G3 | 须 深度引导 > RGB 引导 |
| 复原效果来自颜色校正，与距离无关 | 灰度世界白平衡 | G6 | 须 深度复原 > 灰度世界 |
| 多加的容量/模块 | 同参数量、无几何输入的同构模块（[GD] 表 5 的做法） | T1 | 须 有几何 > 无几何 |

**剂量—反应**（事先写定档位，全部报告，不挑选）：
- λ_z ∈ {0.125, 0.25, 0.5, 1}；
- 深度质量 DAv2-S < B（若拿到水下专用深度模型，如 Wat3R【未核实】，再加一档）；
- 判据：增益随档位单调不减，Spearman 检验。

**异质性预测**（事先写定，用 GT 计算分组，但不用于调参）：按逐图 `ring_contrast` 的三分位分组，几何方案在高对比组的增益应大于低对比组，交互项的 CI 不含 0。若增益在低对比图上反而一样大，说明不是几何的作用。

### 5.5 统计
- **抽样单位**：图像。每个数据集做配对 bootstrap（1000 次，`protocol_eval.py --bootstrap` 已支持）；六集平均对各数据集的差值再做 bootstrap。
- **去重**：先完成请求 1 的 C1（AquaOV255∩USIS16K 的重复与近重复），报告去重后的数字。
- **多重比较**：每张卡只有一个主指标，跨卡用 Holm 校正。
- **最小有意义效应（SESOI，事先约定）**【推断/约定】：
  - P0 六集平均 +0.5（约为 ViT-B 分组上限 1.3 的 40%，并且远大于 A4−A3 的 +0.09；+0.09 这么小的差值 CI 也能不含 0，所以只看显著性不够）；
  - P2 前景 IoU +3；
  - P1b 前景 mIoU +1；
  - 边界 F +2。

### 5.6 不在测试集上调参：替代办法（按优先级）
1. **文献默认值或无标签规则。** 如 Otsu、MAD 归一、"罚项中位数 = 正 margin 的 1/4"。每张卡只设一个主配置，在本报告里写定。
2. **外部验证集**（只用于类无关的几何超参）：
   - SUIM 官方 TEST 集：基准只用 train_val【本项目事实：cfg 名为 suimtrainval，1487 张】；
   - USIS10K train：基准用的是 val/test【本项目事实：imglist 为 test_*，3189 张】。
   - 两者都要先用 C1 脚本确认与六个测试集无重复【推断，TEST/train 的具体规模未核实】。
3. **留一数据集交叉拟合**：在 5 个数据集上选，在第 6 个上报，轮换一遍。报告时注明"交叉拟合"。
4. **剂量档全部报告**，不挑最好的一档。

---

## 6 优先级、关键未知与需要提供的信息

### 6.1 优先级与依赖

| 顺序 | 卡 | 前置条件 | 成本 | 决策点 |
|---|---|---|---|---|
| 1 | 基础设施：深度缓存、背景码普查、C1 去重 | DAv2-B 权重（已有） | 约 1 GPU·h + 0.5 人日 | — |
| 2 | D0 + G4 多层（改配置） | 1 | 约 2.5 GPU·h + 1 人日 | D0 回答"现有增益是否几何"；G4 失败则停止"DA 特征"方向 |
| 3 | G1 预检 → G1 接入 | 1 | 预检半天；接入 0.5 人日 + 75 min | `G1_pass`/`G1_abandon` |
| 4 | G2 预检（A1）→ G2 加对照 | 1；torch 移植需实跑一次单元比对 | 1 人日 + 约 10 GPU·h | 深度版 − 仅空间版 > 0？ |
| 5 | G3 预检（B1）→ G3 | 1 | 1–2 人日 | `G3_pass` |
| 6 | M1 (a)–(c)；RADIO 视权重情况 | 网格重采样 | 1 人日 + 3×75 min | 只作工程增益 |
| 7 | G6 预检 | 探针流程 | 2–3 人日 | 通过才继续 |
| 8 | G5、G7、T1 | G1 结果；SAM2 权重；外部数据去重 | 1–2 周 | 视前面结果而定 |

**两周最小执行包**（适合 5–6 h/天的执行同学）：
- **第 1 周**：
  - 第 1–2 天：深度缓存 + 背景码普查；
  - 第 3 天：G1/G2/G3 三项预检（同一个脚本一次跑完）；
  - 第 4 天：D0 相关分析；
  - 第 5 天：G4 多层配置 + DINOv2 同构配置。
- **第 2 周**：
  - 按第 1 周的预检结果，只推进 `pass` 的卡（预计是 G1，可能还有 G2）；
  - 跑主配置和全部对照；
  - 填同源表。

### 6.2 关键未知
1. **水下数据上的深度质量。** 没有 GT 深度，只能用预检中的前景/背景 AUROC、深度环对比、边界 F 间接估计。这是 G1–G3 能否成立的决定因素。
2. **各数据集"背景"的定义和编码。** 背景是水体、海床，还是两者都有？SUIM/USIS10K 把海床列为类别。这决定了"远=背景"是否成立。
3. **MAS3K、DUT 里底栖/伪装样本的比例。** 决定 G1 的失败面有多大。
4. **研究分支里 A3（DINOv2）是否支持取中间层。** 决定 G4/D0 的同构对照能否零成本完成。
5. **torch 版 `geo_adjust_similarity_torch` 与 numpy 参考是否一致。** 需在服务器上用同一组随机输入比对一次（`np.allclose`）。
6. **导师和审稿人能否接受 P1/P2 作为主证据。** 这依赖请求 1 的线 C（协议修正）能否成立。如果评审只认 P0，G1 的价值就没法体现。

### 6.3 需要你们或执行同学提供
1. **导师期望的结论类型**：P0 涨分，还是"几何作用可归因"？这决定主线是 G1/G3（线 C 框架），还是 M1/M2（工程增益）。
2. **服务器能否访问** huggingface.co 和 github.com，以便下载 SAM2、RADIOv2.5、NAF、DAv2-S 的权重；以及这些权重的许可是否可用于论文实验。
3. **逐图统计的文件格式**：研究分支保存的 per-image 统计长什么样（字段、图片 id 格式），用于 D0 的 join。
4. **研究分支里 A3 的实现位置**，以及能否取 [2,5,8,11] 层。
5. **各数据集掩码中的背景/void 编码**：跑一次 `dedup_and_mask_consistency.py --census_only` 并把输出贴回来。
6. **是否已保存 A4 的逐图预测图**。如果有，G1 可以完全离线评估，不用重跑。
7. **SUIM 官方 TEST、USIS10K train 能否作为外部验证集**：确认可以拿到，并跑一次 C1 去重。
8. 交接包中 qwen7b 设置下 **GT-Uniform oracle 的确切定义**（它是 §1.4 余量账本的依据）。

---

## 附录 A 引用（按标签）

精读（本轮全文问答）【文献事实】：
- [DF2] Bo-Wen Yin, Jiao-Long Cao, Ming-Ming Cheng, Qibin Hou. *DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation*. arXiv 2504.04701（会议信息【未核实】）。
- [DG] Leon Sick, Dominik Engel, Pedro Hermosilla, Timo Ropinski. *Unsupervised Semantic Segmentation Through Depth-Guided Feature Correlation and Sampling*. CVPR 2024. arXiv 2309.12378.
- [DS] Yang, Wang, Yuan, Ban. *See in Depth: Training-Free Surgical Scene Segmentation with Monocular Depth Priors*. arXiv 2512.05529.
- [DR] Varghese, Hueger, Maag. *Combining Foundation Model Confidence and Monocular Depth for Training-Free Out-of-Distribution Segmentation*. arXiv 2609.22896.
- [XC] Xiang, Zhang, Lv, Li, Zhong, Dai. *Exploring Depth Contribution for Camouflaged Object Detection*. arXiv 2106.13217.
- [Cet] Bedrettin Cetinkaya, Sinan Kalkan, Emre Akbas. *Does Depth Estimation Help Object Detection?* Image and Vision Computing, 2022. arXiv 2204.06512.
- [CM] Cai, Metzler. *Underwater Monocular Metric Depth Estimation: Real-World Benchmarks and Synthetic Fine-Tuning with Vision Foundation Models*. arXiv 2507.02148.
- [RAD] Alama et al. *RADSeg: Unleashing Parameter and Compute Efficient Zero-Shot Open-Vocabulary Segmentation Using Agglomerative Models*. arXiv 2511.19704.
- [NAF] Loïck Chambon, Paul Couairon, Éloi Zablocki, Alexandre Boulch, Nicolas Thome, Matthieu Cord. *NAF: Zero-Shot Feature Upsampling via Neighborhood Attention Filtering*. arXiv 2511.18452.
- [ElB] Mohamed El Banani, Amit Raj, Kevis-Kokitsi Maninis, Abhishek Kar, Yuanzhen Li, Michael Rubinstein, Deqing Sun, Leonidas Guibas, Justin Johnson, Varun Jampani. *Probing the 3D Awareness of Visual Foundation Models*. CVPR 2024. arXiv 2404.08636.
- [GD] Xuehao Wang, Jiaxin Hua, Runmei Li, Zhenyu Wu, Chenglizhao Chen, Ke Gu, Aimin Hao. *When Depth Hurts: Reliability-Aware Geometry Distillation for Depth-Free RGB-D Salient Object Detection*. arXiv 2609.03378.
- [PEARL] Gensheng Pei, Xiruo Jiang, Xinhao Cai, Tao Chen, Yazhou Yao, Byeungwoo Jeon. *PEARL: Geometry Aligns Semantics for Training-Free Open-Vocabulary Semantic Segmentation*. arXiv 2603.21528.
- [BtS] Ali Awad, Ashraf Saleem, Sidike Paheding, Evan Lucas, Serein Al-Ratrout, Timothy C. Havens. *Beneath the Surface: The Role of Underwater Image Enhancement in Object Detection*. arXiv 2411.14626（投稿 Elsevier 的预印本）。
- [GCF] Xian Wu, Xinjin Li, Yiliu Xu, Yining Liu, Yong Jiang. *GeoCueFormer: Geometry-Guided Wavelet Representation and Prediction-Cued Dual-Stage Decoder for Underwater Semantic Segmentation*. arXiv 2609.18069.
- [DA1] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, Hengshuang Zhao. *Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data*. CVPR 2024. arXiv 2401.10891.
- [DA2] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, Hengshuang Zhao. *Depth Anything V2*. NeurIPS 2024. arXiv 2406.09414.

复用请求 1 的阅读：
- [LPOSS] Stojnić et al. *LPOSS*. CVPR 2025. arXiv 2503.19777（请求 1 精读）。
- ProxyCLIP（Lan et al., ECCV 2024, arXiv 2408.04883；请求 1 做过代码级核对）。
- TPA（arXiv 2608.08290）。
- GeoSeg-OV（arXiv 2608.10426，部分阅读）。
- MARIS（arXiv 2510.15398，存在已核对，细节【未核实】）。
- LoCoFuse（【未核实】）。

摘要级【未核实】（只看过检索摘要，没读原文）：
- PopNet 2212.05370；Depth Awakens 2405.05614；DGA-Net 2601.02831；BED-SAM2 2605.24893；DepthPolyp 2605.16519；PDDM 2503.18393；GeomPrompt 2604.11585
- DINOcular 2608.27226；Lexicon3D 2409.03757；Geometry Matters 2605.30093；SeeSE3 2607.14228
- FeatUp 2403.10516；JAFAR 2506.11136；LoftUp 2504.14032；AnyUp 2510.12764；RaysUp 2606.22749
- AM-RADIO 2312.06709；RADIOv2.5 2412.07679
- RSGPNet 2606.28410；SENSE 2604.15946；GAMUS 2305.14914；AMMNet 2507.16158；RMT 2309.11523；Learning Spatial Decay 2508.09525
- Saliency-Depth conditioning 2608.25435；SPDA-SAM 2602.06335；Exploiting depth from single monocular images 1610.01706
- 下游任务导向 UIE 2603.01767；WaterFlow 2510.12605；UniV2D 2605.07146；DY-LUT 2607.22801；Weather-Conditioned DA 2609.04827；Wat3R 2607.08772

## 附录 B 本次新增代码

| 文件 | 作用 | 验证情况 |
|---|---|---|
| `scripts/dump_depth_cache.py` | 用 E2O 自带的 `depth_anything_v2` 和 `infer_image(518)` 缓存整图相对逆深度（float16 .npy） | 需要 torch + GPU，本环境无法运行；代码只调用官方 API |
| `scripts/prevalidate_depth_figure_ground.py` | G1/G2/G3 的离线预检：F1 远近顺序/AUROC、F2 Otsu 前景 IoU、F3 深度环对比、A1 像素对 AUROC、B1 边界 F；含中心先验、打乱深度、亮度三种对照；打印预设判据 | numpy-only；`scripts/tests/test_prevalidate_depth.py` 合成数据测试通过（有信息深度应通过、噪声深度应放弃） |
| `scripts/e2o_patches/geo_affinity.py` | G2 的 3D 感知亲和：numpy 参考实现 + torch 移植 + PCA 平滑图对照 | numpy 参考由 `scripts/tests/test_geo_affinity.py` 覆盖：零权重复现 E2O、对角不变、soft 等价于乘性衰减、对深度仿射变换不变、跨深度层被降权，均通过；torch 版未实跑 |
