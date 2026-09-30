# 水下开放词汇语义分割（UOVS）：深度调研与线 B / 线 C 实验方案

> 对象：Earth2Ocean（E2O，CVPR 2026，arXiv 2511.07923）与 UOVSBench
> 依据：用户交接包（2026-09-30 版，下称"交接包"）+ 本次独立核查
> 日期：2026-09-30

## 标注约定

| 标签 | 含义 |
|---|---|
| **【包】** | 交接包给出的事实（本次未重跑） |
| **【核】** | 本次独立核查得到的事实：读了官方代码包、论文原文、公开代码，或跑了脚本（给出文件/行号或命令） |
| **【推】** | 推断。都写明依据；没有依据的直接写"无依据" |
| **【未核实】** | 没能访问原文、代码或数据 |

**访问范围说明【核】**
- 能访问：官方代码包（经 `raw.githubusercontent.com` 下载，SHA256 `f301c638…e69b`，与交接包 §1.5 一致）、ProxyCLIP 与 MMSeg v1.1.1 源码、alphaXiv 论文全文、HuggingFace 元数据（只到文件列表）。
- 不能访问：
  - 本环境的网络策略拦截了 `github.com`、`arxiv.org`、`huggingface.co`（HTTP 403），所以数据集 zip 没法下载，这一节里凡是需要读图像或掩码的核查，都留给执行同学在服务器上做（脚本已备好）。
  - 本地 `D:\code\…` 和服务器 `/home/jli/…` 也访问不到，交接包里描述的 AquaFree/GeoTest 代码路径与参数**没有核对**。

---

## 0. 一页结论

1. **GMG 与 ProxyCLIP 在代码上一致【核】。** E2O 的 `open_clip/transformer.py::custom_attn`（L618–662）与 ProxyCLIP 的 `custom_attn`（mc-lan/ProxyCLIP，L560–587）逐行相同：`(S − β·mean(S))·γ`、负值置 −inf、softmax，默认 β=1.2、γ=3.0 也一样。两者只在代理特征上不同：
   - E2O 用 DepthAnythingV2-B 编码器的**最后一层**（`geofeats_idx='3'` 对应中间层列表 `[2,5,8,11]` 的第 3 个下标，也就是 block 11，交接包写的"第 3 层"不准确）；
   - ProxyCLIP 默认用 DINO ViT-B/8，也就是 E2O 代码里的 `nongeometric` 分支。

   所以"+GMG 带来 +12.67"的大部分可以归到 ProxyCLIP 式代理注意力本身（交接包里的 appearance 对照已经恢复了 86%）。真正属于"几何/深度"的增量，需要 ProxyCLIP-DINO 和 DINOv2 这两个缺失的对照来界定（B5）。
2. **AquaOV255 基本包含了 USIS16K【核】。**
   - 两份官方图像列表有 16,080 个文件名重合（USIS16K 共 16,151 张，重合占 99.6%）。
   - USIS16K 的 158 类就是 AquaOV255 的第 0–157 类，顺序相同，其中 157 个名称一致。
   - E2O 论文 §3.1 自己也写了 AquaOV255 由网络爬取图像"与 USIS16K、USIS10K 等现有数据集合并"而成。
   - 结果是 UOVSBench 的 6 数据集平均对同一批图像计了两次。图像内容是否逐字节相同，还需要服务器上的 MD5 核查（C1）。
3. **类名格式是 AquaOV255 独有的问题【核】。**
   - AquaOV255 类名没有词边界，254 个里有 253 个带尾随空格。
   - USIS16K 对同一批类用的是带空格的正常名称。
   - CLIP 分词器会先转小写（`tokenizer.py` L85/L122），所以 4o 在 Objects 里输出的驼峰名（占 AquaOV255 提及数的 70.1%）在 CSA 句子里同样丢了词边界。
4. **MLLM Objects 的准确率【核】。** 以文件名编码的类别作代理标签（覆盖 20,434 张），第一个 Object 等于该类的比例：4o 73.12%，Qwen7B 41.04%，Qwen3B 22.43%。4o 的数字与交接包的 72.99%、E2O 论文 Fig.5 的 72.91% 一致。另外，文件名本身编码了类别，是潜在的泄漏通道（见 §1 N6）。
5. **在官方协议下，性能主要由"识别"决定，而不是"定位"。**
   - 【包】AquaOV255、MAS3K、USIS16K 有 98.5–99.9% 的图像只含一个有效类；oracle-name 整图填充能得 99.5–99.97。
   - 【包】GT 完美分组只带来 +1.3 到 +2.5；文本头区域准确率 30–40%，视觉质心 85%。
   - 【推】因此线 B 里收益最大的杠杆都在识别侧：类名规范化、MLLM 候选先验、跨图视觉原型、生物领域 VLM。但这些杠杆在官方协议下的收益，有相当一部分是利用了协议的退化性。必须配合线 C 的多协议报告，否则论文会被认为是在钻协议空子。
6. **建议优先级【推】。**
   - 第 1–2 周先做这些：线 C 的数据完整性审计（C1）和多协议评测器（C2，脚本已写）；线 B 的复现对齐（B0）、类名修复（B1）、候选先验（B2），以及两个前置验证：B3 用现有 probe dump 离线做（只需 CPU），B4 做小规模零样本验证（GPU 10–30 分钟）。
   - 第 5 周按预先登记的判据决定继续还是放弃。
   - 线 C 的证据链最完整、风险最低，建议作为论文或毕业论文的主干。线 B 以"识别侧方法，并在修正协议下同样有效"为目标。

---

## 1. 本次核查的新事实（交接包没有记录，或需要更正）

命令都可以用 `python scripts/audit_official_package.py --code_dir <Earth2Ocean-code>` 复现（输出见附录 A）。

| # | 事实【核】 | 出处 / 方法 | 影响 |
|---|---|---|---|
| N1 | GMG 注意力与 ProxyCLIP `custom_attn` 逐行一致（β=1.2、γ=3.0 默认值也一致）；`vfm_model='nongeometric'` 加载 `dino_vitbase8_pretrain.pth`，就是 ProxyCLIP 的默认 VFM；推理用的 `compute_padsize(H,W,56)`、slide 336/112、`logit_scale=40`、`prob_thd=0` 都沿用 ProxyCLIP | E2O `open_clip/transformer.py` L618–662，`Earth2Ocean.py` L88–92/L352；ProxyCLIP `open_clip/transformer.py` L560–587，`proxyclip_segmentor.py` | "+GMG +12.67" 需要重新归因；缺 ProxyCLIP-DINO/DINOv2 对照（B5） |
| N2 | `geofeats_idx='3'` → `intermediate_layer_idx['vitb']=[2,5,8,11]` → 取 **block 11（最后一层）**，并经过 final norm（`get_intermediate_layers(norm=True)`）；DPT 深度头照常计算但输出没用上 | `depth_anything_v2/dpt.py` L163–183，`dinov2.py` L297–312，`Earth2Ocean.py` L259–268 | 交接包 §1.2 写的"第 3 层"应改成"第 4 个中间输出（下标 3）= 最后一层"。论文 §6.1 "stage 3 (last layer) best" 与此一致；深度头是纯开销 |
| N3 | 官方代码只实现了 `geometric` / `nongeometric` 两个分支；取其他值时 `vfm_feats=None`，但 `vfm_h/vfm_w` 没有定义，L310 的 reshape 会报 NameError | `Earth2Ocean.py` L281–310 | 交接包里的 `none`/`clip`/`appearance` 是本地改动，需要在论文中写明网格与实现（21×21 vs 24×24） |
| N4 | AquaOV255 ∩ USIS16K 文件名有 16,080 个；只在 AquaOV255 的 4,643 张，只在 USIS16K 的 71 张。USIS16K 类表 = AquaOV255 第 0–157 类，顺序相同（157/158 规范化后一致，唯一例外是 `Mask`↔`Surgicalmask`）。USIS10K 文件名（`test_xxxxx`）不重合，内容是否重合未知 | `configs/imglist_*.txt`、`cls_*.txt`；论文 §3.1 原文"combining images crawled … with existing underwater datasets such as USIS16K and USIS10K" | 6 数据集平均存在重复计数；需要 MD5/pHash 核查（C1） |
| N5 | `cls_AquaOV255.txt`：253/254 个类名带尾随空格、没有词中空格、只有首字母大写。`cls_usis16k.txt` 对同一批类有 112 个带空格名。MAS3K 有 21 个驼峰名。CLIP 分词器会先 `.lower()` | `configs/cls_*.txt`；`open_clip/tokenizer.py` L85/L122 | 词边界缺失只在 AquaOV255 出现；4o Objects 的驼峰名（AquaOV255 70.1%、USIS16K 73.8%）在 CSA 里同样丢了词边界 |
| N6 | 用文件名前缀作代理标签（20,434/20,723 张可映射）时，第一个映射 Object 与之相同的比例：4o 73.12%、Qwen7B 41.04%、Qwen3B 22.43%；非空候选率分别为 97.71%、82.11%、69.46%。文件名直接编码类名（如 `AlligatorGar_004.jpg`）。4o 输出的写法有 9,712 次与文件名前缀完全一致，与 `cls_AquaOV255.txt` 写法一致的只有 133 次 | `llm_reasoning_AquaOV255_*.json` | 这些数与交接包 §5、论文 Fig.5 一致。写法偏向驼峰，很可能只是因为 prompt 里嵌的是 `category.txt`（驼峰），**不能**据此断定文件名被送进了 MLLM；但凡是重新生成 MLLM 输出，都应该先把文件名换成哈希 |
| N7 | CSA 融合权重 `w=min(sim, w_max)`。在默认 τ=w_max=0.5 下，每次触发融合的权重都**恰好是 0.5**。论文表 4 的超参消融直接在测试基准上做（没有验证集）：正文说 τ=0.1 最佳但默认值是 0.5；w_max=0.8 得 34.62，高于默认的 34.32 | `Earth2Ocean.py` L247–251；论文表 4 | "权重 ≤0.5"实际就是常数 0.5；原文超参是在测试集上选的 |
| N8 | 论文表 10 与本地复现（平均 mIoU）对比：qwen7b ViT-B 33.35 vs 33.34 ✓；**ViT-L 41.13 vs 39.62（−1.51）✗**；ViT-H 46.66 vs 46.66 ✓；4o ViT-H 49.67 vs 50.78（+1.11）✗。LoCoFuse 表 I 里 E2O 的 ViT-H 49.67 与论文相同，属于直接抄录，不是复现 | 论文表 1、表 10；交接包 §6.3 | 线 B 的所有对比必须以本地同源复现为基线；ViT-L 的差距先要查清（B0） |
| N9 | 论文自身有不一致：表 8 正文写 Fish 154 / Invertebrates 48，但所列 ID 数出来是 148 / 54；表 9 写 General 68 / Special 139，所列 ID 是 70 / 137（与交接包 §2.5 的 subclass.json 一致）。论文说 AquaOV255 有"255 categories (including background)"，但评测时 background 被 ignore | 论文 p.2、表 8/9 | 线 C 的审计材料 |
| N10 | MLLM JSON 覆盖情况：AquaOV255 4o 缺 26 张，USIS16K 4o 缺 25 张，USIS10K 4o 缺 7 张，SUIM 4o 缺 2 张（缺失的图直接不做 CSA）。同一张图在 AquaOV255 与 USIS16K 的 4o JSON 里 Caption 相同的只有 5,023/16,055 张，说明两边是分别生成的 | 审计脚本 | 用 MLLM 的方法必须定义缺失时的回退方式；同一张图在两个数据集里拿到的 MLLM 信息不同 |
| N11 | 分组 mIoU 脚本 `get_miou_maccc_class.py` 用 `category_split.json` 里 `id_to_name` 的小写名去匹配 MMSeg 日志中的类名，匹配不上就**静默丢弃**（返回 −1） | `get_miou_maccc_class.py` L39–50 | 【推】如果 `category_split.json` 与交接包 §2.5 描述的 subclass.json 一样是 1-based 且有 2 处名称差异，论文表 2 的分组数可能漏了类或错位；需要核对服务器上的 `datasets/AquaOV255/category_split.json` |
| N12 | 各数据集 4o Objects 为多目标（≥2 个不同对象）的比例：SUIM 92.3%、USIS10K 78.4%、DUT-USEG 22.2%（另有 8.8% 为空）、MAS3K 4.0%、AquaOV255 5.3%、USIS16K 9.5% | 审计脚本 | 候选先验在 SUIM/USIS10K 上的约束力弱；在 DUT 上要处理空候选 |
| N13 | E2O 的 CLIP 权重从本地 `pretrained_ckpt/vit{b,l,h}.bin` 加载，来源（OpenAI 还是 LAION）没有文档说明（`model_pretrained.txt` 为空文件）；ProxyCLIP 的 B、L 用 `openai`，H 用 `laion2b_s32b_b79k` | `eval.py` L62–67；ProxyCLIP `configs/base_config.py` | 这可能是 N8 中 ViT-L 差距的原因；需要核对服务器权重的 MD5 |
| N14 | MMSeg v1.1.1 的实现：`LoadAnnotations` 做 `.squeeze().astype(np.uint8)`（loading.py L103）；`IoUMetric` 先 `mask=(label!=ignore_index)` 再统计（iou_metric.py L185） | open-mmlab/mmsegmentation@v1.1.1 | 交接包 §1.5 属实 |
| N15 | UW 模板共 100 条，其中大量是生物场景句式（如 "a {c} chasing small fish"、"a {c} swimming in the ocean"），E2O 对 100 条取平均后同样用于 Diver、AUV、PlasticBag 等人造物类 | `prompts/imagenet_template.py` L262 起 | 【推】对人造物类可能不适配；可以用 FLOSS 式的类专家模板来处理（线 B 低优先级） |

---

## 2. 审阅交接包：空白、矛盾与未核实项

**需要更正**
- §1.2 GMG 写"取第 3 层特征"，应改为"`geofeats_idx=3`，即中间输出 `[2,5,8,11]` 的第 4 个，也就是最后一层 block 11"（N2）。
- §2.2 `none`/`clip`/`appearance` 属于本地实现（N3），论文里需要说明网格与插值细节。
- 本地复现与论文的差距（N8）在交接包中没有提到。§6.3 只给了平均值，缺各数据集×各 backbone×各 MLLM 与论文表 1/表 10 的逐项对照。

**空白**
1. **缺 ProxyCLIP 对照。** 交接包 §3 有 none/appearance/geometric 三组，但 `nongeometric`（DINO-B/8，就是 ProxyCLIP 默认）和 DINOv2-B/14-reg 都没跑。后者与 DepthAnything-B 编码器结构相同、网格相同（24×24），能把"深度微调"这个因素单独隔离出来（B5）。
2. **重复计数没有记录。** AquaOV255 与 USIS16K 的重叠（N4）交接包没有提到。
3. **类名修复没有进完整管线。** 词边界修复只在测试台上测过（§4.2），也没有覆盖 MLLM Objects 字符串本身（N5）。
4. **只算过 oracle-name 预测器的"背景计分"**（§6.4）；还缺一个把同一次推理同时按多协议计分的评测器（C2 已给出 `scripts/protocol_eval.py`）。
5. **没有验证集。** 没有任何可以合法调超参的数据，替代方案见 §7。
6. **运行时间与资源没有记录。** 按执行同学的资源估计成本时缺少基准数据（§8 给了校准方法）。
7. **没有和已被接受的基准做对照。** 退化性需要一个标定，比如 PASCAL VOC20 是否也有类似的单类占比（C3-ref，脚本已写）。
8. **没有统计显著性。** 6 数据集平均的差值（例如 +1.88）没有置信区间。`--stats-output` 已经存了逐图面积统计，可以直接做 image-level bootstrap（`protocol_eval.py --bootstrap`）。

**有待确认的矛盾或风险**
- 论文把 AquaOV255 说成"255 类（含背景）"，评测时又 ignore 背景（N9）。
- 论文表 8/9 的计数与它自己列出的 ID 不符（N9）。
- τ 的"最佳值"与默认值不一致（N7）。
- 论文和交接包都没说清 E2O 消融表 3 里的"Baseline 15.31"是什么配置。

**未核实（本次也没能核实）**
- LoCoFuse（IJCNN/WCCI 2026 提交稿）在 alphaXiv 检索不到，只能以交接包的阅读为准。
- Nautilus（NeurIPS 2025）没有核对。
- MarineEVT（2607.24064）、ShellfishNet（2605.07338）没有核对。
- 2604.15652 检索到的页面里没有说明背景怎么处理。
- `category_split.json` 与 MLLM prompt 的原文都不在代码包里。

---

## 3. 文献核对与调研

### 3.1 交接包 §8 条目逐一核对

| 条目 | 出处 | 本次核对结论 | 与本任务的关系 |
|---|---|---|---|
| MaskCLIP（Zhou、Loy、Dai，ECCV 2022） | arXiv 2112.01071 | 题目与出处按公开常识；**注意**检索时撞到同名的另一篇 MaskCLIP（2208.12262，掩码自蒸馏预训练），引用时要区分 | value 路径得到稠密特征 |
| SCLIP | 2312.01597 | 已核对存在（ECCV 2024） | correlative self-attention |
| ClearCLIP | 2407.12442 | 已核对存在；CorrCLIP 与 LPOSS 的描述证实其做法是去掉最后一层的残差与 FFN | E2O 的 custom_attn 同样只返回注意力输出（不含残差和 FFN） |
| ProxyCLIP | 2408.04883 | **代码级核对**：与 GMG 等价（N1）；VOC21/COCO-Object/Context60 的配置用多名称背景行加 `prob_thd`（0.2/0.25/0.15） | GMG 的直接前身；背景约定的先例 |
| CorrCLIP | 2411.10086 | 全文已读：Fig.2 把注意力范围限制在同类（intra-class）patch 内，用 DINO 相似度；表 5 显示在 SAM 范围内用均匀相似度（Unif）也很强。ViT-B 8 基准平均 51.0 | GT-Uniform ≈ "Unif + GT 范围"，与 CorrCLIP 的 SAM 范围不同（交接包说法正确） |
| Trident | 2411.09219（Shi、Dong、Xu） | 全文已读：先拼接再分割（splice-then-segment），用 SAM 编码器做全局相关聚合，最后加 SAM 精修。ViT-B 平均 45.8。会议出处按 E2O 表 1 写的 ICCV 2025 | 对比方法 |
| SegEarth-OV | 2410.01768 | 已核对存在；SimFeatUp 与去全局偏置的细节以交接包阅读为准（本次未逐页核对） | 同类的垂域 training-free 工作 |
| E2O | 2511.07923 | 全文已读（表 1/3/4/8/9/10、§3.1、§5.2–5.3）；CVPR 2026 pp.39879–39889（据 GeoSeg-OV 参考文献） | 本任务对象 |
| MARIS | 2510.15398 | 已核对存在；CVPR 2026 pp.24939–24949（据 GeoSeg-OV 参考文献）；细节以交接包为准 | 同组的实例分割工作 |
| LoCoFuse | — | **未核实**（alphaXiv 检索无果） | 唯一在 UOVSBench 上报告结果的后续工作（据交接包） |
| Nautilus | — | **未核实** | — |
| UW-VOS | 2603.24006 | 已核对存在 | — |
| 2604.15652（同组遥感基准） | 2604.15652 | 已读：OVRSISBenchV2 是**训练式**基准；类别由"描述→解析→掩码"半自动生成；检索到的页面没有说明背景处理 | 与 AquaOV255 的"LVLM 辅助类别判别 + SAM 掩码"标注流程类似（见 §3.3 第 4 点） |
| CuPL、Menon & Vondrick | 2209.03320 / 2210.07183 | 检索没有命中，出处按公开常识（ICCV 2023 / ICLR 2023） | 描述替代类名；交接包 §4.6 已显示人工描述没有收益 |
| CoOp / CoCoOp | 2109.01134 / 2203.05557 | 已核对存在 | 需要训练，不符合 training-free 设定 |
| Mind the Gap（Liang 等，NeurIPS 2022） | 2203.02053 | 检索没有命中（命中的是 2025 年同名的另一篇），出处按公开常识 | 模态间隙；与 §4.3 的 margin 分析相关 |
| Tip-Adapter | 2207.09519 | 已核对存在（ECCV 2022） | 少样本缓存，需要标签 |
| TDA | 2403.18293 | 已核对存在；题目为 "Efficient Test-Time Adaptation of Vision-Language Models"，TDA 是其方法名（CVPR 2024） | 测试时缓存（分类） |
| 2608.29923（DAF，持续 TTA for OVSS） | 2608.29923 | 全文已读：包含 **SUIM**。NACLIP B/32 源模型在 SUIM 上 22.4，MLMP 在持续设定下降到 12.9 | **熵最小化类 TTA 在水下数据上有塌缩风险** |
| OVCamo / 2509.24681 | — | 以交接包的代码核对为准（本次未复核） | 类门控指标的先例 |
| MESS | — | 以交接包为准 | 背景计分的先例 |
| 引用 E2O 的工作 | QMSR 2609.18752、GeoSeg-OV 2608.10426、2608.15363 | QMSR 做的是检索（NDCG），用了 USIS10K，没有 UOVSBench 数值；GeoSeg-OV 做遥感，默认用 **DepthAnythingV2-B 作结构先验**，并声明"所有标注类别**包括背景与杂类**都参与计分"；2608.15363 是水下多标注者噪声研究（未读全文） | 交接包"除 E2O 外只有 LoCoFuse 在 UOVSBench 上报告结果"的判断，在本次检索范围内仍然成立 |

### 3.2 新增的相关文献（按与本任务的关系分组）

**A. 识别侧与词表杠杆（线 B 的依据）**
- **FreeCP**（ICCV 2025，arXiv 2508.00557）Fig.1(b)：只把 SCLIP 的词表限制为 GT 类，就能大幅提升：ADE 16.1→32.8，Stuff 22.4→37.4，PC 30.4→45.3。这说明"图像级类别知识"在通用 OVSS 里本来就是主要杠杆，是 B2 的直接先例。
- **What Holds Back OVS?**（2508.04211）：即使给 CLIP 完美掩码，区域分类仍然落后于同域模型（ConvNeXt-L oracle mask 41.8 PQ）。识别是瓶颈。
- **TPA**（2608.08290）：training-free，但属于直推式（transductive）。它从约 10% 无标签部署图像里挑宿主模型高置信的 patch，建 DINOv2 原型库，用 `log p + (1−α)λ·q̃` 做后融合（τ=2/C，α=0.5，λ=5，K_min=5）。ProxyCLIP B/16 +2.0，Trident +1.9；只用单图原型只有 +0.7。这是 B3 的直接先例。
- **PTC**（2608.03991）：单图内部用原型校准文本嵌入。ProxyCLIP +1.9，CorrCLIP +0.9。部分超参按数据集设了默认值。
- **FLOSS**（ICCV 2025，2504.10487）：用无标签图像的熵为每个类挑"专家模板"。NACLIP +0.8，CLIP-DINOiser +1.2。对应 N15 的模板问题。
- **LPOSS**（CVPR 2025，2503.19777）：标签传播。LPOSS+ 比 ProxyCLIP 高约 +1；把深度作为像素特征只 +0.1（42.2 vs 42.1）。另外，GT 下采样到 patch 分辨率这一 oracle 在 8 个通用集上平均 85.2，说明分辨率本身就有上限。

**B. 测试时适配**
- **MLMP**（NeurIPS 2025，2505.21844）：更新 LayerNorm，用多层、多模板的熵最小化。NACLIP L/14 VOC20 75.91→83.76。
- **DAF**（2608.29923）：持续设定下 MLMP 在 SUIM 上塌缩（见上表）。【推】在 UOVSBench 上用梯度式 TTA 风险高、收益难以预判，放低优先级。

**C. 基于 SAM 3 的 training-free 分割（必须考虑的强基线）**
- **SAM 3**（2511.16719）：概念提示分割，带 presence head。
- **SegEarth-OV3**（2512.08730）：把 SAM 3 用在遥感 OVSS 上，8 个数据集平均 53.4，比 CorrCLIP 的 40.7 高 +12.7；VOC20 上 96.8。注意它的背景阈值和置信阈值是**按数据集调到近似最优**的（原文 §4.1）。【推】检索没找到 SAM 3 在水下的评测。审稿人很可能会问"SAM 3 + MLLM 类名"这类简单管线。

**D. 领域基础模型（识别器候选）**
- **BioCLIP 2**（NeurIPS 2025，2505.23883）：ViT-L/14，在 TreeOfLife-200M 上训练（含 FathomNet 海洋动物，95.2 万个分类单元）。10 个物种任务零样本平均 55.6，CLIP-L 为 25.5；FishNet 39.8 vs 27.9。【推】对 AquaOV255 这类细粒度鱼种可能有很大帮助。但它训练时见过大量网络生物图像，**可能与 AquaOV255 的网络爬取图像重叠**；而且严格说它属于"外部领域模型"设定，不是"陆地 VLM 迁移"。
- **AquaticCLIP**（2502.01785）：ViT-B/16，用 200 万水下图文对微调；论文里的 GitHub 地址写的是 "xxx"，**权重是否公开未核实**。
- **UWBench**（2510.18262，同组）：158 类（来自 USIS16K），用 GPT-5 生成、专家校对。

**E. 评测与审计的先例（线 C 的依据）**
- **VOC20/VOC21、PC59/PC60 两套协议并存**：CorrCLIP、FreeCP、PTC 等都把"含背景"与"不含背景"分开报告。例如 CorrCLIP ViT-B：VOC20 88.8 vs VOC21 74.8。
- **GeoSeg-OV**（2608.10426）：明确让背景与杂类参与计分。
- **OVCamo**（ECCV 2024）：类门控，类别错了就记最差（交接包 §7 已核）。
- **OV3D-Bench**（2608.17110）：指出若干方法"依赖部署时拿不到的逐图类别 oracle"。与本任务里的 MLLM Objects 同构（只读了摘要）。
- **Metrics or Mirage?**（2607.08203）：结肠息肉分割基准的评测不一致审计，是医学领域的审计型论文先例（只读了摘要）。
- **LPOSS**（2503.19777 §4.3）：指出 ProxyCLIP 在 Object/VOC 上"使用数据集特定阈值过滤背景"。
- **SegEarth-OV3**：阈值按数据集调优。

### 3.3 从文献与事实归纳出的几点判断【推】

1. **E2O 的几何模块在方法上属于 ProxyCLIP 族**（N1），新意在于"用深度模型的编码器作代理"。GeoSeg-OV 也选了 DepthAnythingV2 作结构先验，LPOSS 测了深度特征但只 +0.1。因此在线 B 里继续挖几何侧，收益预期低（交接包 GT-Uniform oracle 只有 +1.3~2.5）。
2. **识别侧有大量空间，已有多条 training-free 途径可借鉴**：词表约束（FreeCP）、跨图原型（TPA）、模板选择（FLOSS）、领域识别器（BioCLIP 2）。
3. **UOVSBench 的退化性不是独有的**。VOC20 同样 ignore 背景，而且多为单目标图。线 C 的论证应该是**定量**的："UOVSBench 的退化程度远高于已被接受的 VOC20"（前提是 C3-ref 的数据支持），而不是"ignore 背景就是错的"。
4. **AquaOV255 的类别由"LVLM 辅助判别 + SAM 掩码 + 5 名专家校对"产生**（论文 §3.2），推理时又把 GPT-4o 的类别判断注入模型。如果标注用的 LVLM 也是 GPT-4o 系列，就存在"标注偏差与推理偏差同源"的风险。这一点需要问作者（§10）。

---

## 4. 线 B：在官方协议下提出方法、提升性能

### 4.0 总体设计与依赖

```
B0 复现对齐 ─┬─> B1 类名/Objects 规范化 ──┐
             ├─> B2 MLLM 候选先验解码 ────┼─> B-main：识别优先的 training-free 框架
             ├─> B3 跨图视觉原型（先做离线前置验证）┤    （B1+B2+B3/B4，在 P0 与 P2 下都报告）
             ├─> B4 生物领域识别器（先做离线前置验证）┘
             ├─> B5 代理注意力来源对照（同时服务线 C 的分析）
             └─> B6 SAM 3 强基线（探索性）
```

**统一评测设置**（所有 B 实验都一样，除非另行说明）
- 数据：UOVSBench 6 个数据集。AquaOV255 按 README 删掉 Catfish_112，修正 3 张 Lanternfish。
- Backbone：CLIP ViT-B/16、ViT-L/14、ViT-H/14，权重 MD5 需登记（N13）。
- VFM：DepthAnythingV2-B，block 11。
- 模板：UW，100 条。
- CSA：τ=w_max=0.5。
- 官方协议 P0。
- MLLM 用**官方发布的 JSON**（4o/qwen7b/qwen3b）。**不重新生成**，以保证与 E2O 信息同源。
- 基线：本地复现的 E2O（同一 MLLM、同一类名文件）。论文数字只作为参考列。
- 同时输出 P1/P2（线 C 的评测器），供附录使用。

---

### B0 复现对齐（前提，优先级 P0）

- **动机**：本地 qwen7b ViT-L 39.62 vs 论文 41.13（−1.51）；4o ViT-H 50.78 vs 49.67（+1.11）（N8）。不对齐的话，后续所有增量都缺少同源基线。
- **做法**
  1. 登记 `vitb/vitl/vith.bin` 与 `depth_anything_v2_vitb.pth` 的 MD5，与 open_clip 官方权重（OpenAI / laion2b_s32b_b79k）的哈希比对。
  2. 按数据集×backbone×MLLM，逐项和论文表 1、表 10 对照（4o 的 ViT-B/L 也要补跑）。
  3. 记录环境版本（torch、mmseg、mmcv、CUDA），以及 Catfish/Lanternfish 是否已修正。
- **评测**：P0；只报差值表。
- **预期**：无增益。目标是差距 ≤0.3，或者找到原因。
- **成本**：4o 的 ViT-B 与 ViT-L 两轮全量（6 数据集），约 2 个配置×（1–4）GPU·h【推，需校准，见 §8】。
- **风险**：ViT-L 差距可能来自权重版本；如果无法消除，就在论文里如实写"本地复现值"并统一使用它。
- **前置验证**：服务器上 `md5sum pretrained_ckpt/*`；翻出原始日志核对当时用的类名文件与 JSON。

### B1 类名与 Objects 字符串规范化（优先级 P0，成本最低）

- **动机**
  - 【包】§4.2 测试台上，静态原型 30.20→40.26（+10.06），Special 20.15→33.17，shrimp 5.45→63.82。
  - 【核】N5：USIS16K 对同一批类本来就用带空格的名称；4o Objects 有 70% 是驼峰写法，经分词器小写后同样没有词边界。
- **做法**
  1. 类名文件用 `cls_AquaOV255_split_controlled.txt`（只恢复词边界），同时去掉尾随空格。
  2. 在 `Earth2Ocean.build_prompts_from_json` 里对 Object 字符串做驼峰拆分（如 `RaccoonButterflyfish`→`Raccoon Butterflyfish`），再用同样的别名表映射。
  3. MAS3K 的 21 个驼峰名、USIS16K 的 `Porcupinefish` 也做同样处理。
  4. 不改任何超参。
- **评测**：3 个 backbone × {4o, qwen7b} × {原类名, 词边界, 官方 category.txt 名}，P0（以及 P1/P2）。交接包说的"5 个 CamelCase 名"与本次统计的 21 个驼峰名口径不同，需要统一。
- **预期【推】**
  - AquaOV255：ViT-B mIoU +3~8。依据是测试台区域准确率 +10，以及 AquaOV255 上 mIoU/aAcc≈0.55（17.82/32.62）。
  - 4o+CSA 下增益可能更小，因为 CSA 已经部分注入了类名。
  - 6 数据集平均 +0.5~1.5（只有 AquaOV255 和少数 MAS3K/USIS16K 类受影响）。
  - ViT-L/H 下的增益没有依据。
- **成本**：若有特征缓存（S2），每个配置几分钟；没有的话等于一次全量推理。
- **风险**：官方名与恢复词边界的名称之间有 2 处措辞差异（WeedySeaDragon、Mask）。**只能预先固定一种作主结果**，不能按测试集结果挑。
- **前置验证**：核对 `cls_AquaOV255_split_controlled.txt` 与 `cls_AquaOV255.txt` 逐行一一对应（类 ID 不能变）。
- **说明**：B1 更像"必要的基线修正"，本身不够作为贡献，但它同时是线 C 的一个发现（N5）。

### B2 MLLM 候选先验解码（优先级 P0）

- **动机**
  - 【包】§5.2/§5.3：候选内由 CLIP 选择时，4o 从 30.20→79.76（含回退），Qwen7B 从 30.20→46.66。现有 CSA 只把 Objects 融进原型（33.17），利用率很低。
  - 【核】N7：CSA 权重其实是常数 0.5。
  - 【文献】FreeCP 显示 GT 词表约束能带来 +10~16 mIoU。
- **做法**（在 `forward_feature` 里 logits 算出之后、`postprocess_result` 之前）
  1. 用规范化加别名表把 Objects 映射到类 ID（N6 的规则）。
  2. **硬约束**：非候选类的 logit 置为 −100，已实现为 `--restrict-to-llm-objects`。
  3. **软先验**：`logit_c += b·1[c∈O]`，b 取固定值：候选类加上相当于 `logit_scale` 下 0.05 余弦差的偏置，即 b=2.0。这个值是预先登记的，不按测试集调；如需调参，按 §7 的交叉拟合。
  4. 候选为空（DUT 8.8%）或 JSON 缺失（N10）时，回退到无先验。
  5. 多候选时不做额外处理。
- **评测**：3 个 backbone × {4o, qwen7b, qwen3b} × {硬, 软}，P0、P1、P2。报告 6 数据集平均，同时**单列三个单目标数据集（AquaOV255/MAS3K/USIS16K）和三个多类数据集**。
- **预期【推】**
  - AquaOV255 在 4o 硬约束下，ViT-B 可能从 17.81 升到 50~66。上限参照是 4o 名称整图填充的 66.05（§6.2）：单候选图占 93%，官方协议又忽略背景，所以只要候选对了，这张图就≈满分。
  - Qwen7B 候选准确率 41%，预计 30~40。
  - SUIM、USIS10K、DUT 大约 −3~+3（多目标，约束力弱；有漏召回风险）。
  - 这些都是推断。**P2（类门控）下的增益应该明显小于 P0**，这正是线 C 要展示的现象。
- **成本**：特征缓存可以复用；硬、软约束都只改分类头。
- **风险**：
  - 这相当于把 4o 的图像分类能力"直接灌进"分割结果，审稿人会认为是利用协议漏洞。**必须同时报告 P2**，并把 qwen3b/qwen7b 的结果一起给出，展示对 MLLM 质量的依赖。
  - 4o 在本环境里无法调用，只能用官方 JSON。
- **前置验证**：用现有 `--restrict-to-llm-objects` 在 ViT-B、AquaOV255 上先跑一轮，确认 −100 填充不会让滑窗插值出现异常（交接包 §2.2 已说明用 −100 而非 −inf）。

### B3 跨图视觉原型（交叉拟合，无标签）（优先级：先 P0 做离线验证，再按判据决定）

- **动机**
  - 【包】§4.1：文本头 33.17 vs 按图 held-out 的训练集视觉质心 82.60；簇内原型的余弦 T 为 0.65–0.77，C 为 0.84–0.92。
  - 【包】§4.3：不可能学到跨类通用的文本→视觉映射（mean-shift/Procrustes/ridge/kNN 都不如 identity）。
  - 【推】所以每个类都需要自己的**视觉证据**。在不用标签的前提下，只能从测试流里的伪标签得到，这正是 TPA 的做法。
- **做法**
  1. **伪标签种子**（预先登记三种）：
     - (a) MLLM 第一候选；
     - (b) MLLM 候选 ∩ CLIP 的 top-5；
     - (c) 只用 CLIP 的高置信预测（p>2/C，TPA 规则），不用 MLLM。
  2. **锚点**：伪标签类在该图中的预测像素里，p>2/C 的那些；每类至少 5 个锚点（K_min=5）。
  3. **原型空间**：
     - (i) GMG 输出的 CLIP 稠密特征（V_corr，512/768/1024 维）；
     - (ii) DepthAnything-B 的 block-11 特征（已在前向中算出，零额外成本）；
     - (iii) DINOv2-B/14-reg（TPA 的默认选择）。
  4. **融合**：`logit = log p + (1−α)λ·q̃`，q̃ 做均值中心化。λ 有两种取法：
     - 固定 TPA 默认值 α=0.5、λ=5；
     - 或者用**无标签的尺度匹配规则**（让 q̃ 在 top-5 类上的分布宽度与 log p 相当，实现见 `scripts/prevalidate_pseudolabel_prototypes.py::auto_lambda`），因为 E2O 的 `logit_scale=40` 会让 log p 的动态范围远大于 TPA 的设定。
  5. **交叉拟合**：按图像 MD5 奇偶把图像分成 A/B 两折。用 A 折建的库预测 B 折，反过来再做一次，最后合并统计。每个数据集单独建库。AquaOV255 与 USIS16K 用同一个哈希划分，保证重复图像落在同一折。
- **评测**：先 ViT-B × {4o, qwen7b, 无 MLLM} × 3 种原型空间，P0/P1/P2；再扩展到 ViT-L/H。
- **预期**：**没有可靠依据**。TPA 在通用基准上 +1.9~2.0，但本任务的文本/视觉差距大得多（30–40 vs 85），伪标签精度也更低（Qwen7B 41%）。所以必须先做离线验证。
- **前置验证（仅需 CPU，半天）**：
  - 在现有 probe dump（41,824 个区域）上运行 `python scripts/prevalidate_pseudolabel_prototypes.py --probe <dump> --mllm_json configs/llm_reasoning_AquaOV255_qwen7b.json --class_file configs/cls_AquaOV255.txt`。
  - **判据**（预先登记）：用 Qwen7B 伪标签建的原型，交叉拟合后的区域准确率（仅原型或融合）比文本头（33.17）高 ≥5，就进入完整管线；否则放弃 B3。
  - **注意**：probe 的区域是在 GT 范围内池化得到的（完美分组），所以验证结果偏乐观，只能用来做"否决"。
- **成本**：离线验证 CPU 约 10 分钟。完整管线下，建库需要对一半图像做一次前向（和一次推理相当）；推理时只多一次 C×D 的余弦查表。
- **风险**：
  - 确认偏差：错误伪标签会强化错误的类。
  - 长尾类锚点不足（K_min 门控会跳过这些类）。
  - 直推式设定需要导师认可，并在论文中写清楚（TPA 附录 D 有论证可以参考）。

### B4 生物领域识别器 BioCLIP 2（优先级：先 P1 做离线验证）

- **动机**
  - 【包】E2O 论文 Fig.5：CLIP ViT-L 在 AquaOV255 上的图像级分类只有 34.14%，4o 为 72.91%。
  - 【文献】BioCLIP 2 零样本物种分类平均 55.6，CLIP-L 为 25.5。
  - 【推】它可能是一个"不依赖闭源 MLLM"的识别器。
- **做法**
  1. 为 254 个类准备"学名 + 常用名"的文本（可以离线用 GBIF 名称解析，不需要 MLLM；人造物类保留原名）。
  2. 对整张图、以及 GMG 预测的前景区域裁剪图，做 BioCLIP 2 零样本分类，得到图像级先验 p_bio(c)。
  3. 按 B2 的软先验方式融合（b 预先固定），或者作为 B3 的伪标签种子 (d)。
- **评测**：图像级 top-1（以 GT 主类计）对比 CLIP-L 与 4o；再用完整管线测 P0/P2。
- **预期**：**没有依据**（BioCLIP 2 没有水下分割的评测）。
- **前置验证（GPU 约 10–30 分钟）**：在 AquaOV255 上跑整图零样本 top-1。**判据**：比同分辨率下 CLIP ViT-L 零样本至少高 10 个点，才继续。
- **风险**：
  - 数据重叠：TreeOfLife-200M 来自 GBIF/EOL 等网络图源，可能与 AquaOV255 的爬取图像重叠。需要做 pHash 抽查，并在论文里单列成"外部领域模型"设定。
  - 人造物类（32 类）不在它擅长的范围内。
  - ViT-L 的显存和速度开销。

### B5 代理注意力来源对照（优先级 P1；同时是线 C/分析章节的核心证据）

- **动机**：N1/N2；交接包 §3.1 appearance 已恢复 86%；E2O 表 1 的 ProxyCLIP ViT-B 平均 26.29，而表 3 的"w/ GMG" 是 27.98（Δ≈+1.7，但两者是否同源没有核实）。
- **做法**（全部走 E2O 同一套代码，只换 `ex_feats`）
  1. `nongeometric`：DINO-B/8，就是 ProxyCLIP 默认，42×42 网格。
  2. DINOv2-B/14-reg 最后一层：24×24 网格，**结构与网格都和 DepthAnything-B 相同**。
  3. DepthAnything-B 的 block {2,5,8,11}。
  4. DepthAnything-L 最后一层（MARIS 表 5 报告跨域时 vitb 优于 vitl，可作对照）。
  5. appearance 对照补一个 24×24 网格的版本（把 CLIP 特征插值到 24×24 后再算自相似），用来消除网格混杂。
- **评测**：ViT-B 上 6 个数据集，UW 模板，分别在无 CSA 与 4o CSA 两种设置下测；P0/P1/P2；逐图 bootstrap 95% CI。
- **预期【推】**：GMG 与 ProxyCLIP-DINO-B/8 的平均差在 ±3 以内；GMG 与 DINOv2-reg 的差大概率很小（这一项正好隔离了"深度微调"的作用）。依据：LPOSS 表 5 里 ProxyCLIP 用 DINOv2-B/14 平均 41.1、DINO-B/8 平均 42.3（通用集）。
- **成本**：每个配置一次 ViT-B 全量。约 6 个配置（1–4 GPU·h/配置【推】）。
- **风险**：DINO-B/8 的 42×42 网格使注意力矩阵大约 9.4 倍（1764² vs 576²），显存会增加。

### B6 SAM 3 强基线（优先级 P2，探索性）

- **动机**：SegEarth-OV3 在遥感上比 CorrCLIP 高 +12.7；审稿人可能会问。
- **做法**：
  - (a) SAM 3 对全部类逐个提示（254 次提示/图，开销很大），按 presence 分数门控；
  - (b) SAM 3 只对 MLLM 候选（1–3 个/图）提示。
  - 阈值用 SAM 3 的默认值，**不做逐数据集调优**；要调的话按 §7。
- **评测**：P0/P1/P2。
- **预期**：**没有依据**（水下没有先例）。【推】(b) 在 P0 下会接近整图填充的水平。它在 P2 下的前景 IoU 才是有信息量的。
- **成本**：(b) 很便宜；(a) 约 254 次解码/图，需要实测。
- **风险**：环境依赖（SAM 3 权重的获取与许可）；如果出现"简单管线大幅超过 E2O"的结果，会改变整篇论文的叙事。这对线 C 是正面证据，对线 B 是挑战。

### B-main 线 B 的方法主张（第 5 周决策后成形）

- **候选名称**："识别优先（Recognition-first）的 training-free UOVS"。
- **构成**：
  - 定位用 GMG/ProxyCLIP 族（不改动）；
  - 识别用 B1、B2 软先验，再加 B3 或 B4 中通过前置验证的那一个。
- **必须同时满足**：
  1. 在 P0 上超过同源 E2O；
  2. 在 P2（类门控）上也有显著提升（bootstrap CI 不跨 0）；
  3. 给出 qwen7b 与"无 MLLM"两个版本；
  4. 所有超参来自文献默认值、无标签规则或交叉拟合（§7）。

---

## 5. 线 C：修正评测协议，并在修正后的协议下研究方法

### C1 数据完整性审计（优先级 P0，只需 CPU，约 1 天）

- **动机**：N4（重复）、N6（文件名编码类别）、N9（论文的计数不一致）、N10（JSON 缺失）；交接包 §1.4 的 README 勘误（Catfish、Lanternfish）说明标注有噪声。
- **做法**（`scripts/dedup_and_mask_consistency.py`）
  1. **掩码取值普查**（`--census_only`）：确认 6 个数据集的背景码与 void 码（例如 AquaOV255 为 65535；其他数据集尚未确认）。**这是 C2 的前置条件。**
  2. **MD5 与 dHash**：统计 6 个数据集内部及跨数据集的精确重复与近重复。重点是 AquaOV255↔USIS16K；USIS10K↔AquaOV255（论文 §3.1 说包含了 USIS10K）；MAS3K↔其他。
  3. **重叠图像的掩码一致性**：对 16,080 张共享图像，比较两套 GT。先按 census 确认 USIS16K 的背景码和 ID 起点（0 起还是 1 起）。报告前景 IoU、类别一致率、像素一致率。这等于免费得到一个"标注一致性/噪声"估计（可与 2608.15363 的多标注者研究对照）。
  4. 把 USIS16K 的 71 张独有图、AquaOV255 的 4,643 张独有图列成清单。
- **产出**：去重清单、`audit_c1.json`，以及论文里的"数据完整性"表。
- **风险**：掩码格式不统一（调色板 PNG 或 RGB），需要按 census 结果调整读取代码。

### C2 协议族定义与离线评测器（优先级 P0，脚本已完成）

**评测器**：`scripts/protocol_eval.py`（附单元测试 `scripts/tests/test_protocol_eval.py`，已通过）。它把逐图面积统计存进 NPZ，然后离线计算以下协议：

| 协议 | 定义 | 适用 |
|---|---|---|
| **P0 官方** | 等价于 MMSeg IoUMetric：先丢掉背景/void 像素，再对 C 类取 nanmean（单元测试已与参考实现对齐到 1e-9） | 与 E2O 对齐 |
| **P1a** | 背景作为第 C+1 类，mIoU 对 C+1 类取平均（VOC21 风格） | 语义分割型数据集（SUIM、DUT-USEG） |
| **P1b** | 用 P1a 的同一个混淆矩阵，但只对 C 个前景类取平均（背景像素照样产生 FP） | 所有数据集，作主报告 |
| **P2 类门控** | 逐图计算：主类识别准确率、类别无关的前景 IoU、门控 IoU（主类识别对了才计 IoU，否则记 0），再对图像取平均（OVCamo 思路） | 显著/单目标型数据集（AquaOV255、MAS3K、USIS16K、USIS10K） |
| **P3 去重与加权** | 平均时按"去掉重复"计算（例如 AquaOV255 只用 4,643 张独有图，或者不再单列 USIS16K）；同时给出图像加权与数据集加权两种平均 | 基准级汇总 |

**关键设计点**
- 一次推理对应所有协议，不会因为改协议而重跑模型。
- 所有协议都支持逐图 bootstrap 95% CI（`--bootstrap 1000`）。
- 3 个以上方法时，自动计算 P0 与 P1/P2 排名之间的 Kendall τ。

**协议的效度验证（"修正验证"本身）**，用单元测试与合成数据证明协议能区分"识别"和"定位"：
- 整图填充在 P0 下得 100，在 P1b/P2 下很低（已测）；
- 完美掩码但类别一半错时，P2 识别率 50%、前景 IoU 100%、门控 IoU 50（已测）；
- GT 本身在所有协议下得 100。

**注意事项**
- P1 会把**没有标注但确实存在的目标**（显著性数据集里没标的鱼群）记成背景 FP，所以 P1 本身也有偏差。建议：
  - 显著性数据集以 **P2** 为主；
  - SUIM、DUT 以 **P1a** 为主；
  - 并在论文中说明这一点。
- **背景码必须在 C1 核实之后**再写进各数据集的配置。

### C3 方法 × 协议 重评测矩阵（优先级 P0/P1）

- **做法**：下面每个方法跑一次推理，把逐图统计存成 NPZ，离线计算 P0–P3。

  | 方法 | 说明 |
  |---|---|
  | none（CLIP 原版） | 下界 |
  | appearance | 交接包已实现 |
  | ProxyCLIP-DINO-B/8 | 即 `nongeometric` |
  | GMG（无 CSA） | |
  | E2O 完整版（4o/qwen7b） | ViT-B/L/H |
  | E2O + B1 | |
  | E2O + B2 | 硬、软两版 |
  | 整图填充：oracle-name / 4o-name / qwen7b-name | 交接包 §6.1/§6.2 |
  | GT 掩码 + 文本头分类 | 交接包的 GT-Uniform |
  | SAM 3 + MLLM（B6-b） | 若可用 |

- **输出**：
  1. 各协议下的主表；
  2. 排名稳定性（Kendall τ）；
  3. 识别/定位分解图（x 轴主类识别率，y 轴前景 IoU）。
- **C3-ref 标定**：用 `scripts/oracle_fill_reference.py` 在 PASCAL VOC2012 val 上算 VOC20/VOC21 的"单类占比、oracle 填充"，并在 UOVSBench 上复现交接包 §6.1（作为回归测试：AquaOV255 P0 应为 99.66）。
- **预期【推】**：
  - 整图填充在 P0 下能排进前列（4o-name 在 AquaOV255 上 66.05，高于 E2O 4o ViT-H 的 39.78【包】），但在 P1b/P2 前景 IoU 上会垫底；
  - E2O 各变体在 P2 下的差距会主要由识别率决定。
  - 具体数值没有依据。
- **成本**：只多一次 NPZ 保存。如果已有 B 系列的推理，这部分几乎零成本。VOC 标定只需 CPU。

### C4 背景建模基线（为 P1/P2 服务，优先级 P1）

- **动机**：E2O 这类不带背景类的方法在 P1 下会把所有背景像素判成某个类别。需要一个**事先固定、有文献先例**的背景处理方式，才能公平比较。
- **做法**（预先登记，不在测试集上调参）：
  1. **背景名称扩展**：在类名文件末尾加一行背景名（用 `; ` 分隔，这是 SCLIP/ProxyCLIP VOC21 的约定，E2O 的 `get_cls_idx` 已支持多名称类）。按数据集剔除与现有类语义冲突的名称：
     - SUIM 已有 "Sand/sea-floor"，USIS10K 已有 "sea-floor"，这两个数据集的背景只用水体类名称；
     - 其余数据集用"water; ocean; seabed; sand; rock; seaweed; background"。
  2. `prob_thd` 取 0（纯名称竞争）作主设定；另报一个变体：`prob_thd`=ProxyCLIP VOC21 的 0.2。敏感性曲线只作分析，不用于选参。
  3. P2 的另一种前景来源：SAM 3 通用概念（如 "marine animal"），或 GMG 的 fg/bg 二分类（背景名对前景类的并集）。
- **预期**：没有依据。

### C5 在修正协议下研究方法："先定位、后识别"（优先级 P1，在 C2/C3 之后）

- **动机**：在 P2 下，分数≈识别率 × 前景 IoU【推】。两部分可以分开优化，也可以分开归因。
- **做法**：
  1. 类别无关的前景掩码：SAM 3 通用概念，或 C4 的 GMG fg/bg。
  2. 对每个前景区域做识别：B2/B3/B4 的分数在区域内做均值池化后取 argmax。
  3. 对比 E2O+背景（C4）在 P1b/P2 上的表现。
- **预期【推】**：如果前景 IoU 约 0.8、4o 识别率约 0.73，那么 P2 门控 IoU 约 58。这是粗略的乘积估计，**没有实验依据**。
- **成本**：主要是 SAM 3 的前向。

### C6 开发集与发布（优先级 P1）

- 生成并发布：
  - 按图像哈希划分的 2 折清单；
  - 去重清单（C1）；
  - 背景码表；
  - 评测器；
  - 名称规范化后的类名文件。
- **开发集建议**（UOVSBench 没用到的数据）：
  - USIS10K train split：UOVSBench 只用了 val+test 的 3,190 张。据 USIS10K 原文，train 约 7,442 张，**需核实**；
  - SUIM TEST：110 张。
- **与作者联系**：问题清单已在服务器上（`/home/jli/AquaFree/analysis/literature_audit_20260717.md`），建议补充 §10 的第 4、5 项。

---

## 6. 线 B 与线 C 共享的实验基础设施

| 编号 | 组件 | 内容 | 状态 |
|---|---|---|---|
| S1 | 结果登记表 | 每次运行一行：实验 ID、数据集、协议、backbone（权重 MD5）、VFM/层、类名文件（MD5）、模板、MLLM（JSON 文件名与 MD5）、超参来源、划分、git commit、mIoU/aAcc/mAcc、95%CI、GPU 型号、耗时 | 待建（CSV 或表格） |
| S2 | 特征缓存（可选但强烈建议） | 每个 crop 缓存 GMG 输出的稠密特征（24×24×D，fp16）。logit = F·Tᵀ、双线性插值、滑窗平均都是线性运算，所以任何只改分类头的实验（B1/B2/B3、C4 的背景名）都可以从缓存精确复现，**必须先验证复现 E2O 官方数值到 0.01**。存储【推】：ViT-B 在 AquaOV255 上约 35–40 GB（20.7k 图 × 约 3 crop × 576 × 512 × 2 B），ViT-L 约 55 GB，ViT-H 约 75 GB | 待实现 |
| S3 | 逐图统计 NPZ | 在现有 `--stats-output` 基础上增加一行"背景 GT"、一列"背景预测"（`protocol_eval.RunAccumulator.add` 的输入格式） | 已写评测端 |
| S4 | 划分清单 | 按图像 MD5 奇偶分 2 折（AquaOV255 与 USIS16K 共用，保证重复图在同一折）；开发集（C6） | 待生成 |
| S5 | 去重、背景码表 | C1 的输出 | 待运行 |
| S6 | 预先登记文档 | 每个实验跑之前，把超参、判据、主结果设定提交到 git（带时间戳） | 本文档 §4/§5 即初版 |
| S7 | 审计脚本 | `audit_official_package.py`（已在官方包上跑通） | 完成 |

---

## 7. 不在测试集上调超参的替代方案

UOVSBench 没有验证集；E2O 的原始超参本身也是在测试基准上选的（N7）。建议按下面的顺序处理：

1. **沿用已发表的默认值**：E2O 自己的 β、γ、τ、w_max 保持原值，不重新调；TPA 用 α=0.5、λ=5、τ=2/C、K_min=5；ProxyCLIP 背景阈值用 0.2。
2. **无标签规则**：例如 B3 的尺度匹配 λ、TPA 的 τ=2/C。规则在看到结果之前就写进预先登记文档。
3. **交叉拟合**：按图像哈希分 2 折，在 A 折上选参（用的是无标签指标或伪标签指标）、在 B 折上评测，再交换。需要标签的调参**不允许**这样做，因为那等于在测试集上调参。
4. **留一数据集（LODO）**：在其余数据集上选参，再用到目标数据集。AquaOV255 与 USIS16K 视为一组。
5. **外部开发集**：USIS10K train、SUIM TEST（C6）。
6. **敏感性曲线**只能放附录作为分析，不能用来挑主结果。

---

## 8. 计算成本估计与校准方法

执行同学的 GPU 型号和数量不明（交接包只提到使用过 GPU 6–9）。以下都是**【推】**，请先按下面的方法校准：

- **校准方法**：从已有的 mmengine 日志（`work_logs/*/`）读出每个 iter 的 `time`（batch=1，也就是每张图），乘以图像数，得到 GPU·h/配置。用 ViT-B、ViT-H 各取一个 AquaOV255 日志。
- **粗估**（按 A100/4090 这一档）：
  - ViT-B + DA-B 一次 AquaOV255 全量约 1–5 GPU·h。瓶颈可能在 254 通道 logit 上采样到原图分辨率（`forward_slide` L374），而不是 backbone。
  - 6 个数据集合计约为 AquaOV255 的 2.2 倍（按图像数）。
  - ViT-H 约为 ViT-B 的 2–4 倍。

| 阶段 | 主要运行 | 估计 |
|---|---|---|
| B0 | 4o ViT-B/L 两轮 6 数据集 | 5–40 GPU·h |
| B1/B2/C4（有 S2 缓存时） | 若干分类头变体 | 建缓存等于一次全量；之后每个变体为分钟级 |
| B3 | 建库（半数图像一次前向）加推理 | 每个 backbone 约 1.5 次全量 |
| B4 | BioCLIP 2 整图零样本 | AquaOV255 约 10–30 分钟 |
| B5 | ViT-B 上约 6 个 VFM 配置 | 6–30 GPU·h |
| B6 | SAM 3（候选提示） | 需实测 |
| C1/C2/C3-ref | 纯 CPU | 约 1 天，含下载 VOC |

**本环境没有 OpenAI 凭据【包】。** 所有方案都只用官方 JSON。如果需要新的 MLLM 输出（例如科描述路由），用本地 Qwen2.5-VL-7B（与 E2O 同版本）生成，并先把文件名改成哈希（N6）。粗估 AquaOV255 需要 10–25 GPU·h（vLLM 批处理）【推】。

---

## 9. 同源对比规范与表格模板

规则：同一张表里只能出现同一 backbone 权重、同一 MLLM JSON、同一类名文件、同一数据集版本、同一协议的数字。论文里的数字单独放一列"参考"。交接包 §9.3 出现过的口径错误（把 qwen7b 当 4o、把 46.66 当 4o 最优），可以用登记表 S1 避免。

| 实验ID | 数据集 | 协议 | Backbone（MD5 前 8 位） | VFM/层 | 类名文件 | 模板 | MLLM JSON | 超参来源 | 划分 | mIoU | aAcc | mAcc | 95%CI | commit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0-L-4o | AquaOV255 | P0 | ViT-L-14（…） | DA-B/b11 | cls_AquaOV255.txt | UW100 | AquaOV255_4o | E2O 默认 | 全量 | | | | | |

---

## 10. 按周执行计划（建议从国庆后第一周开始）

分工：**用户**每天 5–6 小时，负责离线分析（CPU，NPZ/JSON）、写作与文献；**执行同学**负责 GPU 运行与服务器端的审计。

| 周 | 用户 | 执行同学 | 交付与判据 |
|---|---|---|---|
| 1 | 建 S1 登记表；读懂 `protocol_eval.py` 与单元测试；跑 `audit_official_package.py` 并核对附录 A；准备 VOC2012 val 掩码 | C1：`--census_only` 确认 6 个数据集的背景码；MD5/dHash 与掩码一致性；B0：权重 MD5 与环境版本 | C1 报告；背景码表；B0 差异来源假设 |
| 2 | 在 probe dump 上跑 B3 离线验证（`prevalidate_pseudolabel_prototypes.py`，qwen7b/4o）；跑 C3-ref（VOC20/21 与 UOVSBench 的 oracle 填充） | 给推理代码加 NPZ 保存（S3）与特征缓存（S2），并验证复现到 0.01；B0 补跑 4o 的 ViT-B/L | B3 继续/放弃判据；C3-ref 标定表；缓存验证 |
| 3 | 把 B1/B2 的离线统计填进登记表；起草线 C 的"数据完整性"与"协议"两节 | B1（3 个 backbone × 2 MLLM × 3 类名）；B2 硬/软（ViT-B 起步） | B1/B2 在 P0/P1/P2 下的表 |
| 4 | 整理 C3 矩阵的初版与 Kendall τ；B4 的名称表（学名 + 常用名） | B5（ViT-B 上 6 个 VFM 配置）；B4 整图零样本 | B5 对照表；B4 判据 |
| **5** | **决策会（与导师）**：按预先登记的判据定下 B-main 的组成，以及线 C 主协议（P1b 还是 P2） | C4 背景名基线（有缓存就是分钟级） | 决策纪要 |
| 6 | 写 B-main 的方法描述；做消融设计 | B3 完整管线（如果通过）或 B4 融合；ViT-B 全量加交叉拟合 | B-main ViT-B 结果 |
| 7 | 做 bootstrap CI 与显著性；写线 C 结果节 | 扩展到 ViT-L/H；B6 SAM 3（候选提示） | 全 backbone 主表 |
| 8 | 写 C5 设计；汇总失败实验（负结果也写进论文或附录） | C5 先定位后识别；补做消融 | C5 结果 |
| 9 | 初稿（线 C 为主干，线 B 为方法章） | 复现检查：换一块 GPU 重跑主表中的一行 | 初稿 |
| 10 | 修改；准备发布清单（划分、去重、评测器） | 按需补实验 | 投稿或毕业论文章节 |

---

## 11. 需要用户或执行同学补充的信息

1. 服务器的 GPU 型号、数量、可用磁盘（决定 S2 缓存能不能做、以及 B6 是否可行），以及已有日志中的每图耗时（§8 校准）。
2. `pretrained_ckpt/*.bin` 的 MD5 和来源（B0/N13）；本地 qwen7b ViT-L 那次运行的完整配置（N8）。
3. 各数据集掩码的背景码与 void 码（C1 普查），尤其是 USIS16K 的 ID 是 0 起还是 1 起。
4. `datasets/AquaOV255/category_split.json` 的内容（N11），以及官方 `category.txt` 与 `subclass.json` 的关系。
5. 要问作者的问题：
   - MLLM 的 prompt 原文，以及是否传入了文件名（N6）；
   - AquaOV255 标注时用的是哪个 LVLM（§3.3 第 4 点）；
   - 表 3 里的"Baseline"是什么配置；
   - 表 1 里 ProxyCLIP 用的是哪个 VFM 和哪套模板；
   - 背景为什么被 ignore；
   - AquaOV255 与 USIS16K 的重复是否是有意的。
6. 导师是否接受直推式（B3）和外部领域模型（B4）两种设定。
7. 毕业形式（期刊、会议还是学位论文）与截止时间，这决定第 9–10 周的安排。
8. probe dump 的确切格式：npz 还是 pt，字段名是否与交接包 §2.2 一致。`prevalidate_pseudolabel_prototypes.py` 需要据此调整。
9. MAS3K 在 UOVSBench 中具体用了哪 2,909 张（官方共 3,103 张，**需核实**），剩下的图是否可以作开发集。

---

## 附录 A：官方代码包审计输出（`scripts/audit_official_package.py`）

```
======== Class-name formatting ========
AquaOV255  classes= 254 trailing_space= 253 internal_space=   0 camelcase=   0
USIS16K    classes= 158 trailing_space=   0 internal_space= 112 camelcase=   0
MAS3K      classes=  37 trailing_space=   0 internal_space=   0 camelcase=  21
USIS10K    classes=   7 trailing_space=   0 internal_space=   2 camelcase=   0
SUIM       classes=   7 trailing_space=   0 internal_space=   4 camelcase=   0
DUT-USEG   classes=   4 trailing_space=   0 internal_space=   0 camelcase=   0
======== Class-list lineage AquaOV255 vs USIS16K ========
USIS16K classes whose normalized name equals AquaOV255 class at the same index: 157/158
mismatches: [('Mask', 'Surgicalmask')]
======== Image-list overlap (by file name) ========
AquaOV255 & USIS16K: 16080  (AquaOV255=20723, USIS16K=16151)
AquaOV255 images not in USIS16K: 4643
USIS16K images not in AquaOV255: 71
======== MLLM JSON coverage / Objects statistics ========
AquaOV255  4o      keys= 20697 missing_vs_imglist=  26 empty=  1.7% multi=  5.3% camelcase_mentions= 70.1% in_class_list= 99.1%
AquaOV255  qwen7b  keys= 20723 missing_vs_imglist=   0 empty=  0.2% multi=  8.9% camelcase_mentions= 53.5% in_class_list= 80.7%
AquaOV255  qwen3b  keys= 20723 missing_vs_imglist=   0 empty=  5.5% multi= 41.0% camelcase_mentions= 40.2% in_class_list= 63.4%
USIS16K    4o      keys= 16126 missing_vs_imglist=  25 empty=  0.3% multi=  9.5% camelcase_mentions= 73.8% in_class_list= 99.8%
MAS3K      4o      keys=  2910 missing_vs_imglist=   0 empty=  0.4% multi=  4.0% camelcase_mentions= 44.7% in_class_list=100.0%
USIS10K    4o      keys=  3183 missing_vs_imglist=   7 empty=  1.0% multi= 78.4% camelcase_mentions=  0.0% in_class_list=100.0%
SUIM       4o      keys=  1486 missing_vs_imglist=   2 empty=  0.0% multi= 92.3% camelcase_mentions=  0.0% in_class_list= 99.6%
DUT-USEG   4o      keys=  1487 missing_vs_imglist=   0 empty=  8.8% multi= 22.2% camelcase_mentions=  0.0% in_class_list=100.0%
（qwen7b/qwen3b 的其余行见脚本输出）
======== AquaOV255: MLLM first Object vs class encoded in file name ========
images with file-name class mapped: 20434/20723; unmapped prefixes: humanbiology 148, humanobject 72, weedyseadragon 69
4o      n=20410 first==fname 73.12% | any 74.03% | non-empty 97.71% | single-candidate 93.42%
qwen7b  n=20434 first==fname 41.04% | any 41.65% | non-empty 82.11% | single-candidate 76.74%
qwen3b  n=20434 first==fname 22.43% | any 24.72% | non-empty 69.46% | single-candidate 53.97%
======== Same image, two JSONs (AquaOV255 vs USIS16K, 4o) ========
shared images present in both JSONs: 16055; identical captions: 5023
```

## 附录 B：本仓库脚本一览

| 脚本 | 用途 | 依赖 | 测试情况 |
|---|---|---|---|
| `scripts/audit_official_package.py` | §1 的包级事实 | 仅 Python 标准库 | 已在官方代码包上运行 |
| `scripts/protocol_eval.py` | P0–P3、bootstrap、Kendall τ | numpy | 单元测试通过（P0 与 MMSeg 参考实现一致） |
| `scripts/tests/test_protocol_eval.py` | 协议效度的单元测试 | numpy | 通过 |
| `scripts/dedup_and_mask_consistency.py` | C1：普查、MD5/dHash、掩码一致性 | numpy, Pillow | 合成数据冒烟测试通过；真实数据待服务器运行 |
| `scripts/oracle_fill_reference.py` | C3-ref：任意数据集的 oracle 填充多协议分数 | numpy, Pillow | 合成数据冒烟测试通过 |
| `scripts/prevalidate_pseudolabel_prototypes.py` | B3 离线前置验证 | numpy（pt 格式需要 torch） | 合成数据冒烟测试通过；probe 格式待确认 |

## 附录 C：主要参考文献（均可核查）

- Earth2Ocean：arXiv 2511.07923；代码 `LiBingyu01/Earth2Ocean`（zip SHA256 f301c638…e69b）
- ProxyCLIP：arXiv 2408.04883；代码 `mc-lan/ProxyCLIP`
- CorrCLIP：arXiv 2411.10086
- Trident：arXiv 2411.09219
- SCLIP：arXiv 2312.01597
- ClearCLIP：arXiv 2407.12442
- NACLIP：arXiv 2404.08181
- SegEarth-OV：arXiv 2410.01768
- SegEarth-OV3：arXiv 2512.08730
- SAM 3：arXiv 2511.16719
- LPOSS：arXiv 2503.19777
- FLOSS：arXiv 2504.10487
- FreeCP：arXiv 2508.00557
- TPA：arXiv 2608.08290
- PTC：arXiv 2608.03991
- MLMP：arXiv 2505.21844
- DAF：arXiv 2608.29923
- What Holds Back OVS?：arXiv 2508.04211
- BioCLIP 2：arXiv 2505.23883
- AquaticCLIP：arXiv 2502.01785
- UWBench：arXiv 2510.18262
- MARIS：arXiv 2510.15398
- OVRSISBenchV2：arXiv 2604.15652
- GeoSeg-OV：arXiv 2608.10426
- QMSR：arXiv 2609.18752
- UW-VOS：arXiv 2603.24006
- OV3D-Bench：arXiv 2608.17110
- Metrics or Mirage?：arXiv 2607.08203
- Tip-Adapter：arXiv 2207.09519
- TDA：arXiv 2403.18293
- CoOp：arXiv 2109.01134
- CoCoOp：arXiv 2203.05557
- MMSeg v1.1.1：`mmseg/datasets/transforms/loading.py` L103，`mmseg/evaluation/metrics/iou_metric.py` L185
