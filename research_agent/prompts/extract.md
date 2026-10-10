任务：精读一篇论文，填写证据卡片。不需要运行任何命令，全文在下面。

## 论文元数据（脚本核实，只读）
{{PAPER_META}}

## 项目背景
{{CONTEXT_TEXT}}

## 填写要求
- id 填 `{{PAPER_ID}}`。
- one_line / problem / method：中文，简洁；method 写清楚"用了什么信息、注入在哪里、是否需要训练"。
- geometry_use：按背景里的 I0–I6 和注入位置分类；没用几何就填 ["none"]。
- claims：3–8 条最重要的结论，**优先**挑带数字的主结果和消融。每条：
  - claim：中文转述；
  - quote：**逐字**照抄支持这条结论的原文（英文，一句或半句，至少 6 个词，从下面全文里复制）；
  - location：Sec./Table/Fig./页码；
  - numbers：从原文照抄的数字（metric、dataset、value、baseline、setting），没有就给空数组。不要自己算差值。
- negative_or_limits：论文里报告的失败、负面消融、局限（同样逐字引文）。没有就给空数组，不要编。
- controls：论文是否做了这些对照：same_arch_non_geometric（同构非几何特征）、spatial_only（仅空间先验）、shuffled_or_mismatched_depth（打乱/错配深度）、duplicate_feature_or_capacity（特征复制/容量对照）、depth_quality_variation（不同深度质量）。yes/no/unclear/na，notes 写依据。
- relevance：score 0–3；to_project 写对本项目的启发（这是推断，写明依据）；which_cards 选卡片编号。
- rule_links：这篇论文对 R1–R8 是 supports / contradicts / refines，why 写清楚依据哪条 claim。没有就给空数组。
- open_questions：读完仍不清楚、需要看代码或补充材料才能确认的点。

引文对不上原文会被脚本自动降级；数字必须能在原文里找到。表格在 PDF 转文本后可能错位，表格数字请引用表格标题或正文中提到该数字的句子。

## 来自整合方（Claude）的反馈
{{FEEDBACK}}

## 论文全文（pdftotext 输出，已去掉参考文献；同内容文件：`{{INPUT}}`）
<<<PAPER
{{INPUT_TEXT}}
PAPER>>>
