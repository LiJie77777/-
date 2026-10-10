任务：筛选候选论文（第 {{TODAY}} 批）。不需要运行任何命令，所需内容都在下面。

## 项目背景
{{CONTEXT_TEXT}}

## 候选论文
JSON Lines，每行一篇：id、title、abstract、venue、venue_status、topics_hint（同内容文件：`{{INPUT}}`）。

```jsonl
{{INPUT_TEXT}}
```

## 输出要求
对**每一行**输出一条 item（不能漏、不能多、id 原样照抄）：
- relevance：0–3，按背景里"相关度怎么打"的标准。
- topics：从下列主题名里选（可多选，可为空）：
{{TOPICS}}
- evidence_type：method / positive / negative / analysis / benchmark / survey / other。
- deep_read：只有 relevance ≥2、且全文很可能给出可比数字、消融或对照时才为 true。每批最多 10 篇为 true。
- reason：一句中文，≤60 字，说明它和哪个问题/规则/卡片有关；负面结果要点明"负面"。

只根据标题和摘要判断，不要用你记忆里对这篇论文的了解。venue 一栏只供参考，不要因为是预印本就压低相关度。

## 来自整合方（Claude）的反馈，按它调整判断
{{FEEDBACK}}
