# research-feed 数据分支

服务器上的调研 agent（rafeed + Codex）每天写入这里；Claude 读取后整合进主文档。这个分支只放数据，代码在主分支的 `research_agent/`。

| 路径 | 内容 | 谁写 |
|---|---|---|
| `registry.jsonl` | 所有筛选过的论文：ID、标题、venue（含核实状态与证据链接）、相关度、主题 | 服务器 |
| `cards/*.md`、`cards/*.json` | 精读卡片；每条结论带原文引文，已与 PDF 文本比对 | 服务器 |
| `rejected/*.json` | 没通过校验的卡片和原因（用来改进提示词） | 服务器 |
| `digests/YYYY-MM-DD.md` | 日报/周报 | 服务器 |
| `runs/YYYY-MM-DD.json` | 运行统计和 token 用量 | 服务器 |
| `feedback.md` | 给调研 agent 的反馈：判错的地方、要补的方向 | Claude |
| `integration_log.md` | 哪些卡片已整合进主文档、改了什么 | Claude |

venue 标记：`[CVPR'25]` 已核实；`[CVPR'25?]` 待核实（只有 Semantic Scholar 或 arXiv 注释）；`[preprint]` 未找到证据。
