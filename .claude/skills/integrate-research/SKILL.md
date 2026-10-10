---
name: integrate-research
description: 把服务器调研 agent（rafeed + Codex）写在 research-feed 分支上的论文卡片整合进 docs/geometry_fusion_research.md；核对 venue 与关键结论，更新规则 R1–R8 和方案卡片，把反馈写回 research/feedback.md。用户说"整合调研""看看新论文""更新调研报告"时使用。
---

# 整合 research-feed（Claude 侧）

分工：服务器上的 GPT（Codex）负责广度——每天筛选、精读、每周扩展检索。你负责深度——核对、整合、下结论、给反馈。
数据分支只放数据（orphan 分支 `research-feed`，目录 `research/`），代码与主文档在开发分支。

## 0. 准备
- `git ls-remote --heads origin research-feed`。没有这个分支 → 告诉用户先在服务器上运行 `research_agent/deploy/setup.sh`，然后停止。
- `git fetch origin research-feed` 后，用工作树读取：`git worktree add /tmp/feed origin/research-feed`（已存在就 `git -C /tmp/feed checkout --detach origin/research-feed`）。
- 读 `/tmp/feed/research/README.md`（布局）、`integration_log.md`（上次整合到哪天）、`feedback.md`（上次给的反馈）。

## 1. 取增量
- 新卡片：`cards/*.json` 中 `generated` 晚于上次整合日期的。
- 新的相关论文：`registry.jsonl` 中 `screen.relevance >= 2` 且 `screen.date` 晚于上次整合的（包括没精读的）。
- 近期日报 `digests/`、被拒卡片 `rejected/`、`runs/` 的 token 用量。

## 2. 证据分级（与主文档纳入标准一致）
- 正文证据：`venue.status == "verified"` 且 `tier` 为 `top` 或 `vertical`。`findings` 标 [XXX'26 Findings]，作辅助证据。
- `likely`（只有 Semantic Scholar 或 arXiv 注释）和 `unverified`：只进附录 C 线索，等服务器每 30 天重查升级。
- 卡片里只有 `tag == "文献事实"` 的条目能进正文（脚本已确认引文在 PDF 文本中）；`未核实` 条目不用。

## 3. 核对（每张要进正文的卡片都做）
1. venue：打开 `venue.evidence`（dblp / OpenReview / CVF / ECVA）。云端直连这些站点会被网络策略拦截时，用 WebSearch 并限定官方域名核对。
2. 解释是否正确：脚本只保证引文存在，不保证 claim 正确转述了引文。逐条看 claim ↔ quote 是否一致，数字是否对应正确的数据集和设置。
3. 关键数字（要写进规则或结论的）：有 arXiv 号时用 alphaXiv 全文问答再核一次。
4. 记录抽查结果：核对 N 条、发现 M 条错误、错误类型。

## 4. 更新主文档 `docs/geometry_fusion_research.md`
- §4.2 分类总表：新增或修改行；§4.3/§4.4 正负证据；§4.5 规则：标明新证据是 supports / contradicts / refines，**反例优先写**；§5 卡片的"顶会依据"；附录 A 引用（按 venue）；附录 C 线索。
- 沿用标注：【文献事实】【本项目事实】【推断】【未核实】，每篇文献后带已核实 venue 标记。
- 顶部版本说明写清本次整合的日期范围和新增文献数。
- 规则或卡片有变化时，同步更新 `research_agent/context.md`（服务器下次运行会拉取）。

## 5. 写回反馈（research-feed 分支，只改这两个文件）
- `research/feedback.md`：整体覆盖写，≤60 行。内容：
  - 判错模式（相关度给高/给低的例子各 1–3 个，带论文 ID）；
  - 被拒卡片的共性原因，以及提示词该怎么改；
  - 需要补的方向和建议检索式（尤其能反驳现有规则的）；
  - 下周优先精读的论文 ID。
- `research/integration_log.md`：追加一节，包括日期、整合了哪些卡片、主文档改了哪些节、抽查结果（N/M）。
- 抽查错误率 > 10% 时，在反馈里要求收紧，并考虑修改 `research_agent/prompts/`。

## 6. 提交
- 主文档、`context.md`、`prompts/` 的修改：在当前开发分支提交并推送。
- `feedback.md` 和 `integration_log.md`：在 /tmp/feed 里提交并 `git push origin HEAD:research-feed`。这是向另一个分支推送，先确认用户在本次会话中同意过。服务器只写其他文件，所以不会冲突；推送前先 `git pull --rebase origin research-feed`。
- 主文档有实质变化时，用同一路径重新发布网页版。

## 7. 可选：Notion 镜像
用户要求时，用 Notion MCP 在其工作区建一个"论文卡片"数据库（只建一次），把本次整合的论文写进去：标题、venue 标记、相关度、关联规则、卡片的 GitHub 链接。服务器不持有 Notion 凭据。
