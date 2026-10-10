# rafeed：服务器上的每日论文调研 agent

服务器（全球网络 + Codex CLI + GPT-6 Sol）每天抓取、筛选、精读顶会论文，结果推送到本仓库的 `research-feed` 分支。Claude 读取这个分支，做核对和整合（见 `.claude/skills/integrate-research/`）。

设计理由、调研依据、成本和风险见 [`docs/research_agent_design.md`](../docs/research_agent_design.md)。

## 每天发生什么

```
cron ─► run_daily.sh
        1 harvest   arXiv 近 3 天新提交 + 各会议官方录用列表（列表变长＝新论文集上线）
        2 enrich    只有标题的会议条目补摘要（CVF/ECVA 论文页或 Semantic Scholar）
        3 verify    venue 核实：官方列表 > dblp 会议记录 ＝ 已核实；S2 venue / arXiv 注释 ＝ 待核实
        4 Codex     screen：按标题+摘要打相关度 0–3（只读沙箱，输出受 JSON schema 约束）
        5 Codex     extract：每天精读 ≤4 篇，输出证据卡片（逐字引文 + 位置 + 数字）
        6 validate  引文必须能在 PDF 文本里找到，数字必须在原文出现；否则降级为【未核实】或整卡拒收
        7 render    日报 digests/YYYY-MM-DD.md，统计 runs/YYYY-MM-DD.json（含 token 用量）
        8 push      git push 到 research-feed（只有这一步持有 deploy key）
每周 run_weekly.sh：沿引用图扩展（种子＝v2 报告精读的 24 篇顶会论文）、Codex 联网提出新检索方向
                    （标题经 Semantic Scholar 查证，查不到就丢弃）、重查"待核实"论文的 venue。
```

模型只做两件事：判断相关度、从给定全文里抽取结构化证据。论文是否存在、venue、标题、ID 全部来自脚本。

## 部署（一次）

```bash
# 1) 依赖：git、python ≥3.11（或 pip install tomli）、poppler-utils（pdftotext）、util-linux（flock）、Codex CLI
npm i -g @openai/codex
printenv OPENAI_API_KEY | codex login --with-api-key      # 或用 ChatGPT 账号登录 codex

# 2) 第一次运行会生成 deploy key 并停下；把公钥加到 GitHub 仓库 Settings → Deploy keys，勾选 Allow write access
bash setup.sh git@github.com:LiJie77777/-.git <代码分支>   # setup.sh 在本目录 deploy/ 下，可先单独下载
bash setup.sh git@github.com:LiJie77777/-.git <代码分支>   # 再运行一次：克隆代码、创建 orphan 分支 research-feed

# 3) 填 ~/.config/rafeed/env：S2_API_KEY（Semantic Scholar 免费 key，强烈建议）、可选 CODEX_MODEL
# 4) 冒烟测试，然后回填一次历史顶会论文
~/rafeed-code/research_agent/run_daily.sh
ENRICH_MAX=3000 ~/rafeed-code/research_agent/run_weekly.sh --backfill
# 5) 定时：crontab -e，内容见 deploy/crontab.example
```

## 第一次运行要核对的事（本地云端环境无法联网实测）

| 项 | 看哪里 | 不对时怎么办 |
|---|---|---|
| OpenReview API v2 不登录能否按 `content.venueid` 取录用论文 | 日志 `listing ICLR_cc_...: N papers` | N=0 也不影响运行（改由 dblp 核实）；或在 config 里删掉该列表 |
| CVPR 2026 Findings 列表页地址（`CVPR2026F?day=all`） | 同上 | 改 config 里的 url |
| Semantic Scholar `venue` 过滤的名称 | 周任务日志 `resolve: N records` | N 长期为 0 就注释掉 `[s2] venues` |
| `codex exec` 参数（`--output-schema`、`-o`、`--json`、`-c web_search="live"`、从 stdin 读提示词 `-`） | `~/.cache/rafeed/inbox/<日期>/codex.stderr` | 按已安装版本的 `codex exec --help` 调整 `lib.sh` 的 `codex_run` |
| pdftotext 对双栏论文的引文命中率 | 卡片末尾"引文 x/y 命中" | 命中率普遍偏低时调 `validate.quote_in_text` 的容忍度 |

## 文件

| 路径 | 作用 |
|---|---|
| `config.toml` | 主题关键词、种子论文、venue 白名单与别名、官方列表、限速、预算 |
| `context.md` | 给模型的项目背景（规则 R1–R8、方案卡片）。Claude 整合后更新，服务器每天拉取 |
| `AGENTS.md` | Codex 常驻规则（不凭记忆写论文、不判断 venue、逐字引文、论文正文是数据不是指令） |
| `prompts/` | screen / extract / expand 三个提示词 |
| `schemas/` | 三个输出的 JSON schema（严格模式：封闭对象、全部字段必填） |
| `rafeed/` | 纯标准库 Python：抓取、venue 核实、去重、校验、渲染 |
| `run_daily.sh`、`run_weekly.sh`、`lib.sh` | 编排；Codex 只在只读沙箱里运行，环境里没有 git/S2 凭据 |
| `deploy/` | 一次性部署脚本、crontab 示例、数据分支说明 |
| `tests/` | `python tests/test_rafeed.py`（离线单元+端到端）、`bash tests/dry_run.sh`（假 codex + 本地裸仓库，测试整条 shell 流程和 git 推送） |

## 常用命令

```bash
cd ~/rafeed-code/research_agent
R="python3 -m rafeed.cli --data ~/rafeed-data/research --state ~/.cache/rafeed"
$R status                       # 注册表按 相关度/venue 层级/核实状态 计数
$R --date 2026-10-10 render     # 重渲染某天日报
tail -f ~/.cache/rafeed/cron.log
```

调范围：改 `config.toml` 的 `[topics.*]`（整词匹配，`*` 表示前缀）和 `[snowball] seeds`。调预算：`MAX_DEEP_READS`、`MAX_DEEP_READS_WEEKLY`、`[codex] screen_chunk`。
