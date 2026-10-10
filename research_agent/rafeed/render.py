"""Deterministic Markdown rendering (cards, daily digest). No LLM prose is added here."""
from .venue import tag

STATUS_ZH = {"verified": "已核实", "likely": "待核实", "unverified": "预印本"}


def _venue_line(v):
    if not v:
        return "预印本（未找到 venue 证据）"
    ev = v.get("evidence") or ""
    ev = f"[证据]({ev})" if ev.startswith("http") else ev
    return f"{tag(v)} {STATUS_ZH.get(v.get('status'), v.get('status'))}，tier={v.get('tier')}；{ev}"


def _claim_lines(claims):
    out = []
    for c in claims:
        nums = "；".join(f"{n.get('dataset', '')} {n.get('metric', '')} {n.get('value', '')}"
                        + (f"（基线 {n['baseline']}）" if n.get("baseline") else "")
                        for n in c.get("numbers") or [])
        out.append(f"- 【{c.get('tag', '未核实')}】{c.get('claim', '')}"
                   + (f"（{nums}）" if nums else ""))
        out.append(f"  > \"{c.get('quote', '')}\" — {c.get('location', '')}")
    return out or ["- （无）"]


def card_markdown(card):
    g = card.get("geometry_use") or {}
    rel = card.get("relevance") or {}
    ctr = card.get("controls") or {}
    val = card.get("validation") or {}
    ids = card.get("ids") or {}
    links = [f"[{k}]({u})" for k, u in (("arXiv", f"https://arxiv.org/abs/{ids['arxiv']}" if ids.get("arxiv") else None),
                                       ("OpenReview", f"https://openreview.net/forum?id={ids['openreview']}" if ids.get("openreview") else None),
                                       ("page", card.get("url"))) if u]
    lines = [
        f"# {card.get('title', '')} {tag(card.get('venue'))}",
        "",
        f"- venue：{_venue_line(card.get('venue'))}",
        f"- 链接：{' · '.join(links) or '—'}",
        f"- 一句话：{card.get('one_line', '')}",
        f"- 设定：{card.get('setting', '')}；几何信息：{', '.join(g.get('info', [])) or '—'}；"
        f"注入位置：{', '.join(g.get('injection', [])) or '—'}",
        f"- 问题：{card.get('problem', '')}",
        f"- 方法：{card.get('method', '')}",
        "",
        "## 主要结论（引文已与原文比对）",
        *_claim_lines(card.get("claims") or []),
        "",
        "## 负面结果与局限",
        *_claim_lines(card.get("negative_or_limits") or []),
        "",
        "## 对照（能否排除\"通用特征更强\"）",
        *[f"- {k}：{v}" for k, v in ctr.items() if k != "notes"],
        f"- 说明：{ctr.get('notes', '')}",
        "",
        "## 与本项目",
        f"- 相关度：{rel.get('score')}；关联卡片：{', '.join(rel.get('which_cards', [])) or '—'}",
        f"- 【推断】{rel.get('to_project', '')}",
        *[f"- 规则 {l.get('rule')} {l.get('stance')}：{l.get('why', '')}" for l in card.get("rule_links") or []],
        "",
        "## 开放问题",
        *([f"- {q}" for q in card.get("open_questions") or []] or ["- （无）"]),
        "",
        f"> 自动生成（rafeed + Codex）。校验：引文 {val.get('quotes_ok', 0)}/{val.get('claims', 0)} 命中原文，"
        f"数字 {val.get('numbers_ok', 0)}/{val.get('numbers', 0)} 在原文中出现。未命中的条目已标为【未核实】。",
    ]
    return "\n".join(lines) + "\n"


def digest_markdown(day, stats, screened, cards, rejected, upgrades, errors):
    rows = sorted((s for s in screened if s["relevance"] >= 2),
                  key=lambda s: (-s["relevance"], s.get("title", "")))
    lines = [f"# 调研日报 {day}", "", "## 计数", ""]
    lines += [f"- {k}：{v}" for k, v in stats.items()]
    lines += ["", "## 相关度 ≥2 的新论文", ""]
    if rows:
        lines += ["| 相关度 | 论文 | venue | 主题 | 理由 |", "|---|---|---|---|---|"]
        for s in rows:
            lines.append(f"| {s['relevance']} | {s.get('title', '')} | {tag(s.get('venue'))} "
                         f"| {', '.join(s.get('topics', []))} | {s.get('reason', '').replace('|', '/')} |")
    else:
        lines.append("（无）")
    lines += ["", "## 新卡片", ""] + ([f"- [{c['title']}](../cards/{c['file']}.md) {tag(c.get('venue'))}"
                                       f"，引文命中 {c['quotes']}" for c in cards] or ["（无）"])
    lines += ["", "## 被拒的卡片", ""] + ([f"- {r['id']}：{r['why']}" for r in rejected] or ["（无）"])
    lines += ["", "## venue 升级（待核实/预印本 → 已核实）", ""] + ([f"- {u}" for u in upgrades] or ["（无）"])
    if errors:
        lines += ["", "## 运行问题", ""] + [f"- {e}" for e in errors]
    return "\n".join(lines) + "\n"
