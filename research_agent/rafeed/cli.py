"""rafeed command line. run_daily.sh / run_weekly.sh call these steps in order; Codex is
invoked by the shell scripts between `screen-input` and `apply-screen`, and between
`fetch-text` and `apply-card`.

  python -m rafeed.cli --config config.toml --data <data-repo>/research --state <cache> <cmd> ...
"""
import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

from . import render, sources, store, validate, venue
from .net import Http

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- context
class Ctx:
    def __init__(self, args, http=None):
        with open(args.config, "rb") as f:
            self.cfg = tomllib.load(f)
        self.data, self.state = args.data, args.state
        self.day = args.date or dt.datetime.now(dt.timezone.utc).date().isoformat()
        self.inbox = os.path.join(self.state, "inbox", self.day)
        os.makedirs(self.inbox, exist_ok=True)
        self.book = venue.VenueBook({**self.cfg["venues"], "min_year": self.cfg["project"]["min_year"]})
        self.min_year = int(self.cfg["project"]["min_year"])
        hh = {}
        if os.environ.get("S2_API_KEY"):
            hh["api.semanticscholar.org"] = {"x-api-key": os.environ["S2_API_KEY"]}
        self.http = http or Http(self.cfg["project"].get("contact", ""),
                                 self.cfg.get("http", {}).get("min_interval", {}), hh)
        self.registry = store.Registry(os.path.join(self.data, "registry.jsonl"))

    def p(self, *parts):
        return os.path.join(self.inbox, *parts)

    def bump(self, key, n=1):
        path = self.p("stats.json")
        s = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
        s[key] = s.get(key, 0) + n
        json.dump(s, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    def error(self, msg):
        with open(self.p("errors.txt"), "a", encoding="utf-8") as f:
            f.write(msg.replace("\n", " ") + "\n")
        print("WARN", msg, file=sys.stderr)


def _topics(ctx, rec):
    return store.prefilter(rec, ctx.cfg["topics"], ctx.cfg["prefilter"].get("require", []), ctx.min_year)


def _add_candidates(ctx, recs, label):
    """Prefilter, drop papers already screened, append to today's candidate file."""
    path = ctx.p("candidates.jsonl")
    cands = {c["cid"]: c for c in store.read_jsonl(path)}
    seen_alias = store.AliasIndex()
    for c in cands.values():
        seen_alias.add(c, c["cid"])
    added = 0
    for r in recs:
        topics = _topics(ctx, r)
        if not topics:
            continue
        pid = ctx.registry.find(r)
        if pid and ctx.registry.recs[pid].get("screen"):
            continue
        dup = seen_alias.find(r)
        if dup:
            old = cands[dup]
            old["sources"] = sorted(set(old.get("sources", [])) | set(r.get("sources", [])))
            old["venue"] = venue.better(old.get("venue"), r.get("venue"))
            for k, v in r.get("ids", {}).items():
                old.setdefault("ids", {}).setdefault(k, v)
            if len(r.get("abstract") or "") > len(old.get("abstract") or ""):
                old["abstract"] = r["abstract"]
            continue
        r = dict(r, topics=topics, cid=r.get("cid") or store.canonical_id(r.get("ids", {}), r["title"]))
        cands[r["cid"]] = r
        seen_alias.add(r, r["cid"])
        added += 1
    store.write_jsonl(path, cands.values())
    ctx.bump(f"candidates_{label}", added)
    return added


# ---------------------------------------------------------------- harvest
def _carry_over(ctx, days=7):
    """Candidates from the last week that never got screened (Codex failure, chunk errors)."""
    day = dt.date.fromisoformat(ctx.day)
    recs = []
    for k in range(1, days + 1):
        p = os.path.join(ctx.state, "inbox", (day - dt.timedelta(days=k)).isoformat(), "candidates.jsonl")
        recs += store.read_jsonl(p)
    return recs


def cmd_harvest(ctx, args):
    day = dt.date.fromisoformat(ctx.day)
    print("carried over:", _add_candidates(ctx, _carry_over(ctx), "carry_over"))
    a = ctx.cfg["arxiv"]
    if not args.skip_arxiv:
        q = sources.arxiv_window_query(a["categories"], day - dt.timedelta(days=a["window_days"]),
                                       day - dt.timedelta(days=1))
        try:
            recs = list(sources.arxiv_search(ctx.http, q, a.get("max_results", 2000)))
            ctx.bump("arxiv_fetched", len(recs))
            print("arxiv new candidates:", _add_candidates(ctx, recs, "arxiv"))
        except Exception as e:
            ctx.error(f"arxiv: {e}")
    counts_path = os.path.join(ctx.state, "listing_counts.json")
    counts = json.load(open(counts_path)) if os.path.exists(counts_path) else {}
    pages = {}
    for L in ctx.cfg.get("listings", []):
        slug = re.sub(r"[^A-Za-z0-9]+", "_", (L.get("venueid") or L["url"]) + "_" + L.get("href_contains", "")).strip("_")
        cache = os.path.join(ctx.state, "listings", slug + ".jsonl")
        try:
            if L["kind"] == "openreview":
                recs = list(sources.openreview_listing(ctx.http, L["venueid"]))
            else:
                if L["url"] not in pages:
                    pages[L["url"]] = ctx.http.get(L["url"]).decode("utf-8", "replace")
                html = pages[L["url"]]
                recs = [sources.record(t, year=L["year"], ids={"cvf": u.rsplit("/", 1)[-1].rsplit(".", 1)[0]},
                                       url=u, pdf_url=sources.cvf_pdf_url(u) if "thecvf.com" in u else None,
                                       source="listing")
                        for t, u in sources.parse_ptitle_listing(html, L["url"], L.get("href_contains"))]
        except Exception as e:
            ctx.error(f"listing {slug}: {e}")
            continue
        ev = (f"https://openreview.net/group?id={L['venueid']}" if L["kind"] == "openreview" else L["url"])
        for r in recs:
            r["year"] = L["year"]
            r["venue"] = ctx.book.entry(L["venue"], L["year"], L.get("track", "main"), "verified", ev, ctx.day)
        old = store.read_jsonl(cache)
        old_titles = {venue.norm_title(r["title"]) for r in old}
        if len(recs) >= len(old):
            store.write_jsonl(cache, recs)
        new = recs if args.backfill else [r for r in recs if venue.norm_title(r["title"]) not in old_titles]
        counts[slug] = len(recs)
        if new:
            print(f"listing {slug}: {len(recs)} papers, {len(new)} new ->",
                  _add_candidates(ctx, new, "listing"), "candidates")
    json.dump(counts, open(counts_path, "w"), indent=1)


def _listing_index(ctx):
    idx = venue.ListingIndex()
    d = os.path.join(ctx.state, "listings")
    for fn in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        for r in store.read_jsonl(os.path.join(d, fn)):
            if r.get("venue"):
                idx.add(r["title"], r["venue"])
    return idx


# ---------------------------------------------------------------- snowball / resolve / enrich
def _s2_id(rec):
    ids = rec.get("ids", {})
    for k, pre in (("s2", ""), ("arxiv", "arXiv:"), ("doi", "DOI:")):
        if ids.get(k):
            return pre + ids[k]
    return None


def cmd_snowball(ctx, args):
    sb = ctx.cfg["snowball"]
    state_path = os.path.join(ctx.state, "snowball.json")
    done = json.load(open(state_path)) if os.path.exists(state_path) else {}
    cutoff = (dt.date.fromisoformat(ctx.day) - dt.timedelta(days=sb.get("repeat_days", 28))).isoformat()
    queue = list(sb.get("seeds", []))
    queue += [_s2_id(r) for r in ctx.registry.recs.values()
              if (r.get("screen") or {}).get("relevance", 0) >= sb.get("min_relevance", 2) and _s2_id(r)]
    queue = [q for q in dict.fromkeys(queue) if done.get(q, "") < cutoff][: sb.get("max_papers_per_run", 30)]
    total = 0
    for sid in queue:
        recs = []
        for direction in ("citations", "references"):
            try:
                recs += list(sources.s2_neighbors(ctx.http, sid, direction, sb.get("max_neighbors_per_paper", 400)))
            except Exception as e:
                ctx.error(f"snowball {sid} {direction}: {e}")
        total += _add_candidates(ctx, recs, "snowball")
        done[sid] = ctx.day
    json.dump(done, open(state_path, "w"), indent=1)
    print(f"snowball: {len(queue)} papers expanded, {total} new candidates")


def cmd_resolve(ctx, args):
    obj = json.load(open(args.file, encoding="utf-8"))
    recs = []
    for q in obj.get("queries", [])[: args.max_queries]:
        try:
            recs += list(sources.s2_bulk_search(ctx.http, q["query"], f"{ctx.min_year}-",
                                                ctx.cfg["s2"].get("venues"), args.per_query))
        except Exception as e:
            ctx.error(f"resolve query {q.get('query')!r}: {e}")
    unresolved = []
    for s in obj.get("seed_titles", []):
        r = sources.s2_match_title(ctx.http, s["title"])
        if r and venue.titles_match(r["title"], s["title"], 0.8):
            recs.append(r)
        else:
            unresolved.append(s["title"])
    n = _add_candidates(ctx, recs, "expansion")
    with open(ctx.p("unresolved_titles.txt"), "a", encoding="utf-8") as f:
        f.writelines(t + "\n" for t in unresolved)
    with open(os.path.join(ctx.state, "expansion_queries.txt"), "a", encoding="utf-8") as f:
        f.writelines(f"{ctx.day}\t{q['query']}\n" for q in obj.get("queries", []))
    print(f"resolve: {len(recs)} records, {n} new candidates, {len(unresolved)} titles not found (dropped)")


def cmd_enrich(ctx, args):
    path = ctx.p("candidates.jsonl")
    cands = store.read_jsonl(path)
    kept, n = [], 0
    for c in cands:
        if not c.get("abstract") and n < args.max:
            n += 1
            try:
                if c.get("url") and ("thecvf.com" in c["url"] or "ecva.net" in c["url"]):
                    c["abstract"] = sources.parse_abstract_page(ctx.http.get(c["url"]).decode("utf-8", "replace"))
                else:
                    r = sources.s2_match_title(ctx.http, c["title"])
                    if r and venue.titles_match(r["title"], c["title"]):
                        c["abstract"] = r["abstract"]
                        for k, v in r["ids"].items():
                            c["ids"].setdefault(k, v)
            except Exception as e:
                ctx.error(f"enrich {c['cid']}: {e}")
        topics = _topics(ctx, c)
        if topics:
            c["topics"] = topics
            kept.append(c)
    store.write_jsonl(path, kept)
    ctx.bump("dropped_after_enrich", len(cands) - len(kept))
    print(f"enrich: fetched {n} abstracts, kept {len(kept)}/{len(cands)}")


# ---------------------------------------------------------------- verify
def _verify_one(ctx, rec, idx, dblp_budget):
    v = venue.better(rec.get("venue"), idx.lookup(rec["title"]))
    if (v or {}).get("status") != "verified" and dblp_budget[0] > 0:
        dblp_budget[0] -= 1
        try:
            v = venue.better(v, venue.from_dblp(sources.dblp_search(ctx.http, rec["title"]), rec, ctx.book, ctx.day))
        except Exception as e:
            ctx.error(f"dblp {rec['title'][:60]!r}: {e}")
    v = venue.better(v, venue.from_s2_hint(rec, ctx.book, ctx.day))
    v = venue.better(v, venue.from_arxiv_comment(rec, ctx.book, ctx.day))
    return v or ctx.book.entry(None, None, "main", "unverified", None, ctx.day)


def cmd_verify(ctx, args):
    vc = ctx.cfg["verify"]
    idx, budget = _listing_index(ctx), [vc.get("max_dblp_per_run", 150)]
    path = ctx.p("candidates.jsonl")
    cands = store.read_jsonl(path)
    for c in cands:
        c["venue"] = _verify_one(ctx, c, idx, budget)
    store.write_jsonl(path, cands)
    # re-check relevant registry papers whose venue is not yet verified
    cutoff = (dt.date.fromisoformat(ctx.day) - dt.timedelta(days=vc.get("recheck_days", 30))).isoformat()
    upgrades, n = [], 0
    for r in ctx.registry.recs.values():
        v = r.get("venue") or {}
        if v.get("status") == "verified" or (r.get("screen") or {}).get("relevance", 0) < 2 \
                or (v.get("checked") or "") > cutoff or n >= vc.get("max_recheck_per_run", 100):
            continue
        n += 1
        nv = _verify_one(ctx, r, idx, budget)
        if venue.RANK[nv["status"]] > venue.RANK[v.get("status")]:
            upgrades.append(f"{r['title']}：{venue.tag(v)} → {venue.tag(nv)}")
            r["venue"] = nv
        else:
            r.setdefault("venue", nv)["checked"] = ctx.day
    ctx.registry.save()
    with open(ctx.p("upgrades.txt"), "a", encoding="utf-8") as f:
        f.writelines(u + "\n" for u in upgrades)
    st = {}
    for c in cands:
        st[c["venue"]["status"]] = st.get(c["venue"]["status"], 0) + 1
    print("verify:", st, f"rechecked {n}, upgraded {len(upgrades)}")


# ---------------------------------------------------------------- Codex I/O
def cmd_screen_input(ctx, args):
    cands = [c for c in store.read_jsonl(ctx.p("candidates.jsonl"))
             if not (ctx.registry.find(c) and ctx.registry.recs[ctx.registry.find(c)].get("screen"))]
    rows = [{"id": c["cid"], "title": c["title"], "abstract": (c.get("abstract") or "")[:1500],
             "venue": venue.tag(c.get("venue")), "venue_status": (c.get("venue") or {}).get("status"),
             "topics_hint": c.get("topics", [])} for c in cands]
    paths = []
    for i in range(0, len(rows), args.chunk):
        p = ctx.p(f"screen_in_{i // args.chunk:02d}.jsonl")
        store.write_jsonl(p, rows[i:i + args.chunk])
        paths.append(p)
    print("\n".join(paths))


def _read(path, default=""):
    return open(path, encoding="utf-8").read() if path and os.path.exists(path) else default


def cmd_prompt(ctx, args):
    tpl = _read(os.path.join(HERE, "prompts", f"{args.kind}.md"))
    # Inputs are embedded in the prompt (Codex truncates long shell output, so asking the
    # model to `cat` a 150k-character paper is unreliable); paths are kept for reference.
    subs = {
        "{{INPUT_TEXT}}": _read(args.input),
        "{{INPUT}}": args.input or "",
        "{{CONTEXT_TEXT}}": _read(os.path.join(HERE, "context.md")),
        "{{CONTEXT}}": os.path.join(HERE, "context.md"),
        "{{FEEDBACK}}": _read(os.path.join(ctx.data, "feedback.md"), "（暂无反馈）"),
        "{{TOPICS}}": "\n".join(f"- {k}: {v.get('desc', '')}" for k, v in ctx.cfg["topics"].items()),
        "{{TODAY}}": ctx.day,
    }
    if args.kind == "extract":
        r = ctx.registry.recs[args.id]
        subs["{{PAPER_ID}}"] = r["id"]
        subs["{{PAPER_META}}"] = json.dumps({k: r.get(k) for k in ("id", "title", "year", "venue", "url")},
                                            ensure_ascii=False)
    if args.kind == "expand":
        subs["{{COVERAGE}}"] = _coverage(ctx)
    # single pass, so placeholders that happen to occur inside embedded paper text stay as they are
    sys.stdout.write(re.sub(r"\{\{[A-Z_]+\}\}", lambda m: subs.get(m.group(0), m.group(0)), tpl))


def _coverage(ctx):
    by_topic, by_tier = {}, {}
    for r in ctx.registry.recs.values():
        if (r.get("screen") or {}).get("relevance", 0) < 2:
            continue
        for t in r.get("topics", []):
            by_topic[t] = by_topic.get(t, 0) + 1
        k = (r.get("venue") or {}).get("tier", "none")
        by_tier[k] = by_tier.get(k, 0) + 1
    hist = _read(os.path.join(ctx.state, "expansion_queries.txt"))
    return (f"相关度≥2 的论文按主题：{json.dumps(by_topic, ensure_ascii=False)}\n"
            f"按 venue 层级：{json.dumps(by_tier, ensure_ascii=False)}\n"
            f"以前用过的检索式（不要重复）：\n{hist[-4000:] or '（无）'}")


def cmd_apply_screen(ctx, args):
    cands = {c["cid"]: c for c in store.read_jsonl(ctx.p("candidates.jsonl"))}
    chunk_ids = {r["id"] for r in store.read_jsonl(args.input)} if args.input else set(cands)
    obj = json.load(open(args.file, encoding="utf-8"))
    items, errors, missing = validate.validate_screen(obj, chunk_ids, set(ctx.cfg["topics"]))
    for e in errors:
        ctx.error(f"screen: {e}")
    if missing:
        ctx.error(f"screen: {len(missing)} candidates not screened (will retry tomorrow)")
    out = store.read_jsonl(ctx.p("screened.jsonl"))
    for it in items:
        c = dict(cands[it["id"]])
        c.pop("cid", None)
        c["topics"] = it["topics"] or c.get("topics", [])
        c["screen"] = {k: it[k] for k in ("relevance", "evidence_type", "deep_read", "reason")}
        c["screen"]["date"] = ctx.day
        if it["relevance"] <= 1:  # keep the registry small: no abstracts for irrelevant papers
            c["abstract"], c["authors"] = "", c.get("authors", [])[:3]
        pid = ctx.registry.upsert(c, ctx.day)
        reg = ctx.registry.recs[pid]
        reg["screen"], reg["topics"] = c["screen"], c["topics"]
        out.append({"id": pid, "title": reg["title"], "venue": reg.get("venue"), **c["screen"],
                    "topics": c["topics"]})
    ctx.registry.save()
    store.write_jsonl(ctx.p("screened.jsonl"), out)
    ctx.bump("screened", len(items))
    print(f"apply-screen: {len(items)} ok, {len(errors)} errors, {len(missing)} missing")


def cmd_select_deep(ctx, args):
    k = args.k
    pool = [r for r in ctx.registry.recs.values()
            if (r.get("screen") or {}).get("deep_read") and not r.get("card") and not r.get("card_rejected")
            and sum((r.get("failures") or {}).values()) < 2]
    tier_rank = {"top": 3, "vertical": 3, "findings": 2}
    pool.sort(key=lambda r: (-r["screen"]["relevance"], -venue.RANK[(r.get("venue") or {}).get("status")],
                             -tier_rank.get((r.get("venue") or {}).get("tier"), 0), r["screen"].get("date", "")))
    pick = [r["id"] for r in pool[:k]]
    open(ctx.p("deep_read.txt"), "w").write("".join(p + "\n" for p in pick))
    print("\n".join(pick))


def strip_references(text):
    m = None
    for m in re.finditer(r"\n\s*(References|REFERENCES|Bibliography)\s*\n", text):
        pass
    if not m or m.start() < 0.3 * len(text):
        return text
    tail = text[m.end():]
    app = re.search(r"\n\s*(Appendix|APPENDIX|Supplementary|SUPPLEMENTARY|A\s+[A-Z][a-z]+.*)\n", tail)
    return text[:m.start()] + ("\n[references removed]\n" + tail[app.start():] if app else "\n[references removed]\n")


def _mark_failed(ctx, pid, what):
    f = ctx.registry.recs[pid].setdefault("failures", {})
    f[what] = f.get(what, 0) + 1
    ctx.registry.save()


def cmd_mark_failed(ctx, args):
    _mark_failed(ctx, args.id, args.what)


def cmd_fetch_text(ctx, args):
    r = ctx.registry.recs[args.id]
    name = store.safe_name(args.id)
    d = os.path.join(ctx.state, "text")
    os.makedirs(d, exist_ok=True)
    pdf, txt = os.path.join(d, name + ".pdf"), os.path.join(d, name + ".txt")
    try:
        if not os.path.exists(txt):
            pdf_url = r.get("pdf_url") or (f"https://arxiv.org/pdf/{r['ids']['arxiv']}" if r["ids"].get("arxiv") else None)
            if not pdf_url and r.get("url"):  # e.g. ECVA: the pdf link is on the paper page
                pdf_url = sources.find_pdf_link(ctx.http.get(r["url"]).decode("utf-8", "replace"), r["url"])
            if not pdf_url:
                raise RuntimeError("no pdf url")
            data = ctx.http.get(pdf_url)
            if not data.startswith(b"%PDF"):
                raise RuntimeError(f"not a pdf: {pdf_url}")
            open(pdf, "wb").write(data)
            subprocess.run(["pdftotext", "-enc", "UTF-8", pdf, txt], check=True)
    except Exception as e:
        _mark_failed(ctx, args.id, "fetch")
        sys.exit(f"fetch-text {args.id}: {e}")
    full = _read(txt)
    llm = strip_references(full)[: ctx.cfg["codex"].get("max_paper_chars", 150000)]
    open(os.path.join(d, name + ".llm.txt"), "w", encoding="utf-8").write(llm)
    print(os.path.join(d, name + ".llm.txt"))


def cmd_apply_card(ctx, args):
    rec = ctx.registry.recs[args.id]
    name = store.safe_name(args.id)
    text = _read(os.path.join(ctx.state, "text", name + ".txt"))
    try:
        card = json.load(open(args.file, encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        card, text = {"id": None}, ""
        ctx.error(f"card {args.id}: unreadable output ({e})")
    card, report, ok = validate.validate_card(card, text, rec)
    card["model"] = args.model
    card["generated"] = ctx.day
    log = {"id": args.id, "title": rec["title"], "venue": rec.get("venue"), "file": name,
           "quotes": f"{report['quotes_ok']}/{report['claims']}"}
    if ok:
        os.makedirs(os.path.join(ctx.data, "cards"), exist_ok=True)
        json.dump(card, open(os.path.join(ctx.data, "cards", name + ".json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        open(os.path.join(ctx.data, "cards", name + ".md"), "w", encoding="utf-8").write(render.card_markdown(card))
        rec["card"] = f"cards/{name}.md"
        with open(ctx.p("cards.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(log, ensure_ascii=False) + "\n")
    else:
        os.makedirs(os.path.join(ctx.data, "rejected"), exist_ok=True)
        json.dump({"card": card, "problems": report["problems"]},
                  open(os.path.join(ctx.data, "rejected", name + ".json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        rec["card_rejected"] = ctx.day
        with open(ctx.p("rejected.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": args.id, "why": "; ".join(report["problems"][:3])}, ensure_ascii=False) + "\n")
    ctx.registry.save()
    print(("accepted" if ok else "REJECTED"), args.id, log["quotes"])


def cmd_render(ctx, args):
    stats = json.load(open(ctx.p("stats.json"))) if os.path.exists(ctx.p("stats.json")) else {}
    md = render.digest_markdown(
        ctx.day, stats, store.read_jsonl(ctx.p("screened.jsonl")), store.read_jsonl(ctx.p("cards.jsonl")),
        store.read_jsonl(ctx.p("rejected.jsonl")), _read(ctx.p("upgrades.txt")).splitlines(),
        _read(ctx.p("errors.txt")).splitlines()[-30:])
    os.makedirs(os.path.join(ctx.data, "digests"), exist_ok=True)
    open(os.path.join(ctx.data, "digests", ctx.day + ".md"), "w", encoding="utf-8").write(md)
    os.makedirs(os.path.join(ctx.data, "runs"), exist_ok=True)
    json.dump({"date": ctx.day, **stats, "usage": _usage(ctx)},
              open(os.path.join(ctx.data, "runs", ctx.day + ".json"), "w"), indent=1)
    print("digest:", os.path.join(ctx.data, "digests", ctx.day + ".md"))


def _usage(ctx):
    """Sum token usage from `codex exec --json` logs (keys named *_tokens anywhere)."""
    tot = {}

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k.endswith("tokens") and isinstance(v, int):
                    tot[k] = tot.get(k, 0) + v
                else:
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    for fn in sorted(os.listdir(ctx.inbox)):
        if fn.startswith("codex_") and fn.endswith(".jsonl"):
            for line in open(os.path.join(ctx.inbox, fn), encoding="utf-8", errors="replace"):
                try:
                    o = json.loads(line)
                except ValueError:
                    continue
                if isinstance(o, dict) and "completed" in str(o.get("type", "")):
                    walk(o)
    return tot


def cmd_status(ctx, args):
    st = {}
    for r in ctx.registry.recs.values():
        v = r.get("venue") or {}
        key = f"rel{(r.get('screen') or {}).get('relevance', '-')}/{v.get('tier', '-')}/{v.get('status', '-')}"
        st[key] = st.get(key, 0) + 1
    for k in sorted(st):
        print(f"{k:32s} {st[k]}")


def main(argv=None, http=None):
    ap = argparse.ArgumentParser(prog="rafeed")
    ap.add_argument("--config", default=os.path.join(HERE, "config.toml"))
    ap.add_argument("--data", required=True, help="<data repo>/research")
    ap.add_argument("--state", required=True, help="local cache dir (not committed)")
    ap.add_argument("--date", help="UTC date, default today")
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("harvest")
    h.add_argument("--backfill", action="store_true", help="treat every listed paper as new")
    h.add_argument("--skip-arxiv", action="store_true")
    sub.add_parser("snowball")
    r = sub.add_parser("resolve")
    r.add_argument("--file", required=True)
    r.add_argument("--max-queries", type=int, default=8)
    r.add_argument("--per-query", type=int, default=100)
    e = sub.add_parser("enrich")
    e.add_argument("--max", type=int, default=400)
    sub.add_parser("verify")
    s = sub.add_parser("screen-input")
    s.add_argument("--chunk", type=int, default=120)
    pr = sub.add_parser("prompt")
    pr.add_argument("--kind", choices=["screen", "extract", "expand"], required=True)
    pr.add_argument("--input")
    pr.add_argument("--id")
    a = sub.add_parser("apply-screen")
    a.add_argument("--file", required=True)
    a.add_argument("--input", help="the screen_in chunk the output answers")
    d = sub.add_parser("select-deep")
    d.add_argument("--k", type=int, default=4)
    f = sub.add_parser("fetch-text")
    f.add_argument("--id", required=True)
    c = sub.add_parser("apply-card")
    c.add_argument("--id", required=True)
    c.add_argument("--file", required=True)
    c.add_argument("--model", default="")
    m = sub.add_parser("mark-failed")
    m.add_argument("--id", required=True)
    m.add_argument("--what", required=True)
    sub.add_parser("render")
    sub.add_parser("status")
    args = ap.parse_args(argv)
    if http is None and os.environ.get("RAFEED_FIXTURES"):  # offline dry run of the shell pipeline
        from .net import FakeHttp
        d = os.environ["RAFEED_FIXTURES"]
        http = FakeHttp([(k, open(os.path.join(d, f), "rb").read())
                         for k, f in json.load(open(os.path.join(d, "routes.json")))])
    ctx = Ctx(args, http)
    {"harvest": cmd_harvest, "snowball": cmd_snowball, "resolve": cmd_resolve, "enrich": cmd_enrich,
     "verify": cmd_verify, "screen-input": cmd_screen_input, "prompt": cmd_prompt,
     "apply-screen": cmd_apply_screen, "select-deep": cmd_select_deep, "fetch-text": cmd_fetch_text,
     "apply-card": cmd_apply_card, "mark-failed": cmd_mark_failed, "render": cmd_render,
     "status": cmd_status}[args.cmd](ctx, args)


if __name__ == "__main__":
    main()
