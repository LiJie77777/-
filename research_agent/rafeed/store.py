"""Paper registry (research/registry.jsonl on the data branch): dedup across sources,
stable ids, merge of ids/venue evidence, keyword prefilter."""
import hashlib
import json
import os
import re

from .venue import better, norm_title

ID_ORDER = ("doi", "arxiv", "openreview", "s2", "dblp", "cvf")


def canonical_id(ids, title):
    for k in ID_ORDER:
        v = ids.get(k)
        if v:
            return f"{k}:{v.lower() if k == 'doi' else v}"
    return "t:" + hashlib.sha1(norm_title(title).encode()).hexdigest()[:12]


def safe_name(pid):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", pid)


def _aliases(rec):
    out = {f"{k}:{(v.lower() if k == 'doi' else v)}" for k, v in rec.get("ids", {}).items() if v}
    nt = norm_title(rec.get("title", ""))
    if nt:
        out.add("title:" + nt)
    return out


class AliasIndex:
    """Any shared id (doi/arxiv/openreview/s2/dblp/cvf) or normalized title = same paper."""

    def __init__(self):
        self.index = {}

    def add(self, rec, key):
        for a in _aliases(rec):
            self.index[a] = key

    def find(self, rec):
        for a in _aliases(rec):
            if a in self.index:
                return self.index[a]
        return None


class Registry(AliasIndex):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.recs = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        self.recs[r["id"]] = r
        for pid, r in self.recs.items():
            self.add(r, pid)

    def upsert(self, rec, today):
        pid = self.find(rec)
        if pid is None:
            pid = canonical_id(rec.get("ids", {}), rec["title"])
            new = {k: v for k, v in rec.items()}
            new.update(id=pid, first_seen=today)
            self.recs[pid] = new
        else:
            old = self.recs[pid]
            for k, v in rec.get("ids", {}).items():
                old.setdefault("ids", {}).setdefault(k, v)
            for k in ("url", "pdf_url", "published", "comment", "venue_hint", "year"):
                if not old.get(k) and rec.get(k):
                    old[k] = rec[k]
            if len(rec.get("abstract") or "") > len(old.get("abstract") or ""):
                old["abstract"] = rec["abstract"]
            if not old.get("authors") and rec.get("authors"):
                old["authors"] = rec["authors"]
            old["sources"] = sorted(set(old.get("sources", [])) | set(rec.get("sources", [])))
            old["venue"] = better(old.get("venue"), rec.get("venue"))
        self.add(self.recs[pid], pid)
        return pid

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for pid in sorted(self.recs):
                f.write(json.dumps(self.recs[pid], ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(tmp, self.path)


_KW = {}


def kw_match(keyword, text):
    """Whole-word/phrase match ("fusion" does not match "diffusion", "sam" not "same");
    a trailing * allows any ending ("geometr*" matches geometry/geometric)."""
    rx = _KW.get(keyword)
    if rx is None:
        k = keyword.lower().strip()
        tail = "" if k.endswith("*") else r"(?![a-z0-9])"
        rx = _KW[keyword] = re.compile(r"(?<![a-z0-9])" + re.escape(k.rstrip("*")) + tail)
    return rx.search(text) is not None


def prefilter(rec, topics_cfg, require_terms, min_year):
    """Cheap keyword gate before any LLM call. Returns matched topic names ([] = drop).
    Title-only records (CVF listings) skip the `require` gate until an abstract is fetched."""
    if rec.get("year") and rec["year"] < min_year:
        return []
    text = (rec.get("title", "") + " " + rec.get("abstract", "")).lower()
    topics = [name for name, t in topics_cfg.items()
              if any(kw_match(k, text) for k in t.get("include", []))
              and not any(kw_match(k, text) for k in t.get("exclude", []))]
    if not topics:
        return []
    if rec.get("abstract") and require_terms and not any(kw_match(k, text) for k in require_terms):
        return []
    return topics


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
