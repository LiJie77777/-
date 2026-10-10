"""Venue verification. The LLM never decides a venue: a paper's venue comes only from
sources checked here, in this order of strength.

  verified   accepted-paper listing of the venue (OpenReview venueid, CVF/ECVA open access)
             or a dblp conference record (not CoRR) whose title matches
  likely     Semantic Scholar venue field, or an arXiv comment such as "Accepted to CVPR 2025"
  unverified anything else (preprint)

Tier comes from config [venues]: top / vertical / findings / workshop / other."""
import re
import unicodedata

RANK = {"verified": 3, "likely": 2, "unverified": 1, None: 0}
_ACCEPT_WORDS = re.compile(r"accept|to appear|published|camera[- ]ready|\boral\b|\bposter\b|"
                           r"spotlight|highlight|proceedings", re.I)


def norm_title(t):
    t = unicodedata.normalize("NFKD", t or "").encode("ascii", "ignore").decode().lower()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", t).split())


def _norm_venue(s):
    s = re.sub(r"\([^)]*\)", " ", (s or "").lower())   # "ECCV (12)", "MICCAI (3)"
    s = re.sub(r"['’]?\b(19|20)?\d{2}\b", " ", s)       # years, '25
    return " ".join(re.sub(r"[^a-z/&]+", " ", s).split())


def titles_match(a, b, thr=0.9):
    na, nb = norm_title(a), norm_title(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    ta, tb = set(na.split()), set(nb.split())
    return len(ta & tb) / len(ta | tb) >= thr


class VenueBook:
    def __init__(self, venues_cfg):
        self.top = set(venues_cfg.get("top", []))
        self.vertical = set(venues_cfg.get("vertical", []))
        self.aliases = {k: [_norm_venue(a) for a in v] for k, v in venues_cfg.get("aliases", {}).items()}
        self.min_year = int(venues_cfg.get("min_year", 2022))

    def classify(self, text):
        """-> (canonical venue or None, kind in {'main','workshop','findings'})."""
        s = (text or "").lower()
        kind = "main"
        if re.search(r"workshop|\bwksp\b|\bw\)", s):
            kind = "workshop"
        elif "findings" in s:
            kind = "findings"
        s = _norm_venue(s)
        for canon, alist in self.aliases.items():
            for a in alist:
                if re.search(rf"(^|\s){re.escape(a)}(\s|$)", s):
                    return canon, kind
        return None, kind

    def tier(self, canon, kind):
        if kind == "workshop":
            return "workshop"
        if kind == "findings":
            return "findings"
        if canon in self.top:
            return "top"
        if canon in self.vertical:
            return "vertical"
        return "other"

    def entry(self, canon, year, kind, status, evidence, checked, raw=None):
        return {"name": canon or raw, "year": year, "kind": kind, "tier": self.tier(canon, kind),
                "status": status, "evidence": evidence, "checked": checked}


def year_in_text(text):
    m = re.search(r"\b(20\d{2})\b", text or "")
    if m:
        return int(m.group(1))
    m = re.search(r"['’](\d{2})\b", text or "")
    return 2000 + int(m.group(1)) if m else None


def from_arxiv_comment(rec, book, today):
    c = rec.get("comment") or ""
    if not c or not _ACCEPT_WORDS.search(c):
        return None
    canon, kind = book.classify(c)
    if not canon:
        return None
    return book.entry(canon, year_in_text(c), kind, "likely", f"arXiv comment: {c[:160]}", today)


def from_s2_hint(rec, book, today):
    hint = rec.get("venue_hint")
    if not hint or not any(s.startswith("s2") for s in rec.get("sources", [])):
        return None
    canon, kind = book.classify(hint)
    if not canon:
        return None
    return book.entry(canon, rec.get("year"), kind, "likely", f"Semantic Scholar venue: {hint}", today)


def from_dblp(hits, rec, book, today):
    for h in hits:
        if not titles_match(rec["title"], h["title"]) or h["key"].startswith("journals/corr"):
            continue
        canon, kind = book.classify(h["venue"])
        if not h["key"].startswith("conf/"):
            canon, kind = None, "main"  # journal: verified, but tier "other"
        return book.entry(canon, h["year"], kind, "verified",
                          f"https://dblp.org/rec/{h['key']}", today, raw=h["venue"])
    return None


def better(old, new):
    """Keep the stronger evidence; never downgrade a verified venue."""
    if not new:
        return old
    if not old or RANK[new.get("status")] > RANK[old.get("status")]:
        return new
    return old


def tag(venue):
    """[CVPR'25] style tag used in the reports; preprints get [preprint]."""
    if not venue or venue.get("status") == "unverified" or not venue.get("name"):
        return "[preprint]"
    y = f"'{str(venue['year'])[-2:]}" if venue.get("year") else ""
    extra = {"findings": " Findings", "workshop": " Workshop"}.get(venue.get("kind"), "")
    mark = "" if venue["status"] == "verified" else "?"
    return f"[{venue['name']}{y}{extra}{mark}]"


class ListingIndex:
    """Title index over cached accepted-paper listings (built by `harvest`)."""

    def __init__(self):
        self.by_title = {}

    def add(self, title, venue_entry):
        self.by_title.setdefault(norm_title(title), venue_entry)

    def lookup(self, title):
        return self.by_title.get(norm_title(title))
