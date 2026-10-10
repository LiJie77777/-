"""Fetchers and parsers for the paper sources. Parsers are pure (bytes/dict -> records)
so they are tested offline against fixtures; fetchers only add paging.

A record is a plain dict:
  {title, authors, year, abstract, ids{arxiv,doi,openreview,s2,dblp,cvf}, url, pdf_url,
   published, comment, venue_hint, venue (set only when the source itself proves it), sources}
"""
import re
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from urllib.parse import urljoin

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"
S2_API = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = "title,authors,year,abstract,venue,publicationVenue,externalIds,publicationDate,url"


def record(title, authors=(), year=None, abstract="", ids=None, url=None, pdf_url=None,
           published=None, comment=None, venue_hint=None, venue=None, source=""):
    return {
        "title": " ".join((title or "").split()),
        "authors": [a for a in authors if a],
        "year": int(year) if year else None,
        "abstract": " ".join((abstract or "").split()),
        "ids": {k: v for k, v in (ids or {}).items() if v},
        "url": url,
        "pdf_url": pdf_url,
        "published": published,
        "comment": comment or None,
        "venue_hint": venue_hint or None,
        "venue": venue,
        "sources": [source] if source else [],
    }


# ---------------------------------------------------------------- arXiv (Atom API)
def parse_arxiv_atom(xml_bytes):
    root = ET.fromstring(xml_bytes)
    out = []
    for e in root.findall(f"{ATOM}entry"):
        m = re.search(r"arxiv\.org/abs/(\S+?)(v\d+)?$", e.findtext(f"{ATOM}id", "").strip())
        if not m:
            continue  # the API returns an error as a pseudo-entry
        aid = m.group(1)
        pdf = next((l.get("href") for l in e.findall(f"{ATOM}link") if l.get("title") == "pdf"),
                   f"https://arxiv.org/pdf/{aid}")
        published = (e.findtext(f"{ATOM}published") or "")[:10] or None
        out.append(record(
            e.findtext(f"{ATOM}title", ""),
            [a.findtext(f"{ATOM}name", "").strip() for a in e.findall(f"{ATOM}author")],
            published[:4] if published else None,
            e.findtext(f"{ATOM}summary", ""),
            {"arxiv": aid, "doi": (e.findtext(f"{ARXIV}doi") or "").strip()},
            url=f"https://arxiv.org/abs/{aid}", pdf_url=pdf, published=published,
            comment=" ".join(filter(None, [(e.findtext(f"{ARXIV}comment") or "").strip(),
                                           (e.findtext(f"{ARXIV}journal_ref") or "").strip()])),
            source="arxiv",
        ))
    return out


def arxiv_window_query(categories, start_day, end_day):
    """Submissions in [start_day, end_day] (datetime.date, UTC) in any of the categories."""
    cats = " OR ".join(f"cat:{c}" for c in categories)
    return (f"({cats}) AND submittedDate:[{start_day:%Y%m%d}0000 TO {end_day:%Y%m%d}2359]")


def arxiv_search(http, query, max_results=2000, page=200):
    for start in range(0, max_results, page):
        recs = parse_arxiv_atom(http.get("https://export.arxiv.org/api/query", {
            "search_query": query, "start": start, "max_results": page,
            "sortBy": "submittedDate", "sortOrder": "descending"}))
        yield from recs
        if len(recs) < page:
            break


# ---------------------------------------------------------------- OpenReview (API v2)
def _or_value(content, key):
    v = content.get(key)
    return v.get("value") if isinstance(v, dict) else v


def parse_openreview_notes(obj, source="openreview"):
    out = []
    for n in obj.get("notes", []):
        c = n.get("content", {}) or {}
        title = _or_value(c, "title")
        if not title:
            continue
        forum = n.get("forum") or n.get("id")
        pdf = _or_value(c, "pdf")
        if pdf and pdf.startswith("/"):
            pdf = "https://openreview.net" + pdf
        out.append(record(title, _or_value(c, "authors") or [], None, _or_value(c, "abstract") or "",
                          {"openreview": forum}, url=f"https://openreview.net/forum?id={forum}",
                          pdf_url=pdf or f"https://openreview.net/pdf?id={forum}",
                          venue_hint=_or_value(c, "venue"), source=source))
    return out


def openreview_listing(http, venueid, page=1000):
    """Accepted papers of a venue: notes whose content.venueid equals the venue id."""
    offset = 0
    while True:
        obj = http.get_json("https://api2.openreview.net/notes",
                            {"content.venueid": venueid, "limit": page, "offset": offset})
        yield from parse_openreview_notes(obj)
        if len(obj.get("notes", [])) < page:
            break
        offset += page


# ---------------------------------------------------------------- CVF / ECVA open access listings
class _PTitleParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_dt = self.in_a = False
        self.href, self.buf, self.items = None, [], []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "dt" and "ptitle" in (a.get("class") or ""):
            self.in_dt = True
        elif tag == "a" and self.in_dt:
            self.in_a, self.href, self.buf = True, a.get("href"), []

    def handle_endtag(self, tag):
        if tag == "a" and self.in_a:
            self.items.append((" ".join("".join(self.buf).split()), self.href))
            self.in_a = False
        elif tag == "dt":
            self.in_dt = False

    def handle_data(self, data):
        if self.in_a:
            self.buf.append(data)


def parse_ptitle_listing(html_text, base_url, href_contains=None):
    """CVF (openaccess.thecvf.com/<CONF>?day=all) and ECVA (ecva.net/papers.php) list
    papers as <dt class="ptitle"><a href=...>Title</a></dt>. Returns [(title, abs_url)]."""
    p = _PTitleParser()
    p.feed(html_text)
    return [(t, urljoin(base_url, h)) for t, h in p.items
            if t and h and (not href_contains or href_contains in h)]


def cvf_pdf_url(html_url):
    """.../content/CVPR2025/html/X_CVPR_2025_paper.html -> .../papers/X_CVPR_2025_paper.pdf"""
    if "/html/" in html_url:
        return re.sub(r"\.(html|php)$", ".pdf", html_url.replace("/html/", "/papers/"))
    return None


def find_pdf_link(html_text, base_url):
    m = re.search(r'href=["\']?([^"\'\s>]+\.pdf)', html_text, re.I)
    return urljoin(base_url, m.group(1)) if m else None


_ABS_RE = re.compile(r'<div[^>]*id="abstract"[^>]*>(.*?)</div>', re.S | re.I)


def parse_abstract_page(html_text):
    m = _ABS_RE.search(html_text)
    return " ".join(re.sub(r"<[^>]+>", " ", m.group(1)).split()) if m else ""


# ---------------------------------------------------------------- Semantic Scholar
def parse_s2_paper(p, source="s2"):
    if not p or not p.get("title"):
        return None
    ext = p.get("externalIds") or {}
    pv = p.get("publicationVenue") or {}
    return record(p["title"], [a.get("name") for a in (p.get("authors") or [])], p.get("year"),
                  p.get("abstract") or "",
                  {"s2": p.get("paperId"), "arxiv": ext.get("ArXiv"), "doi": ext.get("DOI"),
                   "dblp": ext.get("DBLP")},
                  url=p.get("url"), published=p.get("publicationDate"),
                  venue_hint=pv.get("name") or p.get("venue") or None, source=source)


def s2_neighbors(http, paper_id, direction="citations", max_n=1000, page=1000):
    """direction: 'citations' (papers citing it) or 'references' (papers it cites)."""
    key = "citingPaper" if direction == "citations" else "citedPaper"
    offset = 0
    while offset < max_n:
        obj = http.get_json(f"{S2_API}/paper/{paper_id}/{direction}",
                            {"fields": S2_FIELDS, "limit": page, "offset": offset})
        data = obj.get("data") or []
        for row in data:
            r = parse_s2_paper(row.get(key), source=f"s2-{direction}")
            if r:
                yield r
        if obj.get("next") is None or not data:
            break
        offset = obj["next"]


def s2_bulk_search(http, query, year="2022-", venues=None, max_n=500):
    params = {"query": query, "fields": S2_FIELDS, "year": year}
    if venues:
        params["venue"] = ",".join(venues)
    n, token = 0, None
    while n < max_n:
        obj = http.get_json(f"{S2_API}/paper/search/bulk", {**params, **({"token": token} if token else {})})
        for p in obj.get("data") or []:
            r = parse_s2_paper(p, source="s2-search")
            if r:
                n += 1
                yield r
        token = obj.get("token")
        if not token:
            break


def s2_match_title(http, title):
    try:
        obj = http.get_json(f"{S2_API}/paper/search/match", {"query": title, "fields": S2_FIELDS})
    except Exception:  # 404 = no match
        return None
    data = obj.get("data") or []
    return parse_s2_paper(data[0], source="s2-match") if data else None


# ---------------------------------------------------------------- dblp
def parse_dblp_hits(obj):
    hits = ((obj.get("result") or {}).get("hits") or {}).get("hit") or []
    out = []
    for h in hits:
        info = h.get("info") or {}
        venue = info.get("venue")
        if isinstance(venue, list):
            venue = venue[0]
        authors = (info.get("authors") or {}).get("author") or []
        if isinstance(authors, dict):
            authors = [authors]
        out.append({"title": info.get("title") or "", "venue": venue or "",
                    "year": int(info["year"]) if str(info.get("year", "")).isdigit() else None,
                    "key": info.get("key") or "", "type": info.get("type") or "",
                    "doi": info.get("doi"), "url": info.get("url"),
                    "authors": [a.get("text", "") if isinstance(a, dict) else str(a) for a in authors]})
    return out


def dblp_search(http, title, h=5):
    q = " ".join(re.sub(r"[^\w\s-]", " ", title).split())
    return parse_dblp_hits(http.get_json("https://dblp.org/search/publ/api",
                                         {"q": q, "format": "json", "h": h}))
