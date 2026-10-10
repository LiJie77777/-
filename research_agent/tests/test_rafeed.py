"""Offline tests for rafeed (run: python research_agent/tests/test_rafeed.py).
No network: every HTTP call is served from tests/fixtures by FakeHttp."""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from rafeed import cli, render, sources, store, validate, venue  # noqa: E402
from rafeed.net import FakeHttp  # noqa: E402

FX = os.path.join(HERE, "fixtures")


def fx(name):
    return open(os.path.join(FX, name), "rb").read()


VENUES = {
    "top": ["CVPR", "ICCV", "ECCV", "NeurIPS", "ICLR", "ICML", "AAAI", "IJCAI", "ACMMM"],
    "vertical": ["MICCAI", "ICRA", "IROS"],
    "aliases": {
        "CVPR": ["cvpr", "conference on computer vision and pattern recognition", "computer vision and pattern recognition"],
        "ICCV": ["iccv", "international conference on computer vision"],
        "ECCV": ["eccv", "european conference on computer vision"],
        "NeurIPS": ["neurips", "nips", "neural information processing systems"],
        "ICLR": ["iclr", "international conference on learning representations"],
        "AAAI": ["aaai"],
        "ACMMM": ["acm multimedia", "acm mm", "acm international conference on multimedia"],
        "MICCAI": ["miccai", "medical image computing and computer assisted intervention"],
        "IROS": ["iros", "intelligent robots and systems"],
    },
}
BOOK = venue.VenueBook(VENUES)


# ---------------------------------------------------------------- parsers
def test_parse_arxiv():
    recs = sources.parse_arxiv_atom(fx("arxiv.xml"))
    assert [r["ids"]["arxiv"] for r in recs] == ["2610.01234", "2610.05678", "2610.09999"]
    r = recs[0]
    assert r["title"] == "Depth-Gated Attention for Training-Free Open-Vocabulary Segmentation"
    assert r["year"] == 2026 and r["authors"] == ["Ada Lovelace", "Alan Turing"]
    assert "Accepted to CVPR 2026 Findings" in r["comment"] and r["pdf_url"].endswith("2610.01234v2")


def test_parse_openreview():
    recs = sources.parse_openreview_notes(json.loads(fx("openreview.json")))
    assert len(recs) == 2 and recs[0]["ids"] == {"openreview": "abcd1234"}
    assert recs[0]["pdf_url"] == "https://openreview.net/pdf/0123456789abcdef.pdf"
    assert recs[0]["venue_hint"] == "ICLR 2026 Poster"
    assert recs[1]["pdf_url"] == "https://openreview.net/pdf?id=efgh5678"


def test_parse_cvf_listing_and_pdf():
    items = sources.parse_ptitle_listing(fx("cvf.html").decode(), "https://openaccess.thecvf.com/CVPR2026?day=all",
                                         "/content/CVPR2026/")
    assert [t for t, _ in items] == ["Depth Anything Meets SAM for Open-Vocabulary Segmentation", "Efficient Video Diffusion"]
    url = items[0][1]
    assert url.startswith("https://openaccess.thecvf.com/content/CVPR2026/html/")
    assert sources.cvf_pdf_url(url).endswith("/content/CVPR2026/papers/Smith_Depth_Anything_Meets_SAM_for_Open_Vocabulary_Segmentation_CVPR_2026_paper.pdf")
    assert sources.parse_abstract_page(fx("cvf_paper.html").decode()).startswith("We fuse SAM regions")
    assert sources.find_pdf_link(fx("cvf.html").decode(), url).endswith("_paper.pdf")


def test_parse_dblp_and_s2():
    hits = sources.parse_dblp_hits(json.loads(fx("dblp.json")))
    assert hits[1]["key"] == "conf/iccv/Carson25" and hits[1]["authors"] == ["Rachel Carson"] and hits[1]["year"] == 2025
    rows = json.loads(fx("s2_citations.json"))["data"]
    r = sources.parse_s2_paper(rows[0]["citingPaper"])
    assert r["ids"] == {"s2": "s2aaa", "arxiv": "2601.00001", "dblp": "conf/cvpr/Foo26"}
    assert r["venue_hint"] == "Computer Vision and Pattern Recognition"


# ---------------------------------------------------------------- venue logic
def test_classify_venues():
    cases = {
        "CVPR": ("CVPR", "main", "top"),
        "IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops": ("CVPR", "workshop", "workshop"),
        "ECCV (12)": ("ECCV", "main", "top"),
        "MICCAI (3)": ("MICCAI", "main", "vertical"),
        "CVPR Findings": ("CVPR", "findings", "findings"),
        "Proceedings of the 31st ACM International Conference on Multimedia": ("ACMMM", "main", "top"),
        "Medical Image Computing and Computer-Assisted Intervention": ("MICCAI", "main", "vertical"),
        "2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)": ("IROS", "main", "vertical"),
        "ICLR 2026 Poster": ("ICLR", "main", "top"),
    }
    for text, (canon, kind, tier) in cases.items():
        c, k = BOOK.classify(text)
        assert (c, k, BOOK.tier(c, k)) == (canon, kind, tier), (text, c, k)
    for text in ("arXiv.org", "IEEE Transactions on Pattern Analysis and Machine Intelligence",
                 "Asian Conference on Computer Vision", "WACV", "BMVC"):
        assert BOOK.classify(text)[0] is None, text


def test_arxiv_comment_and_dblp_and_better():
    v = venue.from_arxiv_comment({"comment": "Accepted to CVPR 2026 Findings. 12 pages"}, BOOK, "2026-10-10")
    assert (v["name"], v["year"], v["kind"], v["status"]) == ("CVPR", 2026, "findings", "likely")
    assert venue.tag(v) == "[CVPR'26 Findings?]"
    assert venue.from_arxiv_comment({"comment": "Accepted by ICCV'25 workshop"}, BOOK, "d")["tier"] == "workshop"
    assert venue.from_arxiv_comment({"comment": "10 pages, 5 figures"}, BOOK, "d") is None
    rec = {"title": "Underwater salient object segmentation with relative depth order"}
    d = venue.from_dblp(sources.parse_dblp_hits(json.loads(fx("dblp.json"))), rec, BOOK, "d")
    assert (d["name"], d["year"], d["status"], d["tier"]) == ("ICCV", 2025, "verified", "top")
    assert d["evidence"] == "https://dblp.org/rec/conf/iccv/Carson25" and venue.tag(d) == "[ICCV'25]"
    assert venue.from_dblp([], rec, BOOK, "d") is None
    assert venue.better(d, v) is d and venue.better(v, d) is d and venue.better(None, v) is v


def test_titles_match():
    assert venue.titles_match("DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation.",
                              "DFormerv2 - geometry self-attention for RGBD semantic segmentation")
    assert not venue.titles_match("Depth helps segmentation", "Depth hurts detection")


# ---------------------------------------------------------------- registry / prefilter
def test_registry_dedup_and_merge():
    tmp = tempfile.mkdtemp()
    try:
        reg = store.Registry(os.path.join(tmp, "registry.jsonl"))
        a = sources.record("Depth Helps: A Study", ["X"], 2025, "short", {"arxiv": "2501.00001"}, source="arxiv")
        pid = reg.upsert(a, "2026-01-01")
        assert pid == "arxiv:2501.00001"
        b = sources.record("Depth helps - a study.", [], 2025, "a much longer abstract here",
                           {"doi": "10.1/XYZ", "s2": "s2x"}, source="s2-citations")
        b["venue"] = BOOK.entry("CVPR", 2025, "main", "verified", "https://dblp.org/rec/conf/cvpr/x", "d")
        assert reg.upsert(b, "2026-01-02") == pid          # same paper via normalized title
        c = sources.record("Different title", ids={"doi": "10.1/xyz"})
        assert reg.find(c) == pid                         # DOI alias, case-insensitive
        reg.save()
        r = store.Registry(os.path.join(tmp, "registry.jsonl")).recs[pid]
        assert r["ids"] == {"arxiv": "2501.00001", "doi": "10.1/XYZ", "s2": "s2x"}
        assert r["abstract"] == "a much longer abstract here" and r["venue"]["status"] == "verified"
        assert r["sources"] == ["arxiv", "s2-citations"] and r["first_seen"] == "2026-01-01"
    finally:
        shutil.rmtree(tmp)


def test_keyword_matching():
    assert store.kw_match("fusion", "feature fusion for segmentation")
    assert not store.kw_match("fusion", "efficient video diffusion")
    assert store.kw_match("sam", "we prompt sam with depth") and not store.kw_match("sam", "the same model")
    assert store.kw_match("geometr*", "geometric priors") and not store.kw_match("geometr", "geometric priors")
    assert store.kw_match("depth", "depth-aware attention") and not store.kw_match("depth", "depthwise convolution")
    assert store.kw_match("open-vocabulary", "an open-vocabulary segmenter")


def test_prefilter():
    topics = {"geometry": {"include": ["depth"]}, "underwater": {"include": ["underwater"]}}
    req = ["segment*"]
    assert store.prefilter({"title": "Depth for segmentation", "abstract": "x", "year": 2024}, topics, req, 2022) == ["geometry"]
    assert store.prefilter({"title": "Depth for detection", "abstract": "boxes", "year": 2024}, topics, req, 2022) == []
    assert store.prefilter({"title": "Underwater depth", "abstract": "", "year": 2024}, topics, req, 2022) == ["geometry", "underwater"]
    assert store.prefilter({"title": "Depth segmentation", "abstract": "", "year": 2019}, topics, req, 2022) == []


# ---------------------------------------------------------------- validation
TEXT = fx("paper.txt").decode()
GOOD_CARD = {
    "id": "arxiv:2610.01234", "one_line": "x", "problem": "x", "method": "x", "setting": "training-free",
    "geometry_use": {"info": ["I1"], "injection": ["attention"], "notes": ""},
    "claims": [
        {"claim": "深度门控提升 mIoU", "kind": "positive", "location": "Sec. 4.2",
         "quote": "Adding the depth gate improves mIoU from 40.87 to 42.31 on ADE20K",
         "numbers": [{"metric": "mIoU", "dataset": "ADE20K", "value": "42.31", "baseline": "40.87", "setting": "full"}]},
        {"claim": "识别不变", "kind": "neutral", "location": "Sec. 4.2",
         "quote": "Recognition accuracy is unchanged (80.02 vs. 80.00) when depth features are added to the classifier.",
         "numbers": [{"metric": "acc", "dataset": "-", "value": "80.00", "baseline": "80.02", "setting": "-"}]},
        {"claim": "SUIM 58.0", "kind": "positive", "location": "Table 3",
         "quote": "Table 3: Results on SUIM. Our method obtains 58.0 mIoU.",
         "numbers": [{"metric": "mIoU", "dataset": "SUIM", "value": "58.0", "baseline": None, "setting": "-"}]},
        {"claim": "编造的结论", "kind": "positive", "location": "Sec. 5",
         "quote": "Depth improves recognition by five points on every dataset we tried",
         "numbers": [{"metric": "acc", "dataset": "all", "value": "+5.0", "baseline": None, "setting": "-"}]},
    ],
    "negative_or_limits": [
        {"claim": "增益主要来自局部性", "kind": "negative", "location": "Sec. 4.2",
         "quote": "a spatial-only prior already reaches 41.95, which suggests that most of the gain comes from locality",
         "numbers": [{"metric": "mIoU", "dataset": "ADE20K", "value": "41.95", "baseline": None, "setting": "spatial-only"}]}],
    "controls": {"same_arch_non_geometric": "no", "spatial_only": "yes", "shuffled_or_mismatched_depth": "no",
                 "duplicate_feature_or_capacity": "no", "depth_quality_variation": "unclear", "notes": ""},
    "relevance": {"score": 3, "to_project": "x", "which_cards": ["G2", "Z9"]},
    "rule_links": [{"rule": "R5", "stance": "supports", "why": "spatial-only"}, {"rule": "R42", "stance": "supports", "why": ""}],
    "open_questions": [],
}
REG = {"id": "arxiv:2610.01234", "title": "Depth-Gated Attention", "ids": {"arxiv": "2610.01234"},
       "venue": {"name": "CVPR", "year": 2026, "kind": "findings", "tier": "findings", "status": "likely",
                 "evidence": "arXiv comment", "checked": "d"}, "url": "https://arxiv.org/abs/2610.01234"}


def test_quote_matching_tolerates_pdf_artifacts():
    t = validate.canon_text(TEXT)
    assert validate.quote_in_text("when depth features are added to the classifier", t)       # de-hyphenation
    assert validate.quote_in_text("improves mIoU from 40.87 to 42.31 on ADE20K, while a spatial-only prior", t)  # line break
    assert not validate.quote_in_text("Depth improves recognition by five points", t)
    assert not validate.quote_in_text("Ablation", t)                                              # too short
    assert validate.numbers_in_text("42.31", t) and not validate.numbers_in_text("2.31", t)
    assert not validate.numbers_in_text("+5.0", t) and not validate.numbers_in_text("n/a", t)


def test_validate_card_accepts_and_downgrades():
    card, rep, ok = validate.validate_card(json.loads(json.dumps(GOOD_CARD)), TEXT, REG)
    assert ok, rep
    tags = [c["tag"] for c in card["claims"]]
    assert tags == ["文献事实", "文献事实", "文献事实", "未核实"]
    assert card["negative_or_limits"][0]["tag"] == "文献事实"
    assert rep["claims"] == 5 and rep["quotes_ok"] == 4 and rep["numbers_ok"] == 4
    assert card["relevance"]["which_cards"] == ["G2"] and [l["rule"] for l in card["rule_links"]] == ["R5"]
    assert card["venue"] == REG["venue"] and card["title"] == REG["title"]   # registry owns bibliography
    md = render.card_markdown(card)
    assert "[CVPR'26 Findings?]" in md and "【未核实】编造的结论" in md and "引文 4/5" in md


def test_validate_card_rejects():
    bad = json.loads(json.dumps(GOOD_CARD))
    for c in bad["claims"][:3]:
        c["quote"] = "this sentence does not appear anywhere in the paper at all"
    _, rep, ok = validate.validate_card(bad, TEXT, REG)
    assert not ok and rep["quotes_ok"] == 1
    _, _, ok = validate.validate_card(dict(GOOD_CARD, id="arxiv:other"), TEXT, REG)
    assert not ok
    _, _, ok = validate.validate_card(dict(GOOD_CARD, claims=[], negative_or_limits=[]), TEXT, REG)
    assert not ok


def test_validate_screen():
    obj = {"items": [
        {"id": "a", "relevance": 3, "topics": ["geometry", "nope"], "evidence_type": "method", "deep_read": True, "reason": "r"},
        {"id": "b", "relevance": 1, "topics": [], "evidence_type": "other", "deep_read": True, "reason": "r"},
        {"id": "a", "relevance": 2, "topics": [], "evidence_type": "other", "deep_read": False, "reason": "dup"},
        {"id": "zzz", "relevance": 2, "topics": [], "evidence_type": "other", "deep_read": False, "reason": "x"},
        {"id": "c", "relevance": 7, "topics": [], "evidence_type": "other", "deep_read": False, "reason": "x"},
    ]}
    items, errors, missing = validate.validate_screen(obj, {"a", "b", "c", "d"}, {"geometry"})
    assert [i["id"] for i in items] == ["a", "b"]
    assert items[0]["topics"] == ["geometry"] and items[1]["deep_read"] is False
    assert len(errors) == 3 and missing == ["c", "d"]


def test_schemas_are_strict():
    for n in ("screen", "card", "expand"):
        sch = json.load(open(os.path.join(os.path.dirname(HERE), "schemas", f"{n}.schema.json")))
        assert validate.schema_is_strict(sch) == [], n


def test_strip_references():
    t = "x" * 1000 + "\nReferences\n[1] a\n[2] b\nAppendix\nA. More ablations\n"
    s = cli.strip_references(t)
    assert "[1] a" not in s and "More ablations" in s
    assert cli.strip_references("References\nearly") == "References\nearly"


# ---------------------------------------------------------------- end to end (offline)
CONFIG = """
[project]
name = "test"
contact = ""
min_year = 2022
[codex]
model = "test-model"
max_paper_chars = 100000
[http.min_interval]
"*" = 0
[arxiv]
categories = ["cs.CV"]
window_days = 3
max_results = 200
[verify]
max_dblp_per_run = 50
recheck_days = 30
max_recheck_per_run = 10
[snowball]
seeds = ["arXiv:2504.04701"]
max_papers_per_run = 5
max_neighbors_per_paper = 100
min_relevance = 2
[s2]
venues = ["CVPR"]
[prefilter]
require = ["segment*", "salien*"]
[topics.geometry]
desc = "depth"
include = ["depth", "geometr*"]
[topics.ovss]
desc = "ovss"
include = ["open-vocabulary", "training-free"]
[topics.underwater]
desc = "uw"
include = ["underwater", "marine"]
[[listings]]
kind = "openreview"
venueid = "ICLR.cc/2026/Conference"
venue = "ICLR"
year = 2026
[[listings]]
kind = "ptitle"
url = "https://openaccess.thecvf.com/CVPR2026?day=all"
href_contains = "/content/CVPR2026/"
venue = "CVPR"
year = 2026
""" + "[venues]\ntop = %s\nvertical = %s\n[venues.aliases]\n" % (json.dumps(VENUES["top"]), json.dumps(VENUES["vertical"])) \
    + "".join(f"{k} = {json.dumps(v)}\n" for k, v in VENUES["aliases"].items())


def _fake():
    return FakeHttp([
        ("export.arxiv.org/api/query", fx("arxiv.xml")),
        ("api2.openreview.net/notes", fx("openreview.json")),
        ("/content/CVPR2026/html/Smith_", fx("cvf_paper.html")),
        ("openaccess.thecvf.com/CVPR2026", fx("cvf.html")),
        ("dblp.org/search/publ/api", fx("dblp.json")),
        ("arXiv:2504.04701/citations", fx("s2_citations.json")),
        ("arXiv:2504.04701/references", b'{"data": []}'),
    ])


def _run(tmp, *argv, http=None):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cli.main(["--config", os.path.join(tmp, "config.toml"), "--data", os.path.join(tmp, "data"),
                  "--state", os.path.join(tmp, "state"), "--date", "2026-10-10", *argv], http=http or _fake())
    return buf.getvalue()


def test_end_to_end_offline():
    tmp = tempfile.mkdtemp()
    try:
        open(os.path.join(tmp, "config.toml"), "w").write(CONFIG)
        inbox = os.path.join(tmp, "state", "inbox", "2026-10-10")
        _run(tmp, "harvest")
        _run(tmp, "snowball")
        _run(tmp, "enrich")
        out = _run(tmp, "verify")
        cands = {c["title"]: c for c in store.read_jsonl(os.path.join(inbox, "candidates.jsonl"))}
        assert set(cands) == {
            "Depth-Gated Attention for Training-Free Open-Vocabulary Segmentation",
            "Underwater Salient Object Segmentation with Relative Depth Order",
            "Geometry Priors Do Not Help Recognition: A Controlled Study of Depth in Open-Vocabulary Segmentation",
            "Depth Anything Meets SAM for Open-Vocabulary Segmentation",
            "Relative Depth Cues for Open-Vocabulary Segmentation of Marine Scenes"}, out
        tags = {t: venue.tag(c["venue"]) for t, c in cands.items()}
        assert tags["Depth-Gated Attention for Training-Free Open-Vocabulary Segmentation"] == "[CVPR'26 Findings?]"
        assert tags["Underwater Salient Object Segmentation with Relative Depth Order"] == "[ICCV'25]"
        assert tags["Depth Anything Meets SAM for Open-Vocabulary Segmentation"] == "[CVPR'26]"
        assert tags["Relative Depth Cues for Open-Vocabulary Segmentation of Marine Scenes"] == "[CVPR'26?]"
        assert cands["Depth Anything Meets SAM for Open-Vocabulary Segmentation"]["abstract"].startswith("We fuse SAM")

        chunk = _run(tmp, "screen-input", "--chunk", "50").split()[0]
        rows = store.read_jsonl(chunk)
        prompt = _run(tmp, "prompt", "--kind", "screen", "--input", chunk)
        assert chunk in prompt and "- geometry: depth" in prompt and "（暂无反馈）" in prompt
        assert rows[0]["title"] in prompt and "R1 深度主要改善定位" in prompt   # rows and context.md embedded

        rel = {"Depth-Gated": 3, "Geometry Priors": 3, "Underwater": 2, "Depth Anything": 1, "Relative Depth": 0}
        items = [{"id": r["id"], "relevance": next(v for k, v in rel.items() if r["title"].startswith(k)),
                  "topics": ["geometry"], "evidence_type": "method", "deep_read": True, "reason": "test"} for r in rows]
        json.dump({"items": items}, open(os.path.join(inbox, "screen.out.json"), "w"))
        _run(tmp, "apply-screen", "--file", os.path.join(inbox, "screen.out.json"), "--input", chunk)
        reg = store.Registry(os.path.join(tmp, "data", "registry.jsonl"))
        assert len(reg.recs) == 5
        low = next(r for r in reg.recs.values() if r["title"].startswith("Relative Depth"))
        assert low["abstract"] == "" and low["screen"]["relevance"] == 0

        picks = _run(tmp, "select-deep", "--k", "3").split()
        assert picks == ["openreview:abcd1234", "arxiv:2610.01234", "arxiv:2610.09999"]  # verified before likely
        trap = os.path.join(tmp, "trap.txt")
        open(trap, "w").write("Body text mentioning {{FEEDBACK}} literally.")
        ep = _run(tmp, "prompt", "--kind", "extract", "--id", picks[1], "--input", trap)
        assert '"id": "arxiv:2610.01234"' in ep and "Body text mentioning {{FEEDBACK}} literally." in ep
        assert ep.count("{{") == 1   # only the literal inside the paper text survives

        os.makedirs(os.path.join(tmp, "state", "text"), exist_ok=True)
        open(os.path.join(tmp, "state", "text", "arxiv_2610.01234.txt"), "w").write(TEXT)
        open(os.path.join(tmp, "state", "text", "openreview_abcd1234.txt"), "w").write("Unrelated text only.")
        json.dump(GOOD_CARD, open(os.path.join(inbox, "card_good.json"), "w"))
        json.dump(dict(GOOD_CARD, id="openreview:abcd1234"), open(os.path.join(inbox, "card_bad.json"), "w"))
        assert _run(tmp, "apply-card", "--id", "arxiv:2610.01234", "--file", os.path.join(inbox, "card_good.json")).startswith("accepted")
        assert _run(tmp, "apply-card", "--id", "openreview:abcd1234", "--file", os.path.join(inbox, "card_bad.json")).startswith("REJECTED")
        assert os.path.exists(os.path.join(tmp, "data", "cards", "arxiv_2610.01234.md"))
        assert os.path.exists(os.path.join(tmp, "data", "rejected", "openreview_abcd1234.json"))
        card = json.load(open(os.path.join(tmp, "data", "cards", "arxiv_2610.01234.json")))
        assert card["venue"]["name"] == "CVPR" and card["model"] == ""

        _run(tmp, "render")
        dg = open(os.path.join(tmp, "data", "digests", "2026-10-10.md"), encoding="utf-8").read()
        assert "调研日报 2026-10-10" in dg and "[ICLR'26]" in dg and "Depth-Gated" in dg and "被拒的卡片" in dg
        assert _run(tmp, "select-deep", "--k", "5").split() == ["arxiv:2610.09999"]   # carded/rejected leave the queue
        _run(tmp, "mark-failed", "--id", "arxiv:2610.09999", "--what", "extract")
        assert _run(tmp, "select-deep", "--k", "5").split() == ["arxiv:2610.09999"]   # one failure: retried
        _run(tmp, "mark-failed", "--id", "arxiv:2610.09999", "--what", "fetch")
        assert _run(tmp, "select-deep", "--k", "5").split() == []                     # two failures: dropped
        # a second harvest the same day re-offers nothing that is already screened
        _run(tmp, "harvest")
        assert _run(tmp, "screen-input", "--chunk", "50").strip() == ""
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            n += 1
            print("ok", name)
    print(f"{n} tests passed")
