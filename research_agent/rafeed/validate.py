"""Gates between the LLM and the repository. Nothing the model writes reaches the data
branch without passing here.

Card checks (against the pdftotext output of the paper itself):
  * every claim carries a verbatim quote; the quote must be found in the paper text
    (whitespace/hyphenation/case tolerant). Found -> tag 文献事实, not found -> 未核实.
  * every reported number must occur in the paper text (no computed deltas).
  * venue/title come from the registry, never from the model.
  * too many failed quotes -> the card is rejected (kept under research/rejected/)."""
import re
import unicodedata

CARD_IDS = {"D0", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "M1", "M2", "T1", "new"}
RULES = {f"R{i}" for i in range(1, 9)}
EVIDENCE = {"method", "positive", "negative", "analysis", "benchmark", "survey", "other"}
_NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")


def canon_text(s):
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("­", "").replace("-\n", "").replace("ﬁ", "fi").replace("ﬂ", "fl")
    s = s.lower().replace("−", "-").replace("–", "-")
    s = re.sub(r"[^a-z0-9.%+\-]+", " ", s)
    return " ".join(s.split())


def quote_in_text(quote, text_c, min_tokens=4, chunk=5, need=0.8):
    q = canon_text(quote)
    toks = q.split()
    if len(toks) < min_tokens:
        return False
    if q in text_c:
        return True
    # tolerate pdftotext artefacts (column breaks, dropped glyphs): most 5-token chunks
    # must occur, in order, inside a window not much longer than the quote
    chunks = [" ".join(toks[i:i + chunk]) for i in range(0, len(toks), chunk)]
    chunks = [c for c in chunks if len(c.split()) >= 3] or chunks
    pos, found, first = 0, 0, None
    for c in chunks:
        j = text_c.find(c, pos)
        if j >= 0:
            found += 1
            first = j if first is None else first
            pos = j + len(c)
    return found / len(chunks) >= need and first is not None and pos - first <= 3 * len(q) + 50


def numbers_in_text(value, text_c):
    nums = _NUM.findall(value or "")
    if not nums:
        return False
    for n in nums:
        bare = n.lstrip("+")
        if not re.search(rf"(?<![\d.]){re.escape(bare)}(?![\d])", text_c):
            return False
    return True


def validate_screen(obj, candidate_ids, topic_names):
    errors, items = [], []
    seen = set()
    for it in (obj or {}).get("items", []):
        pid = it.get("id")
        if pid not in candidate_ids:
            errors.append(f"unknown id {pid!r}")
            continue
        if pid in seen:
            errors.append(f"duplicate id {pid}")
            continue
        if it.get("relevance") not in (0, 1, 2, 3):
            errors.append(f"{pid}: relevance {it.get('relevance')!r}")
            continue
        if it.get("evidence_type") not in EVIDENCE:
            errors.append(f"{pid}: evidence_type {it.get('evidence_type')!r}")
            continue
        it = dict(it)
        it["topics"] = [t for t in it.get("topics", []) if t in topic_names]
        it["reason"] = (it.get("reason") or "")[:300]
        it["deep_read"] = bool(it.get("deep_read")) and it["relevance"] >= 2
        seen.add(pid)
        items.append(it)
    missing = sorted(set(candidate_ids) - seen)
    return items, errors, missing


def _check_claims(claims, text_c, label, report):
    out = []
    for c in claims or []:
        c = dict(c)
        ok_q = quote_in_text(c.get("quote", ""), text_c)
        nums = c.get("numbers") or []
        bad_nums = [n.get("value") for n in nums if not numbers_in_text(n.get("value"), text_c)]
        c["tag"] = "文献事实" if ok_q and not bad_nums else "未核实"
        c["check"] = {"quote_found": ok_q, "numbers_not_found": bad_nums}
        report["claims"] += 1
        report["quotes_ok"] += int(ok_q)
        report["numbers"] += len(nums)
        report["numbers_ok"] += len(nums) - len(bad_nums)
        if not ok_q:
            report["problems"].append(f"{label}: quote not found: {c.get('quote', '')[:80]!r}")
        for v in bad_nums:
            report["problems"].append(f"{label}: number not in paper: {v!r}")
        out.append(c)
    return out


def validate_card(card, paper_text, reg_rec, max_quote_fail=0.34):
    """Returns (card_with_checks, report, accepted)."""
    report = {"claims": 0, "quotes_ok": 0, "numbers": 0, "numbers_ok": 0, "problems": []}
    text_c = canon_text(paper_text)
    if card.get("id") != reg_rec["id"]:
        report["problems"].append(f"id mismatch {card.get('id')!r} != {reg_rec['id']!r}")
        return card, report, False
    card = dict(card)
    card["claims"] = _check_claims(card.get("claims"), text_c, "claim", report)
    card["negative_or_limits"] = _check_claims(card.get("negative_or_limits"), text_c, "limit", report)
    rel = card.get("relevance") or {}
    rel["which_cards"] = [c for c in rel.get("which_cards", []) if c in CARD_IDS]
    card["relevance"] = rel
    links = []
    for l in card.get("rule_links") or []:
        if l.get("rule") in RULES and l.get("stance") in ("supports", "contradicts", "refines"):
            links.append(l)
        else:
            report["problems"].append(f"bad rule link {l!r}")
    card["rule_links"] = links
    # the registry owns bibliographic facts
    card["title"], card["venue"] = reg_rec["title"], reg_rec.get("venue")
    card["ids"], card["url"] = reg_rec.get("ids", {}), reg_rec.get("url")
    accepted = report["claims"] > 0 and \
        (report["claims"] - report["quotes_ok"]) / report["claims"] <= max_quote_fail
    if report["claims"] == 0:
        report["problems"].append("no claims")
    card["validation"] = {k: v for k, v in report.items() if k != "problems"}
    card["validation"]["problems"] = report["problems"][:30]
    return card, report, accepted


def schema_is_strict(schema, path="$"):
    """Structured-output style check: every object closes additionalProperties and lists
    every property as required. Returns a list of violations."""
    bad = []
    if isinstance(schema, dict):
        if schema.get("type") == "object" or "properties" in schema:
            props = schema.get("properties", {})
            if schema.get("additionalProperties") is not False:
                bad.append(f"{path}: additionalProperties must be false")
            if set(schema.get("required", [])) != set(props):
                bad.append(f"{path}: required must list all properties")
        for k, v in schema.items():
            if isinstance(v, (dict, list)):
                bad += schema_is_strict(v, f"{path}.{k}")
    elif isinstance(schema, list):
        for i, v in enumerate(schema):
            bad += schema_is_strict(v, f"{path}[{i}]")
    return bad
