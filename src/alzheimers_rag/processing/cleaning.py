"""Reconstruct BioC sections and classify research content."""

import re
from collections import Counter
from dataclasses import dataclass, field

DROP_SECTION_TYPES = {"REF", "COMP_INT", "ACK_FUND", "AUTH_CONT", "REVIEW_INFO", "ABBR"}
HEADING_TYPE_RE = re.compile("^(abstract_)?title_(\\d+)$")


@dataclass
class Section:
    heading_text: str | None
    level: int
    section_type: str
    paragraphs: list[dict] = field(default_factory=list)
    children: list["Section"] = field(default_factory=list)
    synthetic: bool = False

    def merged_text(self) -> str:
        return "\n\n".join((p["text"] for p in self.paragraphs))


def heading_level(passage_type: str) -> int | None:
    m = HEADING_TYPE_RE.match(passage_type)
    return int(m.group(2)) if m else None


def build_section_tree(passages: list[dict]) -> Section:
    root = Section(heading_text=None, level=0, section_type="ROOT", synthetic=True)
    stack: list[Section] = [root]
    passages = sorted(passages, key=lambda p: p.get("offset", 0))
    for p in passages:
        infons = p.get("infons", {})
        section_type = infons.get("section_type", "UNKNOWN")
        ptype = infons.get("type", "")
        text = p.get("text", "").strip()
        if section_type in DROP_SECTION_TYPES or not text:
            continue
        lvl = heading_level(ptype)
        if lvl is not None:
            while len(stack) - 1 > lvl - 1:
                stack.pop()
            while len(stack) - 1 < lvl - 1:
                synth = Section(None, len(stack), section_type, synthetic=True)
                stack[-1].children.append(synth)
                stack.append(synth)
            new_section = Section(text, lvl, section_type, synthetic=False)
            stack[-1].children.append(new_section)
            stack.append(new_section)
        else:
            if ptype == "front":
                title_section = Section(None, 1, "TITLE", synthetic=False)
                title_section.paragraphs.append({"type": ptype, "text": text})
                root.children.append(title_section)
                continue
            if len(stack) == 1:
                synth = Section(None, 1, section_type, synthetic=True)
                root.children.append(synth)
                stack.append(synth)
            elif stack[-1].section_type != section_type:
                while len(stack) > 1:
                    stack.pop()
                synth = Section(None, 1, section_type, synthetic=True)
                stack[-1].children.append(synth)
                stack.append(synth)
            stack[-1].paragraphs.append({"type": ptype, "text": text})
    return root


def flatten_sections(node: Section, path: list[str] | None = None) -> list[dict]:
    path = path or []
    out = []
    current_path = path + ([node.heading_text] if node.heading_text else [])
    if node.paragraphs:
        out.append(
            {
                "heading_path": [h for h in current_path if h],
                "section_type": node.section_type,
                "level": node.level,
                "paragraphs": node.paragraphs,
                "text": node.merged_text(),
                "synthetic": node.synthetic,
            }
        )
    for child in node.children:
        out.extend(flatten_sections(child, current_path))
    return out


CANONICAL_SECTION_MAP = {
    "TITLE": "title",
    "ABSTRACT": "abstract",
    "INTRO": "introduction",
    "METHODS": "methods",
    "RESULTS": "results",
    "DISCUSS": "discussion",
    "CONCL": "conclusion",
    "CASE": "case_report",
    "KEYWORD": "keywords",
    "APPENDIX": "appendix",
}

TABLE_KEEP_TYPE_MAP = {
    "table_caption": "table_caption",
    "table_title_caption": "table_caption",
    "table_footnote": "table_footnote",
    "table_foot": "table_footnote",
}
TABLE_DROP_TYPES = {"table"}
TABLE_JUNK_RE = re.compile("^\\s*references?\\s*:?\\.?\\s*$", re.IGNORECASE)


def classify_table_passage(ptype: str, text: str) -> str | None:
    if TABLE_JUNK_RE.match(text):
        return None
    if ptype in TABLE_DROP_TYPES:
        return None
    if ptype in TABLE_KEEP_TYPE_MAP:
        return TABLE_KEEP_TYPE_MAP[ptype]
    return "table_caption"


FIG_KEEP_TYPE_MAP = {
    "fig_caption": "figure_caption",
    "fig_title_caption": "figure_caption",
    "fig_footnote": "figure_footnote",
}
FIG_BARE_LABEL_RE = re.compile("^\\s*fig(ure)?\\.?\\s*\\d+\\.?\\s*$", re.IGNORECASE)


def classify_fig_passage(ptype: str, text: str, diagnostics=None) -> str | None:
    if ptype in FIG_KEEP_TYPE_MAP:
        return FIG_KEEP_TYPE_MAP[ptype]
    if ptype == "fig":
        if FIG_BARE_LABEL_RE.match(text.strip()):
            return None
        return "figure_caption"
    if ptype in ("paragraph", "title_1"):
        return "figure_caption"
    if diagnostics is not None:
        diagnostics["unmapped_fig_types"][ptype] += 1
    return "figure_other"


SUPPL_DROP_HEADING_PATTERNS = re.compile(
    "ethics statement|supplementary material$|conflict of interest|author contribution",
    re.IGNORECASE,
)

SUPPL_JUNK_HEADING_PATTERNS = re.compile(
    "^\\s*$|supplementary (information|material|data|files?)s?:?$|supplemental (information|material|data)s?$|supporting information$|full text availability$|code availability$|data (and (code|software))? ?(sharing|availabilit(y|ty)|of availability)( statement:?)?$|source data$|extended data$|orcid\\s*id s?$|resource availability( (lead|technical) contact| (materials|data and code) availability)?$|declarations?$|declaration of (the use of )?generative ai.*$|consent (statement|for publication|to participate)$|ethic(al|s) (statement|considerations?|approval.*)$|informed consent$|approval and registration number$|studies on human and/or animal statement$|research data transparency and availability$|.*consortium affiliations$orcid\\s+(.*)|.*data.*$|.*supplementary.*$",
    re.IGNORECASE,
)


def classify_suppl_section(heading_path: list[str], diagnostics=None) -> str | None:
    label = " ".join(heading_path).strip() if heading_path else ""
    if SUPPL_JUNK_HEADING_PATTERNS.match(label):
        return None
    if SUPPL_DROP_HEADING_PATTERNS.search(label):
        return None
    if re.search("data availability", label, re.IGNORECASE):
        return "data_availability"
    if re.match("^\\d+(\\.\\d+)*\\.?\\s+\\S", label):
        return "body_misfiled_as_suppl"
    if diagnostics is not None:
        diagnostics["unmapped_suppl_headings"][label] += 1
    return "supplementary_other"


def classify_section(flat_section: dict, diagnostics=None) -> list[dict]:
    section_type = flat_section["section_type"]
    heading_path = flat_section["heading_path"]
    records = []
    if section_type == "TABLE":
        for para in flat_section["paragraphs"]:
            canonical = classify_table_passage(para["type"], para["text"])
            if canonical:
                records.append(
                    _make_record(
                        heading_path,
                        section_type,
                        canonical,
                        flat_section["level"],
                        para["text"],
                        flat_section["synthetic"],
                    )
                )
        return records
    if section_type == "FIG":
        for para in flat_section["paragraphs"]:
            canonical = classify_fig_passage(para["type"], para["text"], diagnostics)
            if canonical:
                records.append(
                    _make_record(
                        heading_path,
                        section_type,
                        canonical,
                        flat_section["level"],
                        para["text"],
                        flat_section["synthetic"],
                    )
                )
        return records
    if section_type == "SUPPL":
        canonical = classify_suppl_section(heading_path, diagnostics)
        if canonical:
            records.append(
                _make_record(
                    heading_path,
                    section_type,
                    canonical,
                    flat_section["level"],
                    flat_section["text"],
                    flat_section["synthetic"],
                )
            )
        return records
    canonical = CANONICAL_SECTION_MAP.get(section_type)
    if canonical is None:
        if diagnostics is not None:
            diagnostics["unmapped_section_types"][section_type] += 1
        canonical = "other"
    records.append(
        _make_record(
            heading_path,
            section_type,
            canonical,
            flat_section["level"],
            flat_section["text"],
            flat_section["synthetic"],
        )
    )
    return records


def _make_record(heading_path, section_type, canonical, level, text, synthetic) -> dict:
    return {
        "heading_path": heading_path,
        "section_type": section_type,
        "canonical_section_type": canonical,
        "level": level,
        "text": text,
        "n_words": len(text.split()),
        "n_chars": len(text),
        "synthetic": synthetic,
    }


def process_document(doc: dict) -> dict:
    doc_id = doc.get("id")
    if not doc_id:
        raise ValueError("BioC document is missing its id")
    diagnostics = {
        name: Counter()
        for name in ("unmapped_section_types", "unmapped_fig_types", "unmapped_suppl_headings")
    }
    passages = doc.get("passages", [])
    meta = {"doi": None, "pmc_id": doc_id, "pmid": None, "year": None, "kwd": [], "title": None}
    for p in passages:
        infons = p.get("infons", {})
        if infons.get("type") == "front":
            meta["doi"] = infons.get("article-id_doi")
            meta["pmc_id"] = infons.get("article-id_pmc", doc_id)
            meta["pmid"] = infons.get("article-id_pmid")
            meta["year"] = infons.get("year")
            raw_kwd = infons.get("kwd") or ""
            meta["kwd"] = (
                [k.strip() for k in re.split("[;,]", raw_kwd) if k.strip()]
                if ";" in raw_kwd or "," in raw_kwd
                else raw_kwd.split()
            )
            meta["title"] = p.get("text", "").strip()
            break
    tree = build_section_tree(passages)
    flat_sections = flatten_sections(tree)
    sections = []
    for flat in flat_sections:
        sections.extend(classify_section(flat, diagnostics))
    for i, s in enumerate(sections):
        s["section_id"] = f"{doc_id}_s{i}"
    n_synthetic = sum((1 for s in sections if s["synthetic"]))
    return {
        "doc_id": doc_id,
        "pmc_id": meta["pmc_id"],
        "doi": meta["doi"],
        "pmid": meta["pmid"],
        "year": meta["year"],
        "kwd": meta["kwd"],
        "title": meta["title"],
        "n_sections": len(sections),
        "n_synthetic_sections": n_synthetic,
        "sections": sections,
        "diagnostics": diagnostics,
    }
