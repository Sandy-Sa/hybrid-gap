"""Leaf -> block -> chapter lookup for ICD-9-CM/PCS and ICD-10-CM/PCS.

Backs the hierarchical error-analysis notebook. Reference data:

  * ICD-10-CM tabular XML (block = `<section id=...>` literal id; chapter from `<chapter><name>`).
  * ICD-10-PCS tables XML (block = first 3 chars; chapter = first char / Section).
  * ICD-9-CM Tabular RTF (block-range headers; chapter from `guideline_parser.ICD9_CHAPTER_RANGES`).
  * ICD-9-PCS literal Volume 3 chapter dict; block = 2-digit category.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Iterable

import polars as pl

from coding.utils.helpers import (
    is_procedure_code,
    reformat_icd9cm_code,
    reformat_icd9pcs_code,
    reformat_icd10cm_code,
)
from coding.utils.guideline_parser import (
    ICD9_CHAPTER_RANGES,
    ICD10CM_CHAPTER_RANGES,
)

# ---------------------------------------------------------------------------
# Reference data paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ICD10CM_XML = _REPO_ROOT / "data/reference/icd10cm_table_2022/icd10cm_tabular_2022.xml"
_ICD10PCS_XML = _REPO_ROOT / "data/reference/icd10pcs_table_2022/icd10pcs_tables_2022.xml"
_ICD9CM_RTF = _REPO_ROOT / "data/reference/icd9cm.rtf"

# ICD-9-PCS chapters (Volume 3 leading-digit table).
_ICD9PCS_CHAPTERS: dict[int, str] = {
    0: "Procedures Not Elsewhere Classified",
    1: "Operations on the Nervous System",
    2: "Operations on the Endocrine System",
    3: "Operations on the Eye",
    4: "Operations on the Ear",
    5: "Operations on the Nose, Mouth, and Pharynx",
    6: "Operations on the Respiratory System",
    7: "Operations on the Cardiovascular System",
    8: "Operations on the Hemic and Lymphatic System",
    9: "Operations on the Digestive System",
    10: "Operations on the Urinary System",
    11: "Operations on the Male Genital Organs",
    12: "Operations on the Female Genital Organs",
    13: "Obstetrical Procedures",
    14: "Operations on the Musculoskeletal System",
    15: "Operations on the Integumentary System",
    16: "Miscellaneous Diagnostic and Therapeutic Procedures",
}

# Volume-3 leading-digit -> chapter number.
_ICD9PCS_DIGIT_TO_CHAPTER: dict[str, int] = {
    "0": 0,
    "1": 1, "2": 1,
    "3": 2,
    # 4 covers eye (08), ear (18-20); reuse Volume 3 split below.
}
# The Volume 3 first-digit groupings are not a clean 1-1 map; the canonical mapping
# below works on the *2-digit category* not the leading digit. We therefore index
# on category and fall back on a small lookup. Categories outside this table are
# considered chapter 0 ("Procedures Not Elsewhere Classified").
_ICD9PCS_CATEGORY_TO_CHAPTER: dict[int, int] = {
    **{c: 0 for c in range(0, 1)},                         # 00
    **{c: 1 for c in range(1, 6)},                         # 01-05
    **{c: 2 for c in range(6, 8)},                         # 06-07
    **{c: 3 for c in range(8, 17)} | {c: 3 for c in (16,)},  # 08-16 eye
    **{c: 4 for c in (18, 19, 20)},                        # 18-20 ear
    **{c: 5 for c in range(21, 30)},                       # 21-29 nose/mouth/pharynx
    **{c: 6 for c in range(30, 35)},                       # 30-34 respiratory
    **{c: 7 for c in range(35, 40)},                       # 35-39 cardiovascular
    **{c: 8 for c in range(40, 42)},                       # 40-41 hemic/lymphatic
    **{c: 9 for c in range(42, 55)},                       # 42-54 digestive
    **{c: 10 for c in range(55, 60)},                      # 55-59 urinary
    **{c: 11 for c in range(60, 65)},                      # 60-64 male genital
    **{c: 12 for c in range(65, 72)},                      # 65-71 female genital
    **{c: 13 for c in range(72, 76)},                      # 72-75 obstetrical
    **{c: 14 for c in range(76, 85)},                      # 76-84 musculoskeletal
    **{c: 15 for c in range(85, 87)},                      # 85-86 integumentary
    **{c: 16 for c in range(87, 100)},                     # 87-99 misc dx/tx
    17: 3,                                                 # 17 reserved (rare)
}

# ---------------------------------------------------------------------------
# ICD-10-CM
# ---------------------------------------------------------------------------

_icd10cm_block_for_code: dict[str, str] | None = None
_icd10cm_chapter_for_code: dict[str, int] | None = None


def _norm_icd10cm(code: str) -> str:
    """With-dot canonical, uppercase."""
    return reformat_icd10cm_code(code.strip().upper())


def _walk_diags(parent, codes: list[str]) -> None:
    for d in parent.findall("diag"):
        name_el = d.find("name")
        if name_el is not None and name_el.text:
            codes.append(_norm_icd10cm(name_el.text))
        _walk_diags(d, codes)


def _load_icd10cm() -> tuple[dict[str, str], dict[str, int]]:
    global _icd10cm_block_for_code, _icd10cm_chapter_for_code
    if _icd10cm_block_for_code is not None and _icd10cm_chapter_for_code is not None:
        return _icd10cm_block_for_code, _icd10cm_chapter_for_code

    tree = ET.parse(_ICD10CM_XML)
    root = tree.getroot()
    block_for_code: dict[str, str] = {}
    chapter_for_code: dict[str, int] = {}
    for chapter in root.findall("chapter"):
        name_el = chapter.find("name")
        if name_el is None or not name_el.text:
            continue
        chapter_num = int(name_el.text.strip())
        for section in chapter.findall("section"):
            block_id = section.get("id")
            if not block_id:
                continue
            codes: list[str] = []
            _walk_diags(section, codes)
            for c in codes:
                # First-seen wins (chapters do not overlap on real codes).
                block_for_code.setdefault(c, block_id)
                chapter_for_code.setdefault(c, chapter_num)

    _icd10cm_block_for_code = block_for_code
    _icd10cm_chapter_for_code = chapter_for_code
    return block_for_code, chapter_for_code


def _icd10cm_lookup(code: str) -> tuple[str | None, int | None]:
    block_map, chapter_map = _load_icd10cm()
    norm = _norm_icd10cm(code)
    if norm in block_map:
        return block_map[norm], chapter_map[norm]
    # Cascading parent-prefix match (subcategory codes may not all enumerate in
    # the tabular XML; fall back to truncated parents).
    if "." in norm:
        prefix, sub = norm.split(".", 1)
        for length in range(len(sub), 0, -1):
            cand = prefix + "." + sub[:length] if length else prefix
            if cand in block_map:
                return block_map[cand], chapter_map[cand]
        if prefix in block_map:
            return block_map[prefix], chapter_map[prefix]
    # 3-char prefix scan against ICD10CM_CHAPTER_RANGES for chapter resolution.
    head = norm[:3].upper()
    for num, (lo, hi, _title) in ICD10CM_CHAPTER_RANGES.items():
        if lo <= head <= hi:
            return None, num
    return None, None


# ---------------------------------------------------------------------------
# ICD-10-PCS
# ---------------------------------------------------------------------------

_icd10pcs_codes: set[str] | None = None


def _load_icd10pcs() -> set[str]:
    global _icd10pcs_codes
    if _icd10pcs_codes is not None:
        return _icd10pcs_codes

    tree = ET.parse(_ICD10PCS_XML)
    root = tree.getroot()
    codes: set[str] = set()
    for table in root.findall("pcsTable"):
        # Axes 1, 2, 3 attached to the table; each has exactly one label.
        head_axes = sorted(
            (a for a in table.findall("axis") if a.get("pos") in {"1", "2", "3"}),
            key=lambda a: int(a.get("pos")),
        )
        if len(head_axes) != 3:
            continue
        head_chars = []
        for ax in head_axes:
            label = ax.find("label")
            if label is None or label.get("code") is None:
                head_chars.append(None)
                break
            head_chars.append(label.get("code"))
        if any(c is None for c in head_chars):
            continue
        head = "".join(head_chars)
        for row in table.findall("pcsRow"):
            row_axes = sorted(
                (a for a in row.findall("axis") if a.get("pos") in {"4", "5", "6", "7"}),
                key=lambda a: int(a.get("pos")),
            )
            if len(row_axes) != 4:
                continue
            label_lists = [
                [lab.get("code") for lab in ax.findall("label") if lab.get("code")]
                for ax in row_axes
            ]
            if any(len(ls) == 0 for ls in label_lists):
                continue
            for tail in product(*label_lists):
                codes.add(head + "".join(tail))

    _icd10pcs_codes = codes
    return codes


def _icd10pcs_lookup(code: str) -> tuple[str | None, int | None]:
    norm = code.strip().upper()
    if len(norm) < 3:
        return None, None
    block = norm[:3]
    section = norm[0]
    # Chapter id is the leading character (Section): keep as-is in string form.
    # We map to a stable integer for compatibility with ICD10CM chapter numbers
    # by hashing 0-9, A-Z to 23..58 — but that conflates with CM. Instead the
    # surface API returns chapters via `code_to_chapter` (string), so just keep
    # the section character through the pipeline.
    # For the public chapter-number API we return None here; pipeline code uses
    # `code_to_chapter` which formats `"ICD10PCS-<section>"`.
    return block, None  # chapter resolution handled upstream via section char


# ---------------------------------------------------------------------------
# ICD-9-CM
# ---------------------------------------------------------------------------

_icd9cm_blocks: list[tuple[str, str, int]] | None = None  # (lo, hi, chapter)


def _strip_rtf(rtf_text: str) -> str:
    text = rtf_text
    text = re.sub(r"\\par[d]?\b", "\n", text)
    text = re.sub(r"\\'[0-9a-fA-F]{2}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+-?\d* ?", "", text)
    text = text.replace("{", "").replace("}", "")
    return text


def _code_in_icd9_range(code: str, lo: str, hi: str) -> bool:
    """Compare 3-char ICD-9 codes (numeric, V-, or E-) lexicographically.

    The chapter and block tables are constructed so all entries within a single
    range share a prefix family (numeric, V, or E), which makes lex order safe.
    """
    fam = lambda s: s[0] if s and s[0] in "VE" else "N"
    if fam(code) != fam(lo) or fam(code) != fam(hi):
        return False
    return lo <= code <= hi


def _load_icd9cm_blocks() -> list[tuple[str, str, int]]:
    global _icd9cm_blocks
    if _icd9cm_blocks is not None:
        return _icd9cm_blocks

    rtf = _ICD9CM_RTF.read_text(errors="ignore")
    flat = _strip_rtf(rtf)
    seen: set[tuple[str, str]] = set()
    ranges: list[tuple[str, str]] = []
    # Numeric/V (3-char) block ranges, plus E-code (4-char) ranges.
    for pat in (
        re.compile(r"\(([V0-9]{3})-([V0-9]{3})\)"),
        re.compile(r"\((E[0-9]{3})-(E[0-9]{3})\)"),
    ):
        for m in pat.finditer(flat):
            lo, hi = m.group(1), m.group(2)
            if (lo, hi) not in seen:
                seen.add((lo, hi))
                ranges.append((lo, hi))

    # Drop chapter ranges (exact match in ICD9_CHAPTER_RANGES); keep blocks.
    chapter_set = {(lo, hi) for (lo, hi, _t) in ICD9_CHAPTER_RANGES.values()}
    blocks: list[tuple[str, str, int]] = []
    for lo, hi in ranges:
        if (lo, hi) in chapter_set:
            continue
        # Find the chapter that fully contains this block.
        owners = [
            num
            for num, (clo, chi, _t) in ICD9_CHAPTER_RANGES.items()
            if _code_in_icd9_range(lo, clo, chi) and _code_in_icd9_range(hi, clo, chi)
        ]
        if len(owners) != 1:
            raise ValueError(
                f"ICD-9-CM block range ({lo}-{hi}) does not lie inside exactly one chapter "
                f"(owners={owners}). Check icd9cm.rtf parser."
            )
        blocks.append((lo, hi, owners[0]))

    # Sort blocks ascending by span so the most specific (narrowest) block wins
    # the lookup when multiple intermediate aggregates contain the same code.
    def _span(rng: tuple[str, str, int]) -> int:
        lo, hi, _ = rng
        # Ranks by family then numeric tail.
        try:
            if lo[0] == "E":
                return int(hi[1:]) - int(lo[1:])
            if lo[0] == "V":
                return int(hi[1:]) - int(lo[1:])
            return int(hi) - int(lo)
        except ValueError:
            return 10**6

    blocks.sort(key=_span)
    _icd9cm_blocks = blocks
    return blocks


def _icd9cm_lookup(code: str) -> tuple[str | None, int | None]:
    norm = reformat_icd9cm_code(code.strip().upper())
    head = norm.split(".", 1)[0]
    # E-codes are 4-char heads (E000-E999); numeric/V codes are 3-char.
    if (head.startswith("E") and len(head) < 4) or (not head.startswith("E") and len(head) < 3):
        # Shouldn't happen with reformat_icd9cm_code, but guard anyway.
        return None, None
    blocks = _load_icd9cm_blocks()
    block_id: str | None = None
    chapter_num: int | None = None
    for lo, hi, num in blocks:
        if _code_in_icd9_range(head, lo, hi):
            block_id = f"{lo}-{hi}"
            chapter_num = num
            break
    if chapter_num is None:
        # Fall back to chapter-only lookup against ICD9_CHAPTER_RANGES.
        for num, (clo, chi, _t) in ICD9_CHAPTER_RANGES.items():
            if _code_in_icd9_range(head, clo, chi):
                chapter_num = num
                break
    if block_id is None and chapter_num is not None:
        # Some chapters have no sub-blocks in the Tabular (e.g. 280-289 Blood),
        # and some headings are singletons (e.g. 042 HIV). Use the chapter range
        # as the block so every code has a non-null block.
        clo, chi, _ = ICD9_CHAPTER_RANGES[chapter_num]
        block_id = f"{clo}-{chi}"
    return block_id, chapter_num


# ---------------------------------------------------------------------------
# ICD-9-PCS
# ---------------------------------------------------------------------------

def _icd9pcs_lookup(code: str) -> tuple[str | None, int | None]:
    norm = reformat_icd9pcs_code(code.strip())
    head = norm.split(".", 1)[0]
    if not head.isdigit() or len(head) != 2:
        return None, None
    cat = int(head)
    chapter = _ICD9PCS_CATEGORY_TO_CHAPTER.get(cat)
    return head, chapter


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _resolve(code: str, code_type: str) -> tuple[str | None, str | None]:
    """Return (block_id, chapter_id) where chapter_id is a stable string.

    chapter_id format:
      ICD-9 diagnosis  -> 'ICD9-<num>'
      ICD-9 procedure  -> 'ICD9PCS-<num>'
      ICD-10 diagnosis -> 'ICD10-<num>'
      ICD-10 procedure -> 'ICD10PCS-<section_char>'
    """
    if code is None:
        return None, None
    key = code_type.upper().replace("-", "").replace("_", "")
    is_proc = is_procedure_code(code, code_type) if code is not None else False

    if key == "ICD10":
        if is_proc:
            block, _ = _icd10pcs_lookup(code)
            section = code.strip().upper()[:1] if code else None
            chapter_id = f"ICD10PCS-{section}" if section else None
            return block, chapter_id
        block, chapter_num = _icd10cm_lookup(code)
        chapter_id = f"ICD10-{chapter_num:02d}" if chapter_num is not None else None
        return block, chapter_id

    if key == "ICD9":
        if is_proc:
            block, chapter_num = _icd9pcs_lookup(code)
            chapter_id = f"ICD9PCS-{chapter_num}" if chapter_num is not None else None
            return block, chapter_id
        block, chapter_num = _icd9cm_lookup(code)
        chapter_id = f"ICD9-{chapter_num}" if chapter_num is not None else None
        return block, chapter_id

    raise ValueError(f"Unsupported code_type: {code_type!r}. Use 'ICD-9' or 'ICD-10'.")


@lru_cache(maxsize=None)
def code_to_block(code: str, code_type: str) -> str | None:
    return _resolve(code, code_type)[0]


@lru_cache(maxsize=None)
def code_to_chapter(code: str, code_type: str) -> str | None:
    return _resolve(code, code_type)[1]


def chapter_name(chapter_id: str) -> str | None:
    """Human-readable title for a chapter id from `code_to_chapter`."""
    if chapter_id is None:
        return None
    if chapter_id.startswith("ICD10PCS-"):
        return f"PCS Section {chapter_id.split('-', 1)[1]}"
    if chapter_id.startswith("ICD10-"):
        num = int(chapter_id.split("-", 1)[1])
        entry = ICD10CM_CHAPTER_RANGES.get(num)
        return entry[2] if entry else None
    if chapter_id.startswith("ICD9PCS-"):
        num = int(chapter_id.split("-", 1)[1])
        return _ICD9PCS_CHAPTERS.get(num)
    if chapter_id.startswith("ICD9-"):
        num = int(chapter_id.split("-", 1)[1])
        entry = ICD9_CHAPTER_RANGES.get(num)
        return entry[2] if entry else None
    return None


def build_hierarchy_df(codes: Iterable[str], code_type: str) -> pl.DataFrame:
    """Return DataFrame[code, block, chapter, chapter_name] for the given codes."""
    rows = []
    for c in codes:
        if c is None:
            continue
        block, chapter = _resolve(c, code_type)
        rows.append(
            {
                "code": c,
                "block": block,
                "chapter": chapter,
                "chapter_name": chapter_name(chapter) if chapter else None,
            }
        )
    return pl.DataFrame(rows, schema={
        "code": pl.String,
        "block": pl.String,
        "chapter": pl.String,
        "chapter_name": pl.String,
    })


def validate_pcs_against_class_map(class_map_codes: Iterable[str]) -> list[str]:
    """Cross-validate enumerated ICD-10-PCS codes against a class-map. Returns
    the list of class-map PCS-shaped codes NOT covered by the enumeration. A
    non-empty result is logged as a warning by callers (not raised), since the
    class map only contains codes the model predicts over.
    """
    enum = _load_icd10pcs()
    misses: list[str] = []
    for c in class_map_codes:
        if c is None:
            continue
        norm = c.strip().upper()
        # PCS-shaped: 7 chars, no dot, alnum, ICD-10 alphabet (no I, O).
        if len(norm) == 7 and "." not in norm and norm.isalnum():
            if norm not in enum:
                misses.append(norm)
    return misses
