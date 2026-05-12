"""Deterministic parser for ICD-9-CM / ICD-10-CM Official Guidelines PDFs.

Extracts hierarchical coding rules into structured GuidelineEntry records
for downstream search and prompt injection. Code-mention, code-range, and
chapter-range extraction is parameterized by `code_system` so the same
hierarchical-section logic can handle either CM variant. ICD-10-PCS is
parsed by the separate PCSGuidelineParser below.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import tiktoken
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# ICD-9 chapter → code-range mapping (fixed; does not change across editions)
# ---------------------------------------------------------------------------
ICD9_CHAPTER_RANGES: dict[int, tuple[str, str, str]] = {
    1: ("001", "139", "Infectious and Parasitic Diseases"),
    2: ("140", "239", "Neoplasms"),
    3: ("240", "279", "Endocrine, Nutritional, Metabolic, and Immunity Disorders"),
    4: ("280", "289", "Diseases of Blood and Blood-Forming Organs"),
    5: ("290", "319", "Mental Disorders"),
    6: ("320", "389", "Diseases of Nervous System and Sense Organs"),
    7: ("390", "459", "Diseases of Circulatory System"),
    8: ("460", "519", "Diseases of Respiratory System"),
    9: ("520", "579", "Diseases of Digestive System"),
    10: ("580", "629", "Diseases of Genitourinary System"),
    11: ("630", "679", "Complications of Pregnancy, Childbirth, and the Puerperium"),
    12: ("680", "709", "Diseases of Skin and Subcutaneous Tissue"),
    13: ("710", "739", "Diseases of Musculoskeletal System and Connective Tissue"),
    14: ("740", "759", "Congenital Anomalies"),
    15: ("760", "779", "Certain Conditions Originating in the Perinatal Period"),
    16: ("780", "799", "Symptoms, Signs, and Ill-Defined Conditions"),
    17: ("800", "999", "Injury and Poisoning"),
    18: ("V01", "V91", "Supplementary Classification of Factors Influencing Health Status"),
    19: ("E000", "E999", "Supplementary Classification of External Causes of Injury and Poisoning"),
}

# ICD-10-CM chapters (22 chapters; ranges from the Tabular List).
ICD10CM_CHAPTER_RANGES: dict[int, tuple[str, str, str]] = {
    1: ("A00", "B99", "Certain Infectious and Parasitic Diseases"),
    2: ("C00", "D49", "Neoplasms"),
    3: ("D50", "D89", "Diseases of the Blood and Blood-Forming Organs"),
    4: ("E00", "E89", "Endocrine, Nutritional and Metabolic Diseases"),
    5: ("F01", "F99", "Mental, Behavioral and Neurodevelopmental Disorders"),
    6: ("G00", "G99", "Diseases of the Nervous System"),
    7: ("H00", "H59", "Diseases of the Eye and Adnexa"),
    8: ("H60", "H95", "Diseases of the Ear and Mastoid Process"),
    9: ("I00", "I99", "Diseases of the Circulatory System"),
    10: ("J00", "J99", "Diseases of the Respiratory System"),
    11: ("K00", "K95", "Diseases of the Digestive System"),
    12: ("L00", "L99", "Diseases of the Skin and Subcutaneous Tissue"),
    13: ("M00", "M99", "Diseases of the Musculoskeletal System and Connective Tissue"),
    14: ("N00", "N99", "Diseases of the Genitourinary System"),
    15: ("O00", "O9A", "Pregnancy, Childbirth and the Puerperium"),
    16: ("P00", "P96", "Certain Conditions Originating in the Perinatal Period"),
    17: ("Q00", "Q99", "Congenital Malformations, Deformations and Chromosomal Abnormalities"),
    18: ("R00", "R99", "Symptoms, Signs and Abnormal Clinical and Laboratory Findings"),
    19: ("S00", "T88", "Injury, Poisoning and Certain Other Consequences of External Causes"),
    20: ("V00", "Y99", "External Causes of Morbidity"),
    21: ("Z00", "Z99", "Factors Influencing Health Status and Contact with Health Services"),
    22: ("U00", "U85", "Codes for Special Purposes"),
}

# Map Section I.C chapter numbers to our ICD9_CHAPTER_RANGES keys
_SECTION_CHAPTER_RE = re.compile(
    r"^(\d{1,2})\.\s+Chapter\s+\d+", re.IGNORECASE
)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass
class GuidelineEntry:
    rule_id: str
    section_id: str
    parent_section: str | None
    section_title: str
    guideline_type: str  # convention | general | chapter_specific | principal_dx | additional_dx | outpatient
    code_ranges: list[tuple[str, str]]
    code_mentions: list[str]
    rule_text: str
    hierarchy_depth: int
    source_page: int
    source_document: str
    token_count: int = 0

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "section_id": self.section_id,
            "parent_section": self.parent_section,
            "section_title": self.section_title,
            "guideline_type": self.guideline_type,
            "code_ranges": self.code_ranges,
            "code_mentions": self.code_mentions,
            "rule_text": self.rule_text,
            "hierarchy_depth": self.hierarchy_depth,
            "source_page": self.source_page,
            "source_document": self.source_document,
            "token_count": self.token_count,
        }


# ---------------------------------------------------------------------------
# Internal tree node used during parsing
# ---------------------------------------------------------------------------

@dataclass
class _SectionNode:
    section_id: str          # e.g. "I.A.1" or "I.C.7.a"
    title: str
    depth: int
    body_lines: list[str] = field(default_factory=list)
    children: list[_SectionNode] = field(default_factory=list)
    page: int = 0
    parent: _SectionNode | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Regex patterns for section headers
# ---------------------------------------------------------------------------

# Page header/footer to strip
_PAGE_HEADER_RE = re.compile(
    r"^\s*ICD-9-CM Official Guidelines for Coding and Reporting\s*$"
)
_PAGE_SUBHEADER_RE = re.compile(
    r"^\s*Effective October 1, 20\d{2}\s*$"
)
_PAGE_NUMBER_RE = re.compile(
    r"^\s*Page \d+ of \d+\s*$"
)

# Top-level: "Section I.", "Section II.", etc.
_SECTION_TOP_RE = re.compile(
    r"^Section\s+(I{1,3}V?|IV)\.\s+(.+)"
)

# Letter subsection under a section: "A. ...", "B. ..."
_LETTER_SUBSECTION_RE = re.compile(
    r"^([A-Z])\.\s+(.+)"
)

# Numbered item: "1. ...", "2. ...", etc.
_NUMBERED_ITEM_RE = re.compile(
    r"^(\d{1,2})\.\s+(.+)"
)

# Lettered sub-item: "a. ...", "b. ...", etc.
_LETTERED_SUBITEM_RE = re.compile(
    r"^([a-z])\.\s+(.+)"
)

# Parenthesized numbered: "1) ...", "2) ..."
_PAREN_NUMBERED_RE = re.compile(
    r"^(\d{1,2})\)\s+(.+)"
)

# Parenthesized lettered: "(a) ...", "(b) ..."
_PAREN_LETTERED_RE = re.compile(
    r"^\(([a-z])\)\s+(.+)"
)

# Parenthesized roman: "(i) ...", "(ii) ..."
_PAREN_ROMAN_RE = re.compile(
    r"^\(([ivxlc]+)\)\s+(.+)", re.IGNORECASE
)

# ICD-9 code mentions in text. Each branch is anchored to its own valid digit
# count: V-codes are 2 digits (V01–V99) optionally with subdivisions; E-codes
# are 3 digits with at most one decimal; numeric codes are 3 digits with up to
# two decimals (and an optional placeholder 'x' some compendia use).
_ICD9_CODE_RE = re.compile(
    r"\b(V\d{2}(?:\.\d{1,2})?|E\d{3}(?:\.\d)?|\d{3}(?:\.\d{1,2})?x?)\b",
    re.IGNORECASE,
)

# Code range in chapter headers like "(240-279)" or "(V01-V91)"
_CODE_RANGE_IN_HEADER_RE = re.compile(
    r"\(([VvEe]?\d{2,3})\s*[-–]\s*([VvEe]?\d{2,3})\)"
)

# ICD-10-CM code mentions: [A-Z] + 2 digits + optional .[alnum]{1,4} or X placeholders.
# Excludes leading U-codes from wildcard filter only if needed; kept permissive here.
_ICD10CM_CODE_RE = re.compile(
    r"\b([A-TV-Z]\d{2}(?:\.[A-Z0-9]{1,4})?)\b"
)

# ICD-10-CM code range in chapter headers like "(A00-B99)" or "(S00-T88)".
_ICD10CM_RANGE_IN_HEADER_RE = re.compile(
    r"\(([A-Z]\d{2}[A-Z]?)\s*[-–]\s*([A-Z]\d{2}[A-Z]?)\)"
)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_SUPPORTED_CODE_SYSTEMS = {"icd9cm", "icd10cm"}


class GuidelineParser:
    """Parse ICD-9-CM / ICD-10-CM Official Guidelines PDFs into structured entries.

    The section-header parser is shared; code-mention / code-range extraction
    and chapter ranges are selected by `code_system` ("icd9cm" | "icd10cm").
    """

    def __init__(
        self,
        source_document: str = "ICD-9-CM Official Guidelines 2011",
        encoding_name: str = "cl100k_base",
        max_rule_tokens: int = 200,
        code_system: str = "icd9cm",
    ):
        if code_system not in _SUPPORTED_CODE_SYSTEMS:
            raise ValueError(
                f"Unsupported code_system: {code_system!r}. "
                f"Use one of {sorted(_SUPPORTED_CODE_SYSTEMS)}."
            )
        self.source_document = source_document
        self.code_system = code_system
        if code_system == "icd9cm":
            self._code_re = _ICD9_CODE_RE
            self._range_re = _CODE_RANGE_IN_HEADER_RE
            self._chapter_ranges = ICD9_CHAPTER_RANGES
        else:
            self._code_re = _ICD10CM_CODE_RE
            self._range_re = _ICD10CM_RANGE_IN_HEADER_RE
            self._chapter_ranges = ICD10CM_CHAPTER_RANGES
        self._enc = tiktoken.get_encoding(encoding_name)
        self._max_rule_tokens = max_rule_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, pdf_path: str | Path) -> list[dict]:
        """Extract all guideline entries from the PDF.

        Returns a list of dicts (one per rule), ready for JSONL/Parquet output.
        """
        pages = self._extract_pages(Path(pdf_path))
        full_text, page_map = self._join_pages(pages)
        root_nodes = self._build_tree(full_text, page_map)
        entries = self._tree_to_entries(root_nodes)
        return [e.to_dict() for e in entries]

    # ------------------------------------------------------------------
    # PDF text extraction
    # ------------------------------------------------------------------

    def _extract_pages(self, pdf_path: Path) -> list[tuple[int, str]]:
        """Return [(page_number, cleaned_text), ...]."""
        reader = PdfReader(str(pdf_path))
        pages = []
        for i, page in enumerate(reader.pages):
            raw = page.extract_text() or ""
            cleaned = self._clean_page(raw)
            if cleaned.strip():
                pages.append((i + 1, cleaned))
        return pages

    def _clean_page(self, text: str) -> str:
        """Remove repeating headers/footers."""
        lines = text.split("\n")
        out = []
        for line in lines:
            if _PAGE_HEADER_RE.match(line):
                continue
            if _PAGE_SUBHEADER_RE.match(line):
                continue
            if _PAGE_NUMBER_RE.match(line):
                continue
            out.append(line)
        return "\n".join(out)

    def _join_pages(
        self, pages: list[tuple[int, str]]
    ) -> tuple[list[str], dict[int, int]]:
        """Join all pages into a single list of lines.

        Returns (lines, line_idx -> page_number mapping).
        """
        all_lines: list[str] = []
        line_to_page: dict[int, int] = {}
        for page_num, text in pages:
            for line in text.split("\n"):
                line_to_page[len(all_lines)] = page_num
                all_lines.append(line)
        return all_lines, line_to_page

    # ------------------------------------------------------------------
    # Tree building
    # ------------------------------------------------------------------

    def _build_tree(
        self,
        lines: list[str],
        page_map: dict[int, int],
    ) -> list[_SectionNode]:
        """Build a forest of section nodes from the document lines."""
        root_nodes: list[_SectionNode] = []
        stack: list[_SectionNode] = []  # current ancestry

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            page = page_map.get(i, 0)

            if not line:
                # blank line — append to current node body if exists
                if stack:
                    stack[-1].body_lines.append("")
                i += 1
                continue

            # Try to match section headers in priority order
            node = self._try_match_header(line, lines, i, page, stack)

            if node is not None:
                # Attach to parent or root
                if stack:
                    parent = self._find_parent(stack, node.depth)
                    if parent:
                        parent.children.append(node)
                        node.parent = parent
                        # Rebuild stack: keep up to parent, add node
                        idx = stack.index(parent)
                        stack = stack[: idx + 1]
                    else:
                        root_nodes.append(node)
                        stack = []
                else:
                    root_nodes.append(node)
                    stack = []
                stack.append(node)
            else:
                # Body text — append to current deepest node
                if stack:
                    stack[-1].body_lines.append(line)
                # else: preamble text before first section — skip

            i += 1

        return root_nodes

    @staticmethod
    def _is_toc_line(line: str) -> bool:
        """Detect Table of Contents lines (contain runs of dots)."""
        return "...." in line or "…" in line

    def _try_match_header(
        self,
        line: str,
        lines: list[str],
        idx: int,
        page: int,
        stack: list[_SectionNode],
    ) -> _SectionNode | None:
        """Try to match a line as a section header. Returns node or None."""
        # Skip TOC lines (contain runs of dots like "A. Conventions ..... 6")
        if self._is_toc_line(line):
            return None

        # Top-level section: "Section I. ..."
        m = _SECTION_TOP_RE.match(line)
        if m:
            numeral = m.group(1)
            title = m.group(2).strip()
            title = self._gather_title(title, lines, idx)
            return _SectionNode(
                section_id=numeral, title=title, depth=0, page=page
            )

        # We need at least one section on the stack to anchor sub-headers
        if not stack:
            return None

        # Letter subsection: "A. Conventions...", "B. General Coding..."
        # These are depth=1, children of the top-level Section node
        m = _LETTER_SUBSECTION_RE.match(line)
        if m:
            letter = m.group(1)
            title = m.group(2).strip()
            # Guard: reject if next word after letter is lowercase (e.g. "S. aureus")
            # Real headers have capitalized titles
            if title and title[0].islower():
                pass  # Not a real header
            else:
                title = self._gather_title(title, lines, idx)
                parent_id = self._nearest_ancestor_id(stack, 0)
                sid = f"{parent_id}.{letter}" if parent_id else letter
                return _SectionNode(
                    section_id=sid, title=title, depth=1, page=page
                )

        # Numbered item: "1. Chapter 1: ..." or "1. Use of Both..."
        # These are depth=2, children of a letter subsection
        m = _NUMBERED_ITEM_RE.match(line)
        if m:
            num = m.group(1)
            title = m.group(2).strip()
            title = self._gather_title(title, lines, idx)
            parent_id = self._nearest_ancestor_id(stack, 1)
            sid = f"{parent_id}.{num}" if parent_id else num
            return _SectionNode(
                section_id=sid, title=title, depth=2, page=page
            )

        # Lettered sub-item: "a. Diabetes mellitus"
        # depth=3, children of numbered items
        m = _LETTERED_SUBITEM_RE.match(line)
        if m:
            letter = m.group(1)
            title = m.group(2).strip()
            title = self._gather_title(title, lines, idx)
            parent_id = self._nearest_ancestor_id(stack, 2)
            sid = f"{parent_id}.{letter}" if parent_id else letter
            return _SectionNode(
                section_id=sid, title=title, depth=3, page=page
            )

        # Parenthesized numbered: "1) Fifth-digits..."
        # depth=4, children of lettered sub-items
        m = _PAREN_NUMBERED_RE.match(line)
        if m:
            num = m.group(1)
            title = m.group(2).strip()
            title = self._gather_title(title, lines, idx)
            parent_id = self._nearest_ancestor_id(stack, 3)
            sid = f"{parent_id}.{num})" if parent_id else f"{num})"
            return _SectionNode(
                section_id=sid, title=title, depth=4, page=page
            )

        # Parenthesized lettered: "(a) Documentation..."
        # depth=5
        m = _PAREN_LETTERED_RE.match(line)
        if m:
            letter = m.group(1)
            title = m.group(2).strip()
            title = self._gather_title(title, lines, idx)
            parent_id = self._nearest_ancestor_id(stack, 4)
            sid = f"{parent_id}.({letter})" if parent_id else f"({letter})"
            return _SectionNode(
                section_id=sid, title=title, depth=5, page=page
            )

        # Parenthesized roman: "(i) Secondary diabetes..."
        # depth=6
        m = _PAREN_ROMAN_RE.match(line)
        if m:
            roman = m.group(1).lower()
            title = m.group(2).strip()
            title = self._gather_title(title, lines, idx)
            parent_id = self._nearest_ancestor_id(stack, 5)
            sid = f"{parent_id}.({roman})" if parent_id else f"({roman})"
            return _SectionNode(
                section_id=sid, title=title, depth=6, page=page
            )

        return None

    def _gather_title(self, title: str, lines: list[str], idx: int) -> str:
        """If a title seems truncated (no period, short), peek at next lines."""
        # Titles can span multiple lines in the PDF. Simple heuristic:
        # if the title line doesn't end with common terminators and is short,
        # and the next line(s) don't match a header pattern, append them.
        max_peek = 3
        for offset in range(1, max_peek + 1):
            if idx + offset >= len(lines):
                break
            next_line = lines[idx + offset].strip()
            if not next_line:
                break
            # If next line looks like a header, stop
            if any(p.match(next_line) for p in [
                _SECTION_TOP_RE, _LETTER_SUBSECTION_RE, _NUMBERED_ITEM_RE,
                _LETTERED_SUBITEM_RE, _PAREN_NUMBERED_RE, _PAREN_LETTERED_RE,
                _PAREN_ROMAN_RE,
            ]):
                break
            # If current title looks complete (ends with period, colon, or
            # closing paren of a code range), stop
            if title.rstrip().endswith((".",":", ")")):
                break
            title = title + " " + next_line
        return title.strip()

    def _find_parent(
        self, stack: list[_SectionNode], child_depth: int
    ) -> _SectionNode | None:
        """Walk stack backwards to find a node whose depth < child_depth."""
        for node in reversed(stack):
            if node.depth < child_depth:
                return node
        return None

    def _nearest_ancestor_id(
        self, stack: list[_SectionNode], max_depth: int
    ) -> str:
        """Get section_id of nearest ancestor at depth <= max_depth."""
        for node in reversed(stack):
            if node.depth <= max_depth:
                return node.section_id
        return ""

    # ------------------------------------------------------------------
    # Tree → GuidelineEntry conversion
    # ------------------------------------------------------------------

    def _tree_to_entries(
        self, nodes: list[_SectionNode]
    ) -> list[GuidelineEntry]:
        """Convert the parsed tree into flat GuidelineEntry records."""
        entries: list[GuidelineEntry] = []
        for node in nodes:
            self._process_node(node, entries)
        return entries

    def _process_node(
        self,
        node: _SectionNode,
        entries: list[GuidelineEntry],
    ) -> None:
        """Recursively convert a node and its children into entries."""
        body = "\n".join(node.body_lines).strip()

        if body:
            guideline_type = self._classify_type(node.section_id)
            code_ranges = self._detect_code_ranges(node)
            code_mentions = self._extract_code_mentions(body)
            parent_id = node.parent.section_id if node.parent else None

            # Split long bodies into smaller rules
            chunks = self._split_body(body)
            for i, chunk in enumerate(chunks):
                rule_id = node.section_id if len(chunks) == 1 else f"{node.section_id}#{i+1}"
                token_count = len(self._enc.encode(chunk))
                entries.append(GuidelineEntry(
                    rule_id=rule_id,
                    section_id=node.section_id,
                    parent_section=parent_id,
                    section_title=node.title,
                    guideline_type=guideline_type,
                    code_ranges=code_ranges,
                    code_mentions=self._extract_code_mentions(chunk),
                    rule_text=chunk,
                    hierarchy_depth=node.depth,
                    source_page=node.page,
                    source_document=self.source_document,
                    token_count=token_count,
                ))

        for child in node.children:
            # Children inherit code_ranges from parent if they don't have their own
            self._process_node(child, entries)

    def _classify_type(self, section_id: str) -> str:
        """Determine guideline_type from the section_id."""
        sid = section_id.upper()
        if sid.startswith("I.A"):
            return "convention"
        if sid.startswith("I.B"):
            return "general"
        if sid.startswith("I.C"):
            return "chapter_specific"
        # Check III before II (III starts with II)
        if sid.startswith("III"):
            return "additional_dx"
        if sid.startswith("II"):
            return "principal_dx"
        if sid.startswith("IV"):
            return "outpatient"
        return "general"

    def _detect_code_ranges(self, node: _SectionNode) -> list[tuple[str, str]]:
        """Infer code ranges from the node's title and ancestry."""
        ranges: list[tuple[str, str]] = []

        # Check title for explicit range like "(240-279)" or "(A00-B99)"
        m = self._range_re.search(node.title)
        if m:
            ranges.append((m.group(1).upper(), m.group(2).upper()))
            return ranges

        # For Section I.C nodes, try to map the chapter number
        if node.section_id.upper().startswith("I.C."):
            parts = node.section_id.split(".")
            if len(parts) >= 3:
                try:
                    chapter_num = int(parts[2])
                    if chapter_num in self._chapter_ranges:
                        start, end, _ = self._chapter_ranges[chapter_num]
                        ranges.append((start, end))
                        return ranges
                except ValueError:
                    pass

        # Inherit from parent
        if node.parent:
            return self._detect_code_ranges(node.parent)

        return ranges

    def _extract_code_mentions(self, text: str) -> list[str]:
        """Find specific ICD codes mentioned in text (per self.code_system)."""
        matches = self._code_re.findall(text)
        codes: list[str] = []
        seen: set[str] = set()
        for code in matches:
            code = code.upper().rstrip("X") if self.code_system == "icd9cm" else code.upper()
            # Filter out things that are clearly not ICD codes
            # (e.g. page numbers, years, simple integers).
            if re.match(r"^\d{1,2}$", code):
                continue
            if re.match(r"^(19|20)\d{2}$", code):
                continue
            if code not in seen:
                seen.add(code)
                codes.append(code)
        return codes

    def _split_body(self, body: str) -> list[str]:
        """Split a body text into chunks if it exceeds max_rule_tokens."""
        tokens = len(self._enc.encode(body))
        if tokens <= self._max_rule_tokens:
            return [body]

        # Split on double newlines first (paragraph breaks)
        paragraphs = re.split(r"\n\s*\n", body)
        if len(paragraphs) > 1:
            return self._merge_small_chunks(paragraphs)

        # Split on sentences
        sentences = re.split(r"(?<=[.!?])\s+", body)
        if len(sentences) > 1:
            return self._merge_small_chunks(sentences)

        # Can't split further — return as-is
        return [body]

    def _merge_small_chunks(self, parts: list[str]) -> list[str]:
        """Merge consecutive small parts to get closer to max_rule_tokens."""
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for part in parts:
            part = part.strip()
            if not part:
                continue
            part_tokens = len(self._enc.encode(part))
            if current_tokens + part_tokens > self._max_rule_tokens and current:
                chunks.append("\n\n".join(current))
                current = [part]
                current_tokens = part_tokens
            else:
                current.append(part)
                current_tokens += part_tokens

        if current:
            chunks.append("\n\n".join(current))

        return chunks if chunks else parts


# ---------------------------------------------------------------------------
# ICD-10-PCS parser
# ---------------------------------------------------------------------------

# Top-level section letter: "A. Conventions", "B. Medical and Surgical ..."
_PCS_TOP_RE = re.compile(r"^([A-F])\.\s+(.+)$")
# B-subgroup: "B2.  Body System", "B3. Root Operation", ..., "B6. Device"
_PCS_SUBGROUP_RE = re.compile(r"^B([2-6])\.\s+(Body System|Root Operation|Body Part|Approach|Device)\b")
# Rule IDs on their own line: A1, B2.1a, B3.10b, C1, D1.a, E1.b
_PCS_RULE_RE = re.compile(r"^([A-F]\d+(?:\.\d+[a-z]?|\.[a-z])?)\s*$")
# F numbered list items: "1. Procedure performed..."
_F_NUMBERED_RE = re.compile(r"^(\d+)\.\s+([A-Z].+)$")
# 7-character PCS code, alphabet excludes I and O
_PCS_CODE_RE = re.compile(r"\b([0-9A-HJ-NP-Z]{7})\b")

_PCS_TYPE_MAP = {
    "A": "convention",
    "B": "medical_surgical",
    "C": "obstetrics",
    "D": "radiation_therapy",
    "E": "new_technology",
    "F": "principal_procedure",
}


class PCSGuidelineParser:
    """Parse ICD-10-PCS Official Guidelines PDF into structured entries.

    Emits the same schema as `GuidelineParser` so outputs can be concatenated.
    """

    def __init__(
        self,
        source_document: str = "ICD-10-PCS Official Guidelines 2022",
        encoding_name: str = "cl100k_base",
        max_rule_tokens: int = 200,
    ):
        self.source_document = source_document
        self._enc = tiktoken.get_encoding(encoding_name)
        self._max_rule_tokens = max_rule_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, pdf_path: str | Path) -> list[dict]:
        reader = PdfReader(str(pdf_path))
        lines: list[tuple[str, int]] = []
        for i, page in enumerate(reader.pages):
            raw = page.extract_text() or ""
            # Skip TOC pages (dot-leader runs)
            if raw.count("…") > 5 or raw.count("....") > 5:
                continue
            for ln in raw.split("\n"):
                s = ln.strip()
                # Strip standalone page-number lines
                if re.fullmatch(r"\d{1,2}", s):
                    continue
                lines.append((ln, i + 1))

        entries: list[GuidelineEntry] = []

        current_top: tuple[str, str, int] | None = None  # (letter, title, page)
        current_b_sub: tuple[str, str, int] | None = None  # (num, title, page)
        current_rule: dict | None = None

        def parent_for_rule(rule_id: str) -> str | None:
            # Derive from rule_id so B* rules work even when the phantom
            # "B. Medical and Surgical..." header is absent from the body.
            if rule_id.startswith("B") and current_b_sub is not None:
                return f"B.{current_b_sub[0]}"
            return rule_id[0] if rule_id else None

        def flush_rule(take_trailing_title: bool = False) -> str | None:
            """Emit the pending rule. If take_trailing_title, return a stolen
            category-label line from the tail of its body to use as the next
            rule's section_title."""
            nonlocal current_rule
            if current_rule is None:
                return None
            body_lines = list(current_rule["body_lines"])
            stolen: str | None = None
            if take_trailing_title and body_lines:
                idx = len(body_lines) - 1
                while idx >= 0 and not body_lines[idx].strip():
                    idx -= 1
                if idx >= 0:
                    cand = body_lines[idx].strip()
                    if (
                        1 < len(cand) < 120
                        and cand[0].isupper()
                        and not cand.endswith((".", ",", ":", ";"))
                        and not _PCS_RULE_RE.match(cand)
                        and not _PCS_TOP_RE.match(cand)
                        and not _PCS_SUBGROUP_RE.match(cand)
                    ):
                        stolen = cand
                        body_lines = body_lines[:idx]
            body = "\n".join(body_lines).strip()
            if body:
                self._emit(current_rule, body, entries)
            current_rule = None
            return stolen

        preamble_title: str | None = None

        for ln, page in lines:
            s = ln.strip()

            # Top-level letter section
            m = _PCS_TOP_RE.match(s)
            if m and not _PCS_RULE_RE.match(s):
                flush_rule()
                current_top = (m.group(1), m.group(2).strip(), page)
                current_b_sub = None
                preamble_title = None
                continue

            # B-subgroup (may appear before any "B." top header, which is absent
            # from the PDF body — only in the TOC).
            m = _PCS_SUBGROUP_RE.match(s)
            if m:
                flush_rule()
                current_b_sub = (m.group(1), m.group(2).strip(), page)
                preamble_title = None
                continue

            # Subheader we want to drop
            if s == "General guidelines":
                flush_rule()
                preamble_title = None
                continue

            # Rule ID on its own line
            m = _PCS_RULE_RE.match(s)
            if m:
                stolen = flush_rule(take_trailing_title=True)
                title = stolen or preamble_title
                if not title:
                    if current_b_sub:
                        title = current_b_sub[1]
                    elif current_top:
                        title = current_top[1]
                    else:
                        title = m.group(1)
                rule_id = m.group(1)
                section_id = self._dotted_section_id(rule_id)
                current_rule = {
                    "rule_id": rule_id,
                    "section_id": section_id,
                    "parent_section": parent_for_rule(rule_id),
                    "title": title,
                    "page": page,
                    "top_letter": current_top[0] if current_top else "A",
                    "body_lines": [],
                }
                preamble_title = None
                continue

            # F section numbered list
            if current_top and current_top[0] == "F":
                m = _F_NUMBERED_RE.match(s)
                if m:
                    flush_rule()
                    num = m.group(1)
                    first_line = m.group(2).strip()
                    current_rule = {
                        "rule_id": f"F.{num}",
                        "section_id": f"F.{num}",
                        "parent_section": "F",
                        "title": current_top[1],
                        "page": page,
                        "top_letter": "F",
                        "body_lines": [first_line],
                    }
                    preamble_title = None
                    continue

            # Otherwise body or preamble
            if current_rule is not None:
                current_rule["body_lines"].append(s)
            elif s:
                # Only keep a short, header-looking line as candidate title
                if (
                    1 < len(s) < 120
                    and s[0].isupper()
                    and not s.endswith((".", ",", ":", ";"))
                ):
                    preamble_title = s

        flush_rule()
        return [e.to_dict() for e in entries]

    # ------------------------------------------------------------------
    # Emission / helpers
    # ------------------------------------------------------------------

    def _emit(
        self,
        rule: dict,
        body: str,
        entries: list[GuidelineEntry],
    ) -> None:
        # Derive top letter from the rule_id itself — some top-level "B." / "D."
        # section headers do not appear in the PDF body (only in the TOC).
        top_letter = rule["rule_id"][0] if rule["rule_id"] else rule["top_letter"]
        guideline_type = _PCS_TYPE_MAP.get(top_letter, "general")
        chunks = self._split_body(body)
        for i, chunk in enumerate(chunks):
            rule_id = rule["rule_id"] if len(chunks) == 1 else f"{rule['rule_id']}#{i+1}"
            token_count = len(self._enc.encode(chunk))
            entries.append(GuidelineEntry(
                rule_id=rule_id,
                section_id=rule["section_id"],
                parent_section=rule["parent_section"],
                section_title=rule["title"],
                guideline_type=guideline_type,
                code_ranges=[],
                code_mentions=self._extract_code_mentions(chunk),
                rule_text=chunk,
                hierarchy_depth=2,
                source_page=rule["page"],
                source_document=self.source_document,
                token_count=token_count,
            ))

    @staticmethod
    def _dotted_section_id(rule_id: str) -> str:
        """Convert natural PCS IDs to dot-delimited section IDs.

        A1 -> A.1, B3.2 -> B.3.2, B2.1a -> B.2.1a, D1.a -> D.1.a
        """
        m = re.match(r"^([A-F])(\d+)(.*)$", rule_id)
        if not m:
            return rule_id
        letter, num, rest = m.group(1), m.group(2), m.group(3)
        return f"{letter}.{num}{rest}"

    def _extract_code_mentions(self, text: str) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for code in _PCS_CODE_RE.findall(text):
            if code not in seen:
                seen.add(code)
                out.append(code)
        return out

    def _split_body(self, body: str) -> list[str]:
        tokens = len(self._enc.encode(body))
        if tokens <= self._max_rule_tokens:
            return [body]
        paragraphs = re.split(r"\n\s*\n", body)
        if len(paragraphs) > 1:
            return self._merge_small_chunks(paragraphs)
        sentences = re.split(r"(?<=[.!?])\s+", body)
        if len(sentences) > 1:
            return self._merge_small_chunks(sentences)
        return [body]

    def _merge_small_chunks(self, parts: list[str]) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0
        for part in parts:
            part = part.strip()
            if not part:
                continue
            part_tokens = len(self._enc.encode(part))
            if current_tokens + part_tokens > self._max_rule_tokens and current:
                chunks.append("\n\n".join(current))
                current = [part]
                current_tokens = part_tokens
            else:
                current.append(part)
                current_tokens += part_tokens
        if current:
            chunks.append("\n\n".join(current))
        return chunks if chunks else parts
