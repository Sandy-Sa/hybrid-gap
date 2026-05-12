"""Runtime retrieval of ICD coding guidelines for prompt injection.

Supports ICD-9-CM, ICD-10-CM, and ICD-10-PCS. Builds code-range,
code-mention, and BM25 indices over parsed guideline entries and returns
formatted guideline text within a configurable token budget.
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

import Stemmer
import bm25s
import polars as pl
import tiktoken

from coding.pipeline.data_loader import _normalize_code_type


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ALWAYS_ON_TYPES = frozenset({
    "general", "principal_dx", "additional_dx", "outpatient", "convention",
})

# Priority scores for ranking tiers (higher = more important)
_TIER_CODE_MENTION = 4.0   # rule explicitly names the candidate code
_TIER_CODE_RANGE = 3.0     # rule belongs to same ICD-9 chapter
_TIER_BM25 = 2.0           # BM25 semantic match
_TIER_ALWAYS_ON = 1.0      # general/principal_dx/additional_dx rules

# Token budgets
_DEFAULT_MAX_TOKENS = 1500
_ALWAYS_ON_BUDGET_FRACTION = 0.2  # 20% of budget reserved for always-on


# ---------------------------------------------------------------------------
# GuidelineSearcher
# ---------------------------------------------------------------------------

class GuidelineSearcher:
    """Retrieve relevant ICD coding guidelines for prompt injection.

    Supports both ICD-9 (ICD-9-CM) and ICD-10 (ICD-10-CM + ICD-10-PCS).
    """

    def __init__(
        self,
        guidelines_path: str | Path,
        encoding_name: str = "cl100k_base",
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        code_type: str = "ICD-9",
    ):
        self.max_tokens = max_tokens
        self.code_type = _normalize_code_type(code_type)
        self._enc = tiktoken.get_encoding(encoding_name)

        # Load guidelines
        path = Path(guidelines_path)
        if path.suffix == ".parquet":
            df = pl.read_parquet(path)
            self.entries = df.to_dicts()
            # Deserialize JSON-encoded list columns
            for entry in self.entries:
                if isinstance(entry["code_ranges"], str):
                    entry["code_ranges"] = json.loads(entry["code_ranges"])
                if isinstance(entry["code_mentions"], str):
                    entry["code_mentions"] = json.loads(entry["code_mentions"])
        else:
            # JSONL
            self.entries = []
            with open(path) as f:
                for line in f:
                    self.entries.append(json.loads(line))

        # Build indices
        self._code_range_index = self._build_code_range_index()
        self._code_mention_index = self._build_code_mention_index()
        self._always_on_indices = [
            i for i, e in enumerate(self.entries)
            if e["guideline_type"] in _ALWAYS_ON_TYPES
        ]

        # BM25 index over rule_text
        self._stemmer = Stemmer.Stemmer("english")
        corpus_tokens = [
            self._tokenize_and_stem(e["rule_text"]) for e in self.entries
        ]
        self._bm25 = bm25s.BM25()
        self._bm25.index(corpus_tokens)

        # Parent lookup for hierarchical context
        self._id_to_idx: dict[str, int] = {}
        for i, e in enumerate(self.entries):
            self._id_to_idx[e["section_id"]] = i

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_verification_context(
        self,
        candidate_codes: list[str],
        note_text: str = "",
        max_tokens: int | None = None,
    ) -> str:
        """Retrieve guidelines for the verification stage.

        Uses code-range + code-mention + BM25 + always-on generals.
        """
        budget = max_tokens or self.max_tokens
        scored: list[tuple[int, float, str]] = []

        for code in candidate_codes:
            cat = self._code_to_category(code)

            # Tier 1: exact code-mention matches
            for idx in self._code_mention_index.get(code.upper(), []):
                scored.append((idx, _TIER_CODE_MENTION, "code_mention"))
            # Also check category-level mentions
            for idx in self._code_mention_index.get(cat, []):
                scored.append((idx, _TIER_CODE_MENTION * 0.9, "code_mention"))

            # Tier 2: code-range chapter matches
            for idx in self._code_range_index.get(cat, []):
                scored.append((idx, _TIER_CODE_RANGE, "code_range"))

        # Tier 3: BM25 semantic search over note text
        if note_text:
            bm25_scored = self._bm25_search(note_text, top_k=20)
            for idx, bm25_score in bm25_scored:
                scored.append((idx, _TIER_BM25 + bm25_score * 0.1, "bm25"))

        # Tier 4: always-on generals
        for idx in self._always_on_indices:
            scored.append((idx, _TIER_ALWAYS_ON, "always_on"))

        return self._rank_and_format(scored, budget)

    def get_per_code_guideline(
        self,
        code: str,
        max_tokens: int = 80,
    ) -> str:
        """Return a concise guideline hint for a single ICD-9 code.

        Retrieves the single most relevant rule for this code, truncated
        at sentence boundaries to fit within max_tokens. Designed to be
        appended inline next to each code in the verification prompt.
        """
        cat = self._code_to_category(code)

        # Gather candidates: code-mention > code-range
        candidates: list[tuple[int, float]] = []
        for idx in self._code_mention_index.get(code.upper(), []):
            candidates.append((idx, _TIER_CODE_MENTION))
        for idx in self._code_mention_index.get(cat, []):
            candidates.append((idx, _TIER_CODE_MENTION * 0.9))
        for idx in self._code_range_index.get(cat, []):
            candidates.append((idx, _TIER_CODE_RANGE))

        if not candidates:
            return ""

        # Deduplicate, sort by score
        best: dict[int, float] = {}
        for idx, score in candidates:
            if idx not in best or score > best[idx]:
                best[idx] = score
        ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)

        # Pick entries that fit in budget
        parts: list[str] = []
        used = 0
        for idx, _score in ranked:
            entry = self.entries[idx]
            text = self._truncate_at_sentence(entry["rule_text"].strip(), max_tokens - used)
            if not text:
                continue
            cost = len(self._enc.encode(text))
            if used + cost > max_tokens:
                break
            parts.append(text)
            used += cost

        return " ".join(parts)

    def get_chunk_context(
        self,
        candidate_codes: list[str],
        max_tokens: int = 500,
    ) -> str:
        """Retrieve shared general guidelines for a chunk of codes.

        Focuses on always-on generals (Section I.B, Section II) and
        cross-code interaction rules (etiology/manifestation pairs).
        Chapter-specific rules are excluded — those go in per-code hints.
        """
        scored: list[tuple[int, float, str]] = []

        # Always-on generals (highest priority for the shared block)
        for idx in self._always_on_indices:
            entry = self.entries[idx]
            # Boost general and convention types for the shared block
            if entry["guideline_type"] in ("general", "convention"):
                scored.append((idx, _TIER_ALWAYS_ON * 2.0, "always_on"))
            else:
                scored.append((idx, _TIER_ALWAYS_ON, "always_on"))

        # Cross-code interaction rules: code-mention entries that reference
        # codes from multiple chapters in this chunk
        chapter_cats = set()
        for code in candidate_codes:
            chapter_cats.add(self._code_to_category(code))

        if len(chapter_cats) > 1:
            # Find rules that mention codes from 2+ chapters in this chunk
            for idx, entry in enumerate(self.entries):
                if entry["guideline_type"] != "chapter_specific":
                    continue
                mentioned_cats = set()
                for mentioned_code in entry["code_mentions"]:
                    mc = self._code_to_category(mentioned_code)
                    if mc in chapter_cats:
                        mentioned_cats.add(mc)
                if len(mentioned_cats) >= 2:
                    scored.append((idx, _TIER_CODE_MENTION * 0.8, "cross_code"))

        return self._rank_and_format(scored, max_tokens)

    def get_extraction_context(
        self,
        diagnoses_text: list[str] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Retrieve guidelines for the extraction stage.

        Uses BM25 over diagnosis text + always-on generals.
        """
        budget = max_tokens or self.max_tokens
        scored: list[tuple[int, float, str]] = []

        # BM25 search over each diagnosis term
        if diagnoses_text:
            combined = " ".join(diagnoses_text)
            bm25_scored = self._bm25_search(combined, top_k=30)
            for idx, bm25_score in bm25_scored:
                scored.append((idx, _TIER_BM25 + bm25_score * 0.1, "bm25"))

        # Always-on generals
        for idx in self._always_on_indices:
            scored.append((idx, _TIER_ALWAYS_ON, "always_on"))

        return self._rank_and_format(scored, budget)

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def _build_code_range_index(self) -> dict[str, list[int]]:
        """Map each category to rule indices."""
        index: dict[str, list[int]] = {}
        for i, entry in enumerate(self.entries):
            for start, end in entry["code_ranges"]:
                for cat in self._expand_range(start.upper(), end.upper()):
                    index.setdefault(cat, []).append(i)
        return index

    def _build_code_mention_index(self) -> dict[str, list[int]]:
        """Map specifically mentioned codes to rule indices."""
        index: dict[str, list[int]] = {}
        for i, entry in enumerate(self.entries):
            for code in entry["code_mentions"]:
                code = code.upper()
                index.setdefault(code, []).append(i)
                # Also index the category (first 3 chars)
                cat = self._code_to_category(code)
                if cat != code:
                    index.setdefault(cat, []).append(i)
        return index

    _ICD10_PCS_FORMAT_RE = re.compile(r"^[0-9A-HJ-NP-Z]{7}$")

    def _expand_range(self, start: str, end: str) -> list[str]:
        """Expand a code range into category keys, dispatching by code_type."""
        if self.code_type == "ICD-9":
            return self._expand_range_icd9(start, end)
        return self._expand_range_icd10cm(start, end)

    @staticmethod
    def _expand_range_icd9(start: str, end: str) -> list[str]:
        """Expand ICD-9 ranges: numeric (001-999), V-codes (V01-V91), E-codes (E000-E999)."""
        categories = []
        if start[0] in ("V", "v"):
            s = int(start[1:])
            e = int(end[1:])
            for n in range(s, e + 1):
                categories.append(f"V{n:02d}")
        elif start[0] in ("E", "e"):
            s = int(start[1:])
            e = int(end[1:])
            for n in range(s, e + 1):
                categories.append(f"E{n:03d}")
        else:
            s = int(start)
            e = int(end)
            for n in range(s, e + 1):
                categories.append(f"{n:03d}")
        return categories

    @staticmethod
    def _expand_range_icd10cm(start: str, end: str) -> list[str]:
        """Expand ICD-10-CM category range, e.g. ('A00','B99') → ['A00'..'B99'].

        Assumes 3-char `[A-Z]\\d\\d` category format. Silently returns an empty
        list when inputs don't match (covers empty or malformed PCS ranges).
        """
        if len(start) != 3 or len(end) != 3 or not start[0].isalpha() or not end[0].isalpha():
            return []
        try:
            s_letter, s_num = start[0], int(start[1:])
            e_letter, e_num = end[0], int(end[1:])
        except ValueError:
            return []
        categories: list[str] = []
        letter = ord(s_letter)
        while letter <= ord(e_letter):
            lo = s_num if letter == ord(s_letter) else 0
            hi = e_num if letter == ord(e_letter) else 99
            for n in range(lo, hi + 1):
                categories.append(f"{chr(letter)}{n:02d}")
            letter += 1
        return categories

    def _code_to_category(self, code: str) -> str:
        """Extract the retrieval-key category for a code, dispatching by code_type.

        - ICD-9: 3-digit root (V + 2 digits, E + 3 digits, or 3 digits).
        - ICD-10-CM: 3-char root [A-Z]\\d\\d (e.g. 'J18.9' → 'J18').
        - ICD-10-PCS: 1st char (section letter, e.g. '0DB68ZX' → '0').
        """
        code = code.upper().replace(".", "")
        if self.code_type == "ICD-9":
            if code.startswith("E"):
                return code[:4]
            if code.startswith("V"):
                return code[:3]
            return code[:3]
        # ICD-10: distinguish PCS (7-char, no dot) from CM.
        if self._ICD10_PCS_FORMAT_RE.match(code):
            return code[:1]
        return code[:3]

    # ------------------------------------------------------------------
    # BM25 search
    # ------------------------------------------------------------------

    def _tokenize_and_stem(self, text: str) -> list[str]:
        tokens = text.lower().split()
        return self._stemmer.stemWords(tokens)

    def _bm25_search(
        self, query: str, top_k: int = 20
    ) -> list[tuple[int, float]]:
        """BM25 search, returns [(entry_index, score), ...]."""
        query_tokens = self._tokenize_and_stem(query)
        results, scores = self._bm25.retrieve([query_tokens], k=min(top_k, len(self.entries)))
        return list(zip(results[0].tolist(), scores[0].tolist()))

    # ------------------------------------------------------------------
    # Ranking and formatting
    # ------------------------------------------------------------------

    def _rank_and_format(
        self,
        scored: list[tuple[int, float, str]],
        max_tokens: int,
    ) -> str:
        """Deduplicate, rank, truncate to token budget, and format."""
        if not scored:
            return ""

        # Deduplicate: keep highest score per entry index
        best: dict[int, tuple[float, str]] = {}
        for idx, score, source in scored:
            if idx not in best or score > best[idx][0]:
                best[idx] = (score, source)

        # Sort by score descending
        ranked = sorted(best.items(), key=lambda x: x[1][0], reverse=True)

        # Greedy packing — account for ~15 tokens formatting overhead per entry
        # (section header line + "- " prefix + newlines)
        PER_ENTRY_OVERHEAD = 15
        selected: list[dict] = []
        used_tokens = 0
        seen_sections: set[str] = set()

        for idx, (score, source) in ranked:
            entry = self.entries[idx]
            cost = entry["token_count"] + PER_ENTRY_OVERHEAD

            if used_tokens + cost > max_tokens:
                continue

            selected.append(entry)
            used_tokens += cost
            seen_sections.add(entry["section_id"])

        # Format and do a final token-count check
        result = self._format_guidelines(selected)
        actual_tokens = len(self._enc.encode(result))

        # If still over budget, trim entries from the end until we fit
        while actual_tokens > max_tokens and len(selected) > 1:
            selected.pop()
            result = self._format_guidelines(selected)
            actual_tokens = len(self._enc.encode(result))

        return result

    def _truncate_at_sentence(self, text: str, max_tokens: int) -> str:
        """Truncate text at the last complete sentence within max_tokens."""
        if len(self._enc.encode(text)) <= max_tokens:
            return text

        # Split into sentences and greedily pack
        sentences = re.split(r"(?<=[.!?])\s+", text)
        result: list[str] = []
        used = 0
        for sent in sentences:
            cost = len(self._enc.encode(sent))
            if used + cost > max_tokens:
                break
            result.append(sent)
            used += cost

        if result:
            return " ".join(result)
        # Single sentence exceeded the token budget — truncate on a word
        # boundary so the prompt doesn't end mid-token.
        return textwrap.shorten(sentences[0], width=200, placeholder="…")

    def _format_guidelines(self, entries: list[dict]) -> str:
        """Format selected entries into prompt-ready text."""
        if not entries:
            return ""

        # Sort by section_id for logical ordering
        entries = sorted(entries, key=lambda e: e["section_id"])

        parts = []
        current_section = ""
        for entry in entries:
            section = entry["section_id"]
            # Add section header when section changes
            parent = section.rsplit(".", 1)[0] if "." in section else section
            if parent != current_section:
                current_section = parent
                parts.append(f"\n[{section} — {entry['section_title'][:80]}]")

            # Add rule text, truncated at sentence boundary if too long
            text = entry["rule_text"].strip()
            if len(self._enc.encode(text)) > 150:
                text = self._truncate_at_sentence(text, max_tokens=150)
            parts.append(f"- {text}")

        return "\n".join(parts).strip()
