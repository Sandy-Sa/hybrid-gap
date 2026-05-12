"""Integration helpers for injecting coding guidelines into LLM prompts.

Works with both extraction (vllm_extraction.py) and verification
(vllm_verify.py) pipelines on the prompt_improvement branch.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coding.utils.guideline_search import GuidelineSearcher


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

GUIDELINE_INJECTION_TEMPLATE = """<CODING_GUIDELINES>
The following {code_system_label} coding guidelines are relevant to this case. Apply them when assigning or verifying codes:

{guidelines}
</CODING_GUIDELINES>"""


_CODE_SYSTEM_LABELS = {
    "ICD-9": "ICD-9-CM",
    "ICD-10": "ICD-10-CM/PCS",
}


def _format_template(guidelines: str, code_type: str) -> str:
    label = _CODE_SYSTEM_LABELS.get(code_type, "ICD")
    return GUIDELINE_INJECTION_TEMPLATE.format(
        guidelines=guidelines, code_system_label=label
    )


# ---------------------------------------------------------------------------
# Singleton searcher management
# ---------------------------------------------------------------------------

_searcher: GuidelineSearcher | None = None

# Template overhead in tokens (pre-computed for cl100k_base)
_TEMPLATE_OVERHEAD_TOKENS = 35


def init_guideline_searcher(
    guidelines_path: str | Path,
    max_tokens: int = 1500,
    code_type: str = "ICD-9",
) -> GuidelineSearcher:
    """Initialize (or reinitialize) the global GuidelineSearcher singleton."""
    global _searcher
    from coding.utils.guideline_search import GuidelineSearcher

    _searcher = GuidelineSearcher(
        guidelines_path=guidelines_path,
        max_tokens=max_tokens,
        code_type=code_type,
    )
    return _searcher


def get_searcher() -> GuidelineSearcher | None:
    """Return the current global GuidelineSearcher, or None if not initialized."""
    return _searcher


# ---------------------------------------------------------------------------
# Injection helpers
# ---------------------------------------------------------------------------

def get_extraction_guidelines(
    diagnoses_text: list[str] | None = None,
    max_tokens: int | None = None,
) -> str:
    """Get formatted guideline block for extraction prompts.

    Returns empty string if searcher is not initialized.
    """
    if _searcher is None:
        return ""
    effective_budget = (max_tokens or _searcher.max_tokens) - _TEMPLATE_OVERHEAD_TOKENS
    raw = _searcher.get_extraction_context(
        diagnoses_text=diagnoses_text,
        max_tokens=effective_budget,
    )
    if not raw:
        return ""
    return _format_template(raw, _searcher.code_type)


def get_verification_guidelines(
    candidate_codes: list[str],
    note_text: str = "",
    max_tokens: int | None = None,
) -> str:
    """Get formatted guideline block for verification prompts.

    Returns empty string if searcher is not initialized.
    """
    if _searcher is None:
        return ""
    effective_budget = (max_tokens or _searcher.max_tokens) - _TEMPLATE_OVERHEAD_TOKENS
    raw = _searcher.get_verification_context(
        candidate_codes=candidate_codes,
        note_text=note_text,
        max_tokens=effective_budget,
    )
    if not raw:
        return ""
    return _format_template(raw, _searcher.code_type)


def get_per_code_guidelines(
    candidate_codes: list[str],
    max_tokens_per_code: int = 80,
) -> list[str]:
    """Return a list of per-code guideline hints, aligned to input codes.

    Each hint is a concise string (~50-80 tokens) with the most relevant
    coding rule for that specific code. Designed for inline injection
    via the ``| Coding Guideline:`` field in ``_format_codes_for_prompt``.

    Returns empty strings for codes with no specific guidelines.
    Returns all empty strings if searcher is not initialized.
    """
    if _searcher is None:
        return [""] * len(candidate_codes)
    return [
        _searcher.get_per_code_guideline(code, max_tokens=max_tokens_per_code)
        for code in candidate_codes
    ]


def get_chunk_guidelines(
    candidate_codes: list[str],
    max_tokens: int = 500,
) -> str:
    """Return shared general guidelines for a chunk of codes.

    Focuses on always-on generals and cross-code interaction rules.
    Chapter-specific rules are excluded — those go in per-code hints.

    Returns empty string if searcher is not initialized.
    """
    if _searcher is None:
        return ""
    effective_budget = max_tokens - _TEMPLATE_OVERHEAD_TOKENS
    raw = _searcher.get_chunk_context(
        candidate_codes=candidate_codes,
        max_tokens=effective_budget,
    )
    if not raw:
        return ""
    return _format_template(raw, _searcher.code_type)


def inject_into_system_prompt(system_prompt_content: str, guideline_block: str) -> str:
    """Append a guideline block to the end of a system prompt string."""
    if not guideline_block:
        return system_prompt_content
    return system_prompt_content.rstrip() + "\n\n" + guideline_block


def inject_into_user_message(user_content: str, guideline_block: str) -> str:
    """Insert guideline block before <CODES_TO_VERIFY> in a verification user message.

    Falls back to prepending if the marker is not found.
    """
    if not guideline_block:
        return user_content
    marker = "<CODES_TO_VERIFY>"
    if marker in user_content:
        return user_content.replace(
            marker,
            guideline_block + "\n\n" + marker,
        )
    # Fallback: prepend
    return guideline_block + "\n\n" + user_content
