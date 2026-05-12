import json
import re
import polars as pl
from pathlib import Path
from coding.models.extraction import (
    ClinicalNoteExtraction,
    Diagnosis,
    Evidence,
    Procedure,
)
from coding.utils.reproducibility import DEFAULT_SEED, set_global_seed
from coding.utils.helpers import get_id_columns, format_id
from coding.utils.prompts import (
    prompt_version,
    WHAT_TO_SCAN,
    INCLUSION_RULES_DIAGNOSES,
    INCLUSION_RULES_PROCEDURES,
    DO_NOT_INVENT,
    EVIDENCE_REQUIREMENTS,
    DEDUPLICATION,
    FEW_SHOT_EXAMPLE,
    OUTPUT_FORMAT,
    normalization_rules,
    output_schema,
)
from coding.llm.client import make_client, call_with_retry, run_concurrent
from coding.pipeline.io import append_jsonl

set_global_seed(DEFAULT_SEED)


def build_system_prompt(code_type: str = "ICD-9") -> dict:
    """Build the extraction system prompt for the given ICD code system."""
    return {
        "role": "system",
        "content": f"""You are a clinical documentation extractor. Your task is to extract ALL diagnoses and ALL procedures/interventions that are relevant to this hospital episode from the discharge summary text.

CRITICAL GOALS
- Aim for HIGH RECALL: capture EVERYTHING relevant that contributed to the episode or was actively evaluated/managed during the stay.
- Every extracted item MUST include auditable evidence: an exact quoted span from the text + the section where it appears.
- Extract the MOST SPECIFIC terms possible as written in the note; do not generalize or infer beyond what is documented.

{WHAT_TO_SCAN}

INCLUSION RULES (what to extract)
{INCLUSION_RULES_DIAGNOSES}

{INCLUSION_RULES_PROCEDURES}

{DO_NOT_INVENT}

{EVIDENCE_REQUIREMENTS}

{DEDUPLICATION}

UNCERTAINTY HANDLING
- If you are unsure whether a finding is a diagnosis or a procedure, extract it under the most appropriate category and move on.
- If you cannot find an exact substring for an evidence quote, use the closest phrase that contains the clinical concept without paraphrasing. Do NOT fabricate quotes.
- Do not stall or repeat reasoning. Make a decision and move on.

{FEW_SHOT_EXAMPLE}

{OUTPUT_FORMAT}

{normalization_rules(code_type)}

{output_schema(code_type)}
""",
    }


def get_user_message(note: str) -> dict:
    return {
        "role": "user",
        "content": f"""Extract diagnoses and procedures from the following discharge summary.
            <DISCHARGE_SUMMARY>
             {note}
            </DISCHARGE_SUMMARY>""",
    }


def generate_messages(note: str, code_type: str = "ICD-9") -> list[dict]:
    return [build_system_prompt(code_type), get_user_message(note)]


_WS_RE = re.compile(r"\s+")


def _normalize_whitespace(s: str) -> str:
    return _WS_RE.sub(" ", s).strip()


def _quote_in_note(quote: str, note_norm: str, note_norm_lower: str) -> bool:
    """Check whether *quote* appears in *note_norm* as a word-bounded, whitespace-
    normalised substring. Exact-case first, then case-insensitive fallback.

    Boundary rule: the char before/after the match must not be a word char
    (letter/digit/underscore). Start/end of string also count as boundaries.
    """
    q = _normalize_whitespace(quote)
    if not q:
        return False
    pattern = re.compile(r"(?:^|(?<=\W))" + re.escape(q) + r"(?:$|(?=\W))")
    if pattern.search(note_norm):
        return True
    return bool(re.search(pattern.pattern, note_norm_lower, flags=re.IGNORECASE))


def verify_evidence_quotes(content: dict, note: str) -> dict:
    """Flag evidence quotes that are not word-bounded substrings of the source note.

    Matching normalises whitespace on both sides (so multi-line / multi-space
    quotes still match), requires word-boundary delimiters so short quotes like
    "hyster" don't spuriously match inside "hysterectomy", and prefers exact-case
    match — falling back to case-insensitive only if the exact-case match fails.
    """
    note_norm = _normalize_whitespace(note)
    note_norm_lower = note_norm.lower()
    for category in ("diagnoses", "procedures", "specificity_upgrades"):
        for item in content.get(category, []):
            for ev in item.get("evidence", []):
                quote = ev.get("quote", "")
                ev["verified"] = _quote_in_note(quote, note_norm, note_norm_lower)
    return content


def postprocess_response(
    content_str: str,
    finish_reason: str,
    token_usage: dict,
    reasoning: str | None,
    parse_error: bool,
    model_name: str,
) -> dict:
    """Build a record dict from call_with_retry() output."""
    if parse_error:
        content = {"diagnoses": [], "procedures": []}
    else:
        try:
            content = json.loads(content_str)
        except json.JSONDecodeError:
            content = {"diagnoses": [], "procedures": []}
            parse_error = True
    return {
        "model": model_name,
        "finish_reason": finish_reason,
        "content": content,
        "reasoning": reasoning,
        "parse_error": parse_error,
        **token_usage,
    }


async def generate_responses(
    model: str,
    notes: pl.DataFrame,
    output_path: Path,
    reasoning_effort: str | None,
    seed: int | None,
    max_output_tokens: int | None,
    max_concurrent: int,
    code_type: str = "ICD-9",
    task_timeout: float | None = None,
) -> dict:

    system_prompt = build_system_prompt(code_type)
    _prompt_version = prompt_version(system_prompt["content"])
    client = make_client()
    id_cols = get_id_columns(notes)
    items = notes.select(id_cols + ["text"]).rows(named=True)

    def coro_factory(item):
        messages = generate_messages(item["text"], code_type=code_type)
        return call_with_retry(
            client,
            model,
            messages,
            ClinicalNoteExtraction,
            seed=seed,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            extra_body={"repetition_penalty": 1.1},
        )

    def on_done(item, result):
        content_str, finish_reason, token_usage, reasoning, parse_error, model_name, req_seed = result
        record = postprocess_response(
            content_str, finish_reason, token_usage, reasoning, parse_error, model_name
        )
        record["content"] = verify_evidence_quotes(record["content"], item["text"])
        record.update(
            seed=req_seed,
            prompt_version=_prompt_version,
            **{c: item[c] for c in id_cols},
        )
        append_jsonl(output_path, record)

    return await run_concurrent(
        items, coro_factory, on_done, max_concurrent, task_timeout=task_timeout
    )
