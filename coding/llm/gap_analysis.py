import json
import polars as pl
from pathlib import Path
from coding.models.extraction import Evidence, Diagnosis, Procedure
from coding.models.gap_analysis import GapAnalysisResult
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
    normalization_rules,
    output_schema,
    combination_code_rule,
    etiology_manifestation_rule,
)
from coding.llm.client import make_client, call_with_retry, run_concurrent
from coding.llm.extraction import verify_evidence_quotes
from coding.pipeline.io import append_jsonl

set_global_seed(DEFAULT_SEED)


def build_system_prompt(code_type: str = "ICD-9") -> dict:
    """Build the gap-analyst system prompt for the given ICD code system.

    Code-system-specific text (combination-code example, etiology-manifestation
    rule, normalization-rules target dictionary, and output schema language)
    is injected via the factory helpers in `coding.utils.prompts`.
    """
    return {
        "role": "system",
        "content": f"""You are a clinical coding gap analyst. You are given a discharge summary together with a list of ICD codes that have already been assigned to it by an automated system. Your task is to identify any diagnoses and procedures that are documented in the note but are NOT covered by the assigned codes.

CRITICAL GOALS
- Aim for HIGH RECALL of MISSING items: find every clinically relevant diagnosis and procedure in the note that is not already represented by at least one of the assigned codes.
- Every identified item MUST include auditable evidence: an exact quoted span from the text + the section where it appears.
- Extract the MOST SPECIFIC terms possible as written in the note; do not generalize or infer beyond what is documented.
- PAY SPECIAL ATTENTION TO RARE AND UNCOMMON CONDITIONS: the automated system that assigned the existing codes is biased toward common diagnoses and procedures and frequently misses rare, atypical, or low-frequency conditions. Hunt specifically for unusual organisms, rare syndromes, uncommon complications, and atypical presentations that are explicitly documented in the note.

HOW TO DECIDE "MISSING"
- For each diagnosis/procedure you find in the note, check whether any of the assigned codes already covers it exactly. Only skip a concept if the assigned codes include an exact code for that same condition.
{combination_code_rule(code_type)}
- RISK FACTORS and UPSTREAM CONDITIONS are always coded separately from their sequelae: if hypertension, hyperlipidemia/dyslipidemia, or diabetes is documented, code them even if coronary artery disease, heart failure, CKD, or other sequelae are already assigned. Do NOT treat an assigned sequela as covering its underlying risk factor.
{etiology_manifestation_rule(code_type)}
- When in doubt, include the item — the downstream verifier will decide.
- If the note provides more specificity than an assigned code allows, still list it as missing — let the search and verification steps determine the best code.

{WHAT_TO_SCAN}
- Past Medical History (PMH) / active chronic conditions — always scan the PMH section. Chronic comorbidities documented there (hypertension, diabetes mellitus, hyperlipidemia, tobacco use, hypothyroidism, depression, anxiety, atrial fibrillation, coronary artery disease, etc.) must be listed as gaps if they are not already explicitly coded in the assigned list.
- Social History (SH) — scan for tobacco use/smoking, alcohol use, and substance use disorders. These are codeable conditions if documented (e.g., "smoker", "1 PPD tobacco use", "history of alcohol abuse").

INCLUSION RULES (what to extract as missing)
{INCLUSION_RULES_DIAGNOSES}

{INCLUSION_RULES_PROCEDURES}

{DO_NOT_INVENT}

{EVIDENCE_REQUIREMENTS}

{DEDUPLICATION}

UNCERTAINTY HANDLING
- If you are unsure whether an assigned code already covers a documented concept, err on the side of inclusion (favour recall). List it as missing and let the reviewer decide.
- If you cannot find an exact substring for an evidence quote, use the closest phrase that contains the clinical concept without paraphrasing. Do NOT fabricate quotes.
- Do not stall or repeat reasoning. Make a decision and move on.

OUTPUT FORMAT (STRICT)
Return ONLY valid JSON. No markdown. No commentary.
The JSON must have exactly two top-level keys: "diagnoses" and "procedures".
If no missing items are found, return empty arrays for both.

{normalization_rules(code_type)}

{output_schema(code_type)}
""",
    }


def _format_assigned_codes(codes: list[str], descriptions: list[str]) -> str:
    lines = []
    for i, (code, desc) in enumerate(zip(codes, descriptions), 1):
        lines.append(f"{i}. {code} - {desc}")
    return "\n".join(lines)


def generate_messages(
    note: str,
    codes: list[str],
    descriptions: list[str],
    code_type: str = "ICD-9",
) -> list[dict]:
    code_list = _format_assigned_codes(codes, descriptions)
    user_message = {
        "role": "user",
        "content": f"""Review the following discharge summary and the list of ICD codes already assigned.
Identify any diagnoses or procedures documented in the note that are NOT covered by the assigned codes.

<ASSIGNED_CODES>
{code_list}
</ASSIGNED_CODES>

<DISCHARGE_SUMMARY>
{note}
</DISCHARGE_SUMMARY>""",
    }
    return [build_system_prompt(code_type), user_message]


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
    items = notes.select(id_cols + ["text", "code", "description"]).rows(named=True)

    def coro_factory(item):
        messages = generate_messages(
            note=item["text"],
            codes=item["code"],
            descriptions=item["description"],
            code_type=code_type,
        )
        return call_with_retry(
            client,
            model,
            messages,
            GapAnalysisResult,
            seed=seed,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            extra_body={"repetition_penalty": 1.1},
        )

    def on_done(item, result):
        content_str, finish_reason, token_usage, reasoning, parse_error, model_name, req_seed = result
        if parse_error:
            content = {"diagnoses": [], "procedures": []}
        else:
            try:
                content = json.loads(content_str)
            except json.JSONDecodeError:
                content = {"diagnoses": [], "procedures": [], "specificity_upgrades": []}
                parse_error = True
        content = verify_evidence_quotes(content, item["text"])
        record = {
            "model": model_name,
            "finish_reason": finish_reason,
            "content": content,
            "reasoning": reasoning,
            "parse_error": parse_error,
            **token_usage,
            "seed": req_seed,
            "prompt_version": _prompt_version,
            **{c: item[c] for c in id_cols},
        }
        append_jsonl(output_path, record)

    return await run_concurrent(
        items, coro_factory, on_done, max_concurrent, task_timeout=task_timeout
    )
