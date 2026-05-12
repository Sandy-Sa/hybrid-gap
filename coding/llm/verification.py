import json
import asyncio
import polars as pl
from pathlib import Path
from coding.models.verification import CodeVerdict, VerificationResponse
from coding.utils.helpers import get_id_columns, format_id
from coding.utils.prompts import prompt_version
from coding.utils.guideline_integration import (
    get_verification_guidelines,
    get_per_code_guidelines,
    inject_into_user_message,
)
from coding.llm.client import make_client, call_with_retry
from coding.pipeline.io import write_merged_jsonl

# NOTE: This module does NOT use run_concurrent() from coding.llm.client because
# verification requires chunk-merging logic: multiple LLM calls per note/term are
# issued in parallel, and their verdicts must be accumulated and merged before a
# single output record is written. This grouped-write pattern doesn't fit
# run_concurrent's simple one-item-one-callback model. The concurrency loops in
# generate_responses / generate_top_n_responses are structurally similar to
# run_concurrent but handle chunk grouping and partial-result merging inline.


# ── Prompt construction ───────────────────────────────────────────────────────

system_prompt = {
    "role": "system",
    "content": """You are a clinical coding verifier. For each candidate ICD code, you decide whether the medical note supports assigning it, grounded in evidence from the note itself.

## Procedure

For every code, follow these steps in order:

1. **Search the note** for evidence of the condition or procedure named by the code. Check all relevant sections: problem list, discharge diagnoses, HPI, past medical history, medications, labs and vitals, procedures, consults, and assessment/plan.

2. **Expand clinical shorthand.** Notes routinely use abbreviations (CHF, COPD, HTN, DM2, CKD V, AF, MI, s/p, h/o, etc.) and document conditions implicitly through medications (metformin → diabetes; furosemide → fluid overload; warfarin → AF/DVT/valve disease), labs (elevated troponin → MI; BNP → heart failure; A1c → diabetes), and procedures (dialysis → ESRD). Implicit evidence is valid evidence.

3. **Extract a verbatim quote** (~5–25 words) from the note that supports the code. Copy it exactly. If no supporting evidence exists anywhere in the note — neither explicit nor implicit — write "unavailable". Never paraphrase, never fabricate.

4. **Check for contradiction.** Does the note explicitly contradict the code? Common contradictions: laterality mismatch (code says left, note says right), acuity mismatch (code says acute, note says chronic stable), type mismatch (code says Type I, note says Type II), or explicit denial of the condition.

5. **Decide:**
   - **Confirmed** — evidence supports the code (explicit or implicit), and nothing in the note contradicts it.
   - **Rule out** — evidence is "unavailable", or the note directly contradicts the code.

## Rules

- **Only the note's content determines your decision.** Do not use a code's prior probability, frequency, or any external information as a reason to confirm or reject it.
- **Never invent evidence.** If you cannot point to a specific quote, the evidence is "unavailable".
- **Do not reject for under-specificity.** If a more specific code could arguably be used but the assigned code is consistent with the note, the code is Confirmed. Combination codes (e.g. 403.91, 250.4x) are Confirmed when one component is documented and the other is not contradicted.
- **Implicit evidence counts.** A code supported only by a medication, lab value, or abbreviation is Confirmed — quote the medication, lab, or abbreviation as the evidence.
- **Direct contradiction overrides supporting evidence.** If both are present, Rule out and record the contradiction quote.

## Output format

Return a JSON object with a single key "verdicts" containing an array. Each verdict has:
- "code": string (copied exactly from input)
- "evidence_quote": string — verbatim quote from the note supporting the code, or "unavailable"
- "contradiction_quote": string or null — verbatim quote directly contradicting the code, or null
- "decision": "Confirmed" or "Rule out"
- "correct": boolean (true iff decision is "Confirmed")

Rules:
- Return exactly ONE verdict for EACH code in the input list, in the SAME ORDER.
- `correct` must be true iff `decision` is "Confirmed".
- `decision` must be "Rule out" if `evidence_quote` is "unavailable" OR `contradiction_quote` is non-null.
- Return ONLY the JSON. No other text.
""",
}


def _format_codes_for_prompt(
    codes: list[str],
    descriptions: list[str],
    sft_prob: list[float],
    per_code_guidelines: list[str] | None = None,
) -> str:
    hints = per_code_guidelines or [""] * len(codes)
    lines = []
    for i, (code, desc, prob, hint) in enumerate(zip(codes, descriptions, sft_prob, hints)):
        line = f"{i + 1}. Code: {code} | Description: {desc} | SFT Probability: {prob:.4f}"
        if hint:
            line += f" | Coding Guideline: {hint}"
        lines.append(line)
    return "\n".join(lines)


def get_user_message(
    note: str,
    codes: list[str],
    descriptions: list[str],
    sft_prob: list[float],
    guideline_block: str = None,
    per_code_guidelines: list[str] | None = None,
    threshold_low: float = None,
    threshold_high: float = None,
) -> dict:
    codes_text = _format_codes_for_prompt(codes, descriptions, sft_prob, per_code_guidelines)
    n = len(codes)

    if threshold_low is not None and threshold_high is not None:
        threshold_context = (
            f"\nNote: These codes have low upstream confidence "
            f"(SFT Probability between {threshold_low:.2f} and {threshold_high:.2f}). "
            f"Low confidence means the upstream model was uncertain — it does NOT mean these codes are wrong. "
            f"Many of these conditions are genuinely present in the note despite the low score. "
            f"Only reject if you find direct clinical contradiction or can fully justify unambiguous absence.\n"
        )
    else:
        threshold_context = ""

    content = f"""For each candidate ICD code below, decide whether the medical note supports assigning it.

Follow this procedure for every code, in order:
1. Search the note for evidence of the condition or procedure named by the code. Check explicit mentions, problem lists, discharge diagnoses, HPI, medications, labs, procedures, consults, and standard clinical abbreviations (CHF, COPD, DM2, CKD, s/p, h/o, etc.).
2. Extract a short verbatim quote (~5–25 words) from the note that supports the code. If no supporting evidence exists anywhere in the note, write "unavailable" — do not fabricate or paraphrase.
3. Decide: Confirmed (evidence supports the code) or Rule out (evidence is "unavailable", or the note directly contradicts the code, e.g. laterality, acuity, or type mismatch).

<MEDICAL_NOTE>
{note}
</MEDICAL_NOTE>

<CODES_TO_VERIFY>
{codes_text}
</CODES_TO_VERIFY>

There are {n} codes. Return exactly {n} verdicts in the same order, as a JSON object with key "verdicts". Each verdict has:
- "code": string (copied from input)
- "evidence_quote": string — verbatim quote from the note, or "unavailable"
- "contradiction_quote": string or null — verbatim quote that directly contradicts the code, if any
- "decision": "Confirmed" or "Rule out"
- "correct": boolean (true iff decision is "Confirmed")

Return only the JSON."""
    content = inject_into_user_message(content, guideline_block)
    return {"role": "user", "content": content}


def generate_messages(
    note: str,
    codes: list[str],
    descriptions: list[str],
    sft_prob: list[float],
    guideline_block: str = None,
    per_code_guidelines: list[str] | None = None,
    threshold_low: float = None,
    threshold_high: float = None,
) -> list[dict]:
    messages = [system_prompt]
    messages.append(
        get_user_message(
            note,
            codes,
            descriptions,
            sft_prob,
            guideline_block,
            per_code_guidelines=per_code_guidelines,
            threshold_low=threshold_low,
            threshold_high=threshold_high,
        )
    )
    return messages


# ── Utilities ─────────────────────────────────────────────────────────────────

DEFAULT_CHUNK_SIZE = 20


def _normalize_code(code: str) -> str:
    """Normalize an ICD code for matching: strip dots/spaces, uppercase."""
    return code.replace(".", "").replace(" ", "").upper()


def _match_verdicts(
    input_codes: list[str], verdicts: list[dict]
) -> tuple[dict[str, dict], list[str]]:
    """
    Match returned verdicts to input codes by code string.

    Returns (matched dict {code: verdict}, list of unmatched input codes).
    """
    matched = {}
    exact_lookup = {}
    norm_lookup = {}
    for v in verdicts:
        vcode = v.get("code", "")
        if vcode not in exact_lookup:
            exact_lookup[vcode] = v
        norm = _normalize_code(vcode)
        if norm not in norm_lookup:
            norm_lookup[norm] = v

    missing = []
    for code in input_codes:
        if code in exact_lookup and code not in matched:
            matched[code] = exact_lookup.pop(code)
        elif _normalize_code(code) in norm_lookup:
            norm = _normalize_code(code)
            matched[code] = norm_lookup.pop(norm)
        else:
            missing.append(code)

    return matched, missing


def _chunk_list(lst: list, chunk_size: int) -> list[list]:
    """Split a list into chunks of at most chunk_size."""
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


# ── Chunk-level verification ──────────────────────────────────────────────────


async def _verify_chunk_with_retry(
    client,
    model: str,
    note: str,
    codes: list[str],
    descriptions: list[str],
    sft_prob: list[float],
    guideline_block: str = None,
    per_code_guidelines: list[str] | None = None,
    seed: int = None,
    reasoning_effort: str = None,
    max_output_tokens: int = None,
    max_retries: int = 3,
    threshold_low: float = None,
    threshold_high: float = None,
) -> tuple[list[dict], str, dict, str | None]:
    """
    Verify a chunk of codes with retry and partial recovery.

    Returns:
        verdicts: ordered list aligned to input codes; failed codes get correct=None
        model_name: resolved model identifier
        token_usage: summed token counts across all attempts
        reasoning: chain-of-thought from the first successful attempt, or None
    """
    all_matched = {}
    code_idx = {c: i for i, c in enumerate(codes)}
    remaining_codes = list(codes)
    remaining_descs = list(descriptions)
    remaining_sft_probs = list(sft_prob)
    remaining_per_code = list(per_code_guidelines) if per_code_guidelines else [""] * len(codes)
    total_tokens = {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}
    model_name = model
    reasoning = None

    for attempt in range(max_retries):
        req_seed = seed + attempt if seed is not None else None
        if attempt > 0:
            print(f"Retry attempt {attempt} with seed {req_seed}")

        prompt = generate_messages(
            note,
            remaining_codes,
            remaining_descs,
            remaining_sft_probs,
            guideline_block,
            per_code_guidelines=remaining_per_code,
            threshold_low=threshold_low,
            threshold_high=threshold_high,
        )

        try:
            content_str, finish_reason, usage, attempt_reasoning, parse_error, model_name, _ = \
                await call_with_retry(
                    client, model, prompt, VerificationResponse,
                    seed=req_seed, reasoning_effort=reasoning_effort,
                    max_output_tokens=max_output_tokens, max_retries=1,
                )
            verdicts = json.loads(content_str).get("verdicts", []) if not parse_error else []
            for k, v in usage.items():
                total_tokens[k] += v

            if reasoning is None and attempt_reasoning is not None:
                reasoning = attempt_reasoning

            matched, missing = _match_verdicts(remaining_codes, verdicts)
            all_matched.update(matched)

            if not missing:
                break

            print(
                f"Attempt {attempt}: matched {len(matched)}/{len(remaining_codes)}, "
                f"missing {len(missing)}"
            )

            remaining_codes = missing
            remaining_descs = [descriptions[code_idx[c]] for c in missing]
            remaining_sft_probs = [sft_prob[code_idx[c]] for c in missing]
            remaining_per_code = [
                (per_code_guidelines[code_idx[c]] if per_code_guidelines else "")
                for c in missing
            ]

        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")

    result = []
    for code in codes:
        if code in all_matched:
            v = dict(all_matched[code])
            v["code"] = code
            result.append(v)
        else:
            result.append(
                {
                    "code": code,
                    "evidence_quote": "unavailable",
                    "contradiction_quote": None,
                    "decision": "Rule out",
                    "correct": None,
                }
            )

    for v in result:
        eq = v.get("evidence_quote")
        if eq and eq not in note:
            v["evidence_quote"] = "unavailable"

    return result, model_name, total_tokens, reasoning


# ── Top-N term verification ───────────────────────────────────────────────────

system_prompt_top_n = {
    "role": "system",
    "content": """You are a clinical coding verifier. A medical term was extracted from a note and a ranked list of candidate ICD codes was retrieved. Your task is to identify which single code (if any) is best supported by the note for that specific term.

## Key constraint
At most ONE code can be correct for a given extracted term. The candidates are mutually exclusive alternatives for the same clinical concept. Either:
- Exactly one is Confirmed (the best-fitting code is present in the note), or
- All are Ruled out (none of the candidates matches what the note actually says about this term).

## Procedure

For every candidate code, follow these steps:

1. **Search the note** for evidence of the specific condition or procedure described by this code. Focus on whether it matches the extracted term.

2. **Expand clinical shorthand.** Abbreviations (CHF, COPD, HTN, DM2, etc.) and implicit evidence (medications, labs, procedures) are all valid.

3. **Extract a verbatim quote** (~5–25 words) supporting the code, or write "unavailable".

4. **Check for contradiction** — laterality, acuity, type mismatch, or explicit denial.

5. **Decide:** Confirmed or Rule out.

## Mutual exclusivity rule
If multiple codes would individually qualify as Confirmed, select only the most specific / best-fitting one and Rule out the rest. The note documents one condition per extracted term — pick the code that matches it most precisely.

## Output format

Return a JSON object with a single key "verdicts" containing an array. Each verdict has:
- "code": string (copied exactly from input)
- "evidence_quote": string — verbatim quote or "unavailable"
- "contradiction_quote": string or null
- "decision": "Confirmed" or "Rule out"
- "correct": boolean (true iff decision is "Confirmed")

Rules:
- Return exactly ONE verdict per code, in the SAME ORDER as the input.
- At most ONE verdict may have correct=true.
- Return ONLY the JSON. No other text.
""",
}


def _format_top_n_codes_for_prompt(
    codes: list[str],
    descriptions: list[str],
    rrf_scores: list[float],
    per_code_guidelines: list[str] | None = None,
) -> str:
    hints = per_code_guidelines or [""] * len(codes)
    lines = []
    for i, (code, desc, score, hint) in enumerate(zip(codes, descriptions, rrf_scores, hints)):
        line = f"{i + 1}. Code: {code} | Description: {desc} | Search Score: {score:.4f}"
        if hint:
            line += f" | Coding Guideline: {hint}"
        lines.append(line)
    return "\n".join(lines)


def get_top_n_user_message(
    note: str,
    term: str,
    codes: list[str],
    descriptions: list[str],
    rrf_scores: list[float],
    guideline_block: str = None,
    per_code_guidelines: list[str] | None = None,
) -> dict:
    codes_text = _format_top_n_codes_for_prompt(codes, descriptions, rrf_scores, per_code_guidelines)
    n = len(codes)

    content = f"""The following medical term was extracted from the note:

<EXTRACTED_TERM>
{term}
</EXTRACTED_TERM>

The candidate ICD codes below are the top-{n} search results for this term. At most ONE can be correct.

<MEDICAL_NOTE>
{note}
</MEDICAL_NOTE>

<CODES_TO_VERIFY>
{codes_text}
</CODES_TO_VERIFY>

There are {n} codes. Return exactly {n} verdicts in the same order, as a JSON object with key "verdicts". At most one verdict may have correct=true. Each verdict has:
- "code": string (copied from input)
- "evidence_quote": string — verbatim quote from the note, or "unavailable"
- "contradiction_quote": string or null — verbatim quote that directly contradicts the code, if any
- "decision": "Confirmed" or "Rule out"
- "correct": boolean (true iff decision is "Confirmed")

Return only the JSON."""
    content = inject_into_user_message(content, guideline_block)
    return {"role": "user", "content": content}


def generate_top_n_messages(
    note: str,
    term: str,
    codes: list[str],
    descriptions: list[str],
    rrf_scores: list[float],
    guideline_block: str = None,
    per_code_guidelines: list[str] | None = None,
) -> list[dict]:
    return [
        system_prompt_top_n,
        get_top_n_user_message(
            note, term, codes, descriptions, rrf_scores,
            guideline_block, per_code_guidelines,
        ),
    ]


async def _verify_top_n_chunk_with_retry(
    client,
    model: str,
    note: str,
    term: str,
    codes: list[str],
    descriptions: list[str],
    rrf_scores: list[float],
    guideline_block: str = None,
    per_code_guidelines: list[str] | None = None,
    seed: int = None,
    reasoning_effort: str = None,
    max_output_tokens: int = None,
    max_retries: int = 3,
) -> tuple[list[dict], str, dict, str | None]:
    """Verify a chunk of top-N candidate codes for a single extracted term.

    Returns:
        verdicts: ordered list aligned to input codes; failed codes get correct=None
        model_name: resolved model identifier
        token_usage: summed token counts across all attempts
        reasoning: chain-of-thought from the first successful attempt, or None
    """
    all_matched = {}
    code_idx = {c: i for i, c in enumerate(codes)}
    remaining_codes = list(codes)
    remaining_descs = list(descriptions)
    remaining_scores = list(rrf_scores)
    remaining_per_code = list(per_code_guidelines) if per_code_guidelines else [""] * len(codes)
    total_tokens = {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}
    model_name = model
    reasoning = None

    for attempt in range(max_retries):
        req_seed = seed + attempt if seed is not None else None
        if attempt > 0:
            print(f"Retry attempt {attempt} with seed {req_seed}")

        prompt = generate_top_n_messages(
            note,
            term,
            remaining_codes,
            remaining_descs,
            remaining_scores,
            guideline_block,
            per_code_guidelines=remaining_per_code,
        )

        try:
            content_str, finish_reason, usage, attempt_reasoning, parse_error, model_name, _ = \
                await call_with_retry(
                    client, model, prompt, VerificationResponse,
                    seed=req_seed, reasoning_effort=reasoning_effort,
                    max_output_tokens=max_output_tokens, max_retries=1,
                )
            verdicts = json.loads(content_str).get("verdicts", []) if not parse_error else []
            for k, v in usage.items():
                total_tokens[k] += v

            if reasoning is None and attempt_reasoning is not None:
                reasoning = attempt_reasoning

            matched, missing = _match_verdicts(remaining_codes, verdicts)
            all_matched.update(matched)

            if not missing:
                break

            print(
                f"Attempt {attempt}: matched {len(matched)}/{len(remaining_codes)}, "
                f"missing {len(missing)}"
            )

            remaining_codes = missing
            remaining_descs = [descriptions[code_idx[c]] for c in missing]
            remaining_scores = [rrf_scores[code_idx[c]] for c in missing]
            remaining_per_code = [
                (per_code_guidelines[code_idx[c]] if per_code_guidelines else "")
                for c in missing
            ]

        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")

    result = []
    for code in codes:
        if code in all_matched:
            v = dict(all_matched[code])
            v["code"] = code
            result.append(v)
        else:
            result.append(
                {
                    "code": code,
                    "evidence_quote": "unavailable",
                    "contradiction_quote": None,
                    "decision": "Rule out",
                    "correct": None,
                }
            )

    for v in result:
        eq = v.get("evidence_quote")
        if eq and eq not in note:
            v["evidence_quote"] = "unavailable"

    return result, model_name, total_tokens, reasoning


# ── Public entrypoints ────────────────────────────────────────────────────────


async def generate_top_n_responses(
    model: str,
    terms_df: pl.DataFrame,
    output_path: Path,
    reasoning_effort: str | None,
    seed: int | None,
    max_output_tokens: int | None,
    max_concurrent: int,
    chunk_size: int,
    top_n: int,
    task_timeout: float | None = None,
) -> list:
    """Generate verification responses for top-N candidate codes per extracted term.

    The `terms_df` DataFrame must contain one row per (note, term) with columns:
    id_cols, text, term, codes (List[str]), descriptions (List[str]),
    scores (List[float]), code_type.

    Each LLM call covers one term's top-N candidates. At most one code per term
    is expected to be correct (mutual exclusivity enforced in the prompt).

    Large candidate lists are split into chunks of `chunk_size` for reliability,
    though typically top_n <= chunk_size so each term is handled in one call.
    """
    if reasoning_effort is None:
        reasoning_effort = "low"

    _prompt_version = prompt_version(system_prompt_top_n["content"])
    client = make_client()

    id_cols = get_id_columns(terms_df)
    select_cols = id_cols + ["text", "term", "codes", "descriptions", "scores", "code_type"]
    rows = terms_df.select(select_cols).rows(named=True)

    ChunkItem = dict
    todo: list[ChunkItem] = []
    for row in rows:
        all_codes = row["codes"]
        all_descs = row["descriptions"]
        all_scores = row["scores"]
        all_per_code = get_per_code_guidelines(all_codes)
        code_chunks = _chunk_list(all_codes, chunk_size)
        desc_chunks = _chunk_list(all_descs, chunk_size)
        score_chunks = _chunk_list(all_scores, chunk_size)
        per_code_chunks = _chunk_list(all_per_code, chunk_size)
        id_fields = {c: row[c] for c in id_cols}
        key = tuple(row[c] for c in id_cols) + (row["term"], row["code_type"])
        for ci, (cc, dc, sc, pc) in enumerate(
            zip(code_chunks, desc_chunks, score_chunks, per_code_chunks)
        ):
            guideline_block = get_verification_guidelines(cc, row["text"])
            todo.append(
                {
                    "_key": key,
                    **id_fields,
                    "text": row["text"],
                    "term": row["term"],
                    "code_type": row["code_type"],
                    "codes": cc,
                    "descriptions": dc,
                    "scores": sc,
                    "per_code_guidelines": pc,
                    "guideline_block": guideline_block,
                    "chunk_idx": ci,
                    "total_chunks": len(code_chunks),
                }
            )

    todo.reverse()
    pending = {}
    note_chunks: dict[tuple, dict] = {}
    note_total_chunks: dict[tuple, int] = {}
    note_failed_chunks: dict[tuple, int] = {}
    results = []

    def _try_write_term(key: tuple, id_fields: dict, term: str, code_type: str):
        """Flush once every chunk for *key* is accounted for (success or failure).

        Verdicts and token_usage are merged only from the chunks that
        succeeded; a partial record is marked parse_error=True so downstream
        code can filter it. State dicts are always cleaned up on flush.
        """
        if key not in note_chunks:
            return
        chunks = note_chunks[key]
        received = len(chunks) + note_failed_chunks[key]
        if received < note_total_chunks[key]:
            return

        merged_verdicts = []
        total_usage = {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}
        model_name = None
        merged_reasoning = None
        for ci in sorted(chunks.keys()):
            verdicts, m, usage, chunk_reasoning = chunks[ci]
            merged_verdicts.extend(verdicts)
            model_name = m
            for k, v in usage.items():
                total_usage[k] += v
            if merged_reasoning is None and chunk_reasoning is not None:
                merged_reasoning = chunk_reasoning

        write_merged_jsonl(
            output_path,
            merged_verdicts=merged_verdicts,
            model=model_name,
            token_usage=total_usage,
            reasoning=merged_reasoning,
            parse_error=(note_failed_chunks[key] > 0),
            prompt_version=_prompt_version,
            term=term,
            code_type=code_type,
            top_n=top_n,
            **id_fields,
        )
        del note_chunks[key]
        del note_total_chunks[key]
        del note_failed_chunks[key]

    while todo or pending:
        while todo and len(pending) < max_concurrent:
            item = todo.pop()
            coro = _verify_top_n_chunk_with_retry(
                client=client,
                model=model,
                note=item["text"],
                term=item["term"],
                codes=item["codes"],
                descriptions=item["descriptions"],
                rrf_scores=item["scores"],
                guideline_block=item["guideline_block"],
                per_code_guidelines=item["per_code_guidelines"],
                seed=seed,
                reasoning_effort=reasoning_effort,
                max_output_tokens=max_output_tokens,
            )
            if task_timeout is not None:
                coro = asyncio.wait_for(coro, timeout=task_timeout)
            task = asyncio.create_task(coro)
            pending[task] = item

        done, _ = await asyncio.wait(
            pending.keys(), return_when=asyncio.FIRST_COMPLETED
        )

        for task in done:
            item = pending.pop(task)
            key = item["_key"]
            id_fields = {c: item[c] for c in id_cols}
            if key not in note_chunks:
                note_chunks[key] = {}
                note_total_chunks[key] = item["total_chunks"]
                note_failed_chunks[key] = 0
            try:
                verdicts, model_name, usage, reasoning = task.result()
                note_chunks[key][item["chunk_idx"]] = (
                    verdicts,
                    model_name,
                    usage,
                    reasoning,
                )
            except Exception as e:
                note_failed_chunks[key] += 1
                result = f"Error processing {format_id(item, id_cols)} term '{item['term']}' chunk {item['chunk_idx']}: {str(e)}"
                results.append(result)
                print(
                    f"  done: {result}  (pending: {len(pending)}, remaining: {len(todo)})"
                )
            _try_write_term(key, id_fields, item["term"], item["code_type"])

    return results


async def generate_responses(
    model: str,
    notes: pl.DataFrame,
    output_path: Path,
    reasoning_effort: str | None,
    seed: int | None,
    max_output_tokens: int | None,
    max_concurrent: int,
    chunk_size: int,
    extra_fields: dict | None,
    threshold_low: float | None,
    threshold_high: float | None,
    task_timeout: float | None = None,
) -> list:
    """
    Generate verification responses for notes with per-note codes.

    The `notes` DataFrame must contain columns: note_id, text, code, description.
    The `code` and `description` columns should be list types aligned by index.

    Relevant ICD-9-CM coding guidelines are retrieved automatically per note
    via the global GuidelineSearcher (initialized by init_guideline_searcher).

    Large code lists are split into chunks of `chunk_size` for reliability.
    """
    if reasoning_effort is None:
        reasoning_effort = "low"

    _prompt_version = prompt_version(system_prompt["content"])
    client = make_client()

    id_cols = get_id_columns(notes)
    select_cols = id_cols + ["text", "code", "description", "predictions"]
    rows = notes.select(select_cols).rows(named=True)

    ChunkItem = dict
    todo: list[ChunkItem] = []
    for row in rows:
        all_codes = row["code"]
        all_descs = row["description"]
        all_sft_probs = row.get("predictions", [0.3] * len(all_codes))
        all_per_code = get_per_code_guidelines(all_codes)
        code_chunks = _chunk_list(all_codes, chunk_size)
        desc_chunks = _chunk_list(all_descs, chunk_size)
        sft_prob_chunks = _chunk_list(all_sft_probs, chunk_size)
        per_code_chunks = _chunk_list(all_per_code, chunk_size)
        id_fields = {c: row[c] for c in id_cols}
        key = tuple(row[c] for c in id_cols)
        for ci, (cc, dc, sp, pc) in enumerate(
            zip(code_chunks, desc_chunks, sft_prob_chunks, per_code_chunks)
        ):
            guideline_block = get_verification_guidelines(cc, row["text"])
            todo.append(
                {
                    "_key": key,
                    **id_fields,
                    "text": row["text"],
                    "codes": cc,
                    "descriptions": dc,
                    "predictions": sp,
                    "per_code_guidelines": pc,
                    "guideline_block": guideline_block,
                    "chunk_idx": ci,
                    "total_chunks": len(code_chunks),
                }
            )

    todo.reverse()
    pending = {}
    note_chunks: dict[tuple, dict] = {}
    note_total_chunks: dict[tuple, int] = {}
    note_failed_chunks: dict[tuple, int] = {}
    results = []

    def _try_write_note(key: tuple, id_fields: dict):
        """Flush once every chunk for *key* is accounted for (success or failure).

        Verdicts and token_usage are merged only from the chunks that
        succeeded; a partial record is marked parse_error=True so downstream
        code can filter it. State dicts are always cleaned up on flush.
        """
        if key not in note_chunks:
            return
        chunks = note_chunks[key]
        received = len(chunks) + note_failed_chunks[key]
        if received < note_total_chunks[key]:
            return

        merged_verdicts = []
        total_usage = {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}
        model_name = None
        merged_reasoning = None
        for ci in sorted(chunks.keys()):
            verdicts, m, usage, chunk_reasoning = chunks[ci]
            merged_verdicts.extend(verdicts)
            model_name = m
            for k, v in usage.items():
                total_usage[k] += v
            if merged_reasoning is None and chunk_reasoning is not None:
                merged_reasoning = chunk_reasoning

        extra = extra_fields or {}
        write_merged_jsonl(
            output_path,
            merged_verdicts=merged_verdicts,
            model=model_name,
            token_usage=total_usage,
            reasoning=merged_reasoning,
            parse_error=(note_failed_chunks[key] > 0),
            prompt_version=_prompt_version,
            **id_fields,
            **extra,
        )
        del note_chunks[key]
        del note_total_chunks[key]
        del note_failed_chunks[key]

    while todo or pending:
        while todo and len(pending) < max_concurrent:
            item = todo.pop()
            coro = _verify_chunk_with_retry(
                client=client,
                model=model,
                note=item["text"],
                codes=item["codes"],
                descriptions=item["descriptions"],
                sft_prob=item.get("predictions", [0.3] * len(item["codes"])),
                guideline_block=item["guideline_block"],
                per_code_guidelines=item["per_code_guidelines"],
                seed=seed,
                reasoning_effort=reasoning_effort,
                max_output_tokens=max_output_tokens,
                threshold_low=threshold_low,
                threshold_high=threshold_high,
            )
            if task_timeout is not None:
                coro = asyncio.wait_for(coro, timeout=task_timeout)
            task = asyncio.create_task(coro)
            pending[task] = item

        done, _ = await asyncio.wait(
            pending.keys(), return_when=asyncio.FIRST_COMPLETED
        )

        for task in done:
            item = pending.pop(task)
            key = item["_key"]
            id_fields = {c: item[c] for c in id_cols}
            if key not in note_chunks:
                note_chunks[key] = {}
                note_total_chunks[key] = item["total_chunks"]
                note_failed_chunks[key] = 0
            try:
                verdicts, model_name, usage, reasoning = task.result()
                note_chunks[key][item["chunk_idx"]] = (
                    verdicts,
                    model_name,
                    usage,
                    reasoning,
                )
            except Exception as e:
                note_failed_chunks[key] += 1
                result = f"Error processing {format_id(item, id_cols)} chunk {item['chunk_idx']}: {str(e)}"
                results.append(result)
                print(
                    f"  done: {result}  (pending: {len(pending)}, remaining: {len(todo)})"
                )
            _try_write_note(key, id_fields)

    return results


async def generate_per_code_responses(
    model: str,
    notes: pl.DataFrame,
    output_path: Path,
    reasoning_effort: str | None,
    seed: int | None,
    max_output_tokens: int | None,
    max_concurrent: int,
    chunk_size: int,
    extra_fields: dict | None,
    threshold_low: float | None,
    threshold_high: float | None,
    task_timeout: float | None = None,
) -> list:
    """
    Generate verification responses for a flat DataFrame with one code per row.

    Unlike generate_responses(), this variant does not chunk or group codes by note.
    Each row is verified independently in a single LLM call.

    The `notes` DataFrame must contain columns: note_id, text, code, description, predictions.
    The `code` and `description` columns may be scalars or single-element lists.
    """
    if reasoning_effort is None:
        reasoning_effort = "low"

    _prompt_version = prompt_version(system_prompt["content"])
    client = make_client()

    id_cols = get_id_columns(notes)
    select_cols = id_cols + ["text", "code", "description", "predictions"]
    rows = notes.select(select_cols).rows(named=True)

    todo = []
    for row in rows:
        code = row["code"]
        description = row["description"]
        if isinstance(code, str):
            code = [code]
        if isinstance(description, str):
            description = [description]
        sft_prob_raw = row["predictions"]
        if isinstance(sft_prob_raw, list):
            sft_prob = sft_prob_raw
        else:
            sft_prob = [sft_prob_raw if sft_prob_raw is not None else 0.3]
        guideline_block = get_verification_guidelines(code, row["text"])
        per_code = get_per_code_guidelines(code)
        id_fields = {c: row[c] for c in id_cols}
        todo.append(
            {
                **id_fields,
                "text": row["text"],
                "codes": code,
                "descriptions": description,
                "predictions": sft_prob,
                "per_code_guidelines": per_code,
                "guideline_block": guideline_block,
            }
        )

    todo.reverse()
    pending = {}
    results = []

    while todo or pending:
        while todo and len(pending) < max_concurrent:
            item = todo.pop()
            coro = _verify_chunk_with_retry(
                client=client,
                model=model,
                note=item["text"],
                codes=item["codes"],
                descriptions=item["descriptions"],
                sft_prob=item.get("predictions", [0.3] * len(item["codes"])),
                guideline_block=item["guideline_block"],
                per_code_guidelines=item["per_code_guidelines"],
                seed=seed,
                reasoning_effort=reasoning_effort,
                max_output_tokens=max_output_tokens,
                threshold_low=threshold_low,
                threshold_high=threshold_high,
            )
            if task_timeout is not None:
                coro = asyncio.wait_for(coro, timeout=task_timeout)
            task = asyncio.create_task(coro)
            pending[task] = item

        done, _ = await asyncio.wait(
            pending.keys(), return_when=asyncio.FIRST_COMPLETED
        )

        for task in done:
            item = pending.pop(task)
            id_fields = {c: item[c] for c in id_cols}
            try:
                verdicts, model_name, usage, reasoning = task.result()
                write_merged_jsonl(
                    output_path,
                    merged_verdicts=verdicts,
                    model=model_name,
                    token_usage=usage,
                    reasoning=reasoning,
                    prompt_version=_prompt_version,
                    **id_fields,
                    **(extra_fields or {}),
                )
            except Exception as e:
                result = f"Error processing {format_id(item, id_cols)}: {str(e)}"
                results.append(result)
                print(
                    f"  done: {result}  (pending: {len(pending)}, remaining: {len(todo)})"
                )

    return results
