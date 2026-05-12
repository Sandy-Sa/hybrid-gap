import json
import os
import asyncio
import polars as pl
from pathlib import Path
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from coding.utils.reproducibility import set_global_seed, DEFAULT_SEED
from coding.utils.prompts import prompt_version
from coding.utils.helpers import get_id_columns, format_id

set_global_seed(DEFAULT_SEED)


system_prompt = {
    "role": "system",
    "content": """You are a clinical coder that reads medical notes and translates them into ICD-9 codes.

## Coding approach
For each condition or procedure in the note, follow these steps in your reasoning:
1. Identify the clinical concept exactly as documented.
2. Determine the most specific level of detail available: site, laterality, type/category, etiology, severity/stage, acuity (acute/chronic), and organism (if infection).
3. Map to the ICD-9 code that matches ALL of the documented specificity. If the note does not provide specificity, use the "unspecified" subcategory code.
4. Verify the code is a valid ICD-9 code using the format rules below.

## Instructions
- Read the medical note provided by the user.
- Identify all relevant diagnoses and procedures mentioned in the note.
- For each diagnosis and procedure, find the corresponding ICD-9 code.
- Provide a list of ICD-9 diagnosis codes and a list of ICD-9 procedure codes separately.
- The FIRST diagnosis code must be the principal diagnosis: the condition established after study to be chiefly responsible for occasioning the admission.
- Secondary diagnoses follow: conditions that coexist at admission, develop subsequently, or affect treatment.
- For each medical note, there must be at least one diagnosis code.
- If no relevant procedure is found, respond with an empty list for procedure codes.
- Use only valid ICD-9 codes.

## ICD-9 format rules
- Diagnosis codes: 3-5 digits, with a decimal point after the 3rd digit (e.g., "038.9", "486", "V58.11", "E878.8"). Range: 001-999, V01-V91, E800-E999.
- Procedure codes: 2-4 digits, with a decimal point after the 2nd digit (e.g., "35.25", "37.23", "88.72"). Range: 00.0-99.99.
- Do NOT use ICD-10 codes (those have a letter prefix like "A41.9" or "0BH17EZ").
- Code to the highest level of specificity supported by the documentation. If the code requires a 4th or 5th digit for specificity, provide it.

## Common errors to avoid
- Do not assign a 3-digit code if 4th/5th digit subcategories exist (e.g., use 428.0 not 428 alone; use 250.00 not 250).
- Do not confuse "acute" and "chronic" variants of the same condition.
- Do not assign codes for conditions that were "ruled out" or considered but not confirmed.
- Do not assign both a combination code and its individual components (e.g., if 403.91 covers hypertensive CKD stage V, do not separately assign 585.5 and 401.9).

## Uncertainty handling
- If you are uncertain about a specific code, use your best estimate and move on. Do NOT repeat yourself.

## Output format
- Return the results in JSON format with keys "diagnosis_codes" and "procedure_codes", each containing an array of code strings.
- Only respond with the JSON ICD-9 codes, without any additional commentary or description for codes.
- ONLY RETURN THE JSON, DO NOT RETURN ANY OTHER INFORMATION.
""",
}


def get_user_message(note: str) -> dict:
    return {"role": "user", "content": f"""Medical Note: {note}"""}


def get_assistant_message(diag_codes, proc_codes) -> dict:
    return {
        "role": "assistant",
        "content": f"""{{"diagnosis_codes": {diag_codes},
                            "procedure_codes": {proc_codes}
    }}""",
    }


def generate_messages(
    note: str,
    example: tuple[str, str, str] = None,
) -> list[dict]:
    messages = [system_prompt]
    if example:
        ex_note, diag_codes, proc_codes = example
        messages.append(get_user_message(ex_note))
        messages.append(get_assistant_message(diag_codes, proc_codes))

    messages.append(get_user_message(note))
    return messages


def postprocess_completion(completion) -> dict:
    try:
        content = json.loads(completion.choices[0].message.content)
    except json.JSONDecodeError:
        content = {"diagnosis_codes": [], "procedure_codes": []}
    reasoning = getattr(completion.choices[0].message, "reasoning", None)
    finish_reason = completion.choices[0].finish_reason
    usage = completion.usage
    return {
        "model": completion.model,
        "finish_reason": finish_reason,
        "content": content,
        "reasoning": reasoning,
        "completion_tokens": usage.completion_tokens,
        "prompt_tokens": usage.prompt_tokens,
        "total_tokens": usage.total_tokens,
    }


def write_completion_jsonl(completion, path: Path, **extra_fields) -> None:
    """
    Append a single completion record as one JSON line to `path`.
    """
    record = postprocess_completion(completion)
    record.update(extra_fields)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


class ICD9Response(BaseModel):
    diagnosis_codes: list[str] = Field(min_length=1)
    procedure_codes: list[str]


async def _generate_response_with_retry(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict],
    reasoning_effort: str,
    seed: int = None,
    max_tokens: int = None,
    max_retries: int = 3,
):
    json_schema = ICD9Response.model_json_schema()

    last_exc = None
    for attempt in range(max_retries):
        req_seed = seed + attempt if seed is not None else None
        if req_seed is not None:
            set_global_seed(req_seed)
        if attempt > 0:
            print(f"Retry attempt {attempt} with seed {req_seed}")

        kwargs = dict(
            model=model,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "icd9_response",
                    "schema": json_schema,
                },
            },
            seed=req_seed,
            temperature=0.0,
            extra_body={"repetition_penalty": 1.1},
        )
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        try:
            completion = await client.chat.completions.create(**kwargs)
            return completion, req_seed
        except Exception as e:
            last_exc = e
            print(f"Attempt {attempt} failed: {e}")

    raise last_exc


async def generate_responses(
    model: str,
    notes: pl.DataFrame,
    max_concurrent: int,
    output_path: Path,
    examples: tuple[str, str, str] = None,
    reasoning_effort: str = "low",
    seed: int = None,
    max_tokens: int = None,
) -> None:
    _prompt_version = prompt_version(system_prompt["content"])

    client = AsyncOpenAI(
        base_url=os.environ["OPENAI_BASE_URL"],
        api_key=os.environ.get("OPENAI_API_KEY", "-"),
    )
    id_cols = get_id_columns(notes)
    todo = list(reversed(notes.select(id_cols + ["text"]).rows(named=True)))
    pending = {}
    results = []

    while todo or pending:
        while todo and len(pending) < max_concurrent:
            item = todo.pop()
            coro = _generate_response_with_retry(
                client=client,
                model=model,
                messages=generate_messages(
                    note=item["text"],
                    example=examples if examples else None,
                ),
                reasoning_effort=reasoning_effort,
                seed=seed,
                max_tokens=max_tokens,
            )
            task = asyncio.create_task(coro)
            pending[task] = item

        # Wait: block until at least one task finishes
        done, _ = await asyncio.wait(
            pending.keys(), return_when=asyncio.FIRST_COMPLETED
        )

        for task in done:
            item = pending.pop(task)
            try:
                completion, req_seed = task.result()
                write_completion_jsonl(
                    completion=completion,
                    path=output_path,
                    seed=req_seed,
                    prompt_version=_prompt_version,
                    **{c: item[c] for c in id_cols},
                )
            except Exception as e:
                result = f"Error processing {format_id(item, id_cols)}: {str(e)}"
                results.append(result)
                print(
                    f"  done: {result}  (pending: {len(pending)}, remaining: {len(todo)})"
                )
    return results
