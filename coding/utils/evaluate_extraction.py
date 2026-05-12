import os
import json
import asyncio
import polars as pl
from openai import AsyncOpenAI, APIStatusError
from pydantic import BaseModel, Field
from coding.utils.helpers import get_id_columns, format_id


def exact_match(text: str, anno_words: list[str]) -> bool:
    text_lower = text.lower().strip()
    return text_lower in [w.lower().strip() for w in anno_words]


class LLMMatchResponse(BaseModel):
    match: bool = Field(..., description="True if there is a match, False otherwise.")


_DIRECTIONS = [
    ("diagnosis_names", "words_icd9cm", "llm_match_diag_to_anno"),
    ("words_icd9cm", "diagnosis_names", "llm_match_anno_to_diag"),
    ("procedure_names", "words_icd9pcs", "llm_match_proc_to_anno"),
    ("words_icd9pcs", "procedure_names", "llm_match_anno_to_proc"),
]


_MATCH_SYSTEM_PROMPT = {
    "role": "system",
    "content": "You are a clinical terminology matching expert evaluating whether predicted clinical terms match reference ICD code descriptions.",
}


async def _llm_match_single(
    client: AsyncOpenAI,
    model: str,
    text: str,
    anno_words: list[str],
    max_retries: int = 3,
) -> bool:
    # Short-circuit on exact match to avoid unnecessary LLM calls
    if exact_match(text, anno_words):
        return True

    prompt = (
        f"Determine if the following predicted clinical term matches any term in the reference list.\n\n"
        f"Reference list (ICD code descriptions): {anno_words}\n"
        f'Predicted term: "{text}"\n\n'
        f"A match means the predicted term refers to the SAME clinical concept as a reference term, even if:\n"
        f'- Wording differs (e.g., "heart attack" matches "acute myocardial infarction")\n'
        f'- Abbreviations are used (e.g., "CHF" matches "congestive heart failure")\n'
        f"- One term is more specific but the general concept is the same\n\n"
        f"NOT a match if:\n"
        f'- Terms are related but describe different conditions (e.g., "atrial fibrillation" != "atrial flutter")\n'
        f'- One is a symptom and the other is the underlying disease (e.g., "chest pain" != "myocardial infarction")\n'
        f'- Terms share words but describe different things (e.g., "chronic kidney disease" != "chronic liver disease")\n\n'
        f"Return True if there is a match, False otherwise."
    )
    json_schema = LLMMatchResponse.model_json_schema()

    last_exc = None
    for attempt in range(max_retries):
        if attempt > 0:
            print(f"Retry attempt {attempt} for text: {text!r}")
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    _MATCH_SYSTEM_PROMPT,
                    {"role": "user", "content": prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "llm_match_response",
                        "schema": json_schema,
                    },
                },
            )
            raw = response.choices[0].message.content
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"LLM returned non-JSON content (first 200 chars): {raw[:200]!r}"
                ) from e
            return LLMMatchResponse(**parsed).match
        except APIStatusError as e:
            if e.status_code in (401, 403):
                raise
            last_exc = e
            print(f"Attempt {attempt} failed ({e.status_code}): {e.message}")
        except Exception as e:
            last_exc = e
            print(f"Attempt {attempt} failed: {e}")

    raise last_exc


async def evaluate_extraction(
    data: pl.DataFrame,
    model: str = "openai/gpt-oss-20b",
    max_concurrent: int = 10,
) -> pl.DataFrame:
    client = AsyncOpenAI(
        base_url=os.environ["OPENAI_BASE_URL"],
        api_key=os.environ.get("OPENAI_API_KEY", "-"),
    )
    id_cols = get_id_columns(data)

    rows = list(
        data.select(
            id_cols
            + [
                "diagnosis_names",
                "procedure_names",
                "words_icd9cm",
                "words_icd9pcs",
            ]
        ).iter_rows(named=True)
    )

    # Build flat work list: (row_idx, direction_name, item_idx, text, reference)
    todo = []
    for row_idx, row in enumerate(rows):
        for items_col, ref_col, direction_name in _DIRECTIONS:
            items = row[items_col]
            reference = row[ref_col]
            for item_idx, text in enumerate(items):
                todo.append((row_idx, direction_name, item_idx, text, reference))

    # Pre-allocate results grid: row_idx -> direction_name -> [None, ...]
    results_grid = {}
    for row_idx, row in enumerate(rows):
        results_grid[row_idx] = {}
        for items_col, _, direction_name in _DIRECTIONS:
            n_items = len(row[items_col])
            results_grid[row_idx][direction_name] = [False] * n_items

    total = len(todo)
    pending = {}
    processed = 0
    errors = []

    while todo or pending:
        while todo and len(pending) < max_concurrent:
            row_idx, direction_name, item_idx, text, reference = todo.pop(0)
            task = asyncio.create_task(_llm_match_single(client, model, text, reference))
            pending[task] = (row_idx, direction_name, item_idx)

        done, _ = await asyncio.wait(
            pending.keys(), return_when=asyncio.FIRST_COMPLETED
        )

        for task in done:
            key = pending.pop(task)
            row_idx, direction_name, item_idx = key
            try:
                result = task.result()
                results_grid[row_idx][direction_name][item_idx] = result
                processed += 1
                if processed % 10 == 0 or processed == total:
                    print(
                        f"  Progress: {processed}/{total} done"
                        f"  (pending: {len(pending)}, remaining: {len(todo)})"
                    )
            except Exception as e:
                err_msg = (
                    f"Error processing row {row_idx}, "
                    f"{direction_name}[{item_idx}]: {e}"
                )
                errors.append(err_msg)
                print(f"  {err_msg}")

    if errors:
        print(f"\n{len(errors)} errors occurred during evaluation.")

    # Reassemble into DataFrame
    collected = []
    for row_idx, row in enumerate(rows):
        row_ids = {c: [row[c]] for c in id_cols}
        row_results = {
            direction_name: [results_grid[row_idx][direction_name]]
            for _, _, direction_name in _DIRECTIONS
        }
        collected.append({**row_ids, **row_results})

    results = pl.concat([pl.DataFrame(d) for d in collected], how="vertical_relaxed")

    data = data.join(results, on=id_cols, how="left").with_columns(
        pl.col("llm_match_diag_to_anno")
        .list.mean()
        .fill_null(0)
        .alias("llm_diag_precision"),
        pl.col("llm_match_anno_to_diag")
        .list.mean()
        .fill_null(0)
        .alias("llm_diag_recall"),
        pl.col("llm_match_proc_to_anno")
        .list.mean()
        .fill_null(0)
        .alias("llm_proc_precision"),
        pl.col("llm_match_anno_to_proc")
        .list.mean()
        .fill_null(0)
        .alias("llm_proc_recall"),
    )

    return data
