import json
import os
import re
import asyncio
from typing import Any, Callable, Coroutine, TypeVar

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
)
from pydantic import BaseModel

from coding.utils.reproducibility import set_global_seed

T = TypeVar("T")
R = TypeVar("R")


def make_client() -> AsyncOpenAI:
    """Create an AsyncOpenAI client from environment variables."""
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not base_url:
        raise RuntimeError(
            "OPENAI_BASE_URL is not set. Source init.sh or export it before running."
        )
    return AsyncOpenAI(
        base_url=base_url,
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
    )


async def call_with_retry(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict],
    schema: type[BaseModel],
    seed: int | None = None,
    reasoning_effort: str | None = None,
    max_output_tokens: int | None = None,
    extra_body: dict | None = None,
    max_retries: int = 3,
) -> tuple[str, str, dict, str | None, bool, str, int | None]:
    """Make an API call with retry on transient failures.

    Args:
        client: AsyncOpenAI client instance.
        model: Model identifier string.
        messages: Chat messages list.
        schema: Pydantic model class used to derive response_format JSON schema.
        seed: Base seed; bumped by attempt index on retries.
        reasoning_effort: Optional reasoning effort level ("low", "medium", "high").
        max_output_tokens: Optional token cap for the response.
        extra_body: Optional extra request body fields (e.g. {"repetition_penalty": 1.1}).
        max_retries: Maximum number of attempts before raising.

    Returns:
        (content_str, finish_reason, token_usage, reasoning, parse_error, model_name, req_seed)
        - content_str: Raw response content string.
        - finish_reason: Stop reason from the API.
        - token_usage: Dict with completion_tokens, prompt_tokens, total_tokens.
        - reasoning: Chain-of-thought text if returned by the model, else None.
        - parse_error: True if content_str could not be parsed as JSON.
        - model_name: Resolved model identifier from the response.
        - req_seed: Actual seed used for the successful attempt.
    """
    if max_retries < 1:
        raise ValueError(f"max_retries must be >= 1, got {max_retries}")

    schema_name = re.sub(r"(?<!^)(?=[A-Z])", "_", schema.__name__).lower()
    json_schema = schema.model_json_schema()

    last_exc: BaseException | None = None
    for attempt in range(max_retries):
        req_seed = seed + attempt if seed is not None else None
        if req_seed is not None:
            set_global_seed(req_seed)
        if attempt > 0:
            print(f"Retry attempt {attempt} with seed {req_seed}")

        kwargs: dict[str, Any] = dict(
            model=model,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": json_schema,
                },
            },
            seed=req_seed,
            temperature=0.0,
        )
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        if max_output_tokens is not None:
            kwargs["max_tokens"] = max_output_tokens
        if extra_body is not None:
            kwargs["extra_body"] = extra_body

        try:
            response = await client.chat.completions.create(**kwargs)
            choice = response.choices[0]

            content_str = choice.message.content
            finish_reason = choice.finish_reason
            reasoning = getattr(choice.message, "reasoning", None)
            usage = response.usage
            token_usage = {
                "completion_tokens": usage.completion_tokens,
                "prompt_tokens": usage.prompt_tokens,
                "total_tokens": usage.total_tokens,
            }

            try:
                json.loads(content_str)
                parse_error = False
            except json.JSONDecodeError:
                parse_error = True

            return content_str, finish_reason, token_usage, reasoning, parse_error, response.model, req_seed

        except APIStatusError as e:
            if e.status_code in (401, 403):
                raise
            last_exc = e
            print(f"Attempt {attempt} failed ({e.status_code}): {e.message}")
        except (APIConnectionError, APITimeoutError, asyncio.TimeoutError) as e:
            last_exc = e
            print(f"Attempt {attempt} failed ({type(e).__name__}): {e}")

    if last_exc is None:
        raise RuntimeError(
            f"call_with_retry exhausted {max_retries} attempts without recording an exception"
        )
    raise last_exc


async def run_concurrent(
    items: list[T],
    coro_factory: Callable[[T], Coroutine[Any, Any, R]],
    on_done: Callable[[T, R], None],
    max_concurrent: int,
    task_timeout: float | None = None,
) -> dict:
    """Run items concurrently with bounded parallelism.

    Args:
        items: Work items to process, consumed in FIFO order.
        coro_factory: Callable that takes one item and returns a coroutine.
        on_done: Callback invoked with (item, result) on success. Handles
                 all module-specific output logic (JSONL writes, chunk merging, etc.).
        max_concurrent: Maximum number of tasks running simultaneously.
        task_timeout: Optional per-task wall-clock timeout in seconds. When set,
                      a hung task raises asyncio.TimeoutError and is recorded in
                      `errors` instead of blocking the queue. Default None = no
                      timeout (legacy behaviour).

    Returns:
        {"processed": int, "errors": list[str]}
    """
    todo = list(reversed(items))
    pending: dict[asyncio.Task, T] = {}
    processed = 0
    errors: list[str] = []
    total = len(items)

    def _wrap(item: T) -> Coroutine[Any, Any, R]:
        coro = coro_factory(item)
        if task_timeout is not None:
            return asyncio.wait_for(coro, timeout=task_timeout)
        return coro

    while todo or pending:
        while todo and len(pending) < max_concurrent:
            item = todo.pop()
            task = asyncio.create_task(_wrap(item))
            pending[task] = item

        done, _ = await asyncio.wait(pending.keys(), return_when=asyncio.FIRST_COMPLETED)

        for task in done:
            item = pending.pop(task)
            try:
                result = task.result()
                on_done(item, result)
                processed += 1
                if processed % 10 == 0 or processed == total:
                    print(
                        f"  Progress: {processed}/{total} done"
                        f"  (pending: {len(pending)}, remaining: {len(todo)})"
                    )
            except Exception as e:
                err_msg = f"Error processing item ({type(e).__name__}): {e}"
                errors.append(err_msg)
                print(f"  {err_msg}  (pending: {len(pending)}, remaining: {len(todo)})")

    return {"processed": processed, "errors": errors}
