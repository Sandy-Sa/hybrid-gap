"""Per-note Stage-2 cost aggregator for §4.5.

Reads the saved Stage-2 jsonl outputs (band-verify, gap-analysis, candidate-verify),
sums prompt/completion tokens and LLM-call counts per event_id, and prints means
plus a per-sub-pass breakdown. Retrieval has no LLM call (parquet only).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from coding.eval.paths import EVAL_DIR, OUTPUT_DIR, REPO_ROOT
from coding.eval.harness import SPEC

SUBPASSES = [
    ("band_verify", "in_band_verification.jsonl"),
    ("gap_analysis", "gap_analysis_pass2.jsonl"),
    ("candidate_verify", "searched_topn_verification.jsonl"),
]


def aggregate_dataset(dataset_dir: Path) -> dict:
    """Aggregate token / call counts per event_id, then mean across events."""
    per_event_subpass: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"prompt": 0, "completion": 0, "calls": 0})
    )
    event_ids: set[str] = set()
    for subpass, fname in SUBPASSES:
        path = dataset_dir / fname
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                eid = row["event_id"]
                event_ids.add(eid)
                per_event_subpass[eid][subpass]["prompt"] += int(row["prompt_tokens"])
                per_event_subpass[eid][subpass]["completion"] += int(row["completion_tokens"])
                per_event_subpass[eid][subpass]["calls"] += 1

    n_events = len(event_ids)
    out = {"n_events": n_events, "subpasses": {}}
    grand_prompt = grand_comp = grand_calls = 0
    for subpass, _ in SUBPASSES:
        tot_prompt = sum(per_event_subpass[e][subpass]["prompt"] for e in event_ids)
        tot_comp = sum(per_event_subpass[e][subpass]["completion"] for e in event_ids)
        tot_calls = sum(per_event_subpass[e][subpass]["calls"] for e in event_ids)
        out["subpasses"][subpass] = {
            "prompt_tokens_per_note": tot_prompt / n_events,
            "completion_tokens_per_note": tot_comp / n_events,
            "calls_per_note": tot_calls / n_events,
            "total_prompt_tokens": tot_prompt,
            "total_completion_tokens": tot_comp,
            "total_calls": tot_calls,
        }
        grand_prompt += tot_prompt
        grand_comp += tot_comp
        grand_calls += tot_calls
    out["totals"] = {
        "prompt_tokens_per_note": grand_prompt / n_events,
        "completion_tokens_per_note": grand_comp / n_events,
        "input_tokens_per_note": grand_prompt / n_events,
        "output_tokens_per_note": grand_comp / n_events,
        "calls_per_note": grand_calls / n_events,
    }
    return out


def fmt(n: float) -> str:
    """Compact human-readable formatter."""
    if n >= 100_000:
        return f"{n/1000:.0f}K"
    if n >= 10_000:
        return f"{n/1000:.1f}K"
    if n >= 1_000:
        return f"{n:,.0f}"
    return f"{n:.1f}"


def main() -> None:
    summary = {}
    dataset = SPEC.name
    dataset_dir = OUTPUT_DIR
    print(f"\n=== {dataset} ===")
    agg = aggregate_dataset(dataset_dir)
    summary[dataset] = agg
    print(f"  n_events = {agg['n_events']}")
    for sp, d in agg["subpasses"].items():
        print(f"  {sp:18s}: "
              f"prompt/note={fmt(d['prompt_tokens_per_note'])}, "
              f"completion/note={fmt(d['completion_tokens_per_note'])}, "
              f"calls/note={d['calls_per_note']:.2f}")
    t = agg["totals"]
    print(f"  TOTAL              : "
          f"prompt/note={fmt(t['prompt_tokens_per_note'])}, "
          f"completion/note={fmt(t['completion_tokens_per_note'])}, "
          f"calls/note={t['calls_per_note']:.2f}")

    out_path = EVAL_DIR / "stage2_cost_per_note.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
