"""Build `data/freq_bands.json` from the active dataset's train split.

Reads `${dataset_path}train_dataset.parquet` (produced by the `prepare` DVC
stage) and writes a JSON with:

  * `codes_per_band` — count of codes per Edin band (`>1000` / `101-1000` /
    `51-100` / `11-50` / `1-10` / `0`) plus the rank-based head/torso/tail
    cutoffs (head ≤ 50, torso ≤ 1000, tail otherwise).
  * `head_codes` — the top-50 most-frequent ICD codes (used by
    `coding.eval.tables.per_class_thresholds` for the head-code threshold
    pilot).

Run with::

    python -m coding.eval.build_freq_bands
"""

from __future__ import annotations

import json
from collections import Counter

import polars as pl

from coding.eval.paths import DATASET_DIR, FREQ_BANDS_PATH


HEAD_MAX = 50
TORSO_MAX = 1000


def _edin_bin(n: int) -> str:
    if n == 0:
        return "0"
    if n <= 10:
        return "1-10"
    if n <= 50:
        return "11-50"
    if n <= 100:
        return "51-100"
    if n <= 1000:
        return "101-1000"
    return ">1000"


def _train_occurrence_counts(train_path) -> Counter:
    df = pl.read_parquet(train_path)
    counts: Counter = Counter()
    for row in df.iter_rows(named=True):
        for c in (row.get("diagnoses_code") or []):
            counts[c] += 1
        for c in (row.get("procedure_code") or []):
            counts[c] += 1
    return counts


def main() -> int:
    train_path = DATASET_DIR / "train_dataset.parquet"
    print(f"counting code occurrences in {train_path} …")
    counts = _train_occurrence_counts(train_path)
    if not counts:
        print(f"warning: no codes counted from {train_path}")

    # Edin bins by count.
    edin_buckets: dict[str, int] = {
        ">1000": 0, "101-1000": 0, "51-100": 0, "11-50": 0, "1-10": 0, "0": 0,
    }
    for n in counts.values():
        edin_buckets[_edin_bin(n)] += 1

    # Rank bins (1-based).
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    n_codes = len(ranked)
    head = min(HEAD_MAX, n_codes)
    torso = max(0, min(TORSO_MAX, n_codes) - head)
    tail = max(0, n_codes - HEAD_MAX - torso)

    payload = {
        "n_codes_in_train": n_codes,
        "codes_per_band": {
            **edin_buckets,
            "head": head,
            "torso": torso,
            "tail_in_train": tail,
        },
        "head_codes": [code for code, _ in ranked[:HEAD_MAX]],
    }

    FREQ_BANDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FREQ_BANDS_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {FREQ_BANDS_PATH} (n_codes={n_codes}, head={head}, torso={torso}, tail={tail})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
