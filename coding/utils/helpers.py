import polars as pl
import re
import json
from typing import Optional


def get_base_model(identifier: str) -> str:
    """
    Extracts the "base model class" (e.g. 'llama3.2', 'qwen3') from strings like
    'meta-llama/Llama-3.2-1B' or 'Qwen/Qwen3-8B'.
    """
    m = re.match(r"^(?:.*/)?([A-Za-z]+)(?:[-_]?(\d+(?:\.\d+)*))?(?=[^/]*$)", identifier)
    if not m:
        raise ValueError(f"couldn't parse model from {identifier!r}")
    model, version = m.groups()
    if version is not None:
        return f"{model.lower()}{version}"
    return model.lower()


# Ordered list of candidate ID columns; checked in this order.
_ID_CANDIDATES = ["note_id", "event_id", "subject_id"]


def get_id_columns(df: pl.DataFrame) -> list[str]:
    """Return the ID columns present in *df*, used as a composite key.

    Raises ValueError if none of the candidate ID columns are found.
    """
    cols = [c for c in _ID_CANDIDATES if c in df.columns]
    if not cols:
        raise ValueError(
            f"DataFrame has no recognised ID columns. "
            f"Expected at least one of {_ID_CANDIDATES}, got {df.columns}"
        )
    return cols


def cast_id_columns_to_str(df: pl.DataFrame) -> pl.DataFrame:
    """Cast any present ID column (from _ID_CANDIDATES) to pl.String.

    No-op on DataFrames without ID columns. Keeps join-key dtypes consistent
    across every data source so `get_id_columns`-based joins never mismatch.
    """
    present = [c for c in _ID_CANDIDATES if c in df.columns]
    if not present:
        return df
    return df.with_columns([pl.col(c).cast(pl.String) for c in present])


def format_id(item: dict, id_cols: list[str]) -> str:
    """Format ID columns from an item dict for logging."""
    return ", ".join(f"{c}={item[c]}" for c in id_cols if c in item)


def ensure_dict(content, *, row_id: str = "<unknown>") -> dict:
    """Return *content* as a dict, parsing it if it's a JSON string.

    Used on `content` columns from LLM JSONL outputs where older rows may hold
    the raw JSON string and newer rows hold a pre-parsed dict. On malformed
    JSON, raises ValueError naming the offending row so batch pipelines fail
    fast with enough context to locate the bad record.
    """
    if isinstance(content, dict):
        return content
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON in LLM content for row {row_id}: {e}") from e


def filter_existing_responses(
    data: pl.DataFrame, existing_responses: pl.DataFrame
) -> pl.DataFrame:
    """Filter out notes that already have responses."""
    data_ids = get_id_columns(data)
    resp_ids = get_id_columns(existing_responses)
    id_cols = [c for c in data_ids if c in resp_ids]
    if not id_cols:
        raise ValueError(
            f"No common ID columns between data ({data_ids}) "
            f"and existing responses ({resp_ids})"
        )
    existing_ids = existing_responses.select([pl.col(c) for c in id_cols]).unique()
    filtered_data = data.join(existing_ids, on=id_cols, how="anti")
    return filtered_data


def one_hot_encode(target: list, class_map: dict) -> list:
    """
    One-hot encode a list of target labels.

    Args:
        target (list): The list of target labels to encode.
        class_map (dict): The mapping from target labels to indices.

    Returns:
        list: The one-hot encoded list.
    """
    num_classes = len(class_map)
    one_hot_list = [0] * num_classes
    target = set(target)
    for label in target:
        one_hot_list[class_map[label]] = 1
    return one_hot_list


def map_class(data: pl.LazyFrame, target_column: str) -> dict:
    """Create a mapping from class labels to indices based on the target column.
    Args:
        data (pl.LazyFrame): The input data containing the target column.
        target_column (str): The name of the target column to map.
    Returns:
        dict: A dictionary mapping class labels to indices.
    """
    # Extract unique class labels from
    # the target column and create a mapping
    # from class label to index.
    data_class = data[target_column].explode().unique().to_list()
    class_map = {data_class[i]: i for i in range(len(data_class))}
    return class_map


# ICD-9-CM / PCS patterns
_ICD9_NUMERIC_DX = re.compile(r"^[0-9]{3}(\.[0-9]{1,2})?$")
_ICD9_V_DX = re.compile(r"^V[0-9]{2}(\.[0-9]{1,2})?$")
_ICD9_E_DX = re.compile(r"^E[0-9]{3}(\.[0-9])?$")
_ICD9_PCS_WITH_DECIMAL = re.compile(r"^[0-9]{2}\.[0-9]{1,2}$")

# ICD-10-CM / PCS patterns (decimal optional for category-only diagnoses like "J18")
_ICD10_CM = re.compile(r"^[A-TV-Z][0-9][0-9A-Z](\.[A-Z0-9]{1,4})?$")
_ICD10_PCS = re.compile(r"^[A-HJ-NP-Z0-9]{7}$")


def validate_icd9(code: Optional[str]) -> bool:
    """Return True if *code* is a valid ICD-9-CM diagnosis, V-code, E-code, or
    ICD-9-PCS procedure code. Does NOT consider ICD-10 — use `validate_icd10`
    or `validate_code` with an explicit code_type instead.
    """
    if code is None:
        return False
    return bool(
        _ICD9_NUMERIC_DX.match(code)
        or _ICD9_V_DX.match(code)
        or _ICD9_E_DX.match(code)
        or _ICD9_PCS_WITH_DECIMAL.match(code)
    )


def validate_icd10(code: Optional[str]) -> bool:
    """Return True if *code* is a valid ICD-10-CM diagnosis or ICD-10-PCS
    procedure code. Does NOT consider ICD-9.
    """
    if code is None:
        return False
    return bool(_ICD10_CM.match(code) or _ICD10_PCS.match(code))


def validate_code(code: Optional[str], code_type: str) -> bool:
    """Validate *code* under the given ICD system. *code_type* is known at
    runtime and drives the choice of validator: callers must specify one of
    'ICD-9' or 'ICD-10' (case/separator-insensitive). Mixed-system validation
    is not supported.
    """
    key = code_type.upper().replace("-", "").replace("_", "")
    if key == "ICD9":
        return validate_icd9(code)
    if key == "ICD10":
        return validate_icd10(code)
    raise ValueError(f"Unsupported code_type: {code_type!r}. Use 'ICD-9' or 'ICD-10'.")


_ICD9_PCS_PATTERN = re.compile(r"^[0-9]{2}\.[0-9]{1,2}$")
_ICD10_PCS_PATTERN = re.compile(r"^[0-9A-HJ-NP-Z]{7}$")


def is_procedure_code(code: str, code_type: str) -> bool:
    """Return True if *code* is a procedure code under the given code system.

    ICD-9 procedures are 2-digit.1-or-2-digit (Volume 3 / ICD-9-PCS).
    ICD-10 procedures are the 7-character alphanumeric ICD-10-PCS codes
    (letters I and O excluded). Everything else is treated as a diagnosis.
    """
    if code is None:
        return False
    key = code_type.upper().replace("-", "").replace("_", "")
    if key == "ICD9":
        return bool(_ICD9_PCS_PATTERN.match(code))
    if key == "ICD10":
        return bool(_ICD10_PCS_PATTERN.match(code))
    raise ValueError(f"Unsupported code_type: {code_type!r}. Use 'ICD-9' or 'ICD-10'.")


def reformat_icd9cm_code(code: str) -> str:
    """
    Adds a punctuation in a ICD-9-CM code that is without punctuations.
    Before: 0019
    After:  001.9
    """
    if "." in code:
        return code

    if code.startswith("E"):
        if len(code) > 4:
            return code[:4] + "." + code[4:]
    elif len(code) > 3:
        return code[:3] + "." + code[3:]
    return code


def reformat_icd9pcs_code(code: str) -> str:
    """
    Adds a punctuation in a ICD-9-PCS code that is without punctuations.

    Before: 0019
    After:  00.19
    """

    if "." in code:
        return code
    if len(code) > 2:
        return code[:2] + "." + code[2:]
    return code


def reformat_icd10cm_code(code: str) -> str:
    """
    Adds a punctuation in a ICD-10-CM code that is without punctuations.
    Before: A019
    After:  A01.9
    """
    if "." in code:
        return code
    if len(code) > 3:
        return code[:3] + "." + code[3:]
    return code


# Manual descriptions for retired/missing ICD-9 codes not in MIMIC-III D_ICD tables.
# 36.01, 36.02, 36.05 were deleted in Oct 2005 and replaced by 00.66.
_MANUAL_ICD9_DESCRIPTIONS: dict[str, dict] = {
    "36.01": {
        "short_title": "PTCA single vessel",
        "long_title": "Single vessel percutaneous transluminal coronary angioplasty [PTCA] without mention of thrombolytic agent",
        "code_type": "procedure",
    },
    "36.02": {
        "short_title": "PTCA single w thrombo",
        "long_title": "Single vessel percutaneous transluminal coronary angioplasty [PTCA] with mention of thrombolytic agent",
        "code_type": "procedure",
    },
    "36.05": {
        "short_title": "PTCA multiple vessel",
        "long_title": "Multiple vessel percutaneous transluminal coronary angioplasty [PTCA]",
        "code_type": "procedure",
    },
}


def build_description_lookup(
    diagnoses_df: pl.DataFrame,
    procedures_df: pl.DataFrame,
) -> dict[str, dict]:
    """Build a flat lookup dict from ICD-9 description DataFrames.

    Expects DataFrames with columns: ICD9_CODE, SHORT_TITLE, LONG_TITLE.
    Codes should already be reformatted (with decimals).
    Includes manual overrides for retired codes missing from description tables.

    Returns dict mapping ICD9_CODE -> {short_title, long_title, code_type}.
    """
    lookup = dict(_MANUAL_ICD9_DESCRIPTIONS)
    for row in diagnoses_df.iter_rows(named=True):
        lookup[row["ICD9_CODE"]] = {
            "short_title": row["SHORT_TITLE"],
            "long_title": row["LONG_TITLE"],
            "code_type": "diagnosis",
        }
    for row in procedures_df.iter_rows(named=True):
        lookup[row["ICD9_CODE"]] = {
            "short_title": row["SHORT_TITLE"],
            "long_title": row["LONG_TITLE"],
            "code_type": "procedure",
        }
    return lookup


def resolve_icd9_description(
    code: str,
    lookup: dict[str, dict],
) -> tuple[dict | None, str]:
    """Resolve an ICD-9 code to its description using cascading match.

    Strategy (in order):
        1. Exact match
        2. Trailing zero padding (e.g. "999.8" -> "999.80")
        3. Parent prefix match, preferring .9 (unspecified) subcategory

    Returns (description_dict, match_strategy).
    match_strategy is one of: "exact", "trailing_zero", "parent_prefix", "unmatched".
    """
    if code is None:
        return None, "unmatched"
    # 1. Exact match
    if code in lookup:
        return lookup[code], "exact"

    # 2. Trailing zero: try padding (e.g. "999.8" -> "999.80")
    #    and stripping (e.g. "719.70" -> "719.7")
    if "." in code:
        padded = code + "0"
        if padded in lookup:
            return lookup[padded], "trailing_zero"
        if code.endswith("0") and len(code) > 4:
            stripped = code.rstrip("0")
            if stripped in lookup:
                return lookup[stripped], "trailing_zero"
    else:
        # Category-only code (e.g. "585") - try adding ".0" then ".00"
        for suffix in (".0", ".00"):
            if code + suffix in lookup:
                return lookup[code + suffix], "trailing_zero"

    # 3. Parent prefix match - find all codes starting with this prefix
    prefix = code + "." if "." not in code else code
    candidates = [k for k in lookup if k.startswith(prefix)]
    if candidates:
        # Prefer .9 (unspecified) variant
        unspecified = [c for c in candidates if c.endswith("9")]
        if unspecified:
            best = sorted(unspecified)[0]
        else:
            best = sorted(candidates)[0]
        return lookup[best], "parent_prefix"

    return None, "unmatched"


def resolve_icd10_description(
    code: str | None,
    lookup: dict[str, dict],
) -> tuple[dict | None, str]:
    """Resolve an ICD-10 code to its description.

    Strategy (in order):
        1. Exact match
        2. ICD-10-CM (has dot): truncate subcategory suffix one char at a time,
           then try category-only (e.g. J18.9X -> J18.9 -> J18)
        3. ICD-10-PCS (7-char, no dot): exact match only — no heuristics apply

    Returns (description_dict, match_strategy).
    match_strategy is one of: "exact", "parent_prefix", "unmatched".
    """
    if code is None:
        return None, "unmatched"
    # 1. Exact match
    if code in lookup:
        return lookup[code], "exact"
    # 2. ICD-10-CM codes have a dot; truncate subcategory suffix
    if "." in code:
        prefix, sub = code.split(".", 1)
        for length in range(len(sub) - 1, 0, -1):
            candidate = prefix + "." + sub[:length]
            if candidate in lookup:
                return lookup[candidate], "parent_prefix"
        if prefix in lookup:
            return lookup[prefix], "parent_prefix"
    # 3. ICD-10-PCS: no further heuristics
    return None, "unmatched"


def resolve_all_descriptions(
    codes_df: pl.DataFrame,
    diagnoses_df: pl.DataFrame,
    procedures_df: pl.DataFrame,
    code_type: str = "ICD-9",
) -> pl.DataFrame:
    """Resolve descriptions for all codes using cascading match.

    Args:
        codes_df: DataFrame with at least an ICD9_CODE column (reformatted).
        diagnoses_df: D_ICD_DIAGNOSES with ICD9_CODE, SHORT_TITLE, LONG_TITLE.
        procedures_df: D_ICD_PROCEDURES with ICD9_CODE, SHORT_TITLE, LONG_TITLE.
        code_type: "ICD-9" or "ICD-10".

    Returns:
        DataFrame with columns: ICD9_CODE, SHORT_TITLE, LONG_TITLE, code_type, match_strategy.
        Also prints a summary of match strategies used.
    """
    lookup = build_description_lookup(diagnoses_df, procedures_df)
    unique_codes = codes_df.select("ICD9_CODE").unique().to_series().to_list()
    resolver = resolve_icd10_description if code_type == "ICD-10" else resolve_icd9_description

    records = []
    for code in unique_codes:
        desc, strategy = resolver(code, lookup)
        if desc is not None:
            records.append(
                {
                    "ICD9_CODE": code,
                    "SHORT_TITLE": desc["short_title"],
                    "LONG_TITLE": desc["long_title"],
                    "code_type": desc["code_type"],
                    "match_strategy": strategy,
                }
            )
        else:
            records.append(
                {
                    "ICD9_CODE": code,
                    "SHORT_TITLE": None,
                    "LONG_TITLE": None,
                    "code_type": None,
                    "match_strategy": strategy,
                }
            )

    result = pl.DataFrame(records)

    # Print summary
    summary = result.group_by("match_strategy").len().sort("match_strategy")
    print(f"{code_type} description resolution summary:")
    for row in summary.iter_rows(named=True):
        print(f"  {row['match_strategy']:20s} {row['len']:>6d} codes")

    unmatched = result.filter(pl.col("match_strategy") == "unmatched")
    if len(unmatched) > 0:
        print(f"\nUnmatched codes ({len(unmatched)}):")
        for code in unmatched["ICD9_CODE"].to_list():
            print(f"  {code}")

    return result


def filter_unverified_evidence(content: dict) -> dict:
    """Drop items where ALL evidence quotes failed verification.

    Items with at least one verified quote are kept.
    The 'specificity_upgrades' field is excluded (not fed back into the loop).
    """
    filtered = {}
    for category in ("diagnoses", "procedures"):
        kept = []
        for item in content.get(category, []):
            evidence = item.get("evidence", [])
            if not evidence:
                continue
            if any(ev.get("verified", False) for ev in evidence):
                kept.append(item)
        filtered[category] = kept
    return filtered


def parse_content(row_content):
    """Parse content from either extraction (JSON string) or gap_analysis (dict) format."""
    if isinstance(row_content, str):
        content = json.loads(row_content)
    else:
        content = row_content

    diag_normalized = []
    proc_normalized = []

    for d in content.get("diagnoses") or []:
        diag_normalized.append(d.get("normalized", d.get("name", "")))

    for p in content.get("procedures") or []:
        proc_normalized.append(p.get("normalized", p.get("name", "")))

    return {
        "diagnosis_normalized": diag_normalized,
        "procedure_normalized": proc_normalized,
    }
