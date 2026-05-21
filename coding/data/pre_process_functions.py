from pathlib import Path
import polars as pl
import re
from coding.utils.model_config import (
    ID_COLUMN,
    TEXT_COLUMN,
    SUBJECT_ID_COLUMN,
)


def pre_process_mimiciii(
    raw_notes: pl.LazyFrame,
    raw_diagnoses: pl.LazyFrame,
    raw_procedures: pl.LazyFrame,
    config,
    remove_lab: bool,
    processed_data_path: Path = Path("data/processed_data.parquet"),
) -> pl.DataFrame:
    """Preprocess MIMIC-III dataset."""

    notes = raw_notes.rename(
        {
            config["event_id"]: ID_COLUMN,
            config["note_text"]: TEXT_COLUMN,
            config["subject_id"]: SUBJECT_ID_COLUMN,
            config["note_id"]: "note_id",
            config["note_type"]: "note_type",
            config["note_subtype"]: "note_subtype",
        }
    )
    diagnoses = raw_diagnoses.rename(
        {
            config["event_id"]: ID_COLUMN,
            config["diagnoses_code"]: "diagnoses_code",
            config["diagnoses_sequence"]: "seq_num",
        }
    ).drop(config["subject_id"])

    procedures = raw_procedures.rename(
        {
            config["event_id"]: ID_COLUMN,
            config["procedure_code"]: "procedure_code",
            config["procedure_sequence"]: "seq_num",
        }
    ).drop(config["subject_id"])

    # Format the code type columns
    diagnoses = diagnoses.with_columns(diagnoses_code_type=pl.lit("icd9cm"))
    procedures = procedures.with_columns(procedure_code_type=pl.lit("icd9pcs"))

    # Reformatting the codes
    diagnoses = diagnoses.with_columns(
        pl.col("diagnoses_code").map_elements(
            reformat_icd9cm_code, return_dtype=pl.String
        )
    )
    procedures = procedures.with_columns(
        pl.col("procedure_code").map_elements(
            reformat_icd9pcs_code, return_dtype=pl.String
        )
    )

    # parse codes
    diagnoses = parse_codes_dataframe(
        diagnoses, "diagnoses_code", "diagnoses_code_type"
    )
    procedures = parse_codes_dataframe(
        procedures, "procedure_code", "procedure_code_type"
    )

    # Parse notes dataframe
    notes = parse_mimiciii_notes_dataframe(notes)

    if remove_lab:
        notes = notes.with_columns(
            pl.col(TEXT_COLUMN).map_elements(remove_lab_results, return_dtype=pl.String)
        )

    full_codes = diagnoses.join(
        procedures, on=ID_COLUMN, how="full", coalesce=True, suffix="_proc"
    )

    # Join notes with codes
    processed_data = notes.join(full_codes, on=ID_COLUMN, how="inner")
    processed_data = processed_data.rename(str.lower)

    result = processed_data.collect(engine="streaming")
    if processed_data_path is not None:
        result.write_parquet(processed_data_path)
    return result


def pre_process_mimiciii_clean(
    raw_notes: pl.LazyFrame,
    raw_diagnoses: pl.LazyFrame,
    raw_procedures: pl.LazyFrame,
    config,
    remove_lab: bool = True,
    processed_data_path: Path = Path("data/processed_data.parquet"),
) -> pl.DataFrame:
    """Preprocess MIMIC-III dataset."""
    data = pre_process_mimiciii(
        raw_notes,
        raw_diagnoses,
        raw_procedures,
        config,
        remove_lab,
        processed_data_path=None,
    )

    # only keep ICD-9 codes
    data = data.filter(
        (pl.col("diagnoses_code_type") == "icd9cm")
        | (pl.col("procedure_code_type") == "icd9pcs")
    )

    data = data.unique(subset=[ID_COLUMN])
    data = data.filter(
        pl.col("diagnoses_code").is_not_null() | pl.col("procedure_code").is_not_null()
    )
    data.write_parquet(processed_data_path)
    return data


def pre_process_mimiciv(
    raw_notes: pl.LazyFrame,
    raw_diagnoses: pl.LazyFrame,
    raw_procedures: pl.LazyFrame,
    config,
    remove_lab: bool = True,
    processed_data_path: Path = Path("data/processed_data.parquet"),
) -> pl.DataFrame:
    """Preprocess MIMIC-IV dataset."""

    # renaming columns to match the expected format
    notes = raw_notes.rename(
        {
            config["event_id"]: ID_COLUMN,
            config["note_text"]: TEXT_COLUMN,
            config["subject_id"]: SUBJECT_ID_COLUMN,
        }
    )

    diagnoses = raw_diagnoses.rename(
        {
            config["event_id"]: ID_COLUMN,
            config["diagnoses_code"]: "diagnoses_code",
            config["diagnoses_sequence"]: "seq_num",
            config["diagnoses_version"]: "diagnoses_code_type",
        }
    ).drop(config["subject_id"])

    procedures = raw_procedures.rename(
        {
            config["event_id"]: ID_COLUMN,
            config["procedure_code"]: "procedure_code",
            config["procedure_sequence"]: "seq_num",
            config["procedure_version"]: "procedure_code_type",
        }
    ).drop(config["subject_id"])

    # reformatting the code type columns
    diag_mapping = {"10": "icd10cm", "9": "icd9cm"}
    proc_mapping = {"10": "icd10pcs", "9": "icd9pcs"}

    diagnoses = diagnoses.with_columns(
        pl.col("diagnoses_code_type").cast(pl.String).replace(diag_mapping)
    )
    procedures = procedures.with_columns(
        pl.col("procedure_code_type").cast(pl.String).replace(proc_mapping)
    )

    # Format the diagnosis codes by adding punctuation points
    formatted_codes = (
        pl.when(pl.col("diagnoses_code_type") == "icd10cm")
        .then(
            pl.col("diagnoses_code").map_elements(
                reformat_icd10cm_code, return_dtype=pl.String
            )
        )
        .otherwise(
            pl.col("diagnoses_code").map_elements(
                reformat_icd9cm_code, return_dtype=pl.String
            )
        )
    )
    diagnoses = diagnoses.with_columns(formatted_codes)

    # Format the procedure codes by adding punctuation points
    formatted_codes = (
        pl.when(pl.col("procedure_code_type") == "icd10pcs")
        .then(pl.col("procedure_code"))
        .otherwise(
            pl.col("procedure_code").map_elements(
                reformat_icd9pcs_code, return_dtype=pl.String
            )
        )
    )
    procedures = procedures.with_columns(formatted_codes)

    # Parse codes
    diagnoses = parse_codes_dataframe(
        diagnoses, "diagnoses_code", "diagnoses_code_type"
    )
    procedures = parse_codes_dataframe(
        procedures, "procedure_code", "procedure_code_type"
    )

    # Parse notes dataframe
    notes = notes.filter(pl.col(TEXT_COLUMN).is_not_null()).unique(
        subset=[ID_COLUMN, TEXT_COLUMN]
    )
    if remove_lab:
        notes = notes.with_columns(
            pl.col(TEXT_COLUMN).map_elements(remove_lab_results, return_dtype=pl.String)
        )

    full_codes = diagnoses.join(procedures, on=ID_COLUMN, how="full", coalesce=True)

    # Join notes with codes
    processed_data = notes.join(full_codes, on=ID_COLUMN, how="inner")
    processed_data = processed_data.rename(str.lower)

    result = processed_data.collect(engine="streaming")
    result = result.filter(
        pl.col("diagnoses_code").is_not_null() | pl.col("procedure_code").is_not_null()
    )
    if processed_data_path is not None:
        result.write_parquet(processed_data_path)
    return result


def parse_mimiciii_notes_dataframe(df: pl.LazyFrame) -> pl.LazyFrame:
    """Parse the notes dataframe by filtering out notes with no text and removing duplicates."""

    df = df.filter(pl.col("note_type") == "Discharge summary")
    df = df.filter(pl.col(TEXT_COLUMN).is_not_null())
    df = df.sort(
        [ID_COLUMN, "note_subtype", "CHARTTIME", "CHARTDATE", "note_id"],
        descending=[False, True, False, False, False],
    )
    # join the notes with the same id and note type. This is to join discharge summaries and addendums.
    df = df.group_by([SUBJECT_ID_COLUMN, ID_COLUMN, "note_type"]).agg(
        pl.col("text").str.join(" "),
        pl.col("note_subtype").str.to_lowercase().str.join("+"),
        pl.col("note_id").str.join("+"),
    )
    return df


def parse_codes_dataframe(
    df: pl.LazyFrame,
    code_column: str = "diagnoses_code",
    code_type_column: str = "diagnoses_code_type",
) -> pl.LazyFrame:
    """Clean a dataframe of codes and aggregate them by event and type.

    The input should contain an identifier column and a column of codes with
    associated code types. Null and duplicate codes are removed and the
    remaining codes are grouped so each row holds a list of codes for a given
    identifier and code type.

    Example:
        Input ``pl.DataFrame``::

                subject_id  _id     target
                       2   163353     V3001
                       2   163353      V053
                       2   163353      V290

        Output ``pl.DataFrame``::

                _id     target
                163353  [V3001, V053, V290]

    Args:
        df (pl.LazyFrame): LazyFrame containing the codes.
        code_column (str): Name of the column with codes to aggregate.
        code_type_column (str): Name of the column describing the code type.

    Returns:
        pl.LazyFrame: LazyFrame grouped by identifier and code type with a list
        of codes in ``code_column``.
    """

    df = df.filter(pl.col(code_column).is_not_null())
    df = df.unique(subset=[ID_COLUMN, code_column])
    df = df.sort(
        [ID_COLUMN, code_type_column, "seq_num"], descending=[False, False, False]
    )
    first_alias = (
        "first_diagnosis_code"
        if code_column == "diagnoses_code"
        else f"first_{code_column}"
    )
    df = df.group_by([ID_COLUMN, code_type_column]).agg(
        pl.col(code_column),
        pl.col(code_column).first().alias(first_alias),
    )
    return df


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


def remove_rare_codes(
    df: pl.DataFrame, code_columns: list[str], min_count: int
) -> pl.DataFrame:
    """Removes codes that appear less than min_count times in the dataframe.

    Args:
        df (pl.DataFrame): dataframe with codes
        code_columns (list[str]): list of columns with codes
        min_count (int): minimum number of times a code has to appear in the dataframe to be kept

    Returns:
        pl.DataFrame: dataframe with codes that appear more than min_count times
    """
    for code_column in code_columns:
        code_exploded = (
            df[[ID_COLUMN, code_column]]
            .explode(code_column)
            .with_columns(
                pl.col(code_column).cum_count().over(ID_COLUMN).alias("seq_num")
            )
        )
        code_counts = code_exploded[code_column].value_counts()
        codes_to_include = set(
            code_counts.filter(code_counts["count"] >= min_count)[code_column]
        )
        code_exploded_filtered = code_exploded.filter(
            pl.col(code_column).is_in(codes_to_include)
        )
        first_alias = (
            "first_diagnosis_code"
            if code_column == "diagnoses_code"
            else f"first_{code_column}"
        )
        code_filtered = code_exploded_filtered.group_by(ID_COLUMN).agg(
            pl.col(code_column),
            pl.col("seq_num").alias(f"seq_num_{code_column}"),
            pl.col(code_column).first().alias(first_alias),
            pl.col("seq_num").first().alias(f"{first_alias}_seq_num"),
        )
        for drop_col in [first_alias, f"{first_alias}_seq_num"]:
            if drop_col in df.columns:
                df = df.drop(drop_col)
        df = df.drop(code_column)
        df = df.join(code_filtered, on=ID_COLUMN, how="left")
    return df


# Pre-compiled regexes for remove_lab_results (avoid recompilation per row)
_LAB_HEADER_RE = re.compile(r"^\[\*\*\d{4}-\d{1,2}-\d{1,2}\*\*\]\s+\d{1,2}:\d{2}[AP]M")
_LAB_CONTINUATION_RE = re.compile(
    r"^\s*"  # optional leading spaces
    r"([A-Z0-9()/]+"  # analyte name
    r"(?:\s+[A-Z0-9()/]+)*"  # optional extra words in the name
    r"-[-A-Za-z0-9.()*]+"  # dash + value (numbers/NEG/NotDone/etc)
    r"(?:\s+|$))+"  # spaces or end of line
)


def remove_lab_results(text: str) -> str:
    out_lines = []
    skipping = False  # are we currently skipping a lab block?

    for line in text.splitlines():
        # 1) Start of a lab block
        if _LAB_HEADER_RE.match(line):
            skipping = True
            continue  # drop header line

        # 2) Continuation of lab block (lab-style line) – drop it
        if (
            skipping
            and _LAB_CONTINUATION_RE.match(line)
            and not line.lstrip().startswith((".", "*"))
        ):
            continue

        # 3) Anything else ends the lab block and is kept
        skipping = False
        out_lines.append(line)

    return "\n".join(out_lines)
