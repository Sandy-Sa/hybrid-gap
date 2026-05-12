"""UMLS MRCONSO synonym loader for ICD code enrichment."""

from collections import defaultdict
from pathlib import Path

import polars as pl


# MRCONSO.RRF column names (pipe-delimited, no header)
MRCONSO_COLUMNS = [
    "CUI",
    "LAT",
    "TS",
    "LUI",
    "STT",
    "SUI",
    "ISPREF",
    "AUI",
    "SAUI",
    "SCUI",
    "SDUI",
    "SAB",
    "TTY",
    "CODE",
    "STR",
    "SRL",
    "SUPPRESS",
    "CVF",
]

# Source vocabularies to bridge synonyms through CUI
SYNONYM_SABS = frozenset(
    {
        "SNOMEDCT_US",
        "MSH",
        "NCI",
        "CSP",
        "AOD",
        "CHV",
        "MEDLINEPLUS",
        "NDFRT",
        "RXNORM",
    }
)


def load_umls_synonyms(
    mrconso_path: str | Path,
    icd_sab: str = "ICD9CM",
    lang: str = "ENG",
    bridge_synonyms: bool = True,
) -> dict[str, list[str]]:
    """
    Parse MRCONSO.RRF and return ICD code -> list of synonym strings.

    Two-stage process:
    1. Direct synonyms: rows where SAB matches ``icd_sab`` (e.g. ICD9CM).
       Groups by CODE to collect all unique STR values.
    2. CUI-bridged synonyms (if ``bridge_synonyms=True``): for each ICD
       code's CUI(s), collect STR from other clinical vocabularies
       (SNOMED-CT, MeSH, NCI, etc.) to capture richer synonym coverage.

    Parameters
    ----------
    mrconso_path : path to MRCONSO.RRF
    icd_sab : UMLS source abbreviation for the ICD version
        (e.g. "ICD9CM", "ICD10CM", "ICD10PCS")
    lang : language filter (default "ENG")
    bridge_synonyms : whether to include CUI-bridged synonyms from
        other vocabularies

    Returns
    -------
    dict mapping ICD code (str) -> list of unique synonym strings
    """
    mrconso_path = Path(mrconso_path)
    if not mrconso_path.exists():
        raise FileNotFoundError(f"MRCONSO.RRF not found at {mrconso_path}")

    # Read only the columns we need for efficiency
    # MRCONSO.RRF is pipe-delimited with trailing pipe, no header
    df = pl.read_csv(
        mrconso_path,
        separator="|",
        has_header=False,
        quote_char=None,
        truncate_ragged_lines=True,
        columns=[0, 1, 11, 13, 14],  # CUI, LAT, SAB, CODE, STR
    )
    df = df.rename(
        {
            "column_1": "CUI",
            "column_2": "LAT",
            "column_12": "SAB",
            "column_14": "CODE",
            "column_15": "STR",
        }
    )

    # Filter to target language
    df = df.filter(pl.col("LAT") == lang)

    # Stage 1: Direct ICD synonyms
    if icd_sab == "ICD9CM":
        sab = [icd_sab, "MTHICD9"]
        icd_rows = df.filter(pl.col("SAB").is_in(sab))
    elif icd_sab == "ICD10AM":
        sab = [icd_sab, "ICD10AMAE"]
        icd_rows = df.filter(pl.col("SAB").is_in(sab))
    elif icd_sab == "ICD10":
        sab = [icd_sab, "ICD10AE"]
        icd_rows = df.filter(pl.col("SAB").is_in(sab))
    else:
        icd_rows = df.filter(pl.col("SAB") == icd_sab)
    code_to_syns: dict[str, set[str]] = defaultdict(set)

    for row in icd_rows.iter_rows(named=True):
        code_to_syns[row["CODE"]].add(row["STR"])

    # Stage 2: CUI-bridged synonyms
    if bridge_synonyms:
        # Get CUI -> CODE mapping for ICD codes
        icd_cui_to_codes: dict[str, set[str]] = defaultdict(set)
        for row in icd_rows.select("CUI", "CODE").iter_rows(named=True):
            icd_cui_to_codes[row["CUI"]].add(row["CODE"])

        # Get all relevant CUIs
        target_cuis = set(icd_cui_to_codes.keys())

        # Filter bridge rows: matching CUIs from synonym vocabularies
        bridge_rows = df.filter(
            pl.col("CUI").is_in(target_cuis) & pl.col("SAB").is_in(SYNONYM_SABS)
        )

        for row in bridge_rows.iter_rows(named=True):
            cui = row["CUI"]
            syn_str = row["STR"]
            for code in icd_cui_to_codes[cui]:
                code_to_syns[code].add(syn_str)

    # Convert sets to sorted lists for determinism
    return {code: sorted(syns) for code, syns in code_to_syns.items()}


def expand_corpus(
    df_icd: pl.DataFrame,
    synonyms: dict[str, list[str]],
    code_col: str = "ICD9_CODE",
    title_col: str = "LONG_TITLE",
    max_synonyms: int = 10,
) -> tuple[list[str], list[int]]:
    """
    Build an expanded corpus where each synonym is a separate entry.

    Parameters
    ----------
    df_icd : DataFrame with ICD codes and titles
    synonyms : output of ``load_umls_synonyms``
    code_col : column name for ICD codes
    title_col : column name for long titles
    max_synonyms : max synonyms to add per code

    Returns
    -------
    corpus : list of strings (titles + individual synonyms)
    corpus_to_icd_idx : parallel list mapping each corpus entry to its
        ``df_icd`` row index
    """
    corpus: list[str] = []
    corpus_to_icd_idx: list[int] = []
    for i, row in enumerate(df_icd.select(code_col, title_col).iter_rows(named=True)):
        title = row[title_col]
        code = row[code_col]
        # Always include the original title
        corpus.append(title)
        corpus_to_icd_idx.append(i)
        # Add each unique synonym as a separate entry
        syns = synonyms.get(code, [])
        unique_syns = [s for s in syns if s.lower() != title.lower()][:max_synonyms]
        for syn in unique_syns:
            corpus.append(syn)
            corpus_to_icd_idx.append(i)
    return corpus, corpus_to_icd_idx
