# hybrid-gap

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)

Research code for a hybrid pipeline that combines a fine-tuned LLM
classifier with LLM-based gap-filling, verification, and embedding-search
code expansion for automated ICD coding over MIMIC-III/IV discharge
summaries.

> **Status.** This repository accompanies a paper under double-blind review. It is mirrored to
> [anonymous.4open.science](https://anonymous.4open.science) for double-blind
> review. Author names, institutions, project paths, and the
> `[tool.uv.sources]` index inside [`pyproject.toml`](pyproject.toml) will be
> filled in after de-anonymisation.

## Pipeline overview

The system runs in three phases, wired together as a DVC DAG
([`dvc.yaml`](dvc.yaml)). Every knob is controlled from
[`params.yaml`](params.yaml); DVC re-runs a stage when one of its declared
params or dependencies changes.

1. **Stage 1 — Supervised fine-tuning (SFT).** A transformer LM
   (default `Qwen/Qwen3-4B-Base`) is fine-tuned with one of three
   label-prediction heads — `classifier`, `label_attention`, or
   `label_cross_attention` — to produce calibrated per-code probabilities.
   The `evaluate_sft` stage sweeps decision thresholds on the validation
   set and writes per-code probabilities for the test set.

2. **Stage 2 — LLM gap-fill.** An LLM (default `openai/gpt-oss-20b`
   served via vLLM) verifies low-confidence SFT codes, identifies missing
   codes ("gaps"), expands the candidate set via BM25 embedding search
   over the ICD vocabulary (optionally enriched with UMLS synonyms),
   and re-verifies the top-N candidates.

3. **Evaluation.** A single `evaluate` stage produces every artifact
   consumed by the [`reproduce_paper.ipynb`](coding/eval/reproduce_paper.ipynb)
   notebook: bootstrap confidence intervals, paired significance tests,
   oracle ablations, per-chapter heatmaps, frequency-band breakdowns,
   and figures. The notebook itself does no recomputation.

```mermaid
flowchart LR
  A[pre_processing] --> B[prepare]
  B --> C[train_classifier]
  C --> D[evaluate_sft]
  D --> E[verification_sft]
  E --> F[gap_analysis_pass2]
  F --> G[search_pass2]
  G --> H[verification_top_n]
  H --> I[evaluate]
```

## Repository layout

```text
.
├── pre_process.py prepare.py train.py evaluate_sft.py    # SFT entry points
├── verification.py gap_analysis.py search.py             # LLM stages
├── extraction.py                                         # standalone term extraction
├── coding/
│   ├── data/         # MIMIC-specific dataset loading and splits
│   ├── modules/      # PyTorch Lightning SFT heads
│   ├── llm/          # LLM orchestration: prompts → client → parsed responses
│   ├── models/       # Pydantic response schemas
│   ├── pipeline/     # Cross-stage glue (data_loader, gap_pipeline, io)
│   ├── utils/        # prompts, search, helpers, reproducibility, ICD/UMLS/guidelines
│   ├── baselines/    # Zero/few-shot LLM-only baseline
│   └── eval/         # Paper-results producers + reproduce_paper.ipynb
├── data/splits/      # Pre-computed MIMIC train/val/test splits (feather)
├── tests/            # CPU-safe smoke tests
├── dvc.yaml          # Pipeline definition
├── params.yaml       # All pipeline parameters
└── pyproject.toml    # Project metadata + ruff config
```

## External dependencies

The code itself is open-source, but reproducing the experiments requires
credentialed access to clinical datasets and to UMLS. None of these
resources are redistributed in this repository; you must obtain them
yourself and point the relevant `params.yaml` keys at your local copies.

### 1. MIMIC datasets (required)

The pipeline supports three dataset enums via the `dataset_name` key in
[`params.yaml`](params.yaml). The expected on-disk layout under
`raw_data_path` is defined in
[`coding/data/dataset_config.py`](coding/data/dataset_config.py):

| `dataset_name`     | Source                                                                                                | Expected sub-paths under `raw_data_path` |
|---|---|---|
| `MIMIC-III`        | [PhysioNet — MIMIC-III Clinical Database v1.4](https://physionet.org/content/mimiciii/1.4/)           | `mimic-iii-clinical-database-1.4/{NOTEEVENTS,DIAGNOSES_ICD,PROCEDURES_ICD}.csv.gz` |
| `MIMIC-III-CLEAN`  | Same as above, with the Edin et al. (2023) cleaning recipe                                            | Same as `MIMIC-III` |
| `MIMIC-IV`         | [PhysioNet — MIMIC-IV v2.2](https://physionet.org/content/mimiciv/2.2/) + [MIMIC-IV-Note v2.2](https://physionet.org/content/mimic-iv-note/2.2/) | `mimic-iv-note-deidentified-free-text-clinical-notes-2.2/note/discharge.csv.gz`, `mimic-iv-2.2/hosp/{diagnoses_icd,procedures_icd}.csv.gz` |

Access to MIMIC requires a PhysioNet account, completion of the CITI
"Data or Specimens Only Research" training, and a signed Data Use
Agreement. ICD code description tables
(`d_icd_diagnoses.csv.gz`, `d_icd_procedures.csv.gz`) come from the same
MIMIC release and are pointed to by `code_description_path`.

### 2. Train/val/test splits

The feather files in [`data/splits/`](data/splits/) are the
**train/val/test split files from Edin et al. (2023)**, included verbatim
so the SFT stage is reproducible without re-running their split
generation. The selected file is set by `prepare.split_path` in
`params.yaml`:

| File                              | Setting                          |
|---|---|
| `mimiciii_clean_splits.feather`   | MIMIC-III with Edin et al. cleaning |
| `mimiciii_full_splits.feather`    | MIMIC-III full code set          |
| `mimiciii_50_splits.feather`      | MIMIC-III top-50 codes           |
| `mimiciv_icd9_split.feather`      | MIMIC-IV (ICD-9 subset)          |
| `mimiciv_icd10_split.feather`     | MIMIC-IV (ICD-10)                |
| `mimiciv_icd10cm_splits.feather`  | MIMIC-IV (ICD-10-CM only)        |

Original split-generation code and methodology:
<https://github.com/JoakimEdin/medical-coding-reproducibility>

If you cite this work, please also cite Edin et al. (2023).

### 3. UMLS Metathesaurus (optional)

The `search_pass2` stage uses [UMLS](https://www.nlm.nih.gov/research/umls/)
`MRCONSO.RRF` to bridge LLM-extracted clinical terms to ICD codes via
SNOMED-CT, MeSH, NCI, and related vocabularies. Required loader columns
and source-vocabulary filter are listed in
[`coding/utils/umls.py`](coding/utils/umls.py).

Point `search.mrconso_path` at your `MRCONSO.RRF`. Set it to `null` to
skip UMLS enrichment entirely (search will fall back to BM25 over the
ICD description table only). UMLS requires a free [UTS account](https://uts.nlm.nih.gov/uts/).

### 4. ICD-9-CM / ICD-10-CM Official Coding Guidelines (optional)

The `verification_sft` and `verification_top_n` stages can inject the
relevant section of the **ICD Official Guidelines for Coding and Reporting**
into the verification prompt. The parser in
[`coding/utils/guideline_parser.py`](coding/utils/guideline_parser.py) reads
a parquet of guideline sections; both ICD-9-CM and ICD-10-CM variants are
supported.

- **ICD-10-CM** guidelines: <https://www.cdc.gov/nchs/icd/icd-10-cm/files.html>
- **ICD-9-CM** guidelines: <https://www.cdc.gov/nchs/icd/icd-9-cm/icd-9-cm-files.html>

Download the PDF for the year of interest and run the parser to produce
the parquet that `guidelines_path` expects. Set `guidelines_path` to
`null` to skip guideline injection.

### 5. LLM endpoint

All LLM stages talk to an **OpenAI-compatible chat-completions endpoint**.
The author's setup uses a local [vLLM](https://github.com/vllm-project/vllm)
server, but any compatible
endpoint will work. Configure with two environment variables, conventionally
placed in a `.env` at the repository root (loaded by `python-dotenv`):

```bash
# .env
OPENAI_BASE_URL=http://localhost:8000/v1
OPENAI_API_KEY=-
```

The model is selected by `model:` in `params.yaml` (default
`openai/gpt-oss-20b`). The same `.env` is consumed by the LLM-only
baseline in [`coding/baselines/`](coding/baselines/).

## Installation

The project pins a CUDA 12.9 PyTorch wheel and Python 3.12.
[`uv`](https://docs.astral.sh/uv/) is the supported installer:

```bash
# Runtime (GPU machine)
uv sync

# Dev tools (CPU is fine)
uv sync --group dev
```

A `pip` fallback exists via [`requirements.txt`](requirements.txt):

```bash
pip install -r requirements.txt
```

The `pip` path does **not** apply the CUDA 12.9 PyTorch wheel pin that
`uv sync` does (see `[tool.uv.sources]` in [`pyproject.toml`](pyproject.toml));
install the matching `torch` wheel manually if you need GPU.

### Configuring `params.yaml`

Before the first `dvc repro`, edit [`params.yaml`](params.yaml) so the
following keys point at your local copies of the external resources
above:

| Key                       | What it should point at |
|---|---|
| `raw_data_path`           | Directory containing the MIMIC archives consumed by `pre_process.py` (see table above). |
| `code_description_path`   | Directory containing the ICD code-description tables (`d_icd_*` from the same MIMIC release). |
| `guidelines_path`         | Parsed ICD coding-guidelines parquet — set to `null` to skip. |
| `search.mrconso_path`     | UMLS `MRCONSO.RRF` — set to `null` to skip synonym enrichment. |
| `dataset_name`            | One of `MIMIC-III`, `MIMIC-III-CLEAN`, `MIMIC-IV`. |
| `prepare.split_path`      | A file from [`data/splits/`](data/splits/) matching the dataset. |
| `model_name`              | Hugging Face ID of the SFT base model. |
| `model`                   | Hugging Face / vLLM ID of the LLM used for gap-fill and verification. |

The defaults reference the original author's scratch paths; override
them for your environment.

## Reproducing the experiments

End-to-end:

```bash
dvc repro
```

This walks the DAG in [`dvc.yaml`](dvc.yaml). To re-run a single stage:

```bash
dvc repro <stage>     # e.g. dvc repro evaluate_sft
```

To run a script directly without DVC, copy the corresponding `cmd:` block
from [`dvc.yaml`](dvc.yaml). Every script also accepts `--help` for the
full CLI (`jsonargparse`).

> **Note.** [`.dvc/config`](.dvc/config) currently points at a local
> symlink cache and does not configure a DVC remote. To share artifacts
> between machines, add one (`dvc remote add …`) and `dvc push`.

### Paper results

The `evaluate` stage produces every artifact under `outputs/eval/`
consumed by the reproducibility notebook:

```bash
dvc repro evaluate
uv run jupyter lab coding/eval/reproduce_paper.ipynb
```

Or render it headless:

```bash
uv run jupyter nbconvert --to notebook --execute \
    coding/eval/reproduce_paper.ipynb --output reproduce_paper.executed.ipynb
```

The dataset under evaluation is whichever is configured in `params.yaml`
(`dataset_name`, `code_type`, `dataset_path`). Re-running the notebook
on a different dataset means re-running `dvc repro` against an updated
`params.yaml`.

## Baselines

[`coding/baselines/`](coding/baselines/) contains a zero/few-shot LLM-only
baseline that bypasses the SFT classifier and the gap-fill pipeline
entirely — it asks the LLM to predict ICD codes directly from each note.
It is intentionally **not** part of `dvc repro`; it exists so the full
pipeline can be benchmarked against a "prompt-only" reference.

```bash
python -m coding.baselines.run_generate \
    --data_dir <path-to-notes.parquet> \
    --output_dir <path-to-output-dir> \
    --model openai/gpt-oss-20b \
    --reasoning_effort low
```

The baseline uses the same `.env` LLM endpoint as the main pipeline.
Outputs are JSONL of `{diagnosis_codes, procedure_codes}` per note,
scorable with the same evaluation code as the full pipeline.

## Testing

```bash
uv run pytest -q
```

Tests are CPU-safe smoke checks (module imports, JSONL round-trip, seed
determinism, evaluation bootstrap). CI runs `ruff check` and `pytest` on
every push and PR.

## Citation

This repository accompanies a paper under double-blind review at EMNLP
2026. A BibTeX entry and citation file will be added after the review
period.

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Acknowledgements

This work relies on the **MIMIC-III** and **MIMIC-IV** datasets
(PhysioNet, MIT-LCP), the **UMLS Metathesaurus** (U.S. National Library
of Medicine), the **ICD-9-CM / ICD-10-CM Official Guidelines for Coding
and Reporting** (CDC/NCHS), the train/val/test splits from
[Edin et al. (2023)](https://github.com/JoakimEdin/medical-coding-reproducibility),
and open-source releases of the foundation models used as the SFT
backbone and the LLM verifier.
