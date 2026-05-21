# hybrid-gap

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)

Hybrid pipeline for automated ICD coding that combines a fine-tuned LLM classifier with LLM-based gap-filling, verification, and embedding-search code expansion over MIMIC-III/IV discharge summaries.

## Overview

The pipeline runs in three phases:

1. **Stage 1 — SFT.** A transformer LM is fine-tuned with a chosen label-prediction head (`classifier`, `label_attention`, or `label_cross_attention`) to produce calibrated per-code probabilities.
2. **Stage 2 — LLM gap-fill.** An LLM verifies low-confidence SFT codes, identifies missing codes ("gaps"), expands the candidate set via embedding search over the ICD vocabulary (optionally enriched with UMLS), and re-verifies the top-N candidates.
3. **Eval — paper-results producers.** Bootstrap CIs, paired tests, oracle ablation, per-chapter heatmaps, frequency-band breakdowns, and figures — every artifact the [reproduce_paper.ipynb](coding/eval/reproduce_paper.ipynb) notebook reads. Runs as a single DVC stage so `dvc repro` regenerates everything.

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

Full stage-by-stage breakdown: [docs/architecture.md](docs/architecture.md).

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
│   ├── utils/        # prompts, search, helpers, reproducibility, ICD validation
│   ├── baselines/    # Legacy vLLM-direct generation
│   └── eval/         # Paper-results producers + reproduce_paper.ipynb
├── data/splits/      # Pre-computed MIMIC train/val/test splits (feather)
├── docs/             # Architecture, configuration, data, usage, known issues
├── tests/            # CPU-safe smoke tests
├── dvc.yaml          # Pipeline definition
├── params.yaml       # All pipeline parameters
└── pyproject.toml    # Project metadata + ruff config
```

## Installation

The project pins a CUDA 12.9 PyTorch wheel and a single Python version (3.12). [`uv`](https://docs.astral.sh/uv/) is the supported installer:

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

Note that the `pip` path does **not** apply the CUDA 12.9 PyTorch wheel pin that `uv sync` does (see `[tool.uv.sources]` in [`pyproject.toml`](pyproject.toml)); install the matching `torch` wheel manually if you need GPU.

### First-time setup

Before the first `dvc repro`, edit [`params.yaml`](params.yaml) so the following keys point at your local copies of the external datasets (see [docs/data.md](docs/data.md) for download instructions):

| Key | Default (author's env) | What it should point at |
|---|---|---|
| `raw_data_path` | `/datasets/mimic/` | Directory containing the MIMIC CSVs consumed by `pre_process.py` |
| `code_description_path` | `/datasets/mimic/mimic-iv-3.1/hosp/` | Directory containing the ICD code description tables (MIMIC-IV `d_icd_*` files) |
| `guidelines_path` | `/home/sandy/repos/prompt-icd/data/guidelines/icd10_guidelines.parquet` | Parsed ICD-10 coding guidelines (optional; only used by the verification stage) |
| `search.mrconso_path` | `/datasets/umls/MRCONSO.RRF` | UMLS MRCONSO file (optional; only used by `search.py` for synonym expansion) |

Then copy [`.env.example`](.env.example) to `.env` and fill in the LLM endpoint.

## Configuration

There are two configuration surfaces:

- [`params.yaml`](params.yaml) — every pipeline knob. DVC re-runs a stage when one of its params changes. Reference: [docs/configuration.md](docs/configuration.md).
- `.env` — endpoint config for the LLM stages. Copy [`.env.example`](.env.example) to `.env` and fill in `OPENAI_BASE_URL` and `OPENAI_API_KEY` (any OpenAI-compatible chat-completions endpoint, e.g. a local vLLM server).

## Data

The pipeline consumes three credentialed external datasets:

- **MIMIC-III** and **MIMIC-IV** (PhysioNet, CITI + DUA required) — clinical notes and gold-standard ICD codes.
- **UMLS MRCONSO** (optional) — synonym enrichment for code search.
- **ICD-10 coding guidelines** (optional) — steers the LLM verification stage.

See [docs/data.md](docs/data.md) for download links, the `dataset_name` enum, and what each pre-computed split file in [`data/splits/`](data/splits/) represents.

## Reproducing the experiments

End-to-end:

```bash
dvc repro
```

This walks the DAG in [`dvc.yaml`](dvc.yaml). To run a single stage:

```bash
dvc repro <stage>     # e.g. dvc repro evaluate_sft
```

To run a script directly without DVC, copy the corresponding `cmd:` block from [`dvc.yaml`](dvc.yaml). Per-script CLI examples are in [docs/usage.md](docs/usage.md).

> **Note:** [`.dvc/config`](.dvc/config) does not currently configure a remote. To share artifacts between machines you must add one (`dvc remote add …`) and `dvc push`.

### Paper results

The `evaluate` stage produces every artifact under `outputs/eval/` consumed by the reproducibility notebook. Once the pipeline has run, open the notebook to render the paper's tables and figures:

```bash
dvc repro evaluate
uv run jupyter lab coding/eval/reproduce_paper.ipynb
```

Or render it headless:

```bash
uv run jupyter nbconvert --to notebook --execute \
    coding/eval/reproduce_paper.ipynb --output reproduce_paper.executed.ipynb
```

The notebook only reads cached artifacts — it does no recomputation. The dataset under evaluation is the one configured in `params.yaml` (`dataset_name`, `code_type`, `dataset_path`).

## Testing

```bash
uv run pytest -q
```

Tests are CPU-safe smoke checks (module imports, JSONL round-trip, seed determinism). CI runs `ruff check` and `pytest` on every push and PR.

## Documentation

| Doc | Contents |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Stage-by-stage data flow with mermaid diagram |
| [docs/configuration.md](docs/configuration.md) | Every `params.yaml` key, type, default, and consumer |
| [docs/data.md](docs/data.md) | External dataset acquisition and split files |
| [docs/usage.md](docs/usage.md) | Direct CLI invocation for every script |
| [docs/known-issues.md](docs/known-issues.md) | Cleanup backlog |

## Citation

If you use this code, please cite via [`CITATION.cff`](CITATION.cff). The citation block will be filled in after de-anonymization.

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Acknowledgements

This project relies on the MIMIC-III/IV datasets (PhysioNet, MIT-LCP), the UMLS Metathesaurus (NLM), and open-source releases of the foundation models used as the SFT backbone and the LLM verifier.
