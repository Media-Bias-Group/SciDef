# SciDef CLI Reference (AI-generated)

DISCLAIMER: This documentation was auto-generated with AI assistance. Please
verify commands, paths, and settings in your environment before relying on it.

This document focuses on practical CLI usage. For full argument lists and more
detail, also see `scripts/README.md`.

## Environment setup (common)

SciDef reads config from `.env` or environment variables. For local vLLM:

```bash
export LLM_PROVIDER=vllm
export VLLM_BASE_URL=http://localhost:8000/v1
export VLLM_MODEL_NAME=openai/gpt-oss-20b
export VLLM_API_KEY=EMPTY
```

Notes:

- NLI models are loaded with `transformers` on first run and may download
  weights from Hugging Face.
- Weights and Biases is used for logging. For offline runs:
  `export WANDB_MODE=offline`

All commands below assume you run them from the SciDef repo root.

## Scripts

### `scripts/pdf_to_grobid.py`

Convert a directory of PDFs into GROBID TEI XML files (requires a running
GROBID server).

```bash
uv run python scripts/pdf_to_grobid.py \
  --input_folder /path/to/pdfs \
  --output_folder /path/to/grobid_out \
  --config /path/to/config.json
```

### `scripts/extract_definitions.py`

Run the extraction pipeline over GROBID TEI XML files.

Key flags:

- `--input-dir` one or more TEI folders
- `--extractors` extraction strategies (DSPy and non-DSPy)
- `--chunk_modes` sentence/paragraph/section/three_sentence
- `--max-papers` limit papers for a quick sample
- `--base-api-url` for vLLM or other OpenAI-compatible servers
- `--dspy-program-path` if using a compiled DSPy program
- `--gt_path` optional JSON ground-truth to filter to known papers

Example (small sample):

```bash
uv run python scripts/extract_definitions.py \
  --input-dir /path/to/grobid_out \
  --output-dir results/extraction \
  --extractors MultiStepExtractor \
  --chunk_modes section \
  --max-papers 1 \
  --llm-model-name openai/gpt-oss-20b \
  --base-api-url http://localhost:8000/v1 \
  --api-key NONE
```

### `scripts/evaluate_extraction.py`

Evaluate extracted definitions using an NLI model.

Key flags:

- `--ground-truth-path` SciDef ground-truth JSON (see [DefExtra](https://huggingface.co/datasets/mediabiasgroup/DefExtra) integration)
- `--extractions-dir` GROBID TEI folders
- `--extracted-definitions-path` evaluate an existing extraction JSON
- `--nli-model-name` HF NLI model
- `--nli-threshold` correctness threshold
- `--skip-train/--skip-dev/--skip-test` to reduce runtime

Example (evaluate pre-extracted results on test split only):

```bash
WANDB_MODE=offline uv run python scripts/evaluate_extraction.py \
  --ground-truth-path /path/to/defextra_hydrated.json \
  --extractions-dir /path/to/grobid_out \
  --extracted-definitions-path results/extraction/results_...json \
  --skip-train --skip-dev
```

### `scripts/dspy_train.py`

Train or optimize a DSPy extractor using [DefExtra](https://huggingface.co/datasets/mediabiasgroup/DefExtra) ground truth and NLI-based
scoring.

Key flags:

- `--ground-truth-path` SciDef ground-truth JSON
- `--extractions-dir` GROBID TEI folders
- `--dspy-optimizer` (miprov2, bootstrap, bootstrap_random)
- `--eval-pre-dspy` to evaluate baseline before training
- `--two-step-extraction` enable section gating
Note: DSPy via LiteLLM expects vLLM model ids to be prefixed with
`hosted_vllm/` (e.g., `hosted_vllm/openai/gpt-oss-20b`).

Example (small, faster optimizer):

```bash
WANDB_MODE=offline uv run python scripts/dspy_train.py \
  --ground-truth-path /path/to/defextra_hydrated.json \
  --extractions-dir /path/to/grobid_out \
  --llm-model-name hosted_vllm/openai/gpt-oss-20b \
  --nli-model-name tasksource/ModernBERT-large-nli \
  --dspy-optimizer bootstrap \
  --mipro-auto light
```

### `scripts/defextra_csv_to_json.py`

Convert a hydrated [DefExtra](https://huggingface.co/datasets/mediabiasgroup/DefExtra) CSV to SciDef ground-truth JSON.

```bash
uv run python scripts/defextra_csv_to_json.py \
  --csv /path/to/defextra_hydrated.csv \
  --output /path/to/defextra_hydrated.json \
  --max-papers 5
```

### Benchmarks

- `scripts/benchmark_embedding.py`
- `scripts/benchmark_judge.py`
- `scripts/benchmark_nli.py`

All use `uv run python scripts/<script>.py` and can be sampled via
`--sample-size` for quick runs.
