# DefExtra / DefSim Integration (AI-generated)

DISCLAIMER: This documentation was auto-generated with AI assistance. Please
verify commands, paths, and settings in your environment before relying on it.

This guide shows how SciDef connects to the two released dataset repos:
DefExtra (definition extraction) and DefSim (definition similarity).

## DefExtra (hydrated) -> SciDef ground truth

SciDef expects ground truth in JSON:
```
{
  "<paper_id>": {
    "<term>": {"definition": "...", "context": "...", "type": "explicit"}
  }
}
```

The DefExtra release ships a legal CSV with markers. You must hydrate it from
your own PDFs using the DefExtra scripts.

### 1) Hydrate DefExtra (DefExtra repo)

From the DefExtra repo:
```bash
uv run python scripts/hydrate_defextra.py \
  --legal-csv data/defextra_legal.csv \
  --pdf-dir /path/to/pdfs \
  --grobid-out /path/to/grobid_out \
  --output-csv /path/to/defextra_hydrated.csv \
  --report /path/to/defextra_hydrated_report.txt
```

Notes:
- `grobid_out` contains TEI XML files used by SciDef.
- The output TEI naming is `<paper_id>.grobid.tei.xml` (no `paper_` prefix).
  SciDef accepts both `<paper_id>.grobid.tei.xml` and
  `paper_<paper_id>.grobid.tei.xml`.

### 2) Convert hydrated CSV to SciDef JSON (SciDef repo)

Use the conversion utility:
```bash
uv run python scripts/defextra_csv_to_json.py \
  --csv /path/to/defextra_hydrated.csv \
  --output /path/to/defextra_hydrated.json
```

For a small sample:
```bash
uv run python scripts/defextra_csv_to_json.py \
  --csv /path/to/defextra_hydrated.csv \
  --output /path/to/defextra_hydrated_sample.json \
  --max-papers 3
```

### 3) Run extraction and evaluation (SciDef repo)

Example sample run against a local vLLM server:
```bash
export LLM_PROVIDER=vllm
export VLLM_BASE_URL=http://localhost:8000/v1
export VLLM_MODEL_NAME=openai/gpt-oss-20b
export VLLM_API_KEY=EMPTY
export WANDB_MODE=offline

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

Evaluate the extracted results with NLI:
```bash
WANDB_MODE=offline uv run python scripts/evaluate_extraction.py \
  --ground-truth-path /path/to/defextra_hydrated_sample.json \
  --extractions-dir /path/to/grobid_out \
  --extracted-definitions-path results/extraction/results_...json \
  --skip-train --skip-dev
```

## DefSim

DefSim is a separate dataset for definition similarity with short excerpts.
SciDef can evaluate similarity via:
- `scripts/benchmark_nli.py` (NLI models)
- `scripts/benchmark_embedding.py` (embedding models)
- `scripts/benchmark_judge.py` (LLM judges)

Use `--sample-size` for quick runs.
