# Custom DSPy Extractors and Training (AI-generated)

DISCLAIMER: This documentation was auto-generated with AI assistance. Please
verify commands, paths, and settings in your environment before relying on it.

This guide explains how to create a custom DSPy extractor and train it with
[DefExtra](https://huggingface.co/datasets/mediabiasgroup/DefExtra) ground truth.

## Where the current DSPy extractor lives

The default DSPy extractor is in:
`scidef/extraction/extractor/dspy_extraction.py`

Key components:
- `ExtractPairsSig` and `DetermineIfDefinitionSig` (DSPy signatures)
- `ExtractFromSection` (section-level module)
- `DSPyPaperExtractor` (paper-level module, returns `merged_json`)

The `DSPyPaperExtractor` returns a JSON list of items with keys:
`term`, `definition`, `context`, `type`.

## Creating a custom DSPy extractor

1) Create a new module file, for example:
`scidef/extraction/extractor/my_dspy_extraction.py`

2) Implement a DSPy Module that accepts `sections` and returns a
`dspy.Prediction` with `merged_json` containing a JSON list of term records.

Minimal pattern:
```python
import json
import dspy

class MySig(dspy.Signature):
    section = dspy.InputField()
    extracted_terms = dspy.OutputField()

class MyExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.step = dspy.ChainOfThought(MySig)

    def forward(self, sections):
        all_pairs = []
        for sec in sections:
            out = self.step(section=sec)
            # Convert out.extracted_terms into list of dicts:
            # [{"term": ..., "definition": ..., "context": ..., "type": ...}, ...]
            all_pairs.extend(out.extracted_terms)
        return dspy.Prediction(merged_json=json.dumps(all_pairs, ensure_ascii=False))
```

3) Wire it into the CLI:
- Add the class to `scripts/extract_definitions.py` in `EXTRACTOR_CLASSES`.
- (Optional) Add any DSPy program load path or two-step gating flags.

## Providing your own ground truth

SciDef's default training/evaluation expects a JSON mapping:
```json
{
  "<paper_id>": {
    "<term>": {
      "definition": "...",
      "context": "...",
      "type": "explicit"
    }
  }
}
```

If you keep this schema, you can train/evaluate with
`scripts/dspy_train.py` and `scripts/evaluate_extraction.py` unchanged.

For a different task (e.g., "sentences about politicians"), update:
- The DSPy signature and output schema
- The dataset loader in `scidef/extraction/utils.py` (or create a new loader)
- The evaluation function in `scidef/evaluation/utils.py` (or create a new one)

### Example: "sentences about politicians"

**Goal**: return all sentences in a paper that mention politicians.

1) Define a new signature and module:
```python
class PoliticianSentenceSig(dspy.Signature):
    section = dspy.InputField()
    sentences = dspy.OutputField(desc="List of sentences mentioning politicians.")

class PoliticianSentenceExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.step = dspy.ChainOfThought(PoliticianSentenceSig)

    def forward(self, sections):
        all_sentences = []
        for sec in sections:
            out = self.step(section=sec)
            all_sentences.extend(out.sentences)
        return dspy.Prediction(
            merged_json=json.dumps(all_sentences, ensure_ascii=False),
        )
```

2) Create ground truth JSON:
```json
{
  "<paper_id>": {
    "sentences": [
      "Senator X said ...",
      "The prime minister ...",
      "... etc ..."
    ]
  }
}
```

3) Replace `load_ground_truth()` with a task-specific loader that produces
`dspy.Example(paper_id=..., sections=..., ground_truth_sentences=[...])`.

4) Replace `evaluate_extraction()` with a scoring function (e.g., F1 over
normalized sentences, or semantic matching with NLI/embeddings).

This keeps the pipeline structure but swaps the task-specific pieces.

## Training a DSPy extractor

The training entrypoint is:
`scripts/dspy_train.py`

Inputs:
- Ground-truth JSON ([DefExtra](https://huggingface.co/datasets/mediabiasgroup/DefExtra) hydrated -> SciDef JSON)
- GROBID TEI XML for each paper

Example (smaller, faster run):
```bash
export LLM_PROVIDER=vllm
export VLLM_BASE_URL=http://localhost:8000/v1
export VLLM_MODEL_NAME=openai/gpt-oss-20b
export VLLM_API_KEY=EMPTY
export WANDB_MODE=offline

uv run python scripts/dspy_train.py \
  --ground-truth-path /path/to/defextra_hydrated.json \
  --extractions-dir /path/to/grobid_out \
  --llm-model-name hosted_vllm/openai/gpt-oss-20b \
  --nli-model-name tasksource/ModernBERT-large-nli \
  --dspy-optimizer bootstrap \
  --mipro-auto light
```

Outputs:
- Compiled DSPy program JSON in `artifacts/`
- Offline W&B logs if `WANDB_MODE=offline`

## Using a compiled DSPy program

### Extraction CLI
```bash
uv run python scripts/extract_definitions.py \
  --input-dir /path/to/grobid_out \
  --extractors DSPyPaperExtractor \
  --dspy-program-path /path/to/compiled_program.json \
  --llm-model-name hosted_vllm/openai/gpt-oss-20b \
  --base-api-url http://localhost:8000/v1 \
  --api-key NONE
```

### Evaluation CLI
```bash
WANDB_MODE=offline uv run python scripts/evaluate_extraction.py \
  --ground-truth-path /path/to/defextra_hydrated.json \
  --extractions-dir /path/to/grobid_out \
  --load-compiled-path /path/to/compiled_program.json
```

## Common pitfalls

- If `Config()` fails, ensure `LLM_PROVIDER` and `LLM_MODEL_NAME` are set.
- First-time NLI usage downloads models from Hugging Face.
- The extracted results file uses keys like `paper_<id>.grobid.tei` or
  `<id>.grobid.tei`; both are supported.
- DSPy uses LiteLLM under the hood. For vLLM, the model id must be prefixed
  with `hosted_vllm/`. Example: `hosted_vllm/openai/gpt-oss-20b`.
  If you see `NotFoundError: The model ... does not exist`, ensure:
  - Your vLLM server is started with a matching `--served-model-name`, and
  - The CLI uses the `hosted_vllm/` prefix.
