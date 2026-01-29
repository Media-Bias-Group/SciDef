# Scripts

This file contains the documentation for running our scripts

Scripts:
- [pdf_to_grobid.py](#pdf_to_grobidpy)
- [extract_definitions.py](#extract_definitionspy)
- [benchmark_embedding.py](#benchmark_embeddingpy)
- [benchmark_judge.py](#benchmark_judgepy)
- [benchmark_nli.py](#benchmark_nlipy)
- [dspy_train.py](#dspy_trainpy)
- [evaluate_extraction.py](#evaluate_extractionpy)
 
## pdf_to_grobid.py
This script processes all PDF files in a given directory using a running GROBID server and extracts structured TEI XML representations of the full text. Each PDF is converted into a corresponding .grobid.tei.xml file.

### Requirements
- GROBID instance (for more information on how to run it see the [official Documentation](https://grobid.readthedocs.io/en/update-documentation/getting_started/))
- grobid configuration file

Example for localhost:
```json
{
  "grobid_server": "http://localhost",
  "grobid_port": "8070"
}
```

### Arguments
- --input_folder
  Path to a directory containing PDF files to process.
- --output_folder
  Path to a directory where the extracted TEI XML files will be saved.
- --config (optional)
  Path to the GROBID client configuration file
  (default: ./config.json)

### Example
```bash
uv run grobid_batch_processor.py \
  --input_folder ./pdfs \
  --output_folder ./output \
  --config ./config.json
```


## extract_definitions.py
This script extracts definitions from academic papers that have been preprocessed with GROBID. It supports multiple extraction strategies (one-step, multi-step, few-shot, and DSPy-based extractors), different text chunking modes, and concurrent asynchronous processing. The extracted results, along with metadata and prompt/DSPy configuration, are saved as structured JSON files.

### Requirements

- GROBID-processed TEI XML files (*.grobid.tei.xml)
- Optional: DSPy program file (when using DSPyPaperExtractor)
- configurated [.env](.env) file with LLM backend information

### Arguments
- --input-dir
  One or more directories containing GROBID-extracted TEI XML files.
- --output-dir
  Directory where extraction results will be saved
  (default: results/extraction).
- --max-papers
  Maximum number of papers to process.
- --chunk_modes
  Text chunking strategies to use during extraction.
  Choices: sentence, paragraph, section, three_sentence.
- --extractors
  Extraction strategies to run.
  Choices: MultiStepExtractor, OneStepExtractor,
  OneStepFewShotExtractor, MultiStepFewShotExtractor,
  DSPyPaperExtractor.
- --llm-model-name
  Name of the LLM model to use for extraction.
- --temperature (optional)
  Sampling temperature for LLM-based extractors.
- --top-p (optional)
  Top-p (nucleus sampling) parameter for LLM-based extractors.
- --max-concurrent-papers
  Maximum number of papers processed concurrently
  (default: 64).
- --dspy-program-path (DSPy only)
  Path to a DSPy program file to load.
- --two-step-extraction (DSPy only)
  Enable two-step extraction for DSPyPaperExtractor.
- --base-api-url
  Base URL for the LLM API
  (default: http://localhost:8000/v1).
- --api-key 
  API key for the LLM service.
- --cache-dir (optional)
  Directory for caching LLM responses.
- --disable-cache
  Disable LLM response caching.
- --log-level
  Logging level: DEBUG, INFO, WARNING, ERROR.

### Example 
```python
uv run extract_definitions.py \
  --input-dir ./grobid_xml_files \
  --output-dir ./results/extraction \
  --llm-model-name meta-llama/Llama-3.1-70B \
  --extractors MultiStepExtractor OneStepFewShotExtractor \
  --chunk_modes sentence paragraph \
```


## benchmark_embedding.py
This script evaluates sentence embedding models using cosine similarity across multiple semantic similarity and paraphrase detection benchmarks. 

**Output**:
- EMBEDDING_BENCHMARK_RESULTS.md overview of results
- results/embedding_benchmark/ collects timestamped detailed results

### Requirements
- Configured embedding backend (via .env)
  Requires EMBEDDING_PROVIDER and related environment variables
- Supported benchmark datasets (automatically loaded by the script on first run)

### Benchmarked Datasets
- STS-B (Semantic Textual Similarity Benchmark)
- SICK (Sentences Involving Compositional Knowledge)
- STS3K (all / non-adversarial / adversarial variants)
- MSR Paraphrases
- Quora Duplicate Questions

### Metrics
Supports all metrics defined in MeasureMethod, including:

- Pearson correlation
- Classification metrics (after thresholding), such as accuracy, F1, precision, recall (depending on implementation)

### Arguments
- --datasets
  One or more datasets to evaluate.
  Choices: stsb, sick, sts3k-all, sts3k-non, sts3k-adv, msr-paraphrases, quora-duplicates (default: all)
- --thresholds
  Similarity thresholds used to binarize or bucket predictions.
  (default: 0.5 0.6 0.7 0.8 0.85 0.9 0.95 [0.6,0.7,0.8,0.9])
- --ground-truth-thresholds
  Thresholds applied to ground-truth similarity labels.
  Accepts floats or lists, same format as --thresholds. (default: 0.8 0.85 0.9 0.95)
- --metrics
  Evaluation metrics to compute.
  Choices: all values defined in MeasureMethod.
  (default: all)
- --split
  Dataset split to evaluate.
  Choices: train, test, validation.
  (default: train)
- --sample-size
  Optional limit on the number of sentence pairs evaluated per dataset.

### Example
```python
uv run benchmark_embedding.py \
  --datasets stsb sick \
  --thresholds 0.8 [0.6,0.7,0.8,0.9] \
  --ground-truth-thresholds 0.85 \
  --metrics PEARSON F1 \
  --split train \
  --sample-size 1000
```

## benchmark_judge.py
This script evaluates LLM-based judge models on semantic similarity and paraphrase detection benchmarks. It systematically explores combinations of system prompts, similarity thresholds, temperature and top-p settings, datasets, and evaluation metrics.

**Output**:
- JUDGE_BENCHMARK_RESULTS.md – latest summary
- results/judge_benchmark/ – timestamped detailed reports

### Requirements
- Configured judge backend (via .env)
  Requires variables start with JUDGE_ and related environment variables
- Supported benchmark datasets (automatically loaded by the script on first run)

### Benchmarked Datasets
- STS-B (Semantic Textual Similarity Benchmark)
- SICK (Sentences Involving Compositional Knowledge)
- STS3K (all / non-adversarial / adversarial variants)
- MSR Paraphrases
- Quora Duplicate Questions

### Prompt Strategies
The script evaluates multiple judge system prompts, including:
- Binary classification
- Ternary classification
- Categorical (4-class) classification
Each system prompt is paired with one or more similarity thresholds to convert judge scores into discrete labels for metric computation.

### Metrics
Supports all metrics defined in MeasureMethod, including:

- Pearson correlation
- Classification metrics (after thresholding), such as accuracy, F1, precision, recall (depending on implementation)

### Arguments
- --datasets
  One or more datasets to evaluate.
  Choices: stsb, sick, sts3k-all, sts3k-non, sts3k-adv, msr-paraphrases, quora-duplicates (default: all)
- --prompt-combinations
  Combinations of judge system prompts and thresholds.
  Format: (system_prompt, threshold) Thresholds may be floats or lists (e.g. [0.8,0.95]).
- --temperatures
  Sampling temperature values to evaluate.
  (default: 0.7)
- --top-p-values
  Top-p (nucleus sampling) values to evaluate.
  (default: 0.95)
- --metrics
  Evaluation metrics to compute.
  Choices: all values defined in MeasureMethod.
  (default: all)
- --ground-truth-thresholds
  Thresholds applied to ground-truth similarity labels.
  Accepts floats or lists (e.g. [0.6,0.7,0.8,0.9]).
  (default: 0.8 0.85 0.9 0.95)
- --split
  Dataset split to evaluate.
  Choices: train, test, validation.
  (default: train)
- --sample-size
  Number of sentence pairs evaluated per dataset.
  (default: 5)
- --max-tokens
  Maximum total input tokens allowed per dataset.
  Evaluation stops early if exceeded.
- --per-pair-concurrency
  Maximum number of concurrent LLM requests per evaluation.
  (default: 12)

```python
uv run benchmark_judge.py \
  --datasets stsb sick \
  --prompt-combinations "(BINARY,[0.94])" "(TERNARY,[0.8,0.94])" \
  --temperatures 0.3 0.7 \
  --top-p-values 0.9 0.95 \
  --metrics PEARSON F1 \
  --sample-size 50
```

## benchmark_nli.py
This script evaluates Natural Language Inference (NLI) models on semantic similarity and paraphrase detection benchmarks.

**Output**:
- NLI_BENCHMARK_RESULTS.md – latest summary
- results/nli_benchmark/ – timestamped detailed reports


### Benchmarked Datasets
- STS-B (Semantic Textual Similarity Benchmark)
- SICK (Sentences Involving Compositional Knowledge)
- STS3K (all / non-adversarial / adversarial variants)
- MSR Paraphrases
- Quora Duplicate Questions

### NLI Evaluation Strategy
- Bidirectional NLI: each sentence pair is evaluated in both directions.

- Score aggregation modes:
  - HMEAN – harmonic mean of bidirectional scores
  - AMEAN – arithmetic mean of bidirectional scores
- Threshold-based classification with optional bucketed thresholds.

### Metrics
Supports all metrics defined in MeasureMethod, including:

- Pearson correlation
- Classification metrics (after thresholding), such as accuracy, F1, precision, recall (depending on implementation)

### Arguments
- --datasets
  One or more datasets to evaluate.
  Choices: stsb, sick, sts3k-all, sts3k-non, sts3k-adv, msr-paraphrases, quora-duplicates (default: all)
- --models
  One or more NLI models to evaluate.
  (default includes multiple DeBERTa, RoBERTa, BART, and ModernBERT-based NLI models)
- --score-modes
  Score aggregation modes for bidirectional NLI.
  Choices: HMEAN, AMEAN.
  (default: both)
- --thresholds
  Similarity thresholds used to binarize predictions.
  Accepts floats or lists (e.g. [0.6,0.7,0.8,0.9]).
  (default: 0.5)
- --ground-truth-thresholds
  Thresholds applied to ground-truth similarity labels.
  Accepts floats or lists.
  (default: 0.8 0.85 0.9 0.95)
- --metrics
  Evaluation metrics to compute.
  Choices: all values defined in MeasureMethod.
  (default: all)
- --split
  Dataset split to evaluate.
  Choices: train, test, validation.
  (default: train)
- --sample-size
  Optional limit on the number of sentence pairs evaluated per dataset.

###  Example
```python
uv run benchmark_nli.py \
  --models facebook/bart-large-mnli roberta-large-mnli \
  --datasets stsb sick \
  --score-modes HMEAN \
  --thresholds 0.5 \
  --ground-truth-thresholds 0.85 \
  --metrics PEARSON F1 \
  --sample-size 1000
```

## dspy_train.py
This script evaluates and optimizes a DSPy-based definition extraction pipeline for academic papers.

**Output**: 
- DSPy extraction programm saved as JSON artifact
  - artifacts/dspy_paper_extractor_*.json
- Weights and Biases logs
  - Hyperparameters
  - Pre- and Post-optimization metrics
  - Metadata


### Workflow Overview

1. Load ground-truth definitions and GROBID-extracted papers
2. Split data into train / dev / test sets
3. (Optional) Evaluate pre-DSPy extraction performance
4. Optimize the DSPy extraction program using labeled data
5. Save the compiled DSPy program artifact
6. Log metrics and parameters to Weights & Biases

### Requirements

- GROBID-processed extraction data
- Ground-truth definitions JSON
- Configured LLM backend for DSPy (specified in .env)
- Configured NLI backend for evaluation (specified in .env)
- Weights & Biases account (for experiment tracking) (key specified in .env)

### Arguments
- --ground-truth-path
  Path to the JSON file containing ground-truth definitions.
  (default: data/defExtra.json)
- --extractions-dir
  One or more directories containing GROBID-extracted paper data.
  (default: ManualPDFsGROBID/manual_pdfs_grobid, ManualPDFsGROBID/new_grobid)
- --log-level
  Logging verbosity.Choices: DEBUG, INFO, WARNING, ERROR. (default: INFO)
- --llm-model-name
  LLM model used by DSPy for definition extraction.
  (default: openai/gpt-oss-20b)
- --nli-model-name
  NLI model used to evaluate extraction correctness.
  (default: tasksource/ModernBERT-large-nli)
- --nli-threshold
  NLI score threshold used to determine correctness. (default: 0.25)
- --allow-out-of-vocab
  Allow extracted terms not present in the ground-truth vocabulary.
- --num-mipro-threads
  Number of concurrent threads used by the DSPy optimizer.
  (default: 4)
- --eval-pre-dspy
  Evaluate extraction performance before DSPy optimization.
- --max-tokens
  Maximum number of tokens for LLM responses. (default: 8192)
- --dspy-optimizer
  DSPy optimization strategy.
  Choices:
  - miprov2 (recommended, requires ~200+ examples)
  - bootstrap
  - bootstrap_random
  - (default: miprov2)
- --two-step-extraction
  Enable a two-step extraction process (section detection → extraction).
- --chunk-mode
  Text chunking strategy used for extraction.
  Choices: values defined in ChunkMode. (default: section)
- --nli-compile
  Compile the NLI model for faster inference.
- --metric-threshold
  Metric threshold used during DSPy optimization. (default: 0.25)
- --disable-cache
  Disable DSPy caching (useful for debugging).
- --base-api-url Base URL for the LLM API.
  (default: http://localhost:8000/v1)
- --api-key
  API key for the LLM backend.
  (default: NONE)
--mipro-auto
  MIPROv2 auto optimization mode.
  Choices: light, medium, heavy.(default: light)

### Example

```python
uv run dspy_definition_extraction_optimization.py \
  --ground-truth-path data/definitions/all_concepts.json \
  --extractions-dir extraction_dir/pdfs_grobid \
  --llm-model-name openai/gpt-oss-20b \
  --nli-model-name tasksource/ModernBERT-large-nli \
  --eval-pre-dspy \
  --two-step-extraction \
  --chunk-mode section \
  --dspy-optimizer miprov2 \
  --mipro-auto medium
```

## evaluate_extraction.py
This script evaluates definition extraction quality for academic papers using either DSPy-based extractors or custom LLM extraction pipelines.

**Output**:
- Weights and Biases
  - pre-split aggregate metrics
  - metadata
- score files:
  - score/<extrated_name>_score.txt
- logs
  - average scores per splot
  - statistics (mean, median, percentiles)

### Workflow Overview

1. Load ground-truth definitions and GROBID-extracted papers
2. Split dataset into train / dev / test
3. Initialize NLI model for semantic correctness scoring
4. Initialize extractor (DSPy or custom), or load pre-extracted results
5. Run extraction + evaluation (or evaluation only)
6. Compute aggregate and distributional metrics
7. Log results to Weights & Biases
8. Optionally save evaluation scores to disk

### Requirements

- GROBID-processed paper extractions
- Ground-truth definitions JSON
- extracted definitions or Configured LLM backend (via .env)
- Configured NLI backend
- Weights & Biases account (via .env)

### Arguments
- --ground-truth-path
  Path to JSON file containing ground-truth definitions
  (default: data/defExtra.json)
- --extractions-dir
  One or more directories containing GROBID XML outputs
  (default: ManualPDFsGROBID/manual_pdfs_grobid, ManualPDFsGROBID/new_grobid)
- --extracted-definitions-path
  Path to a JSON file of pre-extracted definitions
  (skips extraction and runs evaluation only)
- --llm-model-name
  LLM model used for extraction (DSPy or custom)
  (default: openai/gpt-oss-20b)
- --nli-model-name
  NLI model used for evaluation
  (default: tasksource/ModernBERT-large-nli)
- --max-tokens
  Maximum tokens for LLM responses (default: 8192)
- --nli-compile
  Compile NLI model for faster inference
- --nli-threshold
  NLI score threshold for marking predictions as correct (default: 0.60)
- --allow-out-of-vocab
  Allow evaluation of terms not present in the ground-truth vocabulary
- --skip-train / --skip-dev / --skip-test
  Skip evaluation on specific dataset splits
- --extractor-type
  Extraction backend to use( default DSPy Extractor)
- --load-compiled-path
  Path to a compiled DSPy extractor JSON
  (skips optimization and loads program directly)
- --two-step-extraction
  Enable two-stage DSPy extraction
  (section detection → definition extraction)
- --chunk-mode
  Text chunking strategy
  Choices: values defined in ChunkMode (default: section)
- --log-level
  Choices: DEBUG, INFO, WARNING, ERROR(default: INFO)


### Example Usage

#### Evaluate a DSPy extractor
```python
uv run evaluate_definitions.py \
  --ground-truth-path data/defExtra.json \
  --extractions-dir extraction_dir/pdfs_grobid \
  --llm-model-name openai/gpt-oss-20b \
  --nli-model-name tasksource/ModernBERT-large-nli \
  --chunk-mode section
```

#### Evaluation-only (pre-extracted definitions)
```python
uv run evaluate_definitions.py \
  --ground-truth-path data/defExtra.json \
  --extractions-dir extraction_dir/pdfs_grobid \
  --extracted-definitions-path outputs/definitions.json
```
