# SciDef: Automated Definition Extraction from Scientific Literature

![SciDef - Workflow](img/scidef_workflow.png)

## Overview
With the rapid growth of publications, identifying definitions relevant to a given keyword has become increasingly difficult. 
**SciDef** provides resources to support research on **definition extraction and definition similarity** from scientific literature.

This repository contains:
- An LLM-based **definition extraction pipeline**
- **Scripts** for running and evaluating definition extraction
- **DefExtra**, a human-annotated dataset for definition extraction
- **DefSim**, a human-annotated dataset for definition similarity
- **Evaluation scripts** covering multiple models, prompting strategies, and similarity metrics

The goal of SciDef is to provide resources for reproducible research on on definition extraction from scientific articles. 

---

## Datasets
To facilitate future research in Definition Extraction from Scientific articles we publish 2 human annotated datasets.

### DefExtra: Definition Extraction Dataset
[DefExtra](data/defExtra.json) is an human-annotated dataset for the evaluation of definition extaction.

**Content**:
- 268 definitions from 75 papers
- 60 media bias realted and 15 non-media bias related papers

### DefSim: Definition Similarity Dataset
[DefSim](data/DefSim.json) is an human-annotated dataset for the evaluation of definition similarity.

**Content**:
- 60 definition definition pairs
- Similarity rating on a 1-5 scale


## Scripts and Usage
To support user-friendly usage, we provide scripts for running the SciDef pipeline, evaluation methods and other utility functions in the [scripts/](scripts) directory.

The [README.md](scripts/README.md) explains how to use them.


SciDef uses [uv](https://github.com/astral-sh/uv) for the package and enviroment management.

### Example
```bash
uv run ./benchmark_evaluation.py --datasets stsb sick --metrics cosine_similarity nli --sample-size 100
```

## Citation
If you use SciDef or DefExtra, please cite:
```bibtex
TODO
```
