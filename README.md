# Embedding-Level PEFT Alternative to LoRA

This repository contains the code and results for experiments on parameter-efficient fine-tuning for BERT-based passage re-ranking.

The project tests whether small, targeted parameter updates can adapt a BERT cross-encoder for information retrieval, and compares these methods with BM25, public MonoBERT, LoRA, and full MonoBERT fine-tuning.

## Repository Contents

```text
Papers/
  Papers and reference material used for the thesis.

notebooks/
  Notebooks used for plotting and analysis.

scripts/
  training/
    Training scripts for the different fine-tuning approaches.

  evaluation/
    Evaluation scripts for BM25, MonoBERT, LoRA, and the trained models.

  utils/
    Helper scripts, including the final result merge script.

  reference/
    Reference code shared for the original MonoT5-style prompt embedding setup.

slurm/
  training/
    SLURM scripts used to run training jobs on the HPC cluster.

  evaluation/
    SLURM scripts used to run evaluation jobs on the HPC cluster.

results/
  final_results_all_experiments.csv
    Final merged table with all approaches and all datasets.

  individual/
    Separate CSV files for each experiment group.

  raw_jsonl/
    Raw JSONL result outputs.

logs_summary/
  Selected logs from training and evaluation runs.
```

## Task

The task is passage re-ranking.

For each query, a set of candidate passages is given. A model scores each query-passage pair, and the passages are sorted by the predicted relevance score.

The BERT cross-encoder input format is:

```text
[CLS] query [SEP] passage [SEP]
```

## Models and approaches

The final experiments include:

| Approach | Description |
|---|---|
| BM25 | Lexical retrieval baseline |
| Public MonoBERT | Publicly available MonoBERT model evaluated in the same setup |
| CLS/SEP embeddings | Only the `[CLS]` and `[SEP]` embeddings are trained |
| Top-6 embeddings | Only the six most changed embedding rows from the BERT/MonoBERT comparison are trained |
| Top-3 attention matrices | Only the three most changed attention matrices are trained |
| All embeddings | The full embedding layer is trained while the rest of the model is frozen |
| All embeddings except CLS/SEP | The full embedding layer is trained except `[CLS]` and `[SEP]` |
| LoRA all attention | LoRA is applied to all attention matrices |
| LoRA top-3 attention | LoRA is applied only to the top-3 changed attention matrices |
| Full MonoBERT | All BERT parameters are fine-tuned |

## Datasets

The experiments use MS MARCO passage ranking data through `ir_datasets`.

Evaluation was done on:

```text
msmarco-passage/trec-dl-2019/judged
msmarco-passage/trec-dl-2020/judged
msmarco-passage/dev/small
msmarco-passage/trec-dl-hard
```

For DL-Hard, BM25 top-1000 candidates were generated with PyTerrier before neural re-ranking.

## Training setup

Most training runs use the following setup:

| Setting | Value |
|---|---|
| Base model | `bert-large-uncased` |
| Training data | `msmarco-passage/train/triples-small` |
| Steps | 100,000 |
| Max sequence length | 128 |
| Effective batch size | 128 |
| Optimizer | AdamW |
| Precision | Mixed precision |
| Checkpoints | Every 10,000 steps |

The training data is based on MS MARCO triples. Each triple is converted into two binary classification examples:

```text
(query, positive passage) -> label 1
(query, negative passage) -> label 0
```

## Results

The main result file is:

```text
results/final_results_all_experiments.csv
```

It contains all 10 approaches evaluated on all 4 datasets.

Individual result files are in:

```text
results/individual/
```

Raw JSONL outputs are in:

```text
results/raw_jsonl/
```

The reported metrics are:

```text
NDCG@10
MRR@10
MAP@1000
Recall@100
```

## Main observations

The embedding-only methods are very parameter-efficient, but their ranking performance is much lower than attention-based adaptation.

Training only `[CLS]` and `[SEP]` is the most restricted setup. Training all embeddings improves some recall-based results, but still does not match attention-based methods.

LoRA on all attention matrices performs close to Public MonoBERT and Full MonoBERT.

LoRA on only the top-3 attention matrices is much smaller than LoRA on all attention matrices, but still keeps a large part of the ranking performance.

Full MonoBERT was trained mainly to verify that the local training and evaluation setup is comparable to the public MonoBERT setup.

## Requirements

The main Python dependencies are listed in:

```text
requirements.txt
```

The experiments were run on an HPC cluster using SLURM. The SLURM scripts are included so the runs can be reproduced or inspected.

## Notes

Large model checkpoints are not included in this repository. The repository contains the code, job scripts, logs, and result files needed to inspect the experimental setup and final outputs.
```
