# Embedding-Level PEFT for Information Retrieval

This repository contains the code and experimental results for my thesis project on **embedding-level parameter-efficient fine-tuning (PEFT)** for passage re-ranking.

The goal of the project is to investigate whether very small, targeted parameter updates can adapt a BERT cross-encoder for Information Retrieval, and how these updates compare to adapting larger but still restricted parts of the model.

The work is motivated by MonoBERT-style passage re-ranking and by prior work on MonoT5 embedding-level adaptation. Instead of fully fine-tuning BERT, the experiments freeze almost the entire model and train only selected embeddings or selected attention matrices.

## Overview

The task is passage re-ranking. Given a query and a set of candidate passages, the model scores each query-passage pair and reorders the candidates by predicted relevance.

The BERT input format is:

```text
[CLS] query [SEP] passage [SEP]
```

All experiments use `bert-large-uncased` as a cross-encoder sequence classification model.

Three training approaches are evaluated:

| Approach | Description | Trainable Parameters |
|---|---|---:|
| Approach 1 | Train only `[CLS]` and `[SEP]` embeddings | 2,048 |
| Approach 2 | Train the top 6 most changed embedding vectors from BERT/MonoBERT analysis | 6,144 |
| Approach 3 | Train the top 3 most changed attention matrices from BERT/MonoBERT analysis | 3,145,728 |

The rest of the model is frozen in all three approaches.

## Approaches

### Approach 1: CLS/SEP Embedding Tuning

This approach trains only the embeddings of the two BERT structural tokens used in the cross-encoder input:

```text
[CLS] token id = 101
[SEP] token id = 102
```

All transformer layers, all other embeddings, and the classification head are frozen.

This is the most constrained experiment. It tests whether changing only the structural tokens that organize the query-passage input can provide useful adaptation for re-ranking.

### Approach 2: Top-6 Embedding Tuning

This approach trains the six embedding rows that changed the most in the preliminary BERT vs. MonoBERT embedding analysis:

```text
[1000, 2133, 29649, 29658, 29645, 21932]
```

The rest of the model remains frozen.

During training, only four of the six selected rows changed. The remaining two did not receive updates, likely because they did not appear in the tokenized training inputs.

### Approach 3: Top-3 Attention Matrix Tuning

This approach trains the three matrices that changed the most in the BERT vs. MonoBERT matrix-level comparison:

```text
bert.encoder.layer.23.attention.output.dense.weight
bert.encoder.layer.22.attention.output.dense.weight
bert.encoder.layer.23.attention.self.query.weight
```

These matrices are located in the upper layers of BERT-large. This approach is less restrictive than the embedding-only approaches, but still trains less than 1% of the full model.

## Training Setup

Training was performed on MS MARCO passage ranking triples using `ir_datasets`.

Each triple provides:

```text
query
positive passage
negative passage
```

Each triple is converted into two binary classification examples:

```text
(query, positive passage) -> label 1
(query, negative passage) -> label 0
```

Training configuration:

| Setting | Value |
|---|---:|
| Base model | `bert-large-uncased` |
| Training data | `msmarco-passage/train/triples-small` |
| Training steps | 100,000 |
| Micro-batch size | 16 |
| Gradient accumulation | 8 |
| Effective batch size | 128 |
| Max sequence length | 128 |
| Learning rate | 1e-4 |
| Warmup steps | 10,000 |
| Optimizer | AdamW |
| Precision | Mixed precision |

The maximum sequence length was set to 128 tokens for computational feasibility. Truncation was applied only to the passage side of the input.

## Evaluation

The trained models were evaluated as cross-encoder re-rankers. For each query, candidate passages were scored independently and sorted by the relevance logit.

Evaluation datasets:

```text
msmarco-passage/trec-dl-2019/judged
msmarco-passage/trec-dl-2020/judged
msmarco-passage/dev/small
msmarco-passage/trec-dl-hard
```

For TREC DL 2019, TREC DL 2020, and MS MARCO dev small, candidate rankings were loaded directly from `ir_datasets`.

For TREC DL Hard, candidate rankings were not provided directly, so BM25 top-1000 candidates were generated using PyTerrier and then re-ranked with each trained model.

Metrics:

```text
NDCG@10
MRR@10
MAP@1000
Recall@100
```

## Results

| Dataset | Approach | NDCG@10 | MRR@10 | MAP@1000 | Recall@100 |
|---|---|---:|---:|---:|---:|
| TREC DL 2019 | Approach 1: CLS/SEP embeddings | 0.0449 | 0.0659 | 0.0942 | 0.1379 |
| TREC DL 2019 | Approach 2: Top-6 embeddings | 0.0197 | 0.0411 | 0.0517 | 0.0462 |
| TREC DL 2019 | Approach 3: Top-3 attention matrices | 0.5627 | 0.8973 | 0.4050 | 0.4707 |
| TREC DL 2020 | Approach 1: CLS/SEP embeddings | 0.0224 | 0.0640 | 0.0684 | 0.1202 |
| TREC DL 2020 | Approach 2: Top-6 embeddings | 0.0193 | 0.0613 | 0.0380 | 0.0306 |
| TREC DL 2020 | Approach 3: Top-3 attention matrices | 0.5312 | 0.8343 | 0.4023 | 0.5331 |
| MS MARCO Dev Small | Approach 1: CLS/SEP embeddings | 0.0093 | 0.0057 | 0.0117 | 0.1735 |
| MS MARCO Dev Small | Approach 2: Top-6 embeddings | 0.0054 | 0.0035 | 0.0056 | 0.0356 |
| MS MARCO Dev Small | Approach 3: Top-3 attention matrices | 0.3348 | 0.2795 | 0.2856 | 0.7635 |
| TREC DL Hard | Approach 1: CLS/SEP embeddings | 0.0031 | 0.0076 | 0.0242 | 0.0475 |
| TREC DL Hard | Approach 2: Top-6 embeddings | 0.0016 | 0.0129 | 0.0142 | 0.0112 |
| TREC DL Hard | Approach 3: Top-3 attention matrices | 0.3065 | 0.5370 | 0.2163 | 0.4741 |

The results show a clear trend. The embedding-only approaches are highly parameter-efficient, but their ranking performance is weak. The top-3 attention-matrix approach performs substantially better across all evaluated datasets, suggesting that the largest task-specific changes in MonoBERT are concentrated more strongly in upper-layer attention matrices than in isolated embedding vectors.

## Repository Structure

```text
scripts/
  train_cls_sep.py
  train_top6_embeddings.py
  train_top3_attention.py
  eval_approaches.py
  eval_hard_from_candidates.py
  build_hard_bm25_candidates.py

slurm/
  run_cls_sep.sbatch
  run_top6_embeddings.sbatch
  run_top3_attention.sbatch
  run_eval_all.sbatch
  run_build_hard_bm25.sbatch
  run_eval_hard.sbatch

results/
  final_results.csv
  results.jsonl
```

The trained checkpoints are not included in this repository because of file size. They can be regenerated by running the training scripts or shared separately if needed.

## Notes

The `results.jsonl` file may contain a few small debug runs used to verify the evaluation pipeline. The clean final table is stored in:

```text
results/final_results.csv
```

For TREC DL Hard, the candidate file was generated with BM25 using PyTerrier. The candidate file itself is not included in the repository because it is derived from MS MARCO data.

## Requirements

Main Python dependencies:

```text
torch
transformers
ir_datasets
numpy
pandas
pyterrier
```

PyTerrier also requires Java. In the HPC environment used for this project, OpenJDK 17 was installed in a local Conda environment and used to run the BM25 candidate generation step.
```
