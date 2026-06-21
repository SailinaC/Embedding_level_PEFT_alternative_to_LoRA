# Embedding-Level Parameter-Efficient Fine-Tuning as an Alternative to LoRA

The thesis investigates whether very small and targeted parameter updates can adapt a BERT cross-encoder for Information Retrieval. The main idea is inspired by Light-MonoT5, where only selected prompt embeddings are trained instead of fine-tuning the whole model. This repository applies a related idea to MonoBERT-style passage re-ranking and compares it with BM25, Public MonoBERT, LoRA, and full MonoBERT fine-tuning.

## Overview

Neural re-ranking models such as MonoBERT improve retrieval quality by jointly encoding a query and a candidate passage with a Transformer cross-encoder. This allows the model to score the relevance of a passage with respect to the query more accurately than lexical matching alone.

However, full fine-tuning updates hundreds of millions of parameters. This makes training expensive and motivates the use of Parameter-Efficient Fine-Tuning methods.

The starting point of this thesis is the prompt-embedding tuning setup used in Light-MonoT5. In that setting, the model is mostly frozen and only selected embedding vectors are updated through a binary gradient mask. This thesis studies whether a similar idea can be transferred to BERT-based re-ranking.

The main experiment trains only the BERT structural token embeddings used in the MonoBERT input format:

```text
[CLS] query [SEP] passage [SEP]
```

The project then expands this setup by comparing several restricted adaptation strategies, including selected embedding rows, selected attention matrices, LoRA adapters, and full fine-tuning.

## Research Questions

- Can a BERT cross-encoder be adapted for passage re-ranking by training only selected embedding vectors?
- How does training only `[CLS]` and `[SEP]` compare with training the embedding rows that change most between BERT and MonoBERT?
- Are the most important changes from BERT to MonoBERT concentrated more strongly in embeddings or in attention matrices?
- How does embedding-level tuning compare with LoRA-based tuning?
- Can LoRA applied only to the most changed attention matrices retain much of the performance of LoRA applied to all attention matrices?
- How close is the local full fine-tuning setup to Public MonoBERT?

## Repository Structure

```text
Embedding_level_PEFT_alternative_to_LoRA/
├── Papers/
│   └── Research papers and reference material used for the thesis
│
├── scripts/
│   ├── training/
│   │   └── Training scripts for embedding tuning, LoRA, and full MonoBERT
│   │
│   ├── evaluation/
│   │   └── Evaluation scripts for BM25, MonoBERT, LoRA, and trained models
│   │
│   ├── utils/
│   │   └── Utility scripts, including final result merging
│   │
│   └── reference/
│       └── Reference code related to the original MonoT5 prompt-embedding setup
│
├── slurm/
│   ├── training/
│   │   └── SLURM scripts used for training jobs on the HPC cluster
│   │
│   ├── evaluation/
│   │   └── SLURM scripts used for evaluation jobs on the HPC cluster
│   │
│   └── other/
│       └── Additional cluster scripts if needed
│
├── results/
│   ├── final_results_all_experiments.csv
│   ├── individual/
│   │   └── Separate CSV files for each experiment group
│   │
│   └── raw_jsonl/
│       └── Raw JSONL outputs from evaluation scripts
│
├── logs_summary/
│   └── Selected training and evaluation logs
│
├── bert train prompt embedding.py
├── bert_train_lora.py
├── plots.ipynb
├── t5train-prompt-embedding.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Task

The task is passage re-ranking.

For each query, a candidate set of passages is provided. A model scores each query-passage pair independently, and the passages are sorted by the predicted relevance score.

The BERT cross-encoder input format is:

```text
[CLS] query [SEP] passage [SEP]
```

The model is trained as a binary relevance classifier using MS MARCO triples:

```text
(query, positive passage) -> label 1
(query, negative passage) -> label 0
```

## Experiments

The repository contains the following experiment groups.

| Approach | Description |
|---|---|
| CLS/SEP embeddings | Only the `[CLS]` and `[SEP]` embedding vectors are trained |
| Top-6 embeddings | Only the six most changed embedding rows from the BERT/MonoBERT comparison are trained |
| Top-3 attention matrices | Only the three most changed attention matrices are trained directly |
| BM25 | Lexical retrieval baseline without neural re-ranking |
| Public MonoBERT | Publicly available MonoBERT model evaluated in the same setup |
| LoRA all attention | LoRA is applied to all attention matrices |
| LoRA top-3 attention | LoRA is applied only to the top-3 changed attention matrices |
| All embeddings | The full embedding layer is trained while the rest of BERT is frozen |
| All embeddings except CLS/SEP | The full embedding layer is trained except `[CLS]` and `[SEP]` |
| Full MonoBERT | All BERT parameters are fine-tuned |

## Datasets

The experiments use MS MARCO passage ranking data through `ir_datasets`.

Evaluation is performed on:

```text
msmarco-passage/trec-dl-2019/judged
msmarco-passage/trec-dl-2020/judged
msmarco-passage/dev/small
msmarco-passage/trec-dl-hard
```

For TREC DL 2019, TREC DL 2020, and MS MARCO dev small, candidate passages are loaded through `ir_datasets`.

For DL-Hard, BM25 top-1000 candidates are generated with PyTerrier before neural re-ranking.

## Training Setup

Most training runs use the following configuration.

| Setting | Value |
|---|---|
| Base model | `bert-large-uncased` |
| Training data | `msmarco-passage/train/triples-small` |
| Training steps | 100,000 |
| Maximum sequence length | 128 |
| Effective batch size | 128 |
| Optimizer | AdamW |
| Precision | Mixed precision |
| Checkpoint interval | 10,000 steps |

The maximum sequence length is set to 128 tokens for computational feasibility. Query and passage are encoded as a single BERT input pair, with truncation applied when the sequence exceeds the maximum length.

## Evaluation

The final comparison uses the following metrics:

| Metric | Meaning |
|---|---|
| NDCG@10 | Ranking quality in the top 10 results |
| MRR@10 | Reciprocal rank of the first relevant result within the top 10 |
| MAP@1000 | Mean average precision over the top 1000 candidates |
| Recall@100 | Fraction of relevant passages retrieved in the top 100 |

The main result table is:

```text
results/final_results_all_experiments.csv
```

Individual result files are stored in:

```text
results/individual/
```

Raw evaluation outputs are stored in:

```text
results/raw_jsonl/
```

## Key Findings

The final results show a clear difference between embedding-level tuning and attention-based tuning.

The embedding-only methods are extremely parameter-efficient, but their ranking performance is limited. Training only `[CLS]` and `[SEP]` gives the smallest trainable parameter count, but it does not approach the performance of MonoBERT or LoRA.

Training all embeddings improves some recall-oriented results, but still remains much weaker than attention-based adaptation. The experiment that trains all embeddings except `[CLS]` and `[SEP]` provides an additional comparison for understanding the role of the structural BERT tokens.

The attention-based methods perform substantially better. Training the top-3 changed attention matrices gives strong results with a restricted number of updated parameters. LoRA applied only to the top-3 changed attention matrices also retains a large part of the ranking performance while using far fewer trainable parameters than broader LoRA adaptation.

LoRA applied to all attention matrices performs close to Public MonoBERT and Full MonoBERT. This suggests that most of the ranking performance can be recovered without updating the full model.


## Acknowledgments

Under the supervision of:
- Dr. Gabriella Pasi, University of Milano-Biccoca
- Marco Braga, University of Milano-Biccoca



```
