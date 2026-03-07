# Investigating Embedding-Level Parameter-Efficient Fine-Tuning as an Alternative to LoRA

This repository contains the codebase for my Bachelor's thesis in Artificial Intelligence at the University of Milan. The project investigates a highly constrained Parameter-Efficient Fine-Tuning (PEFT) method : specifically, an embedding-level "gradient masking" technique - as an alternative to Low-Rank Adaptation (LoRA) for Information Retrieval (IR) tasks.

## Overview

Fine-tuning massive pre-trained language models like BERT for text ranking requires significant computational resources. While methods like LoRA reduce the number of trainable parameters by injecting low-rank matrices into the attention and feedforward layers, this project explores an even more constrained approach: **Prompt Embedding Tuning**. 

Instead of modifying the network architecture or tuning layer weights, we completely freeze the entire BERT model and *only* train the embeddings of the special structural tokens (`[CLS]` and `[SEP]`). The goal is to determine if highly targeted embedding tuning can match the ranking performance of traditional parameter-efficient methods while further reducing the trainable parameter footprint.

### Main Approaches Evaluated:
1. **Prompt Embedding Tuning (Gradient Masking):** Trains only the `[CLS]` and `[SEP]` token embeddings.
2. **LoRA Baseline:** Standard Low-Rank Adaptation applied to the attention modules.
3. **MonoBERT Baseline:** Full fine-tuning (for comparison).

## Datasets

*   **Training:** MS MARCO Passage Ranking dataset (accessed via `ir_datasets`).
*   **Evaluation:** BEIR Heterogeneous Benchmark (e.g., NFCorpus, SciFact, TREC-COVID).


## Acknowledgements

Supervisors: Dr. Gabriella Pasi & Marco Braga, University of Milano-Bicocca.
