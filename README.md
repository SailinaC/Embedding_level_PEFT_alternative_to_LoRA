# BS Artificial Intelligence - Thesis

# Investigating Embedding-Level Parameter-Efficient Fine-Tuning as an Alternative to LoRA

This repository contains the codebase for my Bachelor's thesis in Artificial Intelligence at the University of Milano-Bicocca. The project investigates a highly constrained Parameter-Efficient Fine-Tuning (PEFT) method—specifically, an embedding-level "gradient masking" technique—as an alternative to Low-Rank Adaptation (LoRA) for Information Retrieval (IR) tasks.

## Overview

Fine-tuning massive pre-trained language models like BERT for text ranking requires significant computational resources. While methods like LoRA reduce the number of trainable parameters by injecting low-rank matrices into the attention and feedforward layers, this project explores an even more constrained approach: **Prompt Embedding Tuning**. 

Instead of modifying the network architecture or tuning layer weights, we completely freeze the entire BERT model and *only* train the embeddings of the special structural tokens (`[CLS]` and `[SEP]`). The goal is to determine if highly targeted embedding tuning can match the ranking performance of traditional parameter-efficient methods while further reducing the trainable parameter footprint.

### Main Approaches Evaluated:
1. **Prompt Embedding Tuning (Gradient Masking):** Trains only the `[CLS]` and `[SEP]` token embeddings.
2. **LoRA Baseline:** Standard Low-Rank Adaptation applied to the attention modules.
3. **MonoBERT Baseline:** Full fine-tuning (for comparison).

## Repository Structure

*   `bert_train_prompt_full.py`: The main training script for the Prompt Embedding approach. It uses a custom PyTorch gradient hook to apply a mask, ensuring only the exact token IDs for `[CLS]` (101) and `[SEP]` (102) receive gradients during the backward pass.
*   `bert_train_lora_full.py`: The baseline training script implementing standard LoRA using the HuggingFace `peft` library.
*   `bert_eval.py`: The evaluation pipeline using the BEIR benchmark. It implements a two-stage retrieval process (Dense Retriever -> Cross-Encoder reranking) to calculate NDCG@10, MRR@10, and Recall@100.
*   `BAI_Thesis LaTeX/`: Contains the LaTeX source code and template for the written thesis document.

## Datasets

*   **Training:** MS MARCO Passage Ranking dataset (accessed via `ir_datasets`).
*   **Evaluation:** BEIR Heterogeneous Benchmark (e.g., NFCorpus, SciFact, TREC-COVID).

## Quick Start
### Prerequisites

Ensure you have a modern GPU environment configured with PyTorch and CUDA.

```bash
pip install torch transformers peft datasets ir_datasets beir
```

### 1. Training

To start training the Prompt Embedding model on MS MARCO for 100,000 steps:

```bash
python bert_train_prompt_full.py \
    --model_name_or_path bert-base-uncased \
    --output_dir checkpoints/prompt_model \
    --max_steps 100000 \
    --batch_size 8 \
    --gradient_accumulation_steps 16 \
    --learning_rate 3e-6
```

Both training scripts support robust checkpointing and resuming in case of cluster preemptions:
```bash
python bert_train_prompt_full.py \
    --output_dir checkpoints/prompt_model \
    --resume_from_checkpoint
```

### 2. Evaluation

To evaluate a trained checkpoint on a BEIR dataset (e.g., NFCorpus):

```bash
python bert_eval.py \
    --dataset nfcorpus \
    --model_name_or_path checkpoints/prompt_model/checkpoint-100000 
```

To evaluate a LoRA checkpoint, simply append the `--is_lora` flag so the script reconstructs the adapter weights over the base model.

## Acknowledgements

Supervisors: Dr. Gabriella Pasi & Marco Braga, University of Milano-Bicocca.
