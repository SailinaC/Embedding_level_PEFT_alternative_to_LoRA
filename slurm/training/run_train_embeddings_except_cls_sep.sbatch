#!/bin/bash
#SBATCH --job-name=emb_no_clssep
#SBATCH --partition=only-one-gpu
#SBATCH --qos=only-one-gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/embeddings_except_cls_sep_%j.out
#SBATCH --error=logs/embeddings_except_cls_sep_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

cd /home/s.chaudhary-thesis

mkdir -p logs
mkdir -p outputs/approach7_all_embeddings_except_cls_sep

export LD_LIBRARY_PATH=/opt/ice4hpc/rel-24.04/conda/envs/huggingface/lib:$LD_LIBRARY_PATH

/opt/ice4hpc/rel-24.04/conda/envs/huggingface/bin/python -u train_embeddings_except_cls_sep.py
