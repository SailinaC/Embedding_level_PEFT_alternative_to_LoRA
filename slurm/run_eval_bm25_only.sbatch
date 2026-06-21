#!/bin/bash
#SBATCH --job-name=eval_bm25
#SBATCH --partition=ice4hpc
#SBATCH --output=logs/eval_bm25_%j.out
#SBATCH --error=logs/eval_bm25_%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

cd /home/s.chaudhary-thesis

mkdir -p logs
mkdir -p eval_results

export LD_LIBRARY_PATH=/opt/ice4hpc/rel-24.04/conda/envs/huggingface/lib:$LD_LIBRARY_PATH

/opt/ice4hpc/rel-24.04/conda/envs/huggingface/bin/python -u eval_bm25_only.py
