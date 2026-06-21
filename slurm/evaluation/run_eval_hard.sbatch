#!/bin/bash
#SBATCH --job-name=eval_hard
#SBATCH --partition=only-one-gpu
#SBATCH --qos=only-one-gpu
#SBATCH --output=logs/eval_hard_%j.out
#SBATCH --error=logs/eval_hard_%j.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

cd /home/s.chaudhary-thesis

PYTHON=/opt/ice4hpc/rel-24.04/conda/envs/huggingface/bin/python

$PYTHON -u eval_hard_from_candidates.py --approach approach1 --candidate_file hard_bm25_top1000.tsv --top_k 1000 --batch_size 64 --max_length 128
$PYTHON -u eval_hard_from_candidates.py --approach approach2 --candidate_file hard_bm25_top1000.tsv --top_k 1000 --batch_size 64 --max_length 128
$PYTHON -u eval_hard_from_candidates.py --approach approach3 --candidate_file hard_bm25_top1000.tsv --top_k 1000 --batch_size 64 --max_length 128
