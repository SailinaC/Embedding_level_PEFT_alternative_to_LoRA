#!/bin/bash
#SBATCH --job-name=eval_a3_dev
#SBATCH --partition=only-one-gpu
#SBATCH --qos=only-one-gpu
#SBATCH --output=logs/eval_a3_dev_%j.out
#SBATCH --error=logs/eval_a3_dev_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

cd /home/s.chaudhary-thesis

/opt/ice4hpc/rel-24.04/conda/envs/huggingface/bin/python -u eval_approaches.py \
  --approach approach3 \
  --dataset msmarco-passage/dev/small \
  --top_k 1000 \
  --batch_size 64 \
  --max_length 128
