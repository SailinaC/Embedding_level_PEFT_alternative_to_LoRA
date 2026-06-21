#!/bin/bash
#SBATCH --job-name=eval_mbert
#SBATCH --partition=only-one-gpu
#SBATCH --qos=only-one-gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/eval_public_monobert_%j.out
#SBATCH --error=logs/eval_public_monobert_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

cd /home/s.chaudhary-thesis

mkdir -p logs
mkdir -p eval_results

export LD_LIBRARY_PATH=/opt/ice4hpc/rel-24.04/conda/envs/huggingface/lib:$LD_LIBRARY_PATH

/opt/ice4hpc/rel-24.04/conda/envs/huggingface/bin/python -u eval_public_monobert.py
