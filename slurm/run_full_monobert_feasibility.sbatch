#!/bin/bash
#SBATCH --job-name=fullbert_test
#SBATCH --partition=only-one-gpu
#SBATCH --qos=only-one-gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/full_monobert_feasibility_%j.out
#SBATCH --error=logs/full_monobert_feasibility_%j.err
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G

cd /home/s.chaudhary-thesis

mkdir -p logs

export LD_LIBRARY_PATH=/opt/ice4hpc/rel-24.04/conda/envs/huggingface/lib:$LD_LIBRARY_PATH

/opt/ice4hpc/rel-24.04/conda/envs/huggingface/bin/python -u train_full_monobert_feasibility.py
