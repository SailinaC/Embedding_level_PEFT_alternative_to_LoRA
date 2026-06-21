#!/bin/bash
#SBATCH --job-name=lora_all
#SBATCH --partition=only-one-gpu
#SBATCH --qos=only-one-gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/lora_all_attention_%j.out
#SBATCH --error=logs/lora_all_attention_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

cd /home/s.chaudhary-thesis

mkdir -p logs
mkdir -p outputs/approach5_lora_all_attention

export LD_LIBRARY_PATH=/opt/ice4hpc/rel-24.04/conda/envs/huggingface/lib:$LD_LIBRARY_PATH

/opt/ice4hpc/rel-24.04/conda/envs/huggingface/bin/python -u train_lora_all_attention.py
