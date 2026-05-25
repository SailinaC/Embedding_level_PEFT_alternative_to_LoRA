#!/bin/bash
#SBATCH --job-name=cls_sep
#SBATCH --partition=only-one-gpu
#SBATCH --qos=only-one-gpu
#SBATCH --output=logs/cls_sep_%j.out
#SBATCH --error=logs/cls_sep_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

cd /home/s.chaudhary-thesis
/opt/ice4hpc/rel-24.04/conda/envs/huggingface/bin/python -u train_cls_sep.py
