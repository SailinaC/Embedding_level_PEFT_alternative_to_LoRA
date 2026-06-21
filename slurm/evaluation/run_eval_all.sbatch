#!/bin/bash
#SBATCH --job-name=eval_all
#SBATCH --partition=only-one-gpu
#SBATCH --qos=only-one-gpu
#SBATCH --output=logs/eval_all_%j.out
#SBATCH --error=logs/eval_all_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

cd /home/s.chaudhary-thesis

PYTHON=/opt/ice4hpc/rel-24.04/conda/envs/huggingface/bin/python

echo "Approach 1: CLS/SEP embeddings only | Dataset: TREC DL 2019"
$PYTHON -u eval_approaches.py --approach approach1 --dataset msmarco-passage/trec-dl-2019/judged --top_k 1000 --batch_size 64 --max_length 128

echo "Approach 2: Top 6 changed embeddings | Dataset: TREC DL 2019"
$PYTHON -u eval_approaches.py --approach approach2 --dataset msmarco-passage/trec-dl-2019/judged --top_k 1000 --batch_size 64 --max_length 128

echo "Approach 3: Top 3 changed attention matrices | Dataset: TREC DL 2019"
$PYTHON -u eval_approaches.py --approach approach3 --dataset msmarco-passage/trec-dl-2019/judged --top_k 1000 --batch_size 64 --max_length 128

echo "Approach 1: CLS/SEP embeddings only | Dataset: TREC DL 2020"
$PYTHON -u eval_approaches.py --approach approach1 --dataset msmarco-passage/trec-dl-2020/judged --top_k 1000 --batch_size 64 --max_length 128

echo "Approach 2: Top 6 changed embeddings | Dataset: TREC DL 2020"
$PYTHON -u eval_approaches.py --approach approach2 --dataset msmarco-passage/trec-dl-2020/judged --top_k 1000 --batch_size 64 --max_length 128

echo "Approach 3: Top 3 changed attention matrices | Dataset: TREC DL 2020"
$PYTHON -u eval_approaches.py --approach approach3 --dataset msmarco-passage/trec-dl-2020/judged --top_k 1000 --batch_size 64 --max_length 128

echo "Approach 1: CLS/SEP embeddings only | Dataset: MS MARCO Dev Small"
$PYTHON -u eval_approaches.py --approach approach1 --dataset msmarco-passage/dev/small --top_k 1000 --batch_size 64 --max_length 128

echo "Approach 2: Top 6 changed embeddings | Dataset: MS MARCO Dev Small"
$PYTHON -u eval_approaches.py --approach approach2 --dataset msmarco-passage/dev/small --top_k 1000 --batch_size 64 --max_length 128

echo "Approach 3: Top 3 changed attention matrices | Dataset: MS MARCO Dev Small"
$PYTHON -u eval_approaches.py --approach approach3 --dataset msmarco-passage/dev/small --top_k 1000 --batch_size 64 --max_length 128
