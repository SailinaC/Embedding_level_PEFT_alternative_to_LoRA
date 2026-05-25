#!/bin/bash
#SBATCH --job-name=hard_bm25
#SBATCH --partition=ice4hpc
#SBATCH --output=logs/hard_bm25_%j.out
#SBATCH --error=logs/hard_bm25_%j.err
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

cd /home/s.chaudhary-thesis

source /opt/ice4hpc/rel-24.04/conda/etc/profile.d/conda.sh
conda activate java17

export JAVA_HOME=/home/s.chaudhary-thesis/.conda/envs/java17
export JVM_PATH=/home/s.chaudhary-thesis/.conda/envs/java17/lib/jvm/lib/server/libjvm.so
export PATH=$JAVA_HOME/bin:$PATH
export LD_LIBRARY_PATH=/opt/ice4hpc/rel-24.04/conda/envs/huggingface/lib:$JAVA_HOME/lib:$LD_LIBRARY_PATH

/opt/ice4hpc/rel-24.04/conda/envs/huggingface/bin/python -u build_hard_bm25_candidates.py
