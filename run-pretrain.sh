#!/bin/bash
#SBATCH --gpus-per-node=a100
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=def-annielee

#############################################################
# install the environment by loading in python and required packages
module load python/3.11
# virtualenv --no-download env
source env/bin/activate

# pip install --no-index --upgrade pip

module load gcc
module load cuda
module load arrow
# pip install transformers datasets accelerate

#############################################################

echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"
python run.py --lang=wuu
