#!/bin/bash
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --account=def-annielee
#SBATCH --cpus-per-task=4
#############################################################
# install the environment by loading in python and required packages
module load python/3.11
# virtualenv --no-download env
source ~/env/bin/activate

# pip install --no-index --upgrade pip

module load gcc
module load cuda
module load arrow
# pip install transformers datasets accelerate

#############################################################

echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"
python run.py --pretrain --model_dir="models/yue-pretrain-wiki"
