#!/bin/bash
#SBATCH -n 1                # node count
#SBATCH --mem-per-cpu=2G    # memory per cpu-core
#SBATCH -t 4:00:00         # total run time limit (HH:MM:SS) (increased to 24 hours)
#SBATCH --array=0-3       # 2000 total combinations (4*5*4*5*5)
#SBATCH --output /om/user/zaho/bfm_ic/reports/slurm-%A_%a.out # STDOUT
#SBATCH --gres=gpu:a100:1
export PATH="/om2/user/zaho/anaconda3/bin:/om2/user/zaho/anaconda3/condabin:$PATH"
conda activate venv

# Define arrays for each hyperparameter
dtype_array=('bfloat16' 'float32')
lr_array=(0.0001 0.001)
# Calculate indices for each hyperparameter
index=$SLURM_ARRAY_TASK_ID
dtype_index=$((index % 2))
lr_index=$((index / 2))

python ttt.py --dtype ${dtype_array[dtype_index]} --lrmax ${lr_array[lr_index]} --lrmin ${lr_array[lr_index]}