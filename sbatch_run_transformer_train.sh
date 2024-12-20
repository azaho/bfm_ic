#!/bin/bash
#SBATCH -n 1                # node count
#SBATCH --mem-per-cpu=2G    # memory per cpu-core
#SBATCH -t 3:00:00         # total run time limit (HH:MM:SS) (increased to 24 hours)
#SBATCH --array=0-26      # 2000 total combinations (4*5*4*5*5)
#SBATCH --output /om/user/zaho/bfm_ic/reports/slurm-%A_%a.out # STDOUT
#SBATCH --gres=gpu:a100:1
export PATH="/om2/user/zaho/anaconda3/bin:/om2/user/zaho/anaconda3/condabin:$PATH"
eval "$(conda shell.bash hook)"
conda activate venv

# Define arrays for each hyperparameter
dtype_array=('bfloat16')
lr_array=(0.001 0.0005 0.0001)
wd_array=(0.000 0.001 0.0001)
max_gradient_norm_array=(-1 1 0.5)
# Calculate indices for each hyperparameter
index=$SLURM_ARRAY_TASK_ID
dtype_index=$((index / 27))
lr_index=$(((index % 27) / 9))
wd_index=$(((index % 9) / 3))
max_gradient_norm_index=$((index % 3))

python ttt.py --dtype ${dtype_array[dtype_index]} --lrmax ${lr_array[lr_index]} --lrmin ${lr_array[lr_index]} --weight_decay ${wd_array[wd_index]} --max_gradient_norm ${max_gradient_norm_array[max_gradient_norm_index]} --wait_n_intervals $SLURM_ARRAY_TASK_ID