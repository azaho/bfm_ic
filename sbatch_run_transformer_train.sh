#!/bin/bash
#SBATCH -n 1                # node count
#SBATCH --mem-per-cpu=16G    # memory per cpu-core
#SBATCH -t 16:00:00         # total run time limit (HH:MM:SS) (increased to 24 hours)
#SBATCH --array=0-0       # 2000 total combinations (4*5*4*5*5)
#SBATCH --output /om/user/zaho/bfm_ic/reports/slurm-%A_%a.out # STDOUT
#SBATCH --gres=gpu:a100:1
export PATH="/om2/user/zaho/anaconda3/bin:$PATH"
conda activate venv

# Define arrays for each hyperparameter
#n_top_pc_llm_array=(100 400 800 -1)
#weight_decay_array=(0.0 0.001 0.005 0.01 0.02)
# Calculate indices for each hyperparameter
#index=$SLURM_ARRAY_TASK_ID
#n_top_pc_llm_index=$((index % 4))

python ttt.py