#!/bin/bash
#SBATCH -n 1                # node count
#SBATCH --mem-per-cpu=2G    # memory per cpu-core
#SBATCH -t 6:00:00         # total run time limit (HH:MM:SS) (increased to 24 hours)
#SBATCH --array=0-35      # 2000 total combinations (4*5*4*5*5)
#SBATCH --output /om/user/zaho/bfm_ic/reports/slurm-%A_%a.out # STDOUT
#SBATCH --gres=gpu:a100:1

export PATH="/om2/user/zaho/anaconda3/bin:/om2/user/zaho/anaconda3/condabin:$PATH"
eval "$(conda shell.bash hook)"
conda activate venv

# Define arrays for each hyperparameter
dtype_array=('bfloat16')
optimizer_array=('AdamW' 'Muon')
electrode_init_array=('normal' 'zeros')
dropout_array=(0.0 0.1 0.2)
batch_size_array=(10 20 50)
# Fixed parameters
lr=0.001
wd=0
max_gradient_norm=-1
# Calculate indices for each hyperparameter
index=$SLURM_ARRAY_TASK_ID
dtype_index=0
optimizer_index=$((index / 18))
electrode_init_index=$(((index % 18) / 9))
dropout_index=$(((index % 9) / 3))
batch_size_index=$((index % 3))

python ttt.py --dtype ${dtype_array[dtype_index]} --optimizer ${optimizer_array[optimizer_index]} --electrode_embedding_init ${electrode_init_array[electrode_init_index]} --dr ${dropout_array[dropout_index]} --bs ${batch_size_array[batch_size_index]} --lrmax $lr --lrmin $lr --weight_decay $wd --max_gradient_norm $max_gradient_norm --wait_n_intervals $SLURM_ARRAY_TASK_ID