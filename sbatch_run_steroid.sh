#!/bin/bash
#SBATCH --job-name=brain foundation model training          # Name of the job
#SBATCH -n 1                # node count
#SBATCH --mem-per-cpu=16G    # memory per cpu-core
#SBATCH -t 16:00:00         # total run time limit (HH:MM:SS) (increased to 24 hours)
#SBATCH --array=0-7      # 8 total combinations (2*2*2*2)
#SBATCH --output /shared/anzah/bfm_ic/reports/%A_%a.out # STDOUT
#SBATCH --gres=gpu:1

#export PATH="/om2/user/zaho/anaconda3/bin:/om2/user/zaho/anaconda3/condabin:$PATH"
#eval "$(conda shell.bash hook)"
#conda activate venv
source .venv/bin/activate

# Define arrays for each hyperparameter
dtype_array=('bfloat16')
optimizer_array=('AdamW' 'Muon')
electrode_init_array=('normal' 'zeros')
dropout_array=(0.0)
batch_size_array=(100)
# Fixed parameters
lr=0.001
wd=0
max_gradient_norm=-1
dtype_index=0
dropout_index=0
batch_size_index=0

# Calculate indices for each hyperparameter
index=$SLURM_ARRAY_TASK_ID
optimizer_index=$((index / 4))
electrode_init_index=$(((index % 4) / 2))
random_string=$((index % 2))

# no need to run through srun because it's already an array job on a single node
python ttt.py --dtype ${dtype_array[dtype_index]} --optimizer ${optimizer_array[optimizer_index]} \
--electrode_embedding_init ${electrode_init_array[electrode_init_index]} --dr ${dropout_array[dropout_index]} \
--bs ${batch_size_array[batch_size_index]} --lrmax $lr --lrmin $lr --weight_decay $wd --max_gradient_norm $max_gradient_norm \
--wait_n_intervals $SLURM_ARRAY_TASK_ID --wandb_project bfm --rs $random_string --wandb_project bfm_steroids