#!/bin/bash
#SBATCH --job-name=brain_foundation_model_training          # Name of the job
#SBATCH -n 1                # node count
#SBATCH --mem-per-cpu=16G    # memory per cpu-core
#SBATCH -t 16:00:00         # total run time limit (HH:MM:SS) (increased to 24 hours)
#SBATCH --array=0-5      # 66 total combinations (2*2*2*2*2*2)
#SBATCH --output /shared/anzah/bfm_ic/reports/%A_%a.out # STDOUT
#SBATCH --gres=gpu:1

#export PATH="/om2/user/zaho/anaconda3/bin:/om2/user/zaho/anaconda3/condabin:$PATH"
#eval "$(conda shell.bash hook)"
#conda activate venv
source .venv/bin/activate

# Define arrays for each hyperparameter
dtype_array=('bfloat16')
optimizer_array=('Muon')
electrode_init_array=('normal')
dropout_array=(0.0 0.1 0.2)
batch_size_array=(100)
subjects_array=('2' '1234567890')
lr_array=(0.001)
nl_array=(10 15 20)
# Fixed parameters
wd=0
max_gradient_norm=-1
dtype_index=0
nl_index=1
batch_size_index=0
optimizer_index=0
electrode_init_index=0
lr_index=0
random_string='1338'

# Calculate indices for each hyperparameter
index=$SLURM_ARRAY_TASK_ID
subjects_index=$((index % 2))
dropout_index=$((index / 2))

# no need to run through srun because it's already an array job on a single node
python ttt_cpc.py --dtype ${dtype_array[dtype_index]} --optimizer ${optimizer_array[optimizer_index]} \
--electrode_embedding_init ${electrode_init_array[electrode_init_index]} --dr ${dropout_array[dropout_index]} \
--bs ${batch_size_array[batch_size_index]} --lrmax ${lr_array[lr_index]} --lrmin ${lr_array[lr_index]} --weight_decay $wd --max_gradient_norm $max_gradient_norm \
--subjects ${subjects_array[subjects_index]} --wait_n_intervals $SLURM_ARRAY_TASK_ID --wandb_project bfm --rs $random_string --wandb_project bfm_steroids_eval1 --nl ${nl_array[nl_index]}