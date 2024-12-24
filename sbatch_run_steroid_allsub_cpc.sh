#!/bin/bash
#SBATCH --job-name=brain_foundation_model_training          # Name of the job
#SBATCH -n 1                # node count
#SBATCH --mem-per-cpu=64G    # memory per cpu-core
#SBATCH -t 16:00:00         # total run time limit (HH:MM:SS) (increased to 24 hours)
#SBATCH --array=0-13      # 14 jobs (108/8 rounded up)
#SBATCH --output /shared/anzah/bfm_ic/reports/%A_%a.out # STDOUT
#SBATCH --gres=gpu:8       # Request 8 GPUs per job
#SBATCH --cpus-per-task=8    # Request 8 CPU cores per GPU

source .venv/bin/activate

# Define arrays for each hyperparameter
filename_array=('ttt_cpc.py' 'ttt_simsiam.py')
dtype_array=('bfloat16')
optimizer_array=('Muon')
electrode_init_array=('normal')
dropout_array=(0.0 0.1 0.2)
batch_size_array=(100)
subjects_array=('2' '12345' '1234567890')
lr_array=(0.001 0.0005)
nl_array=(10 14 16)
d_model_array=(192 384 576)
# Fixed parameters
wd=0
max_gradient_norm=-1
dtype_index=0
batch_size_index=0
optimizer_index=0
electrode_init_index=0
random_string='1339'

# Calculate base index for this job
base_index=$((SLURM_ARRAY_TASK_ID * 8))

# Run 8 combinations in parallel using all 8 GPUs
for gpu_id in {0..7}; do
    index=$((base_index + gpu_id))
    
    # Skip if we've exceeded total combinations
    if [ $index -ge 108 ]; then
        continue
    fi

    # Calculate indices for each hyperparameter
    subjects_index=$((index % 3))
    dropout_index=$((index / 3 % 3))
    filename_index=$((index / 9 % 2))
    lr_index=$((index / 18 % 2))
    nl_index=$((index / 36))
    d_model_index=$((nl_index))

    CUDA_VISIBLE_DEVICES=$gpu_id python ${filename_array[filename_index]} --dtype ${dtype_array[dtype_index]} --optimizer ${optimizer_array[optimizer_index]} \
    --electrode_embedding_init ${electrode_init_array[electrode_init_index]} --dr ${dropout_array[dropout_index]} --dm ${d_model_array[d_model_index]} \
    --bs ${batch_size_array[batch_size_index]} --lrmax ${lr_array[lr_index]} --lrmin ${lr_array[lr_index]} --weight_decay $wd --max_gradient_norm $max_gradient_norm \
    --subjects ${subjects_array[subjects_index]} --wait_n_intervals $index --wandb_project bfm --rs $random_string --wandb_project bfm_steroids_eval2 --nl ${nl_array[nl_index]} &
done

wait # Wait for all parallel processes to complete