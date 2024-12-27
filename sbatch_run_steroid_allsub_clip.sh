#!/bin/bash
#SBATCH --job-name=brain_foundation_model_training          # Name of the job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=8             # 8 tasks total
#SBATCH --cpus-per-task=16    # Request 8 CPU cores per GPU
#SBATCH --gpus-per-task=1
#SBATCH -t 16:00:00         # total run time limit (HH:MM:SS) (increased to 24 hours)
#SBATCH --array=0-1      # 14 jobs (108/8 rounded up)
#SBATCH --output /shared/anzah/bfm_ic/reports/%A_%a.out # STDOUT

source .venv/bin/activate

# Define arrays for each hyperparameter
filename_array=('ttt_clip.py')
dtype_array=('bfloat16')
optimizer_array=('Muon')
electrode_init_array=('coordinates_nograd' 'zeros' 'normal')
dropout_array=(0.0 0.2 0.5)
batch_size_array=(100)
subjects_array=('3' '1234567890')
lr_array=(0.001 0.0015)
nl_array=(10 14)
d_model_array=(192 384)
pushaway_array=(0)
random_string_array=('NE')
wd_array=(0 0.0001)
# Fixed parameters
max_gradient_norm=-1
dtype_index=0
batch_size_index=0
electrode_init_index=0
pushaway_index=0
filename_index=0
random_string_index=0

# Calculate base index for this job
base_index=$((SLURM_ARRAY_TASK_ID * 8))
optimizer_index=0

# Run 8 combinations in parallel using all 8 GPUs
for gpu_id in {0..7}; do
    index=$((base_index + gpu_id))
    
    # Skip if we've exceeded total combinations
    if [ $index -ge 108 ]; then
        continue
    fi

    # Calculate indices for each hyperparameter
    lr_index=$((index % 2))
    nl_index=$((index / 2 % 2))
    wd_index=$((index / 4 % 2))
    subjects_index=$((index / 8 % 2))
    electrode_init_index=$((index / 16 % 3))
    dropout_index=$((index / 32 % 3))
    d_model_index=$((nl_index))

    if [ $subjects_index -ne 1 ]; then
        continue
    fi
    echo "python ${filename_array[filename_index]} --dtype ${dtype_array[dtype_index]} --optimizer ${optimizer_array[optimizer_index]} --electrode_embedding_init ${electrode_init_array[electrode_init_index]} --dr ${dropout_array[dropout_index]} --dm ${d_model_array[d_model_index]} --pushaway ${pushaway_array[pushaway_index]} --bs ${batch_size_array[batch_size_index]} --lrmax ${lr_array[lr_index]} --lrmin 0.0 --weight_decay $wd --max_gradient_norm $max_gradient_norm --subjects ${subjects_array[subjects_index]} --wait_n_intervals $index --wandb_project bfm --rs ${random_string_array[random_string_index]} --wandb_project bfm_clip3 --nl ${nl_array[nl_index]}"

    srun python ${filename_array[filename_index]} --dtype ${dtype_array[dtype_index]} --optimizer ${optimizer_array[optimizer_index]} \
    --electrode_embedding_init ${electrode_init_array[electrode_init_index]} --dr ${dropout_array[dropout_index]} --dm ${d_model_array[d_model_index]} --pushaway ${pushaway_array[pushaway_index]} \
    --bs ${batch_size_array[batch_size_index]} --lrmax ${lr_array[lr_index]} --lrmin 0.0 --weight_decay ${wd_array[wd_index]} --max_gradient_norm $max_gradient_norm \
    --subjects ${subjects_array[subjects_index]} --wait_n_intervals $index --wandb_project bfm --rs ${random_string_array[random_string_index]} --wandb_project bfm_clip3 --nl ${nl_array[nl_index]} &
done

wait # Wait for all parallel processes to complete