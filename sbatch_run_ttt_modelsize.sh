#!/bin/bash
#SBATCH --job-name=brain_foundation_model_training          # Name of the job
#SBATCH --ntasks=8             # 8 tasks total
#SBATCH --cpus-per-task=16    # Request 8 CPU cores per GPU
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=128G
#SBATCH -t 24:00:00         # total run time limit (HH:MM:SS) (increased to 24 hours)
#SBATCH --array=0-26      # 14 jobs (108/8 rounded up)
#SBATCH --output /shared/anzah/bfm_ic/r/%A_%a.out # STDOUT

source .venv/bin/activate

# Define arrays for each hyperparameter
filename_array=('ttt_clip.py')
dtype_array=('bfloat16')
optimizer_array=('Muon' 'AdamW')
electrode_init_array=('coordinates_nograd' 'zeros' 'normal')
dropout_array=(0.0 0.2 0.4)
batch_size_array=(100)
subjects_array=('3') # '123456' '234' '24' '23' '3')
lr_array=(0.001 0.0015)
nl_array=(10 12 14)
d_model_array=(192 216 240)
random_string_array=('MSIZE')
wd_array=(0 0.0001)
n_freq_features_array=(37 64 128)
# Fixed parameters
max_gradient_norm=-1
dtype_index=0
batch_size_index=0
electrode_init_index=0
filename_index=0
random_string_index=0
wd_index=0
subjects_index=0

spectrogram=1
binarize_eval=1
temp_clip_param=1
test_chunks_interleaved=0
n_freq_features_index=0
multisubj_eval=0

# Calculate base index for this job
base_index=$((SLURM_ARRAY_TASK_ID * 8))

# Run 8 combinations in parallel using all 8 GPUs
for gpu_id in {0..7}; do
    index=$((base_index + gpu_id))

    # Calculate indices for each hyperparameter
    lr_index=$((index % 2))
    electrode_init_index=$((index / 2 % 2))
    optimizer_index=$((index / 4 % 2))
    dropout_index=$((index / 8 % 3))
    nl_index=$((index / 24 % 3))
    d_model_index=$((index / 72 % 3))

    srun --exclusive -n1 --mem=128G --cpu-bind=cores python ${filename_array[filename_index]} --dtype ${dtype_array[dtype_index]} --optimizer ${optimizer_array[optimizer_index]} \
    --spectrogram ${spectrogram} --binarize_eval ${binarize_eval} --temp_clip_param ${temp_clip_param} --test_chunks_interleaved ${test_chunks_interleaved} --multisubj_eval ${multisubj_eval} \
    --electrode_embedding_init ${electrode_init_array[electrode_init_index]} --dr ${dropout_array[dropout_index]} --dm ${d_model_array[d_model_index]} --n_freq_features ${n_freq_features_array[n_freq_features_index]} \
    --bs ${batch_size_array[batch_size_index]} --lrmax ${lr_array[lr_index]} --lrmin ${lr_array[lr_index]} --weight_decay ${wd_array[wd_index]} --max_gradient_norm $max_gradient_norm \
    --subjects ${subjects_array[subjects_index]} --wait_n_intervals $index --wandb_project bfm --rs ${random_string_array[random_string_index]} --wandb_project bfbofa3 --nl ${nl_array[nl_index]} &
done

wait # Wait for all parallel processes to complete