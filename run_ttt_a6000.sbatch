#!/bin/bash
#SBATCH --job-name=bfm_run          # Name of the job
#SBATCH --ntasks=1             # 8 tasks total
#SBATCH --cpus-per-task=4    # Request 8 CPU cores per GPU
#SBATCH --gres=gpu:RTXA6000:1 # A6000
#SBATCH --mem-per-gpu=64G
#SBATCH -t 12:00:00         # total run time limit (HH:MM:SS) (increased to 24 hours)
#SBATCH --array=1-5      # 14 jobs (108/8 rounded up)
#SBATCH --output r/%A_%a.out # STDOUT
#SBATCH --error r/%A_%a.err # STDERR
#SBATCH -p use-everything

source .venv/bin/activate

embedding_dim_array=(192 96 48 24 12 6 3)
embedding_dim=${embedding_dim_array[$SLURM_ARRAY_TASK_ID]}

echo "embedding_dim: $embedding_dim"

python ttt_unscuffed.py --embedding_dim $embedding_dim --rs 1_a

# python ttt_clip_old_newer.py --two_views_split_eval $(($SLURM_ARRAY_TASK_ID%2)) --two_views_split $(($SLURM_ARRAY_TASK_ID/2%2)) --wandb_project ne_bofa1 --rs XXXXX \
#     --dtype bfloat16 \
#     --optimizer Muon \
#     --spectrogram 1 \
#     --binarize_eval 1 \
#     --temp_clip_param 1 \
#     --test_chunks_interleaved 0 \
#     --multisubj_eval 1 \
#     --electrode_embedding_init $(if [ $(($SLURM_ARRAY_TASK_ID/4)) -eq 0 ]; then echo "coordinates_nograd"; else echo "zeros"; fi) \
#     --dr 0.2 \
#     --dm 192 \
#     --n_freq_features 64 \
#     --bs 100 \
#     --lrmax 0.0015 \
#     --lrmin 0.0015 \
#     --weight_decay 0 \
#     --max_gradient_norm -1 \
#     --subjects 1234567890 \
#     --wait_n_intervals 0 \
#     --nl 10